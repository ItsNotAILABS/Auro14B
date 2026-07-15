"""Approximate nearest-neighbor matching for scalable spectral retrieval.

Implements Locality-Sensitive Hashing (LSH) and MinHash-based approaches
for sub-linear search over large spectral embedding indices.

These complement the brute-force SpectralRetriever with scalable alternatives
suitable for libraries with thousands to millions of fingerprints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer


# =============================================================================
# Locality-Sensitive Hashing (LSH)
# =============================================================================


class SpectralLSH:
    """Locality-Sensitive Hashing for approximate spectral nearest-neighbor search.

    Uses random hyperplane projections (cosine LSH) to hash embeddings into
    buckets, enabling sub-linear query time at the cost of approximate results.

    Args:
        n_tables: Number of hash tables (more = higher recall, more memory).
        n_bits: Number of hash bits per table (more = higher precision, fewer candidates).
        vectorizer: SpectralVectorizer for embedding computation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_tables: int = 8,
        n_bits: int = 16,
        vectorizer: Optional[SpectralVectorizer] = None,
        seed: int = 42,
    ) -> None:
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.vectorizer = vectorizer or SpectralVectorizer()
        self.seed = seed

        self._hyperplanes: List[np.ndarray] = []
        self._tables: List[Dict[str, List[int]]] = []
        self._embeddings: List[np.ndarray] = []
        self._record_ids: List[str] = []
        self._initialized = False

    def _init_hyperplanes(self, dim: int) -> None:
        """Initialize random hyperplanes for hashing."""
        rng = np.random.default_rng(self.seed)
        self._hyperplanes = [
            rng.standard_normal((self.n_bits, dim)).astype(np.float32)
            for _ in range(self.n_tables)
        ]
        self._tables = [{} for _ in range(self.n_tables)]
        self._initialized = True

    def _hash_vector(self, vec: np.ndarray, table_idx: int) -> str:
        """Compute hash key for a vector in a given table."""
        projections = self._hyperplanes[table_idx] @ vec
        bits = (projections > 0).astype(np.uint8)
        return bits.tobytes().hex()

    def index(self, records: Sequence[RecordInput]) -> None:
        """Index records for approximate nearest-neighbor search.

        Args:
            records: Records to add to the index.
        """
        for r in records:
            rec = load_record(r)
            emb = self.vectorizer.transform(rec)

            if not self._initialized:
                self._init_hyperplanes(len(emb))

            idx = len(self._embeddings)
            self._embeddings.append(emb)
            self._record_ids.append(rec.record_id)

            # Insert into all hash tables
            for t in range(self.n_tables):
                key = self._hash_vector(emb, t)
                if key not in self._tables[t]:
                    self._tables[t][key] = []
                self._tables[t][key].append(idx)

    def query(
        self,
        record: RecordInput,
        top_k: int = 5,
        n_candidates: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Find approximate nearest neighbors by LSH.

        Retrieves candidate set from hash buckets, then re-ranks by
        exact Euclidean distance for the final top-k.

        Args:
            record: Query record.
            top_k: Number of results to return.
            n_candidates: Max candidates to re-rank (default: 10 * top_k).

        Returns:
            List of (record_id, distance) tuples sorted by distance ascending.
        """
        if not self._embeddings:
            return []

        query_emb = self.vectorizer.transform(record)

        if not self._initialized:
            return []

        # Collect candidate indices from all tables
        candidate_indices: Set[int] = set()
        for t in range(self.n_tables):
            key = self._hash_vector(query_emb, t)
            if key in self._tables[t]:
                candidate_indices.update(self._tables[t][key])

        # If no candidates found via LSH, fall back to scanning all
        if not candidate_indices:
            candidate_indices = set(range(len(self._embeddings)))

        # Limit candidates for efficiency
        max_candidates = n_candidates or (10 * top_k)
        candidates = sorted(candidate_indices)[:max_candidates]

        # Re-rank by exact distance
        distances: List[Tuple[str, float]] = []
        for idx in candidates:
            dist = float(np.linalg.norm(query_emb - self._embeddings[idx]))
            distances.append((self._record_ids[idx], dist))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    @property
    def size(self) -> int:
        """Number of indexed records."""
        return len(self._embeddings)

    @property
    def table_stats(self) -> Dict[str, int]:
        """Statistics about hash table bucket distribution."""
        total_buckets = sum(len(t) for t in self._tables)
        max_bucket = max(
            (max(len(v) for v in t.values()) if t else 0)
            for t in self._tables
        )
        return {
            "n_tables": self.n_tables,
            "total_buckets": total_buckets,
            "avg_buckets_per_table": total_buckets // max(self.n_tables, 1),
            "max_bucket_size": max_bucket,
            "n_indexed": len(self._embeddings),
        }


# =============================================================================
# MinHash for spectral similarity
# =============================================================================


class SpectralMinHash:
    """MinHash-based similarity estimation for spectral fingerprints.

    Discretizes spectral features into shingles (binary feature sets)
    and uses MinHash signatures for fast Jaccard similarity estimation.
    Useful for coarse-grained deduplication and candidate generation.

    Args:
        n_hashes: Number of hash functions in the MinHash signature.
        n_bands: Number of LSH bands for candidate generation.
        n_shingles: Number of amplitude quantization levels for shingling.
        vectorizer: SpectralVectorizer for initial feature extraction.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_hashes: int = 128,
        n_bands: int = 16,
        n_shingles: int = 64,
        vectorizer: Optional[SpectralVectorizer] = None,
        seed: int = 42,
    ) -> None:
        self.n_hashes = n_hashes
        self.n_bands = n_bands
        self.n_shingles = n_shingles
        self.vectorizer = vectorizer or SpectralVectorizer()
        self.seed = seed

        if n_hashes % n_bands != 0:
            raise ValueError(f"n_hashes ({n_hashes}) must be divisible by n_bands ({n_bands})")

        self._rows_per_band = n_hashes // n_bands
        self._rng = np.random.default_rng(seed)
        # Hash coefficients for universal hashing: h(x) = (a*x + b) mod p
        self._prime = 2**31 - 1
        self._a = self._rng.integers(1, self._prime, size=n_hashes)
        self._b = self._rng.integers(0, self._prime, size=n_hashes)

        self._signatures: List[np.ndarray] = []
        self._record_ids: List[str] = []
        self._band_buckets: List[Dict[str, List[int]]] = [{} for _ in range(n_bands)]

    def _to_shingle_set(self, record: RecordInput) -> Set[int]:
        """Convert a record to a set of shingle IDs."""
        emb = self.vectorizer.transform(record)
        # Quantize embedding values to create shingles
        min_val, max_val = emb.min(), emb.max()
        span = max_val - min_val
        if span < 1e-12:
            return {0}
        normalized = (emb - min_val) / span  # [0, 1]
        quantized = (normalized * (self.n_shingles - 1)).astype(int)
        # Create positional shingles: (position, quantized_value)
        shingles: Set[int] = set()
        for i, q in enumerate(quantized):
            shingles.add(i * self.n_shingles + q)
        return shingles

    def _compute_signature(self, shingle_set: Set[int]) -> np.ndarray:
        """Compute MinHash signature for a shingle set."""
        if not shingle_set:
            return np.full(self.n_hashes, self._prime, dtype=np.int64)

        shingles = np.array(list(shingle_set), dtype=np.int64)
        # For each hash function, compute min hash value
        signature = np.full(self.n_hashes, self._prime, dtype=np.int64)
        for s in shingles:
            hashes = (self._a * s + self._b) % self._prime
            signature = np.minimum(signature, hashes)
        return signature

    def index(self, records: Sequence[RecordInput]) -> None:
        """Index records for MinHash-based similarity search.

        Args:
            records: Records to add to the index.
        """
        for r in records:
            rec = load_record(r)
            shingle_set = self._to_shingle_set(r)
            sig = self._compute_signature(shingle_set)

            idx = len(self._signatures)
            self._signatures.append(sig)
            self._record_ids.append(rec.record_id)

            # LSH banding for candidate generation
            for band in range(self.n_bands):
                start = band * self._rows_per_band
                end = start + self._rows_per_band
                band_key = sig[start:end].tobytes().hex()
                if band_key not in self._band_buckets[band]:
                    self._band_buckets[band][band_key] = []
                self._band_buckets[band][band_key].append(idx)

    def query(self, record: RecordInput, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find records with highest estimated Jaccard similarity.

        Args:
            record: Query record.
            top_k: Number of results to return.

        Returns:
            List of (record_id, estimated_jaccard_similarity) tuples,
            sorted by similarity descending.
        """
        if not self._signatures:
            return []

        shingle_set = self._to_shingle_set(record)
        query_sig = self._compute_signature(shingle_set)

        # Collect candidates from band buckets
        candidate_indices: Set[int] = set()
        for band in range(self.n_bands):
            start = band * self._rows_per_band
            end = start + self._rows_per_band
            band_key = query_sig[start:end].tobytes().hex()
            if band_key in self._band_buckets[band]:
                candidate_indices.update(self._band_buckets[band][band_key])

        # If no candidates via banding, check all
        if not candidate_indices:
            candidate_indices = set(range(len(self._signatures)))

        # Estimate Jaccard similarity for candidates
        results: List[Tuple[str, float]] = []
        for idx in candidate_indices:
            # Jaccard estimate = fraction of hash functions that agree
            agreement = float(np.sum(query_sig == self._signatures[idx])) / self.n_hashes
            results.append((self._record_ids[idx], agreement))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @property
    def size(self) -> int:
        """Number of indexed records."""
        return len(self._signatures)

    def estimate_similarity(self, record_a: RecordInput, record_b: RecordInput) -> float:
        """Estimate Jaccard similarity between two records.

        Args:
            record_a: First record.
            record_b: Second record.

        Returns:
            Estimated Jaccard similarity in [0, 1].
        """
        sig_a = self._compute_signature(self._to_shingle_set(record_a))
        sig_b = self._compute_signature(self._to_shingle_set(record_b))
        return float(np.sum(sig_a == sig_b)) / self.n_hashes


# =============================================================================
# Hybrid coarse + fine search
# =============================================================================


class HybridSpectralSearch:
    """Two-stage coarse-to-fine spectral search.

    Stage 1 (coarse): LSH retrieves a candidate set in sub-linear time.
    Stage 2 (fine): Exact distance re-ranking over candidates.

    This mirrors modern spectral matching pipelines where speed and accuracy
    are balanced through hierarchical retrieval.

    Args:
        n_tables: Number of LSH tables for coarse stage.
        n_bits: Hash bits per table.
        vectorizer: Shared vectorizer for embeddings.
        seed: Random seed.
    """

    def __init__(
        self,
        n_tables: int = 12,
        n_bits: int = 20,
        vectorizer: Optional[SpectralVectorizer] = None,
        seed: int = 42,
    ) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()
        self._lsh = SpectralLSH(
            n_tables=n_tables,
            n_bits=n_bits,
            vectorizer=self.vectorizer,
            seed=seed,
        )
    def index(self, records: Sequence[RecordInput]) -> None:
        """Index records for hybrid search.

        Args:
            records: Records to add to the index.
        """
        self._lsh.index(records)

    def query(
        self,
        record: RecordInput,
        top_k: int = 5,
        coarse_factor: int = 20,
    ) -> List[Tuple[str, float]]:
        """Two-stage coarse-to-fine query.

        Args:
            record: Query record.
            top_k: Number of final results.
            coarse_factor: Multiplier for coarse candidate count.

        Returns:
            List of (record_id, distance) tuples sorted by distance ascending.
        """
        # Coarse stage: get candidates via LSH
        coarse_k = top_k * coarse_factor
        return self._lsh.query(record, top_k=top_k, n_candidates=coarse_k)

    @property
    def size(self) -> int:
        """Number of indexed records."""
        return self._lsh.size
