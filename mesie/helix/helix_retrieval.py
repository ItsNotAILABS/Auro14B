"""Helix retrieval — nearest-neighbor search in helical vector space.

Provides retrieval capabilities that operate natively in helix coordinates,
supporting phase-aware search, coherence-weighted ranking, and arc-distance
based similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.helix.vector_helix import VectorHelix, HelixConfig, HelixNode


@dataclass
class HelixSearchResult:
    """Result from a helix-based search operation.

    Args:
        node: The matched HelixNode.
        distance: Distance to the query in helix space.
        phase_distance: Angular distance to the query.
        coherence_score: Coherence-weighted relevance score.
    """

    node: HelixNode
    distance: float
    phase_distance: float
    coherence_score: float


class HelixRetriever:
    """Retrieval engine operating in helix vector space.

    Indexes spectral records on a Vector Helix and supports
    nearest-neighbor search using helix-native distance metrics.

    Args:
        helix: VectorHelix instance to search within.
        coherence_weight: Weight for coherence in ranking.
    """

    def __init__(
        self,
        helix: Optional[VectorHelix] = None,
        coherence_weight: float = 0.3,
    ) -> None:
        self._helix = helix or VectorHelix()
        self.coherence_weight = coherence_weight

    @property
    def helix(self) -> VectorHelix:
        """Underlying Vector Helix."""
        return self._helix

    def index(self, records: Sequence[RecordInput]) -> int:
        """Index records onto the helix.

        Args:
            records: Records to index.

        Returns:
            Number of nodes inserted.
        """
        nodes = self._helix.insert_batch(records)
        return len(nodes)

    def search(
        self,
        query: RecordInput,
        top_k: int = 5,
        phase_tolerance: Optional[float] = None,
        min_coherence: Optional[float] = None,
    ) -> List[HelixSearchResult]:
        """Search the helix for records similar to query.

        Uses a combination of embedding distance, phase proximity,
        and coherence weighting for ranking.

        Args:
            query: Query record.
            top_k: Number of results to return.
            phase_tolerance: Optional phase filter (radians).
            min_coherence: Optional minimum coherence filter.

        Returns:
            Ranked list of HelixSearchResult.
        """
        rec = load_record(query)
        query_embedding = self._helix._vectorizer.transform(rec)
        query_phase = self._helix._compute_phase(query_embedding)

        candidates = list(self._helix.nodes)

        # Apply coherence filter
        if min_coherence is not None:
            candidates = [n for n in candidates if n.coherence >= min_coherence]

        # Apply phase filter
        if phase_tolerance is not None:
            filtered = []
            for n in candidates:
                diff = abs(n.phase - query_phase)
                diff = min(diff, 2 * np.pi - diff)
                if diff <= phase_tolerance:
                    filtered.append(n)
            candidates = filtered

        if not candidates:
            return []

        # Compute distances and scores
        results: List[HelixSearchResult] = []
        for node in candidates:
            # Embedding distance
            emb_dist = float(np.linalg.norm(query_embedding - node.embedding))

            # Phase distance
            phase_diff = abs(node.phase - query_phase)
            phase_diff = min(phase_diff, 2 * np.pi - phase_diff)

            # Coherence-weighted score (lower is better)
            coherence_bonus = (1.0 - node.coherence) * self.coherence_weight
            combined_distance = emb_dist + coherence_bonus

            results.append(HelixSearchResult(
                node=node,
                distance=combined_distance,
                phase_distance=phase_diff,
                coherence_score=node.coherence,
            ))

        # Sort by combined distance
        results.sort(key=lambda r: r.distance)
        return results[:top_k]

    def search_by_phase_range(
        self,
        min_phase: float,
        max_phase: float,
        min_coherence: float = 0.0,
    ) -> List[HelixSearchResult]:
        """Search for nodes within a specific phase range.

        Args:
            min_phase: Minimum phase in radians.
            max_phase: Maximum phase in radians.
            min_coherence: Minimum coherence filter.

        Returns:
            List of matching results sorted by coherence.
        """
        candidates = self._helix.query_by_phase(
            target_phase=(min_phase + max_phase) / 2.0,
            tolerance=(max_phase - min_phase) / 2.0,
        )

        if min_coherence > 0:
            candidates = [n for n in candidates if n.coherence >= min_coherence]

        results = [
            HelixSearchResult(
                node=n,
                distance=0.0,
                phase_distance=0.0,
                coherence_score=n.coherence,
            )
            for n in candidates
        ]
        results.sort(key=lambda r: r.coherence_score, reverse=True)
        return results

    @property
    def index_size(self) -> int:
        """Number of indexed nodes."""
        return self._helix.size
