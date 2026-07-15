"""Cross-domain spectral transfer engine.

Provides the high-level interface for transferring spectral knowledge
between domains using CORAL, MMD, and domain-invariant normalization.
Enables a model trained on one spectral domain (e.g., earthquake harmonics)
to generalize to another (e.g., bridge vibration anomalies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.corpora.base import CorpusRecord, SpectralCorpus
from mesie.corpora.domains import DOMAIN_REGISTRY, SpectralDomain
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.transfer.alignment import CORAL, MMD, DomainInvariantNormalizer


@dataclass
class TransferResult:
    """Result of a cross-domain transfer operation.

    Attributes:
        source_domain: Source domain key.
        target_domain: Target domain key.
        aligned_embeddings: Source embeddings aligned to target space.
        mmd_before: MMD distance before alignment.
        mmd_after: MMD distance after alignment.
        coral_distance: CORAL distance between domains.
        transfer_gain: Improvement ratio (mmd_before / mmd_after).
        n_source_samples: Number of source samples.
        n_target_samples: Number of target samples.
    """

    source_domain: str
    target_domain: str
    aligned_embeddings: np.ndarray
    mmd_before: float
    mmd_after: float
    coral_distance: float
    transfer_gain: float
    n_source_samples: int
    n_target_samples: int


class CrossDomainTransferEngine:
    """Engine for cross-domain spectral transfer learning.

    Combines CORAL alignment, MMD measurement, and domain-invariant
    normalization to enable spectral knowledge transfer across heterogeneous
    domains. This is the core mechanism for building a unified spectral
    latent space.

    The engine supports:
        - Pairwise domain alignment (source → target)
        - Multi-domain invariant space construction
        - Domain divergence measurement
        - Transfer feasibility assessment

    Example:
        >>> engine = CrossDomainTransferEngine()
        >>> engine.register_corpus(seismic_corpus)
        >>> engine.register_corpus(vibration_corpus)
        >>> result = engine.transfer('seismic', 'vibration')
        >>> print(f"Transfer gain: {result.transfer_gain:.2f}x")

    Cross-domain transfer examples:
        - earthquake harmonics → bridge vibration anomalies
        - EEG oscillations → audio resonance detection
        - seismic P-waves → financial cycle detection
    """

    def __init__(
        self,
        vectorizer: Optional[SpectralVectorizer] = None,
        coral_regularization: float = 1e-6,
        mmd_kernel: str = "rbf",
        mmd_bandwidth: Optional[float] = None,
    ) -> None:
        """Initialize the cross-domain transfer engine.

        Args:
            vectorizer: Vectorizer for embedding computation.
            coral_regularization: CORAL regularization parameter.
            mmd_kernel: MMD kernel type ('rbf' or 'linear').
            mmd_bandwidth: MMD bandwidth (None = median heuristic).
        """
        self.vectorizer = vectorizer or SpectralVectorizer()
        self._coral = CORAL(regularization=coral_regularization)
        self._mmd = MMD(kernel=mmd_kernel, bandwidth=mmd_bandwidth)
        self._normalizer = DomainInvariantNormalizer(whiten=True)

        self._corpora: Dict[str, SpectralCorpus] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

    def register_corpus(self, corpus: SpectralCorpus) -> None:
        """Register a domain corpus for transfer.

        Args:
            corpus: A SpectralCorpus with loaded records.
        """
        self._corpora[corpus.domain_key] = corpus
        # Compute embeddings for all records in the corpus
        embeddings = []
        for record in corpus:
            emb = self.vectorizer.transform(record.record)
            embeddings.append(emb)
        if embeddings:
            self._embeddings[corpus.domain_key] = np.vstack(embeddings)
        else:
            self._embeddings[corpus.domain_key] = np.empty(
                (0, self.vectorizer.embedding_dim)
            )

    def transfer(
        self,
        source_domain: str,
        target_domain: str,
    ) -> TransferResult:
        """Perform cross-domain transfer from source to target.

        Aligns source domain embeddings to the target domain space using
        CORAL, and measures transfer quality using MMD.

        Args:
            source_domain: Source domain key.
            target_domain: Target domain key.

        Returns:
            TransferResult with aligned embeddings and metrics.

        Raises:
            ValueError: If either domain is not registered.
        """
        if source_domain not in self._embeddings:
            raise ValueError(
                f"Source domain '{source_domain}' not registered. "
                f"Available: {list(self._embeddings.keys())}"
            )
        if target_domain not in self._embeddings:
            raise ValueError(
                f"Target domain '{target_domain}' not registered. "
                f"Available: {list(self._embeddings.keys())}"
            )

        source_emb = self._embeddings[source_domain]
        target_emb = self._embeddings[target_domain]

        # Measure divergence before alignment
        mmd_before = self._mmd.compute(source_emb, target_emb)
        coral_dist = self._coral.coral_distance(source_emb, target_emb)

        # Perform CORAL alignment
        aligned = self._coral.fit_transform(source_emb, target_emb)

        # Measure divergence after alignment
        mmd_after = self._mmd.compute(aligned, target_emb)

        # Transfer gain
        transfer_gain = mmd_before / max(mmd_after, 1e-12)

        return TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            aligned_embeddings=aligned,
            mmd_before=mmd_before,
            mmd_after=mmd_after,
            coral_distance=coral_dist,
            transfer_gain=transfer_gain,
            n_source_samples=source_emb.shape[0],
            n_target_samples=target_emb.shape[0],
        )

    def build_unified_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build a unified domain-invariant spectral latent space.

        Combines all registered domain embeddings and applies
        domain-invariant normalization to create a shared space.

        Returns:
            Tuple of (unified_embeddings, domain_labels) where
            domain_labels is an array of domain keys per sample.

        Raises:
            RuntimeError: If fewer than 2 domains are registered.
        """
        if len(self._embeddings) < 2:
            raise RuntimeError(
                "At least 2 domains must be registered to build a unified space."
            )

        all_embeddings = []
        all_labels = []

        for domain_key, emb in self._embeddings.items():
            all_embeddings.append(emb)
            all_labels.extend([domain_key] * emb.shape[0])

        combined = np.vstack(all_embeddings)
        labels = np.array(all_labels)

        # Apply domain-invariant normalization
        unified = self._normalizer.fit_transform(combined, labels)
        return unified, labels

    def domain_divergence_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise domain divergence matrix.

        Returns:
            Nested dict of domain → domain → MMD distance.
        """
        domains = list(self._embeddings.keys())
        matrix: Dict[str, Dict[str, float]] = {}

        for i, d1 in enumerate(domains):
            matrix[d1] = {}
            for j, d2 in enumerate(domains):
                if i == j:
                    matrix[d1][d2] = 0.0
                elif j < i and d2 in matrix and d1 in matrix[d2]:
                    matrix[d1][d2] = matrix[d2][d1]
                else:
                    matrix[d1][d2] = self._mmd.domain_divergence(
                        self._embeddings[d1], self._embeddings[d2]
                    )

        return matrix

    def assess_transfer_feasibility(
        self,
        source_domain: str,
        target_domain: str,
    ) -> Dict[str, object]:
        """Assess whether transfer between domains is feasible.

        Uses MMD distance and CORAL distance to determine if meaningful
        transfer is possible between source and target domains.

        Args:
            source_domain: Source domain key.
            target_domain: Target domain key.

        Returns:
            Dictionary with feasibility metrics and recommendation.
        """
        if source_domain not in self._embeddings:
            raise ValueError(f"Source domain '{source_domain}' not registered.")
        if target_domain not in self._embeddings:
            raise ValueError(f"Target domain '{target_domain}' not registered.")

        source_emb = self._embeddings[source_domain]
        target_emb = self._embeddings[target_domain]

        mmd_dist = self._mmd.domain_divergence(source_emb, target_emb)
        coral_dist = self._coral.coral_distance(source_emb, target_emb)

        # Heuristic thresholds for feasibility
        if mmd_dist < 0.1:
            feasibility = "high"
            recommendation = "Domains are closely aligned; direct transfer likely effective."
        elif mmd_dist < 0.5:
            feasibility = "medium"
            recommendation = "Moderate divergence; CORAL alignment recommended."
        else:
            feasibility = "low"
            recommendation = (
                "High divergence; consider intermediate domain bridging "
                "or larger training corpus."
            )

        return {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "mmd_distance": mmd_dist,
            "coral_distance": coral_dist,
            "feasibility": feasibility,
            "recommendation": recommendation,
            "n_source": source_emb.shape[0],
            "n_target": target_emb.shape[0],
        }

    @property
    def registered_domains(self) -> List[str]:
        """List of registered domain keys."""
        return list(self._embeddings.keys())
