"""Spectral matching engine with composite scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.io.loaders import RecordInput, load_record
from mesie.matching.metrics import cosine_similarity, log_spectral_distance, spectral_rmse, band_weighted_error
from mesie.processing.interpolate import interpolate_component
from mesie.features.electro_spectral import ElectroSpectralLayer
from mesie.topology.node_mapping import NodeTopologyMapper


@dataclass
class MatchResult:
    """Result of spectral matching between reference and candidate.

    Attributes:
        reference_id: Reference record identifier.
        candidate_id: Candidate record identifier.
        score: Composite match score in [0, 1].
        composite_score: Alias for score.
        metrics: Individual metric values.
        metric_breakdown: Alias for metrics.
        component_scores: Per-component similarity scores.
        warnings: Any warnings generated during matching.
    """

    reference_id: str
    candidate_id: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    component_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Alias for score."""
        return self.score

    @property
    def metric_breakdown(self) -> Dict[str, float]:
        """Alias for metrics."""
        return self.metrics


_FALLBACK_COMPONENT_NAME = "__mesie_fallback_component__"


def _normalize_weights(values: Mapping[str, float], keys) -> Dict[str, float]:
    """Normalize weight values to sum to 1."""
    output = {k: float(values.get(k, 1.0)) for k in keys}
    s = sum(max(v, 0.0) for v in output.values())
    if s <= 0:
        return {k: 1.0 / max(len(output), 1) for k in output}
    return {k: max(v, 0.0) / s for k, v in output.items()}


class SpectralMatcher:
    """Spectral matching engine combining multiple similarity metrics.

    Computes a composite score from cosine similarity, RMSE, log spectral
    distance, band-weighted error, electro-spectral similarity, and
    node topology alignment.

    Args:
        phase_aware: Whether to include phase similarity in scoring.
        band_weights: Frequency band weights as (low, high, weight) tuples.
        channel_weights: Per-component channel weights.
        score_weights: Weights for combining individual metrics.
    """

    def __init__(
        self,
        phase_aware: bool = False,
        band_weights: Optional[Sequence[Tuple[float, float, float]]] = None,
        channel_weights: Optional[Mapping[str, float]] = None,
        score_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.phase_aware = phase_aware
        self.band_weights = band_weights
        self.channel_weights = dict(channel_weights or {})
        self.references: List[MultiElementRecord] = []
        self.mapper = NodeTopologyMapper()
        self.electro = ElectroSpectralLayer()
        self.score_weights = {
            "cosine": 1.0,
            "rmse": 1.0,
            "log_distance": 1.0,
            "band_error": 1.0,
            "coherence": 1.0,
            "electro": 1.0,
            "node": 1.0,
            "phase": 0.5,
            **(score_weights or {}),
        }

    def fit(self, reference_records: Sequence[RecordInput]) -> "SpectralMatcher":
        """Fit matcher with reference records for matching and ranking.

        Args:
            reference_records: Sequence of reference records.

        Returns:
            Self for method chaining.
        """
        self.references = []
        for idx, record in enumerate(reference_records):
            try:
                self.references.append(load_record(record))
            except Exception as exc:
                raise ValueError(f"Failed to load reference record at index {idx}: {exc}") from exc
        return self

    def _common_grid(self, reference: MultiElementRecord, candidate: MultiElementRecord) -> np.ndarray:
        """Compute common frequency grid between two records."""
        min_f = max(
            min(c.frequency[0] for c in reference.components),
            min(c.frequency[0] for c in candidate.components),
        )
        max_f = min(
            max(c.frequency[-1] for c in reference.components),
            max(c.frequency[-1] for c in candidate.components),
        )
        if max_f <= min_f:
            union = np.unique(np.concatenate([
                reference.components[0].frequency,
                candidate.components[0].frequency,
            ]))
            return union
        n = max(len(reference.components[0].frequency), len(candidate.components[0].frequency), 64)
        return np.linspace(min_f, max_f, n)

    def score(self, reference: RecordInput, candidate: RecordInput) -> MatchResult:
        """Compute comprehensive match score between reference and candidate.

        Args:
            reference: Reference record.
            candidate: Candidate record.

        Returns:
            MatchResult with composite score and metric breakdown.
        """
        ref = load_record(reference)
        cand = load_record(candidate)

        freq = self._common_grid(ref, cand)
        ref_map = {c.name: c for c in ref.components}
        cand_map = {c.name: c for c in cand.components}
        common = sorted(set(ref_map) & set(cand_map))
        if not common:
            ref_map[_FALLBACK_COMPONENT_NAME] = next(iter(ref_map.values()))
            cand_map[_FALLBACK_COMPONENT_NAME] = next(iter(cand_map.values()))
            common = [_FALLBACK_COMPONENT_NAME]

        ch_weights = _normalize_weights(self.channel_weights, common)

        ref_mix = np.zeros_like(freq)
        cand_mix = np.zeros_like(freq)
        component_scores: Dict[str, float] = {}
        phase_scores: List[float] = []

        for name in common:
            rc = interpolate_component(ref_map[name], freq)
            cc = interpolate_component(cand_map[name], freq)
            w = ch_weights[name]
            ref_mix += rc.amplitude * w
            cand_mix += cc.amplitude * w
            component_scores[name] = cosine_similarity(rc.amplitude, cc.amplitude)
            if self.phase_aware and rc.phase is not None and cc.phase is not None:
                phase_scores.append(float(np.mean(np.cos(rc.phase - cc.phase))))

        eps = 1e-12
        diff = ref_mix - cand_mix
        abs_diff = np.abs(diff)

        cos_sim = cosine_similarity(ref_mix, cand_mix)
        rmse = spectral_rmse(ref_mix, cand_mix)
        log_dist = log_spectral_distance(ref_mix, cand_mix)
        band_err = band_weighted_error(freq, abs_diff, self.band_weights)
        coherence = float(np.mean(list(component_scores.values()))) if component_scores else 0.0
        electro_sim = 1.0 / (1.0 + self.electro.electro_distance(ref, cand))
        node_align = self.mapper.alignment_score(ref, cand)
        phase_sim = float(np.mean(phase_scores)) if phase_scores else 1.0

        score_terms = {
            "cosine": max(cos_sim, 0.0),
            "rmse": 1.0 / (1.0 + rmse),
            "log_distance": 1.0 / (1.0 + log_dist),
            "band_error": 1.0 / (1.0 + band_err),
            "coherence": max(coherence, 0.0),
            "electro": electro_sim,
            "node": node_align,
            "phase": max(min(phase_sim, 1.0), 0.0),
        }

        wsum = sum(max(float(self.score_weights.get(k, 0.0)), 0.0) for k in score_terms)
        if wsum <= 0:
            final_score = 0.0
        else:
            final_score = sum(
                score_terms[k] * max(float(self.score_weights.get(k, 0.0)), 0.0)
                for k in score_terms
            ) / wsum

        return MatchResult(
            reference_id=ref.record_id,
            candidate_id=cand.record_id,
            score=float(final_score),
            metrics={
                "cosine_similarity": float(cos_sim),
                "rmse": rmse,
                "log_spectral_distance": log_dist,
                "frequency_band_weighted_error": band_err,
                "coherence": coherence,
                "electro_spectral_similarity": electro_sim,
                "node_topology_alignment": node_align,
                "phase_similarity": phase_sim,
            },
            component_scores=component_scores,
        )

    def match(self, candidate_record: RecordInput) -> MatchResult:
        """Match candidate against fitted references, return best result.

        Args:
            candidate_record: Candidate record to match.

        Returns:
            Best MatchResult from all fitted references.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.references:
            raise RuntimeError("SpectralMatcher.fit must be called before match.")
        candidate = load_record(candidate_record)
        results = [self.score(ref, candidate) for ref in self.references]
        return max(results, key=lambda r: r.score)

    def batch_match(self, candidate_records: Sequence[RecordInput]) -> List[MatchResult]:
        """Match a batch of candidates against fitted references.

        Args:
            candidate_records: Sequence of candidate records.

        Returns:
            List of best MatchResult for each candidate.
        """
        return [self.match(c) for c in candidate_records]

    def rank_matches(self, candidate: RecordInput, top_k: int = 10) -> List[MatchResult]:
        """Rank all fitted references against a candidate.

        Args:
            candidate: Candidate record.
            top_k: Number of top results to return.

        Returns:
            List of MatchResults sorted by score descending.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self.references:
            raise RuntimeError("SpectralMatcher.fit must be called before rank_matches.")
        cand = load_record(candidate)
        ranked = sorted(
            (self.score(r, cand) for r in self.references),
            key=lambda x: x.score,
            reverse=True,
        )
        return ranked[:max(int(top_k), 1)]


def match_records(
    reference: RecordInput,
    candidate: RecordInput,
    matcher: Optional[SpectralMatcher] = None,
) -> MatchResult:
    """Match two spectral records and return a comprehensive MatchResult.

    Convenience function for quick matching without explicit matcher setup.

    Args:
        reference: Reference record.
        candidate: Candidate record.
        matcher: Optional pre-configured matcher.

    Returns:
        MatchResult with composite score and metrics.
    """
    m = matcher or SpectralMatcher()
    return m.score(reference, candidate)
