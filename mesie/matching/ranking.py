"""Ranking and top-k retrieval for spectral matching."""

from __future__ import annotations

from typing import List, Sequence

from mesie.io.loaders import RecordInput
from mesie.matching.matcher import MatchResult, SpectralMatcher


def rank_candidates(
    reference: RecordInput,
    candidates: Sequence[RecordInput],
    top_k: int = 10,
    matcher: SpectralMatcher = None,
) -> List[MatchResult]:
    """Rank candidate records against a single reference.

    Args:
        reference: Reference record to compare against.
        candidates: Sequence of candidate records.
        top_k: Number of top results to return.
        matcher: Optional pre-configured matcher instance.

    Returns:
        List of MatchResults sorted by score descending.
    """
    m = matcher or SpectralMatcher()
    results = [m.score(reference, c) for c in candidates]
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    return ranked[:max(int(top_k), 1)]
