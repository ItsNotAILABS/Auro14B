"""Spectral matching and scoring."""

from mesie.matching.matcher import SpectralMatcher, MatchResult, match_records
from mesie.matching.metrics import spectral_rmse, spectral_mae, cosine_similarity, log_spectral_distance
from mesie.matching.ranking import rank_candidates
from mesie.matching.approximate import SpectralLSH, SpectralMinHash, HybridSpectralSearch

__all__ = [
    "HybridSpectralSearch",
    "MatchResult",
    "SpectralLSH",
    "SpectralMatcher",
    "SpectralMinHash",
    "cosine_similarity",
    "log_spectral_distance",
    "match_records",
    "rank_candidates",
    "spectral_mae",
    "spectral_rmse",
]
