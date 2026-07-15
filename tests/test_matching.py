"""Tests for spectral matching engine."""

import numpy as np
import pytest

from mesie.matching.matcher import SpectralMatcher, MatchResult, match_records
from mesie.matching.metrics import spectral_rmse, spectral_mae, cosine_similarity, log_spectral_distance
from mesie.matching.ranking import rank_candidates


class TestMetrics:
    def test_rmse_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert spectral_rmse(a, a) == pytest.approx(0.0)

    def test_rmse_different(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        assert spectral_rmse(a, b) > 0

    def test_mae_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert spectral_mae(a, a) == pytest.approx(0.0)

    def test_cosine_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_log_distance_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert log_spectral_distance(a, a) == pytest.approx(0.0)


class TestSpectralMatcher:
    def _make_payload(self, record_id, amp):
        return {
            "record_id": record_id,
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0], "amplitude": amp}
            ],
        }

    def test_identical_records_high_score(self):
        payload = self._make_payload("r1", [0.4, 0.8, 0.6, 0.2])
        matcher = SpectralMatcher()
        result = matcher.score(payload, payload)
        assert result.score > 0.99
        assert result.reference_id == "r1"

    def test_different_records_lower_score(self):
        ref = self._make_payload("ref", [0.4, 0.8, 0.6, 0.2])
        cand = self._make_payload("cand", [0.1, 0.1, 0.1, 0.1])
        matcher = SpectralMatcher()
        result = matcher.score(ref, cand)
        assert result.score < 0.99

    def test_match_result_has_metrics(self):
        payload = self._make_payload("r1", [0.4, 0.8, 0.6, 0.2])
        result = match_records(payload, payload)
        assert "cosine_similarity" in result.metrics
        assert "rmse" in result.metrics
        assert result.composite_score == result.score

    def test_fit_and_match(self):
        ref1 = self._make_payload("ref1", [0.4, 0.8, 0.6, 0.2])
        ref2 = self._make_payload("ref2", [0.1, 0.1, 0.1, 0.1])
        cand = self._make_payload("cand", [0.39, 0.79, 0.59, 0.19])
        matcher = SpectralMatcher()
        matcher.fit([ref1, ref2])
        result = matcher.match(cand)
        assert result.reference_id == "ref1"

    def test_rank_matches(self):
        ref1 = self._make_payload("ref1", [0.4, 0.8, 0.6, 0.2])
        ref2 = self._make_payload("ref2", [0.1, 0.1, 0.1, 0.1])
        cand = self._make_payload("cand", [0.4, 0.8, 0.6, 0.2])
        matcher = SpectralMatcher()
        matcher.fit([ref1, ref2])
        results = matcher.rank_matches(cand, top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score


class TestRanking:
    def test_rank_candidates(self):
        ref = {
            "record_id": "ref",
            "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.5, 0.6, 0.7]}],
        }
        candidates = [
            {"record_id": "c1", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.5, 0.6, 0.7]}]},
            {"record_id": "c2", "components": [{"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.1, 0.1, 0.1]}]},
        ]
        results = rank_candidates(ref, candidates, top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score
