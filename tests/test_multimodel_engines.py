"""Tests for multi-model Julia engines (mesie.engines.multimodel_julia_engine).

Tests use mocked Julia bridge to verify engine logic without requiring Julia.
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from mesie.engines.multimodel_julia_engine import (
    MultiModelEmbeddingEngine,
    MultiModelFingerprintEngine,
    MultiModelMatchingEngine,
    MultiModelValidationEngine,
)
from mesie.internal_api.messages import MessageEnvelope


# --- Helpers ---


def _msg(target: str, action: str, payload: dict) -> MessageEnvelope:
    from mesie.internal_api.messages import MessageTopic
    return MessageEnvelope(
        topic=MessageTopic.ENGINE_REQUEST,
        source="test",
        target=target,
        action=action,
        payload=payload,
    )


@pytest.fixture
def sample_record():
    """A minimal record dict loadable by mesie."""
    return {
        "frequency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "amplitude": [0.1, 0.5, 0.9, 0.7, 0.3, 0.2, 0.4, 0.6],
    }


@pytest.fixture
def sample_record_b():
    return {
        "frequency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "amplitude": [0.2, 0.4, 0.8, 0.6, 0.4, 0.3, 0.5, 0.7],
    }


# --- MultiModelValidationEngine ---


class TestMultiModelValidationEngine:
    def _engine_with_mock_julia(self):
        engine = MultiModelValidationEngine(julia_backend="subprocess")
        engine._julia = MagicMock()
        return engine

    def test_health(self):
        engine = self._engine_with_mock_julia()
        engine._julia.health.return_value = {"status": "ok"}
        msg = _msg("multimodel_validation", "health", {})
        resp = engine.handle(msg)
        assert resp.ok
        assert "python" in resp.data["models"]
        assert "julia" in resp.data["models"]

    def test_validate_julia(self, sample_record):
        engine = self._engine_with_mock_julia()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.level = 5
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.n_points = 8
        engine._julia.validate.return_value = mock_result

        msg = _msg("multimodel_validation", "validate_julia", {"record": sample_record})
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["is_valid"] is True
        assert resp.data["runtime"] == "julia"

    def test_validate_consensus(self, sample_record):
        engine = self._engine_with_mock_julia()
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.level = 5
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.n_points = 8
        engine._julia.validate.return_value = mock_result

        msg = _msg("multimodel_validation", "validate_consensus", {"record": sample_record})
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["consensus"] is True
        assert resp.data["confidence"] == 1.0
        assert "python" in resp.data["models_used"]
        assert "julia" in resp.data["models_used"]

    def test_ignores_other_targets(self, sample_record):
        engine = self._engine_with_mock_julia()
        msg = _msg("other_engine", "validate", {"record": sample_record})
        assert engine.handle(msg) is None

    def test_unknown_action(self):
        engine = self._engine_with_mock_julia()
        msg = _msg("multimodel_validation", "unknown_action", {})
        resp = engine.handle(msg)
        assert not resp.ok


# --- MultiModelMatchingEngine ---


class TestMultiModelMatchingEngine:
    def _engine_with_mock_julia(self):
        engine = MultiModelMatchingEngine(julia_backend="subprocess")
        engine._julia = MagicMock()
        return engine

    def test_health(self):
        engine = self._engine_with_mock_julia()
        msg = _msg("multimodel_matching", "health", {})
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["fusion_weights"] == {"python": 0.5, "julia": 0.5}

    def test_match_julia(self, sample_record, sample_record_b):
        engine = self._engine_with_mock_julia()
        mock_result = MagicMock()
        mock_result.composite_score = 0.92
        mock_result.cosine = 0.98
        mock_result.rmse = 0.1
        mock_result.pearson = 0.95
        mock_result.n_compared = 8
        engine._julia.match.return_value = mock_result

        msg = _msg("multimodel_matching", "match_julia", {
            "record_a": sample_record, "record_b": sample_record_b,
        })
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["composite_score"] == 0.92
        assert resp.data["runtime"] == "julia"

    def test_match_fused(self, sample_record, sample_record_b):
        engine = self._engine_with_mock_julia()
        mock_result = MagicMock()
        mock_result.composite_score = 0.90
        mock_result.cosine = 0.95
        mock_result.rmse = 0.15
        mock_result.pearson = 0.88
        mock_result.n_compared = 8
        engine._julia.match.return_value = mock_result

        msg = _msg("multimodel_matching", "match_fused", {
            "record_a": sample_record, "record_b": sample_record_b,
        })
        resp = engine.handle(msg)
        assert resp.ok
        assert "fused_score" in resp.data
        assert "python" in resp.data["models_used"]
        assert "julia" in resp.data["models_used"]


# --- MultiModelEmbeddingEngine ---


class TestMultiModelEmbeddingEngine:
    def _engine_with_mock_julia(self):
        engine = MultiModelEmbeddingEngine(julia_backend="subprocess", n_bands=8)
        engine._julia = MagicMock()
        return engine

    def test_health(self):
        engine = self._engine_with_mock_julia()
        msg = _msg("multimodel_embedding", "health", {})
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["fusion_mode"] == "concatenate"

    def test_embed_julia(self, sample_record):
        engine = self._engine_with_mock_julia()
        engine._julia.embed.return_value = np.ones(32)

        msg = _msg("multimodel_embedding", "embed_julia", {"record": sample_record})
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["dim"] == 32
        assert resp.data["runtime"] == "julia"

    def test_embed_fused_concatenate(self, sample_record):
        engine = self._engine_with_mock_julia()
        engine._julia.embed.return_value = np.ones(32) * 0.5

        msg = _msg("multimodel_embedding", "embed_fused", {"record": sample_record})
        resp = engine.handle(msg)
        assert resp.ok
        # Concatenated: python_dim + julia_dim
        assert resp.data["dim"] > 32
        assert "python" in resp.data["models_used"]
        assert "julia" in resp.data["models_used"]

    def test_batch_embed(self, sample_record):
        engine = self._engine_with_mock_julia()
        engine._julia.embed.return_value = np.ones(32) * 0.5

        msg = _msg("multimodel_embedding", "batch_embed", {
            "records": [sample_record, sample_record],
        })
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["count"] == 2


# --- MultiModelFingerprintEngine ---


class TestMultiModelFingerprintEngine:
    def _engine_with_mock_julia(self):
        engine = MultiModelFingerprintEngine(julia_backend="subprocess", resolution=16)
        engine._julia = MagicMock()
        return engine

    def test_health(self):
        engine = self._engine_with_mock_julia()
        msg = _msg("multimodel_fingerprint", "health", {})
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["resolution"] == 16

    def test_fingerprint(self, sample_record):
        engine = self._engine_with_mock_julia()
        engine._julia.fingerprint.return_value = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

        msg = _msg("multimodel_fingerprint", "fingerprint", {"record": sample_record})
        resp = engine.handle(msg)
        assert resp.ok
        assert len(resp.data["fingerprint"]) == 16
        assert resp.data["runtime"] == "julia"

    def test_hamming_distance(self, sample_record):
        engine = self._engine_with_mock_julia()
        # Pre-index two fingerprints
        engine._index["rec_a"] = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        engine._index["rec_b"] = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=np.uint8)

        msg = _msg("multimodel_fingerprint", "hamming_distance", {
            "record_id_a": "rec_a", "record_id_b": "rec_b",
        })
        resp = engine.handle(msg)
        assert resp.ok
        assert resp.data["hamming_distance"] == 2  # 2 bits differ
        assert 0.0 <= resp.data["similarity"] <= 1.0

    def test_hamming_missing_record(self):
        engine = self._engine_with_mock_julia()
        msg = _msg("multimodel_fingerprint", "hamming_distance", {
            "record_id_a": "missing_a", "record_id_b": "missing_b",
        })
        resp = engine.handle(msg)
        assert not resp.ok


# --- Registry integration ---


class TestRegistryIntegration:
    def test_multimodel_engines_registered(self):
        from mesie.engines.registry import build_default_registry
        registry = build_default_registry()
        names = registry.names()
        assert "multimodel_validation" in names
        assert "multimodel_matching" in names
        assert "multimodel_embedding" in names
        assert "multimodel_fingerprint" in names
