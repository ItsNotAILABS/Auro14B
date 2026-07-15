"""Tests for the Python↔Julia bridge (mesie.polyglot.julia_bridge).

Tests the bridge logic using mocked subprocess calls so they run
without Julia installed. Integration tests that require Julia are
marked with pytest.mark.skipif.
"""

import json
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mesie.polyglot.julia_bridge import (
    JuliaBridge,
    JuliaBridgeError,
    JuliaCallBackend,
    JuliaMatchResult,
    JuliaValidationResult,
    SubprocessBackend,
)


# --- Test fixtures ---


@pytest.fixture
def sample_record():
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


@pytest.fixture
def component_record():
    return {
        "components": [
            {
                "name": "test_comp",
                "frequency": [10.0, 20.0, 30.0, 40.0],
                "amplitude": [1.0, 2.0, 3.0, 4.0],
            }
        ]
    }


# --- SubprocessBackend tests ---


class TestSubprocessBackend:
    def test_call_validate(self, sample_record):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "ok": True,
            "data": {
                "is_valid": True,
                "level": 5,
                "errors": [],
                "warnings": [],
                "n_points": 8,
                "runtime": "julia",
            },
        })

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            backend = SubprocessBackend(julia_exec="julia")
            result = backend.call("validate", {"record": sample_record})

        assert result["ok"] is True
        assert result["data"]["is_valid"] is True
        mock_run.assert_called_once()

    def test_call_julia_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            backend = SubprocessBackend(julia_exec="julia_nonexistent")
            with pytest.raises(JuliaBridgeError, match="Julia executable not found"):
                backend.call("health", {})

    def test_call_timeout(self):
        import subprocess as sp

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("julia", 30)):
            backend = SubprocessBackend()
            with pytest.raises(JuliaBridgeError, match="timed out"):
                backend.call("health", {})

    def test_call_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: something went wrong"

        with patch("subprocess.run", return_value=mock_result):
            backend = SubprocessBackend()
            with pytest.raises(JuliaBridgeError, match="Julia process failed"):
                backend.call("health", {})

    def test_call_invalid_json(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not json at all"

        with patch("subprocess.run", return_value=mock_result):
            backend = SubprocessBackend()
            with pytest.raises(JuliaBridgeError, match="Invalid JSON"):
                backend.call("health", {})


# --- JuliaBridge high-level tests ---


class TestJuliaBridge:
    def _mock_bridge(self):
        """Create a bridge with mocked backend."""
        bridge = JuliaBridge(backend="subprocess")
        bridge._backend = MagicMock()
        return bridge

    def test_health(self):
        bridge = self._mock_bridge()
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {"status": "ok", "runtime": "julia", "version": "1.10.0", "threads": 4},
        }
        result = bridge.health()
        assert result["status"] == "ok"
        assert result["runtime"] == "julia"

    def test_validate(self, sample_record):
        bridge = self._mock_bridge()
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {
                "is_valid": True,
                "level": 5,
                "errors": [],
                "warnings": [],
                "n_points": 8,
                "runtime": "julia",
            },
        }
        result = bridge.validate(sample_record)
        assert isinstance(result, JuliaValidationResult)
        assert result.is_valid is True
        assert result.level == 5
        assert result.n_points == 8

    def test_validate_invalid(self):
        bridge = self._mock_bridge()
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {
                "is_valid": False,
                "level": 2,
                "errors": ["negative amplitudes detected"],
                "warnings": [],
                "n_points": 4,
                "runtime": "julia",
            },
        }
        result = bridge.validate({"frequency": [1, 2, 3, 4], "amplitude": [-1, 2, 3, 4]})
        assert result.is_valid is False
        assert "negative amplitudes" in result.errors[0]

    def test_match(self, sample_record, sample_record_b):
        bridge = self._mock_bridge()
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {
                "composite_score": 0.92,
                "metrics": {"cosine": 0.98, "rmse": 0.1, "pearson": 0.95},
                "n_compared": 8,
                "runtime": "julia",
            },
        }
        result = bridge.match(sample_record, sample_record_b)
        assert isinstance(result, JuliaMatchResult)
        assert result.composite_score == 0.92
        assert result.cosine == 0.98
        assert result.n_compared == 8

    def test_embed(self, sample_record):
        bridge = self._mock_bridge()
        embedding = [0.1] * 32  # 8 bands * 4 features
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {"embedding": embedding, "runtime": "julia"},
        }
        result = bridge.embed(sample_record)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32,)
        assert result.dtype == np.float64

    def test_fingerprint(self, sample_record):
        bridge = self._mock_bridge()
        fp = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {"fingerprint": fp, "runtime": "julia"},
        }
        result = bridge.fingerprint(sample_record)
        assert isinstance(result, np.ndarray)
        assert result.shape == (16,)
        assert result.dtype == np.uint8

    def test_batch_validate(self, sample_record):
        bridge = self._mock_bridge()
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {
                "is_valid": True, "level": 5,
                "errors": [], "warnings": [],
                "n_points": 8, "runtime": "julia",
            },
        }
        results = bridge.batch_validate([sample_record, sample_record])
        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_batch_embed(self, sample_record):
        bridge = self._mock_bridge()
        bridge._backend.call.return_value = {
            "ok": True,
            "data": {"embedding": [0.1] * 32, "runtime": "julia"},
        }
        result = bridge.batch_embed([sample_record, sample_record])
        assert result.shape == (2, 32)

    def test_backend_auto_selects_subprocess_when_no_juliacall(self):
        with patch.dict("sys.modules", {"juliacall": None}):
            bridge = JuliaBridge(backend="auto")
            assert bridge.backend_name == "subprocess"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            JuliaBridge(backend="invalid")


# --- Integration tests (require Julia installed) ---


@pytest.mark.skipif(
    shutil.which("julia") is None,
    reason="Julia not installed",
)
class TestJuliaBridgeIntegration:
    """Integration tests that actually call the Julia subprocess."""

    def test_health_subprocess(self):
        bridge = JuliaBridge(backend="subprocess")
        result = bridge.health()
        assert result["status"] == "ok"
        assert result["runtime"] == "julia"

    def test_validate_subprocess(self, sample_record):
        bridge = JuliaBridge(backend="subprocess")
        result = bridge.validate(sample_record)
        assert result.is_valid is True

    def test_match_subprocess(self, sample_record, sample_record_b):
        bridge = JuliaBridge(backend="subprocess")
        result = bridge.match(sample_record, sample_record_b)
        assert 0.0 <= result.composite_score <= 1.0
        assert result.n_compared == 8
