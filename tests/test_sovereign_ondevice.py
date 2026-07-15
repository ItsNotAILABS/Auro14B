"""Tests for sovereign on-device SDK components."""

import numpy as np
import pytest

from mesie.sdk.sovereign_ondevice import (
    OnDeviceFingerprintLibrary,
    OnDeviceStreamingPipeline,
    SovereignConfig,
    SpectralFingerprint,
)


def test_add_guard_when_capacity_zero():
    lib = OnDeviceFingerprintLibrary(capacity=0, embedding_dim=4)
    added = lib.add(
        SpectralFingerprint(
            fingerprint_id="fp1",
            embedding=np.ones(4),
            created_at=1.0,
        )
    )
    assert added is False
    assert lib.size == 0


def test_add_validates_and_casts_embedding_shape():
    lib = OnDeviceFingerprintLibrary(capacity=2, embedding_dim=4)
    with pytest.raises(ValueError, match="embedding must have length"):
        lib.add(SpectralFingerprint(fingerprint_id="fp1", embedding=np.ones(3), created_at=1.0))

    fp = SpectralFingerprint(fingerprint_id="fp2", embedding=np.ones(4, dtype=np.float64), created_at=2.0)
    assert lib.add(fp)
    assert fp.embedding.dtype == np.float32


def test_incremental_update_validates_inputs():
    lib = OnDeviceFingerprintLibrary(capacity=2, embedding_dim=4)
    lib.add(SpectralFingerprint(fingerprint_id="fp1", embedding=np.ones(4), created_at=1.0))

    with pytest.raises(ValueError, match="momentum must be in \\[0, 1\\]"):
        lib.incremental_update("fp1", np.ones(4), momentum=1.5)

    with pytest.raises(ValueError, match="embedding must have length"):
        lib.incremental_update("fp1", np.ones(3), momentum=0.5)


def test_streaming_pipeline_validates_threshold_and_window():
    lib = OnDeviceFingerprintLibrary(capacity=2, embedding_dim=4)
    config = SovereignConfig(embedding_dim=4)

    with pytest.raises(ValueError, match="novelty_threshold must be in \\[0, 1\\]"):
        OnDeviceStreamingPipeline(lib, config, novelty_threshold=1.2)

    with pytest.raises(ValueError, match="drift_window must be >= 4"):
        OnDeviceStreamingPipeline(lib, config, drift_window=3)
