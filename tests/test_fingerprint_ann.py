"""Tests for TF → salient → LSH → ANN fingerprint pipeline."""

import numpy as np
import pytest

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.embeddings import ANNIndex, LSHHasher, SpectralFingerprintPipeline
from mesie.signal import SalientFeatureExtractor, TimeFrequencyTransform


def _record(rid: str, scale: float = 1.0) -> MultiElementRecord:
    f = np.linspace(0.5, 20.0, 64)
    a = scale * (np.exp(-((f - 5.0) ** 2) / 3.0) + 0.1)
    return MultiElementRecord(
        record_id=rid,
        components=[SpectralComponent(name="c", frequency=f, amplitude=a)],
    )


class TestTimeFrequency:
    def test_pseudo_tf_from_record(self):
        tf = TimeFrequencyTransform().from_record(_record("tf-1"))
        assert tf.matrix.shape[0] > 0
        assert tf.method == "spectral_pseudo_tf"

    def test_stft_from_series(self):
        sr = 256.0
        t = np.arange(512) / sr
        x = np.sin(2 * np.pi * 10 * t)
        tf = TimeFrequencyTransform().from_time_series(x, sr)
        assert tf.matrix.ndim == 2


class TestSalient:
    def test_extract_points(self):
        tf = TimeFrequencyTransform().from_record(_record("s-1"))
        sal = SalientFeatureExtractor(max_points=8).extract(tf)
        assert sal.n_points >= 1
        assert len(sal.feature_vector) == 8 * 4


class TestLSHANN:
    def test_lsh_bucket_stable(self):
        h = LSHHasher(dim=16, n_planes=8, seed=1)
        v = np.ones(16)
        s1 = h.hash(v)
        s2 = h.hash(v)
        assert s1.bucket_key == s2.bucket_key

    def test_ann_finds_neighbor(self):
        idx = ANNIndex(use_lsh=True, lsh_planes=12)
        a = np.zeros(20)
        a[0] = 1.0
        b = np.zeros(20)
        b[0] = 0.99
        c = np.zeros(20)
        c[-1] = 1.0
        idx.add("a", a)
        idx.add("b", b)
        idx.add("c", c)
        hits = idx.query(a, top_k=2)
        assert hits[0].item_id in ("a", "b")
        assert hits[0].item_id != "c" or hits[1].item_id in ("a", "b")


class TestFingerprintPipeline:
    def test_index_and_query(self):
        pipe = SpectralFingerprintPipeline()
        r1 = _record("fp-1", 1.0)
        r2 = _record("fp-2", 1.05)
        r3 = _record("fp-3", 0.3)
        pipe.index_records([r1, r2, r3])
        hits = pipe.query(r1, top_k=2)
        assert len(hits) >= 1
        assert hits[0].item_id in ("fp-1", "fp-2")

    def test_fingerprint_engine_bus(self):
        from mesie.internal_api import InternalRouter

        router = InternalRouter()
        r = router.call("fingerprint", "index", {"records": [_record("e-1"), _record("e-2")]})
        assert r.ok
        q = router.call("fingerprint", "query", {"record": _record("e-1"), "top_k": 2})
        assert q.ok
        assert len(q.data["hits"]) >= 1