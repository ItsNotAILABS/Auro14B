"""Tests for neural spectral encoders and approximate matching."""

import numpy as np
import pytest

from mesie.embeddings.neural import NeuralSpectralEncoder, _NumpyAutoencoderFallback, _record_to_signal
from mesie.matching.approximate import SpectralLSH, SpectralMinHash, HybridSpectralSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {"record_id": "eq_1", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.01, 0.08, 0.15, 0.12, 0.04, 0.01]}]},
    {"record_id": "eq_2", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.02, 0.09, 0.18, 0.10, 0.03, 0.01]}]},
    {"record_id": "vib_1", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.001, 0.002, 0.01, 0.5, 0.8, 0.3]}]},
    {"record_id": "vib_2", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.002, 0.003, 0.015, 0.55, 0.75, 0.28]}]},
    {"record_id": "noise", "components": [{"name": "h", "frequency": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], "amplitude": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}]},
]


# ---------------------------------------------------------------------------
# NeuralSpectralEncoder tests
# ---------------------------------------------------------------------------


class TestNeuralSpectralEncoder:
    """Tests for NeuralSpectralEncoder."""

    def test_autoencoder_mode_produces_embedding(self):
        encoder = NeuralSpectralEncoder(mode="autoencoder", embedding_dim=32)
        emb = encoder.encode(SAMPLE_RECORDS[0])
        assert emb.shape == (32,)
        assert not np.all(emb == 0)

    def test_cnn_mode_produces_embedding(self):
        encoder = NeuralSpectralEncoder(mode="cnn", embedding_dim=64)
        emb = encoder.encode(SAMPLE_RECORDS[0])
        assert emb.shape == (64,)

    def test_batch_encode(self):
        encoder = NeuralSpectralEncoder(mode="autoencoder", embedding_dim=16)
        embs = encoder.encode_batch(SAMPLE_RECORDS)
        assert embs.shape == (5, 16)

    def test_deterministic_encoding(self):
        enc1 = NeuralSpectralEncoder(mode="autoencoder", seed=123)
        enc2 = NeuralSpectralEncoder(mode="autoencoder", seed=123)
        e1 = enc1.encode(SAMPLE_RECORDS[0])
        e2 = enc2.encode(SAMPLE_RECORDS[0])
        np.testing.assert_array_almost_equal(e1, e2)

    def test_different_records_produce_different_embeddings(self):
        encoder = NeuralSpectralEncoder(mode="autoencoder", embedding_dim=32)
        e1 = encoder.encode(SAMPLE_RECORDS[0])
        e2 = encoder.encode(SAMPLE_RECORDS[2])
        assert not np.allclose(e1, e2)

    def test_fit_reduces_loss(self):
        encoder = NeuralSpectralEncoder(mode="autoencoder", input_length=64, embedding_dim=16)
        losses = encoder.fit(SAMPLE_RECORDS, epochs=20, lr=1e-2)
        assert len(losses) == 20
        assert losses[-1] < losses[0]  # Loss should decrease
        assert encoder.is_fitted

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            NeuralSpectralEncoder(mode="invalid")

    def test_backend_property(self):
        encoder = NeuralSpectralEncoder()
        assert encoder.backend in ("torch", "numpy")

    def test_cnn_fit_raises_for_torch_backend(self):
        encoder = NeuralSpectralEncoder(mode="cnn", embedding_dim=16)
        if encoder.backend == "torch":
            with pytest.raises(NotImplementedError, match="mode='autoencoder'"):
                encoder.fit(SAMPLE_RECORDS, epochs=1)


class TestNumpyFallback:
    """Tests for the NumPy autoencoder fallback."""

    def test_encode_shape(self):
        ae = _NumpyAutoencoderFallback(input_dim=64, latent_dim=16)
        x = np.random.randn(3, 64).astype(np.float32)
        z = ae.encode(x)
        assert z.shape == (3, 16)

    def test_decode_shape(self):
        ae = _NumpyAutoencoderFallback(input_dim=64, latent_dim=16)
        z = np.random.randn(3, 16).astype(np.float32)
        recon = ae.decode(z)
        assert recon.shape == (3, 64)

    def test_train_step_returns_loss(self):
        ae = _NumpyAutoencoderFallback(input_dim=32, latent_dim=8)
        batch = np.random.randn(10, 32).astype(np.float32)
        loss = ae.train_step(batch)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_fit_returns_decreasing_losses(self):
        ae = _NumpyAutoencoderFallback(input_dim=32, latent_dim=8)
        data = np.random.randn(50, 32).astype(np.float32) * 0.1
        losses = ae.fit(data, epochs=30, lr=1e-2)
        assert losses[-1] < losses[0]

    def test_fit_produces_deterministic_results_with_same_seed(self):
        data = np.random.randn(30, 32).astype(np.float32)
        ae1 = _NumpyAutoencoderFallback(input_dim=32, latent_dim=8, seed=7)
        ae2 = _NumpyAutoencoderFallback(input_dim=32, latent_dim=8, seed=7)
        losses1 = ae1.fit(data, epochs=5, lr=1e-3, batch_size=10)
        losses2 = ae2.fit(data, epochs=5, lr=1e-3, batch_size=10)
        np.testing.assert_allclose(losses1, losses2, rtol=1e-7, atol=1e-7)


class TestRecordToSignal:
    """Tests for signal extraction utility."""

    def test_resamples_to_target_length(self):
        from mesie.io.loaders import load_record
        rec = load_record(SAMPLE_RECORDS[0])
        sig = _record_to_signal(rec, target_length=64)
        assert sig.shape == (64,)
        assert sig.dtype == np.float32

    def test_empty_record_returns_zeros(self):
        from mesie.io.loaders import load_record
        rec = load_record({"record_id": "empty", "components": []})
        sig = _record_to_signal(rec, target_length=32)
        assert np.all(sig == 0)


# ---------------------------------------------------------------------------
# SpectralLSH tests
# ---------------------------------------------------------------------------


class TestSpectralLSH:
    """Tests for LSH-based approximate search."""

    def test_index_and_query(self):
        lsh = SpectralLSH(n_tables=4, n_bits=8, seed=42)
        lsh.index(SAMPLE_RECORDS)
        assert lsh.size == 5

        results = lsh.query(SAMPLE_RECORDS[0], top_k=3)
        assert len(results) <= 3
        # Self should be closest (distance 0)
        assert results[0][0] == "eq_1"
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_similar_records_rank_higher(self):
        lsh = SpectralLSH(n_tables=8, n_bits=12, seed=42)
        lsh.index(SAMPLE_RECORDS)

        results = lsh.query(SAMPLE_RECORDS[0], top_k=5)
        ids = [r[0] for r in results]
        # eq_1 (self) should always be first
        assert ids[0] == "eq_1"
        # eq_2 should be in the results (similar to eq_1)
        assert "eq_2" in ids

    def test_empty_query_returns_empty(self):
        lsh = SpectralLSH()
        assert lsh.query(SAMPLE_RECORDS[0]) == []

    def test_table_stats(self):
        lsh = SpectralLSH(n_tables=4, n_bits=8, seed=42)
        lsh.index(SAMPLE_RECORDS)
        stats = lsh.table_stats
        assert stats["n_tables"] == 4
        assert stats["n_indexed"] == 5

    def test_candidate_truncation_is_deterministic(self):
        lsh = SpectralLSH(n_tables=4, n_bits=8, seed=42)
        lsh.index(SAMPLE_RECORDS)
        r1 = lsh.query(SAMPLE_RECORDS[0], top_k=3, n_candidates=2)
        r2 = lsh.query(SAMPLE_RECORDS[0], top_k=3, n_candidates=2)
        assert r1 == r2

        query_emb = lsh.vectorizer.transform(SAMPLE_RECORDS[0])
        candidate_indices = set()
        for t in range(lsh.n_tables):
            key = lsh._hash_vector(query_emb, t)
            if key in lsh._tables[t]:
                candidate_indices.update(lsh._tables[t][key])
        if not candidate_indices:
            candidate_indices = set(range(lsh.size))

        expected_first_candidate = min(candidate_indices)
        deterministic_single = lsh.query(SAMPLE_RECORDS[0], top_k=1, n_candidates=1)
        assert deterministic_single[0][0] == lsh._record_ids[expected_first_candidate]


# ---------------------------------------------------------------------------
# SpectralMinHash tests
# ---------------------------------------------------------------------------


class TestSpectralMinHash:
    """Tests for MinHash-based similarity search."""

    def test_index_and_query(self):
        mh = SpectralMinHash(n_hashes=128, n_bands=16, seed=42)
        mh.index(SAMPLE_RECORDS)
        assert mh.size == 5

        results = mh.query(SAMPLE_RECORDS[0], top_k=3)
        assert len(results) <= 3
        # Self should have highest similarity
        assert results[0][0] == "eq_1"

    def test_similarity_self_is_one(self):
        mh = SpectralMinHash(n_hashes=128, n_bands=16, seed=42)
        sim = mh.estimate_similarity(SAMPLE_RECORDS[0], SAMPLE_RECORDS[0])
        assert sim == pytest.approx(1.0)

    def test_similar_records_higher_similarity(self):
        mh = SpectralMinHash(n_hashes=256, n_bands=16, seed=42)
        sim_similar = mh.estimate_similarity(SAMPLE_RECORDS[0], SAMPLE_RECORDS[1])
        sim_different = mh.estimate_similarity(SAMPLE_RECORDS[0], SAMPLE_RECORDS[2])
        assert sim_similar > sim_different

    def test_invalid_bands_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            SpectralMinHash(n_hashes=100, n_bands=7)


# ---------------------------------------------------------------------------
# HybridSpectralSearch tests
# ---------------------------------------------------------------------------


class TestHybridSpectralSearch:
    """Tests for two-stage coarse-to-fine search."""

    def test_index_and_query(self):
        hybrid = HybridSpectralSearch(n_tables=4, n_bits=8, seed=42)
        hybrid.index(SAMPLE_RECORDS)
        assert hybrid.size == 5

        results = hybrid.query(SAMPLE_RECORDS[0], top_k=3)
        assert len(results) <= 3
        assert results[0][0] == "eq_1"
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_returns_correct_ranking(self):
        hybrid = HybridSpectralSearch(n_tables=8, n_bits=12, seed=42)
        hybrid.index(SAMPLE_RECORDS)

        results = hybrid.query(SAMPLE_RECORDS[0], top_k=5)
        ids = [r[0] for r in results]
        assert ids[0] == "eq_1"
