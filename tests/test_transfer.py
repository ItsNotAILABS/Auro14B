"""Tests for cross-domain spectral transfer module."""

from __future__ import annotations

import numpy as np
import pytest

from mesie.corpora import SpectralCorpus, SpectralDomain, DOMAIN_REGISTRY, CorpusRecord
from mesie.transfer import CORAL, MMD, DomainInvariantNormalizer, CrossDomainTransferEngine, TransferResult


class TestDomainRegistry:
    """Tests for spectral domain definitions."""

    def test_all_domains_registered(self):
        expected = {"seismic", "vibration", "eeg", "ecg", "audio", "rf", "financial", "satellite_edge", "eco_hz"}
        assert expected == set(DOMAIN_REGISTRY.keys())

    def test_domain_has_required_fields(self):
        for key, domain in DOMAIN_REGISTRY.items():
            assert domain.name
            assert domain.key == key
            assert domain.frequency_range[0] < domain.frequency_range[1]
            assert domain.typical_sampling_rate > 0


class TestSpectralCorpus:
    """Tests for corpus base class."""

    def test_create_corpus(self):
        corpus = SpectralCorpus("seismic")
        assert corpus.domain_key == "seismic"
        assert len(corpus) == 0

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            SpectralCorpus("nonexistent")

    def test_add_record(self):
        corpus = SpectralCorpus("seismic", canonical_n_points=64)
        freq = np.linspace(0.01, 50.0, 100)
        amp = np.random.default_rng(42).random(100)

        record = corpus.add_record("test_001", freq, amp)
        assert isinstance(record, CorpusRecord)
        assert record.domain_key == "seismic"
        assert len(corpus) == 1

    def test_canonical_grid_resampling(self):
        corpus = SpectralCorpus("audio", canonical_n_points=128)
        freq = np.linspace(20.0, 20000.0, 500)
        amp = np.sin(freq / 1000.0)

        record = corpus.add_record("audio_001", freq, amp)
        assert record.record.components[0].frequency.shape[0] == 128
        assert record.record.components[0].amplitude.shape[0] == 128

    def test_unit_energy_normalization(self):
        corpus = SpectralCorpus("vibration", canonical_n_points=64)
        freq = np.linspace(0.1, 1000.0, 200)
        amp = np.random.default_rng(7).random(200) * 100  # Large amplitudes

        record = corpus.add_record("vib_001", freq, amp)
        energy = np.sum(record.record.components[0].amplitude ** 2)
        assert abs(energy - 1.0) < 1e-6

    def test_get_split(self):
        corpus = SpectralCorpus("eeg", canonical_n_points=32)
        freq = np.linspace(0.5, 100.0, 50)
        rng = np.random.default_rng(0)

        corpus.add_record("eeg_train_1", freq, rng.random(50), split="train")
        corpus.add_record("eeg_train_2", freq, rng.random(50), split="train")
        corpus.add_record("eeg_test_1", freq, rng.random(50), split="test")

        assert len(corpus.get_split("train")) == 2
        assert len(corpus.get_split("test")) == 1

    def test_get_embedding_matrix(self):
        corpus = SpectralCorpus("financial", canonical_n_points=32)
        freq = np.linspace(1e-6, 1.0, 50)
        rng = np.random.default_rng(1)

        for i in range(5):
            corpus.add_record(f"fin_{i}", freq, rng.random(50))

        matrix = corpus.get_embedding_matrix()
        assert matrix.shape == (5, 32)


class TestCORAL:
    """Tests for CORAL alignment."""

    def test_basic_alignment(self):
        rng = np.random.default_rng(42)
        # Source: unit Gaussian
        source = rng.standard_normal((50, 10))
        # Target: shifted and scaled
        target = rng.standard_normal((50, 10)) * 2.0 + 3.0

        coral = CORAL()
        aligned = coral.fit_transform(source, target)

        assert aligned.shape == source.shape
        # Aligned mean should be closer to target mean
        target_mean = np.mean(target, axis=0)
        assert np.linalg.norm(np.mean(aligned, axis=0) - target_mean) < \
               np.linalg.norm(np.mean(source, axis=0) - target_mean)

    def test_coral_distance(self):
        rng = np.random.default_rng(0)
        same = rng.standard_normal((30, 5))
        different = rng.standard_normal((30, 5)) * 5.0 + 10.0

        coral = CORAL()
        dist_same = coral.coral_distance(same, same)
        dist_diff = coral.coral_distance(same, different)

        assert dist_same < dist_diff

    def test_not_fitted_raises(self):
        coral = CORAL()
        with pytest.raises(RuntimeError, match="fit"):
            coral.transform(np.zeros((5, 3)))


class TestMMD:
    """Tests for Maximum Mean Discrepancy."""

    def test_same_distribution_low_mmd(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5))
        mmd = MMD(kernel="rbf")
        # Same data should have very low MMD
        dist = mmd.compute(data[:50], data[50:])
        assert dist < 0.5  # Loose bound for same distribution

    def test_different_distributions_higher_mmd(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((50, 5))
        target = rng.standard_normal((50, 5)) + 5.0

        mmd = MMD(kernel="rbf")
        dist = mmd.compute(source, target)
        assert dist > 0.0

    def test_linear_kernel(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((30, 4))
        target = rng.standard_normal((30, 4)) + 2.0

        mmd = MMD(kernel="linear")
        dist = mmd.compute(source, target)
        assert dist >= 0.0

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="Unsupported kernel"):
            MMD(kernel="polynomial")

    def test_domain_divergence(self):
        rng = np.random.default_rng(0)
        source = rng.standard_normal((40, 6))
        target = rng.standard_normal((40, 6)) + 3.0

        mmd = MMD()
        div = mmd.domain_divergence(source, target)
        assert div >= 0.0


class TestDomainInvariantNormalizer:
    """Tests for domain-invariant normalization."""

    def test_basic_normalization(self):
        rng = np.random.default_rng(42)
        # Two domains with different distributions
        domain_a = rng.standard_normal((30, 8))
        domain_b = rng.standard_normal((30, 8)) * 3.0 + 5.0

        embeddings = np.vstack([domain_a, domain_b])
        labels = np.array(["a"] * 30 + ["b"] * 30)

        normalizer = DomainInvariantNormalizer()
        result = normalizer.fit_transform(embeddings, labels)

        assert result.shape == embeddings.shape
        # Result should be approximately zero-mean, unit-variance
        assert abs(np.mean(result)) < 0.5
        assert abs(np.std(result) - 1.0) < 0.5

    def test_with_whitening(self):
        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((60, 5))
        labels = np.array(["x"] * 30 + ["y"] * 30)

        normalizer = DomainInvariantNormalizer(whiten=True)
        result = normalizer.fit_transform(embeddings, labels)
        assert result.shape == embeddings.shape

    def test_not_fitted_raises(self):
        normalizer = DomainInvariantNormalizer()
        with pytest.raises(RuntimeError, match="fit"):
            normalizer.transform(np.zeros((5, 3)))


class TestCrossDomainTransferEngine:
    """Tests for the cross-domain transfer engine."""

    def _make_corpus(self, domain_key: str, n_records: int = 20, seed: int = 0):
        """Helper to create a populated corpus."""
        domain = DOMAIN_REGISTRY[domain_key]
        corpus = SpectralCorpus(domain_key, canonical_n_points=32)
        rng = np.random.default_rng(seed)
        freq = np.linspace(domain.frequency_range[0], domain.frequency_range[1], 100)

        for i in range(n_records):
            amp = rng.random(100) * (1 + 0.5 * np.sin(freq / freq[-1] * np.pi * (i + 1)))
            corpus.add_record(f"{domain_key}_{i:03d}", freq, amp)

        return corpus

    def test_register_and_transfer(self):
        engine = CrossDomainTransferEngine()
        seismic = self._make_corpus("seismic", n_records=25, seed=42)
        vibration = self._make_corpus("vibration", n_records=25, seed=7)

        engine.register_corpus(seismic)
        engine.register_corpus(vibration)

        result = engine.transfer("seismic", "vibration")
        assert isinstance(result, TransferResult)
        assert result.source_domain == "seismic"
        assert result.target_domain == "vibration"
        assert result.aligned_embeddings.shape[0] == 25
        assert result.mmd_before >= 0
        assert result.mmd_after >= 0
        assert result.transfer_gain > 0

    def test_build_unified_space(self):
        engine = CrossDomainTransferEngine()
        engine.register_corpus(self._make_corpus("seismic", 15, seed=1))
        engine.register_corpus(self._make_corpus("eeg", 15, seed=2))
        engine.register_corpus(self._make_corpus("audio", 15, seed=3))

        unified, labels = engine.build_unified_space()
        assert unified.shape[0] == 45
        assert labels.shape[0] == 45
        assert set(labels) == {"seismic", "eeg", "audio"}

    def test_domain_divergence_matrix(self):
        engine = CrossDomainTransferEngine()
        engine.register_corpus(self._make_corpus("seismic", 20, seed=10))
        engine.register_corpus(self._make_corpus("vibration", 20, seed=20))

        matrix = engine.domain_divergence_matrix()
        assert "seismic" in matrix
        assert "vibration" in matrix
        assert matrix["seismic"]["seismic"] == 0.0
        assert matrix["seismic"]["vibration"] == matrix["vibration"]["seismic"]

    def test_assess_transfer_feasibility(self):
        engine = CrossDomainTransferEngine()
        engine.register_corpus(self._make_corpus("seismic", 20, seed=5))
        engine.register_corpus(self._make_corpus("vibration", 20, seed=6))

        assessment = engine.assess_transfer_feasibility("seismic", "vibration")
        assert "feasibility" in assessment
        assert assessment["feasibility"] in ("high", "medium", "low")
        assert "recommendation" in assessment
        assert assessment["mmd_distance"] >= 0

    def test_unregistered_domain_raises(self):
        engine = CrossDomainTransferEngine()
        engine.register_corpus(self._make_corpus("seismic", 10, seed=0))

        with pytest.raises(ValueError, match="not registered"):
            engine.transfer("seismic", "audio")

    def test_unified_space_requires_two_domains(self):
        engine = CrossDomainTransferEngine()
        engine.register_corpus(self._make_corpus("seismic", 10, seed=0))

        with pytest.raises(RuntimeError, match="At least 2"):
            engine.build_unified_space()

    def test_registered_domains_property(self):
        engine = CrossDomainTransferEngine()
        engine.register_corpus(self._make_corpus("eeg", 5, seed=0))
        engine.register_corpus(self._make_corpus("financial", 5, seed=1))

        assert set(engine.registered_domains) == {"eeg", "financial"}
