"""Tests for foundation pretraining objectives and observation encoder."""

import numpy as np
import pytest

from mesie.pretraining.foundation_objectives import (
    AugmentationConfig,
    FoundationObjectiveSuite,
    InfoNCEContrastiveLoss,
    MaskConfig,
    MaskedSpectralModeling,
    TemporalPrediction,
    TemporalPredictionConfig,
)
from mesie.pretraining.observation_encoder import (
    LineageConditionedEncoder,
    ModalityConfig,
    ObservationEncoder,
    SpectralTransform,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def frequencies():
    """Standard frequency axis."""
    return np.linspace(0.1, 100.0, 256)


@pytest.fixture
def spectra_batch():
    """Batch of synthetic spectra."""
    rng = np.random.default_rng(42)
    n_samples, n_freq = 16, 256
    spectra = rng.exponential(1.0, size=(n_samples, n_freq))
    # Add some structure (peaks)
    for i in range(n_samples):
        peak_idx = rng.integers(20, 200)
        spectra[i, peak_idx: peak_idx + 10] += 5.0
    return spectra


@pytest.fixture
def embedding_sequence():
    """Temporal sequence of embeddings with drift."""
    rng = np.random.default_rng(123)
    n_steps, embed_dim = 50, 32
    seq = np.zeros((n_steps, embed_dim))
    for t in range(n_steps):
        drift = 0.02 * t
        seq[t] = rng.normal(drift, 0.1, size=embed_dim)
    return seq


@pytest.fixture
def raw_signal():
    """Synthetic time-domain signal."""
    t = np.linspace(0, 1.0, 1024)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    signal += np.random.default_rng(0).normal(0, 0.1, len(t))
    return signal


# ---------------------------------------------------------------------------
# Tests: Masked Spectral Modeling
# ---------------------------------------------------------------------------


class TestMaskedSpectralModeling:
    """Tests for MaskedSpectralModeling objective."""

    def test_default_mask_generation(self):
        """Test random mask generation."""
        msm = MaskedSpectralModeling()
        mask = msm.generate_mask(256, seed=42)
        assert mask.dtype == bool
        assert mask.shape == (256,)
        expected_masked = int(256 * 0.15)
        assert abs(mask.sum() - expected_masked) <= 1

    def test_contiguous_mask(self):
        """Test contiguous masking strategy."""
        config = MaskConfig(mask_strategy="contiguous", mask_ratio=0.2)
        msm = MaskedSpectralModeling(config)
        mask = msm.generate_mask(100, seed=0)
        # Contiguous: all True values should be adjacent
        indices = np.where(mask)[0]
        if len(indices) > 1:
            diffs = np.diff(indices)
            assert np.all(diffs == 1)

    def test_band_mask(self):
        """Test band-based masking strategy."""
        config = MaskConfig(mask_strategy="band", n_bands=8, mask_ratio=0.25)
        msm = MaskedSpectralModeling(config)
        mask = msm.generate_mask(256, seed=0)
        assert mask.sum() > 0
        assert mask.sum() < 256

    def test_corrupt_preserves_unmasked(self, spectra_batch):
        """Test that unmasked bins are preserved."""
        msm = MaskedSpectralModeling()
        corrupted, mask = msm.corrupt(spectra_batch, seed=42)
        # Unmasked positions unchanged
        np.testing.assert_array_equal(
            corrupted[:, ~mask], spectra_batch[:, ~mask]
        )
        # Masked positions zeroed
        np.testing.assert_array_equal(
            corrupted[:, mask], 0.0
        )

    def test_compute_loss(self, spectra_batch):
        """Test loss computation over masked positions."""
        msm = MaskedSpectralModeling()
        mask = msm.generate_mask(spectra_batch.shape[1], seed=0)
        # Perfect prediction → zero loss
        loss = msm.compute_loss(spectra_batch, spectra_batch, mask)
        assert loss == pytest.approx(0.0)
        # Random prediction → non-zero loss
        random_pred = np.random.default_rng(1).normal(size=spectra_batch.shape)
        loss = msm.compute_loss(random_pred, spectra_batch, mask)
        assert loss > 0.0

    def test_create_training_sample(self, spectra_batch):
        """Test full training sample creation."""
        msm = MaskedSpectralModeling()
        sample = msm.create_training_sample(spectra_batch, seed=99)
        assert "corrupted" in sample
        assert "targets" in sample
        assert "mask" in sample
        assert sample["corrupted"].shape == spectra_batch.shape
        assert sample["targets"].shape == spectra_batch.shape
        assert sample["mask"].dtype == bool

    def test_reproducibility(self, spectra_batch):
        """Test seed-based reproducibility."""
        msm = MaskedSpectralModeling()
        s1 = msm.create_training_sample(spectra_batch, seed=7)
        s2 = msm.create_training_sample(spectra_batch, seed=7)
        np.testing.assert_array_equal(s1["mask"], s2["mask"])
        np.testing.assert_array_equal(s1["corrupted"], s2["corrupted"])


# ---------------------------------------------------------------------------
# Tests: InfoNCE Contrastive Loss
# ---------------------------------------------------------------------------


class TestInfoNCEContrastiveLoss:
    """Tests for InfoNCE contrastive learning."""

    def test_augmentation_produces_different_output(self, spectra_batch):
        """Test that augmentation modifies spectra."""
        cl = InfoNCEContrastiveLoss()
        augmented = cl.augment(spectra_batch, seed=42)
        assert augmented.shape == spectra_batch.shape
        # Should not be identical
        assert not np.allclose(augmented, spectra_batch)

    def test_similarity_matrix_shape(self):
        """Test similarity matrix computation."""
        cl = InfoNCEContrastiveLoss()
        a = np.random.default_rng(0).normal(size=(8, 32))
        b = np.random.default_rng(1).normal(size=(8, 32))
        sim = cl.compute_similarity_matrix(a, b)
        assert sim.shape == (8, 8)

    def test_cosine_similarity_range(self):
        """Test cosine similarity values are in [-1, 1]."""
        cl = InfoNCEContrastiveLoss(similarity="cosine")
        a = np.random.default_rng(0).normal(size=(10, 64))
        b = np.random.default_rng(1).normal(size=(10, 64))
        sim = cl.compute_similarity_matrix(a, b)
        assert sim.min() >= -1.0 - 1e-6
        assert sim.max() <= 1.0 + 1e-6

    def test_loss_positive(self):
        """Test InfoNCE loss is positive for random embeddings."""
        cl = InfoNCEContrastiveLoss(temperature=0.1)
        rng = np.random.default_rng(42)
        anchors = rng.normal(size=(16, 32))
        positives = anchors + rng.normal(0, 0.1, size=(16, 32))
        loss = cl.compute_loss(anchors, positives)
        assert loss > 0.0

    def test_loss_lower_for_similar_pairs(self):
        """Test that highly similar positives yield lower loss."""
        cl = InfoNCEContrastiveLoss(temperature=0.1)
        rng = np.random.default_rng(42)
        anchors = rng.normal(size=(16, 32))
        # Very similar positives
        close_positives = anchors + rng.normal(0, 0.01, size=(16, 32))
        # Dissimilar positives
        far_positives = rng.normal(size=(16, 32))
        loss_close = cl.compute_loss(anchors, close_positives)
        loss_far = cl.compute_loss(anchors, far_positives)
        assert loss_close < loss_far

    def test_create_training_batch(self, spectra_batch):
        """Test full contrastive batch creation."""
        cl = InfoNCEContrastiveLoss()

        def encoder(x):
            return x[:, :32] if x.ndim == 2 else x[:32]

        batch = cl.create_training_batch(spectra_batch, encoder, seed=0)
        assert "anchor_embeddings" in batch
        assert "positive_embeddings" in batch
        assert "loss" in batch
        assert batch["anchor_embeddings"].shape == (16, 32)

    def test_dot_product_similarity(self):
        """Test dot product similarity mode."""
        cl = InfoNCEContrastiveLoss(similarity="dot")
        a = np.random.default_rng(0).normal(size=(8, 16))
        b = np.random.default_rng(1).normal(size=(8, 16))
        sim = cl.compute_similarity_matrix(a, b)
        assert sim.shape == (8, 8)

    def test_temperature_effect(self):
        """Test that lower temperature sharpens the distribution."""
        rng = np.random.default_rng(42)
        anchors = rng.normal(size=(16, 32))
        positives = anchors + rng.normal(0, 0.5, size=(16, 32))

        cl_low_temp = InfoNCEContrastiveLoss(temperature=0.01)
        cl_high_temp = InfoNCEContrastiveLoss(temperature=1.0)

        loss_low = cl_low_temp.compute_loss(anchors, positives)
        loss_high = cl_high_temp.compute_loss(anchors, positives)
        # Low temperature amplifies differences
        assert loss_low != loss_high

    def test_invalid_temperature_raises(self):
        """Test that non-positive temperature raises error."""
        with pytest.raises(ValueError):
            InfoNCEContrastiveLoss(temperature=0.0)
        with pytest.raises(ValueError):
            InfoNCEContrastiveLoss(temperature=-1.0)


# ---------------------------------------------------------------------------
# Tests: Temporal Prediction
# ---------------------------------------------------------------------------


class TestTemporalPrediction:
    """Tests for temporal prediction objective."""

    def test_generate_samples(self, embedding_sequence):
        """Test sample generation from temporal sequence."""
        tp = TemporalPrediction()
        contexts, targets, indices = tp.generate_samples(embedding_sequence)
        assert contexts.shape[0] == targets.shape[0]
        assert contexts.shape[0] == len(indices)
        assert contexts.shape[0] > 0

    def test_context_aggregation_mean(self, embedding_sequence):
        """Test mean aggregation."""
        config = TemporalPredictionConfig(aggregation="mean", context_window=5)
        tp = TemporalPrediction(config)
        ctx = embedding_sequence[:5]
        agg = tp.aggregate_context(ctx)
        np.testing.assert_array_almost_equal(agg, np.mean(ctx, axis=0))

    def test_context_aggregation_last(self, embedding_sequence):
        """Test 'last' aggregation."""
        config = TemporalPredictionConfig(aggregation="last", context_window=5)
        tp = TemporalPrediction(config)
        ctx = embedding_sequence[:5]
        agg = tp.aggregate_context(ctx)
        np.testing.assert_array_equal(agg, ctx[-1])

    def test_context_aggregation_weighted(self, embedding_sequence):
        """Test weighted (exponential decay) aggregation."""
        config = TemporalPredictionConfig(
            aggregation="weighted", context_window=5, decay_factor=0.9
        )
        tp = TemporalPrediction(config)
        ctx = embedding_sequence[:5]
        agg = tp.aggregate_context(ctx)
        assert agg.shape == (embedding_sequence.shape[1],)

    def test_context_aggregation_concat(self, embedding_sequence):
        """Test concat aggregation (flattened window)."""
        config = TemporalPredictionConfig(aggregation="concat", context_window=5)
        tp = TemporalPrediction(config)
        ctx = embedding_sequence[:5]
        agg = tp.aggregate_context(ctx)
        assert len(agg) == 5 * embedding_sequence.shape[1]

    def test_insufficient_sequence(self):
        """Test graceful handling of too-short sequences."""
        config = TemporalPredictionConfig(context_window=10, prediction_horizon=5)
        tp = TemporalPrediction(config)
        short_seq = np.random.default_rng(0).normal(size=(5, 16))
        contexts, targets, indices = tp.generate_samples(short_seq)
        assert contexts.shape[0] == 0

    def test_compute_loss_perfect(self, embedding_sequence):
        """Test zero loss for perfect predictions."""
        tp = TemporalPrediction()
        contexts, targets, _ = tp.generate_samples(embedding_sequence)
        loss = tp.compute_loss(targets, targets)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_compute_loss_random(self, embedding_sequence):
        """Test positive loss for random predictions."""
        tp = TemporalPrediction()
        contexts, targets, _ = tp.generate_samples(embedding_sequence)
        rng = np.random.default_rng(0)
        random_pred = rng.normal(size=targets.shape)
        loss = tp.compute_loss(random_pred, targets)
        assert loss > 0.0

    def test_drift_prediction(self, embedding_sequence):
        """Test drift prediction computation."""
        tp = TemporalPrediction()
        drift = tp.compute_drift_prediction(embedding_sequence)
        assert drift.shape == (embedding_sequence.shape[0],)
        # First step should have zero drift (it's the reference)
        assert drift[0] == pytest.approx(0.0)
        # Later steps should show increasing drift (due to fixture design)
        assert drift[-1] > drift[1]


# ---------------------------------------------------------------------------
# Tests: Foundation Objective Suite
# ---------------------------------------------------------------------------


class TestFoundationObjectiveSuite:
    """Tests for the combined objective suite."""

    def test_initialization(self):
        """Test suite initializes with all objectives."""
        suite = FoundationObjectiveSuite()
        assert suite.masked_modeling is not None
        assert suite.contrastive is not None
        assert suite.temporal is not None

    def test_compute_total_loss(self):
        """Test weighted loss aggregation."""
        suite = FoundationObjectiveSuite(
            weights={"masked": 2.0, "contrastive": 1.0, "temporal": 0.5}
        )
        losses = {"masked": 1.0, "contrastive": 2.0, "temporal": 4.0}
        total = suite.compute_total_loss(losses)
        expected = 2.0 * 1.0 + 1.0 * 2.0 + 0.5 * 4.0
        assert total == pytest.approx(expected)

    def test_custom_weights(self):
        """Test custom weight configuration."""
        suite = FoundationObjectiveSuite(
            weights={"masked": 0.0, "contrastive": 1.0, "temporal": 0.0}
        )
        losses = {"masked": 10.0, "contrastive": 2.0, "temporal": 10.0}
        total = suite.compute_total_loss(losses)
        assert total == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests: Spectral Transform
# ---------------------------------------------------------------------------


class TestSpectralTransform:
    """Tests for SpectralTransform."""

    def test_fft_output_shape(self, raw_signal):
        """Test FFT output dimensionality."""
        st = SpectralTransform(method="fft", n_fft=256)
        result = st.transform(raw_signal)
        assert result.shape == (128,)  # n_fft // 2

    def test_stft_output_shape(self, raw_signal):
        """Test STFT spectrogram shape."""
        st = SpectralTransform(method="stft", n_fft=256, hop_length=128)
        result = st.transform(raw_signal)
        assert result.ndim == 2
        assert result.shape[1] == 128  # n_fft // 2

    def test_welch_output_shape(self, raw_signal):
        """Test Welch PSD output shape."""
        st = SpectralTransform(method="welch", n_fft=256, hop_length=128)
        result = st.transform(raw_signal)
        assert result.shape == (128,)

    def test_multichannel(self, raw_signal):
        """Test multi-channel input handling."""
        multichannel = np.stack([raw_signal, raw_signal * 0.5])
        st = SpectralTransform(method="fft", n_fft=256)
        result = st.transform(multichannel)
        assert result.shape == (128,)

    def test_invalid_method_raises(self):
        """Test invalid transform method raises error."""
        with pytest.raises(ValueError):
            SpectralTransform(method="invalid")


# ---------------------------------------------------------------------------
# Tests: Observation Encoder
# ---------------------------------------------------------------------------


class TestObservationEncoder:
    """Tests for ObservationEncoder."""

    def test_default_observation_dim(self):
        """Test default observation dimensionality."""
        enc = ObservationEncoder()
        # 64 (spectral) + 16 (state) + 8 (semantic) = 88
        assert enc.observation_dim == 88

    def test_encode_spectral(self):
        """Test spectral encoding."""
        enc = ObservationEncoder()
        spectrum = np.random.default_rng(0).exponential(1.0, size=128)
        z_t = enc.encode_spectral(spectrum)
        assert z_t.shape == (64,)

    def test_encode_observation_full(self):
        """Test full observation with all modalities."""
        enc = ObservationEncoder()
        spectrum = np.random.default_rng(0).exponential(1.0, size=128)
        state = np.random.default_rng(1).normal(size=16)
        semantic = np.random.default_rng(2).normal(size=8)
        obs = enc.encode_observation(spectrum, state=state, semantic=semantic)
        assert obs.shape == (88,)

    def test_encode_observation_spectral_only(self):
        """Test observation with spectral modality only."""
        enc = ObservationEncoder()
        spectrum = np.random.default_rng(0).exponential(1.0, size=128)
        obs = enc.encode_observation(spectrum)
        assert obs.shape == (88,)
        # State and semantic parts should be zero
        assert np.allclose(obs[64:], 0.0)

    def test_encode_from_raw_signal(self, raw_signal):
        """Test full pipeline from raw signal."""
        enc = ObservationEncoder()
        obs = enc.encode_from_raw_signal(raw_signal)
        assert obs.shape == (88,)
        assert not np.all(obs[:64] == 0.0)  # Spectral part should be non-zero

    def test_batch_encode(self, spectra_batch):
        """Test batch encoding."""
        enc = ObservationEncoder()
        obs = enc.batch_encode(spectra_batch)
        assert obs.shape == (16, 88)

    def test_custom_modalities(self):
        """Test custom modality configuration."""
        modalities = [
            ModalityConfig(name="spectral", dim=32, normalize=True, weight=1.0),
            ModalityConfig(name="state", dim=8, normalize=False, weight=1.0),
        ]
        enc = ObservationEncoder(modalities=modalities)
        assert enc.observation_dim == 40
        spectrum = np.random.default_rng(0).normal(size=64)
        state = np.random.default_rng(1).normal(size=8)
        obs = enc.encode_observation(spectrum, state=state)
        assert obs.shape == (40,)

    def test_output_projection(self):
        """Test output dimensionality projection."""
        enc = ObservationEncoder(output_dim=32)
        assert enc.observation_dim == 32
        spectrum = np.random.default_rng(0).normal(size=128)
        obs = enc.encode_observation(spectrum)
        assert obs.shape == (32,)

    def test_normalization(self):
        """Test L2 normalization of spectral modality."""
        enc = ObservationEncoder()
        spectrum = np.random.default_rng(0).exponential(1.0, size=128) * 100
        obs = enc.encode_observation(spectrum)
        # Spectral part (first 64) should be normalized
        spectral_part = obs[:64]
        norm = np.linalg.norm(spectral_part)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_modality_names(self):
        """Test modality name listing."""
        enc = ObservationEncoder()
        assert "spectral" in enc.modality_names
        assert "state" in enc.modality_names
        assert "semantic" in enc.modality_names


# ---------------------------------------------------------------------------
# Tests: Lineage Conditioned Encoder
# ---------------------------------------------------------------------------


class TestLineageConditionedEncoder:
    """Tests for LineageConditionedEncoder."""

    def test_observation_dim_includes_lineage(self):
        """Test that lineage modality is included in dimensions."""
        enc = LineageConditionedEncoder(lineage_dim=64)
        # 64 (spectral) + 16 (state) + 8 (semantic) + 64 (lineage) = 152
        assert enc.observation_dim == 152

    def test_encode_with_lineage_no_memory(self):
        """Test encoding without memory store returns zero lineage."""
        enc = LineageConditionedEncoder(memory_store=None, lineage_dim=32)
        spectrum = np.random.default_rng(0).exponential(1.0, size=128)
        obs = enc.encode_with_lineage(spectrum)
        # Should still produce valid output
        expected_dim = 64 + 16 + 8 + 32  # spectral + state + semantic + lineage
        assert obs.shape == (expected_dim,)

    def test_encode_with_lineage_mock_memory(self):
        """Test encoding with a mock memory store."""

        class MockMemoryStore:
            def get_lineage(self, embedding):
                return np.ones(64) * 0.5

        enc = LineageConditionedEncoder(
            memory_store=MockMemoryStore(), lineage_dim=64
        )
        spectrum = np.random.default_rng(0).exponential(1.0, size=128)
        obs = enc.encode_with_lineage(spectrum)
        expected_dim = 64 + 16 + 8 + 64
        assert obs.shape == (expected_dim,)
        # Lineage part should be non-zero
        assert not np.all(obs[-64:] == 0.0)

    def test_integration_with_spectral_memory(self):
        """Test integration with actual SpectralMemoryStore."""
        from mesie.pretraining.spectral_memory import SpectralMemoryStore

        store = SpectralMemoryStore(embed_dim=64)
        # Populate memory
        rng = np.random.default_rng(42)
        for i in range(10):
            store.store(
                timestamp=float(i),
                embedding=rng.normal(size=64),
                event_type="normal",
            )

        enc = LineageConditionedEncoder(
            memory_store=store, lineage_dim=64
        )
        spectrum = np.random.default_rng(0).exponential(1.0, size=128)
        obs = enc.encode_with_lineage(spectrum)
        expected_dim = 64 + 16 + 8 + 64
        assert obs.shape == (expected_dim,)
