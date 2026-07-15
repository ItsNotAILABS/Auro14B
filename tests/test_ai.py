"""Tests for mesie.ai module."""

import numpy as np
import pytest

from mesie.ai.models import SpectralAutoencoder, SpectralClassifier, SpectralTransformer, ModelConfig
from mesie.ai.training import TrainingPipeline, TrainingConfig
from mesie.ai.inference import InferenceEngine, PredictionResult
from mesie.ai.transfer import TransferAdapter, DomainAdaptation, DomainInfo


class TestSpectralAutoencoder:
    """Tests for SpectralAutoencoder."""

    def test_initialization(self):
        model = SpectralAutoencoder()
        assert not model.is_trained
        assert len(model._encoder_weights) > 0
        assert len(model._decoder_weights) > 0

    def test_encode_decode_shapes(self):
        config = ModelConfig(input_dim=64, latent_dim=16)
        model = SpectralAutoencoder(config)
        data = np.random.randn(10, 64)
        latent = model.encode(data)
        assert latent.shape == (10, 16)
        reconstructed = model.decode(latent)
        assert reconstructed.shape == (10, 64)

    def test_reconstruct(self):
        config = ModelConfig(input_dim=32, latent_dim=8)
        model = SpectralAutoencoder(config)
        data = np.random.randn(5, 32)
        result = model.reconstruct(data)
        assert result.shape == (5, 32)

    def test_fit(self):
        config = ModelConfig(input_dim=32, latent_dim=8)
        model = SpectralAutoencoder(config)
        data = np.random.randn(50, 32)
        losses = model.fit(data, epochs=5, batch_size=16)
        assert len(losses) == 5
        assert model.is_trained


class TestSpectralClassifier:
    """Tests for SpectralClassifier."""

    def test_initialization(self):
        model = SpectralClassifier(input_dim=32, n_classes=3)
        assert not model.is_trained
        assert model.n_classes == 3

    def test_predict_proba_shape(self):
        model = SpectralClassifier(input_dim=32, n_classes=5)
        data = np.random.randn(10, 32)
        proba = model.predict_proba(data)
        assert proba.shape == (10, 5)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict(self):
        model = SpectralClassifier(input_dim=16, n_classes=3)
        data = np.random.randn(8, 16)
        labels = model.predict(data)
        assert labels.shape == (8,)
        assert all(0 <= l < 3 for l in labels)

    def test_fit(self):
        model = SpectralClassifier(input_dim=16, n_classes=3)
        features = np.random.randn(30, 16)
        labels = np.random.randint(0, 3, size=30)
        losses = model.fit(features, labels, epochs=5)
        assert len(losses) == 5
        assert model.is_trained


class TestSpectralTransformer:
    """Tests for SpectralTransformer."""

    def test_initialization(self):
        model = SpectralTransformer(input_dim=1, n_heads=4, d_model=32)
        assert not model.is_trained

    def test_forward(self):
        model = SpectralTransformer(input_dim=1, n_heads=4, d_model=32, max_seq_len=64)
        data = np.random.randn(32, 1)
        output = model.forward(data)
        assert output.shape == (32, 32)

    def test_extract_features(self):
        model = SpectralTransformer(input_dim=1, n_heads=2, d_model=16, max_seq_len=64)
        data = np.random.randn(20, 1)
        features = model.extract_features(data)
        assert features.shape == (16,)


class TestTrainingPipeline:
    """Tests for TrainingPipeline."""

    def test_train_autoencoder(self):
        config = TrainingConfig(epochs=5, batch_size=8, early_stopping_patience=3)
        pipeline = TrainingPipeline(config)
        model = SpectralAutoencoder(ModelConfig(input_dim=16, latent_dim=4))
        data = np.random.randn(30, 16)
        result = pipeline.train_autoencoder(model, data)
        assert len(result.train_losses) > 0
        assert len(result.val_losses) > 0
        assert model.is_trained

    def test_train_classifier(self):
        config = TrainingConfig(epochs=5, batch_size=8, early_stopping_patience=3)
        pipeline = TrainingPipeline(config)
        model = SpectralClassifier(input_dim=16, n_classes=3)
        features = np.random.randn(30, 16)
        labels = np.random.randint(0, 3, size=30)
        result = pipeline.train_classifier(model, features, labels)
        assert len(result.train_losses) > 0
        assert "accuracy" in result.metrics_history


class TestInferenceEngine:
    """Tests for InferenceEngine."""

    def test_autoencoder_inference(self):
        model = SpectralAutoencoder(ModelConfig(input_dim=32, latent_dim=8))
        engine = InferenceEngine(model, model_type="autoencoder")
        data = np.random.randn(5, 32)
        result = engine.predict(data)
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape == (5, 32)
        assert result.confidence.shape == (5,)
        assert result.latent_features is not None

    def test_classifier_inference(self):
        model = SpectralClassifier(input_dim=16, n_classes=4)
        engine = InferenceEngine(model, model_type="classifier")
        data = np.random.randn(3, 16)
        result = engine.predict(data)
        assert result.predictions.shape == (3,)
        assert result.confidence.shape == (3,)

    def test_confidence_check(self):
        model = SpectralAutoencoder(ModelConfig(input_dim=16, latent_dim=4))
        engine = InferenceEngine(model, model_type="autoencoder", confidence_threshold=0.1)
        data = np.random.randn(3, 16)
        result = engine.predict(data)
        assert isinstance(engine.is_confident(result), bool)


class TestTransferAdapter:
    """Tests for TransferAdapter."""

    def test_fit_and_transform(self):
        adapter = TransferAdapter(adaptation_strength=0.8)
        source = np.random.randn(20, 16)
        target = np.random.randn(20, 16) + 2.0
        adapter.fit(source, target)
        assert adapter.is_fitted
        transformed = adapter.transform(source)
        assert transformed.shape == source.shape

    def test_no_fit_passthrough(self):
        adapter = TransferAdapter()
        data = np.random.randn(5, 8)
        result = adapter.transform(data)
        np.testing.assert_array_equal(result, data)


class TestDomainAdaptation:
    """Tests for DomainAdaptation."""

    def test_coral_adaptation(self):
        da = DomainAdaptation(strategy="coral")
        source = np.random.randn(30, 10)
        target = np.random.randn(30, 10) * 2 + 1
        da.fit(source, target)
        assert da.is_fitted
        result = da.transform(source)
        assert result.shape == source.shape

    def test_mmd_adaptation(self):
        da = DomainAdaptation(strategy="mmd")
        source = np.random.randn(20, 8)
        target = np.random.randn(20, 8) + 3
        da.fit(source, target)
        result = da.transform(source)
        assert result.shape == source.shape

    def test_domain_distance(self):
        da = DomainAdaptation()
        source = np.zeros((10, 5))
        target = np.ones((10, 5))
        dist = da.compute_domain_distance(source, target)
        assert dist > 0
