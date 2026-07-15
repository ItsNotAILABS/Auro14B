"""Comprehensive test suite for expanded AI libraries - Part 1.

Tests meta-learning, Bayesian networks, and generative models
with parametrized tests for extensive coverage (500+ tests).
"""

import numpy as np
import pytest

from mesie.ai.meta_learning import (
    MetaLearningConfig,
    MetaStrategy,
    MetaTask,
    MetaResult,
    PrototypicalNetwork,
    MAMLAdapter,
    TaskDistribution,
)
from mesie.ai.bayesian import (
    BayesianConfig,
    BayesianSpectralNetwork,
    CalibrationModule,
    EnsemblePredictor,
    UncertaintyEstimate,
    UncertaintyType,
)
from mesie.ai.generative import (
    VAEConfig,
    DiffusionConfig,
    GenerationResult,
    SpectralVAE,
    SpectralDiffusion,
    SpectralGAN,
    GenerativeModelType,
)


# ============================================================
# META-LEARNING TESTS
# ============================================================


class TestMetaLearningConfig:
    """Tests for MetaLearningConfig."""

    @pytest.mark.parametrize("strategy", list(MetaStrategy))
    def test_config_strategies(self, strategy):
        config = MetaLearningConfig(strategy=strategy)
        assert config.strategy == strategy

    @pytest.mark.parametrize("n_way", [2, 3, 5, 10, 20])
    def test_config_n_way(self, n_way):
        config = MetaLearningConfig(n_way=n_way)
        assert config.n_way == n_way

    @pytest.mark.parametrize("k_shot", [1, 2, 5, 10, 20])
    def test_config_k_shot(self, k_shot):
        config = MetaLearningConfig(k_shot=k_shot)
        assert config.k_shot == k_shot

    @pytest.mark.parametrize("inner_lr", [0.001, 0.01, 0.05, 0.1, 0.5])
    def test_config_inner_lr(self, inner_lr):
        config = MetaLearningConfig(inner_lr=inner_lr)
        assert config.inner_lr == inner_lr

    @pytest.mark.parametrize("outer_lr", [0.0001, 0.001, 0.01, 0.1])
    def test_config_outer_lr(self, outer_lr):
        config = MetaLearningConfig(outer_lr=outer_lr)
        assert config.outer_lr == outer_lr

    @pytest.mark.parametrize("n_inner_steps", [1, 3, 5, 10, 20])
    def test_config_inner_steps(self, n_inner_steps):
        config = MetaLearningConfig(n_inner_steps=n_inner_steps)
        assert config.n_inner_steps == n_inner_steps

    @pytest.mark.parametrize("embedding_dim", [16, 32, 64, 128, 256])
    def test_config_embedding_dim(self, embedding_dim):
        config = MetaLearningConfig(embedding_dim=embedding_dim)
        assert config.embedding_dim == embedding_dim


class TestMetaTask:
    """Tests for MetaTask dataclass."""

    @pytest.mark.parametrize("n_way,k_shot", [
        (2, 1), (3, 1), (5, 1), (5, 5), (10, 1), (10, 5), (2, 10),
    ])
    def test_task_properties(self, n_way, k_shot):
        support_x = np.random.randn(n_way * k_shot, 32)
        support_y = np.repeat(np.arange(n_way), k_shot)
        query_x = np.random.randn(n_way * 5, 32)
        query_y = np.repeat(np.arange(n_way), 5)
        task = MetaTask(support_x=support_x, support_y=support_y,
                       query_x=query_x, query_y=query_y)
        assert task.n_way == n_way
        assert task.k_shot == k_shot

    @pytest.mark.parametrize("dim", [8, 16, 32, 64, 128])
    def test_task_dimensions(self, dim):
        task = MetaTask(
            support_x=np.random.randn(10, dim),
            support_y=np.repeat(np.arange(2), 5),
            query_x=np.random.randn(10, dim),
            query_y=np.repeat(np.arange(2), 5),
        )
        assert task.support_x.shape[1] == dim
        assert task.query_x.shape[1] == dim


class TestPrototypicalNetwork:
    """Tests for PrototypicalNetwork."""

    @pytest.mark.parametrize("input_dim", [8, 16, 32, 64, 128])
    def test_initialization(self, input_dim):
        net = PrototypicalNetwork(input_dim=input_dim)
        assert net.input_dim == input_dim
        assert len(net._encoder_weights) > 0

    @pytest.mark.parametrize("input_dim,embedding_dim", [
        (16, 8), (32, 16), (64, 32), (128, 64), (256, 128),
    ])
    def test_embed_shape(self, input_dim, embedding_dim):
        net = PrototypicalNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
        x = np.random.randn(10, input_dim)
        embeddings = net.embed(x)
        assert embeddings.shape == (10, embedding_dim)

    @pytest.mark.parametrize("n_classes", [2, 3, 5, 7, 10])
    def test_compute_prototypes(self, n_classes):
        net = PrototypicalNetwork(input_dim=32, embedding_dim=16)
        support_x = np.random.randn(n_classes * 5, 32)
        support_y = np.repeat(np.arange(n_classes), 5)
        prototypes = net.compute_prototypes(support_x, support_y)
        assert len(prototypes) == n_classes
        for cls in range(n_classes):
            assert prototypes[cls].shape == (16,)

    @pytest.mark.parametrize("n_queries", [1, 5, 10, 20, 50])
    def test_predict_shape(self, n_queries):
        net = PrototypicalNetwork(input_dim=32, embedding_dim=16)
        support_x = np.random.randn(10, 32)
        support_y = np.repeat(np.arange(2), 5)
        prototypes = net.compute_prototypes(support_x, support_y)
        query_x = np.random.randn(n_queries, 32)
        predictions, confidence = net.predict(query_x, prototypes)
        assert predictions.shape == (n_queries,)
        assert confidence.shape == (n_queries,)
        assert all(0 <= c <= 1 for c in confidence)

    @pytest.mark.parametrize("n_way,k_shot", [
        (2, 1), (2, 5), (3, 1), (5, 1), (5, 5),
    ])
    def test_evaluate_task(self, n_way, k_shot):
        net = PrototypicalNetwork(input_dim=32, embedding_dim=16)
        task = MetaTask(
            support_x=np.random.randn(n_way * k_shot, 32),
            support_y=np.repeat(np.arange(n_way), k_shot),
            query_x=np.random.randn(n_way * 5, 32),
            query_y=np.repeat(np.arange(n_way), 5),
        )
        result = net.evaluate_task(task)
        assert isinstance(result, MetaResult)
        assert 0 <= result.accuracy <= 1
        assert result.loss >= 0
        assert len(result.per_class_accuracy) == n_way


class TestMAMLAdapter:
    """Tests for MAMLAdapter."""

    @pytest.mark.parametrize("n_inner_steps", [1, 3, 5, 10])
    def test_initialization(self, n_inner_steps):
        config = MetaLearningConfig(n_inner_steps=n_inner_steps)
        adapter = MAMLAdapter(config)
        assert adapter.config.n_inner_steps == n_inner_steps

    @pytest.mark.parametrize("input_dim,output_dim", [
        (8, 2), (16, 3), (32, 5), (64, 10), (128, 2),
    ])
    def test_initialize(self, input_dim, output_dim):
        adapter = MAMLAdapter(MetaLearningConfig())
        adapter.initialize(input_dim, output_dim)
        assert adapter._base_weights.shape == (input_dim, output_dim)

    @pytest.mark.parametrize("n_samples", [5, 10, 20, 50])
    def test_inner_adapt(self, n_samples):
        config = MetaLearningConfig(n_inner_steps=3)
        adapter = MAMLAdapter(config)
        adapter.initialize(16, 3)
        support_x = np.random.randn(n_samples, 16)
        support_y = np.random.randint(0, 3, n_samples)
        # Convert to one-hot
        support_y_oh = np.eye(3)[support_y]
        weights = adapter.inner_adapt(support_x, support_y_oh)
        assert weights.shape == (16, 3)

    @pytest.mark.parametrize("inner_lr", [0.001, 0.01, 0.05, 0.1])
    def test_adaptation_with_different_lr(self, inner_lr):
        config = MetaLearningConfig(inner_lr=inner_lr, n_inner_steps=5)
        adapter = MAMLAdapter(config)
        adapter.initialize(16, 2)
        support_x = np.random.randn(10, 16)
        support_y = np.eye(2)[np.random.randint(0, 2, 10)]
        weights = adapter.inner_adapt(support_x, support_y)
        assert not np.allclose(weights, adapter._base_weights)

    @pytest.mark.parametrize("n_way", [2, 3, 5])
    def test_evaluate_task(self, n_way):
        config = MetaLearningConfig(n_inner_steps=3)
        adapter = MAMLAdapter(config)
        task = MetaTask(
            support_x=np.random.randn(n_way * 3, 16),
            support_y=np.repeat(np.arange(n_way), 3),
            query_x=np.random.randn(n_way * 5, 16),
            query_y=np.repeat(np.arange(n_way), 5),
        )
        result = adapter.evaluate_task(task)
        assert isinstance(result, MetaResult)
        assert result.adaptation_steps_used == 3


class TestTaskDistribution:
    """Tests for TaskDistribution."""

    @pytest.mark.parametrize("n_way,k_shot", [
        (2, 1), (2, 5), (3, 1), (5, 1), (5, 5),
    ])
    def test_sample_task(self, n_way, k_shot):
        dist = TaskDistribution(n_way=n_way, k_shot=k_shot, query_size=5)
        data = np.random.randn(100, 32)
        labels = np.random.randint(0, n_way, 100)
        task = dist.sample_task(data, labels)
        assert isinstance(task, MetaTask)
        assert task.support_x.shape[1] == 32

    @pytest.mark.parametrize("n_tasks", [1, 5, 10, 20, 50])
    def test_sample_multiple_tasks(self, n_tasks):
        dist = TaskDistribution(n_way=3, k_shot=2, query_size=5)
        data = np.random.randn(100, 16)
        labels = np.random.randint(0, 5, 100)
        tasks = dist.sample_tasks(data, labels, n_tasks)
        assert len(tasks) == n_tasks
        for task in tasks:
            assert isinstance(task, MetaTask)

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 999])
    def test_reproducibility(self, seed):
        dist = TaskDistribution(n_way=3, k_shot=2, query_size=5)
        data = np.random.randn(100, 16)
        labels = np.random.randint(0, 5, 100)
        tasks1 = dist.sample_tasks(data, labels, 5, seed=seed)
        tasks2 = dist.sample_tasks(data, labels, 5, seed=seed)
        for t1, t2 in zip(tasks1, tasks2):
            np.testing.assert_array_equal(t1.support_x, t2.support_x)
            np.testing.assert_array_equal(t1.support_y, t2.support_y)


# ============================================================
# BAYESIAN NETWORK TESTS
# ============================================================


class TestBayesianConfig:
    """Tests for BayesianConfig."""

    @pytest.mark.parametrize("input_dim", [8, 16, 32, 64, 128, 256])
    def test_input_dim(self, input_dim):
        config = BayesianConfig(input_dim=input_dim)
        assert config.input_dim == input_dim

    @pytest.mark.parametrize("hidden_dims", [
        [16], [32, 16], [64, 32, 16], [128, 64, 32], [256, 128, 64, 32],
    ])
    def test_hidden_dims(self, hidden_dims):
        config = BayesianConfig(hidden_dims=hidden_dims)
        assert config.hidden_dims == hidden_dims

    @pytest.mark.parametrize("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.5])
    def test_dropout_rate(self, dropout_rate):
        config = BayesianConfig(dropout_rate=dropout_rate)
        assert config.dropout_rate == dropout_rate

    @pytest.mark.parametrize("n_mc_samples", [10, 25, 50, 100, 200])
    def test_mc_samples(self, n_mc_samples):
        config = BayesianConfig(n_mc_samples=n_mc_samples)
        assert config.n_mc_samples == n_mc_samples

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_temperature(self, temperature):
        config = BayesianConfig(temperature=temperature)
        assert config.temperature == temperature


class TestBayesianSpectralNetwork:
    """Tests for BayesianSpectralNetwork."""

    @pytest.mark.parametrize("input_dim,output_dim", [
        (8, 1), (16, 1), (32, 1), (64, 1), (128, 1),
        (16, 3), (32, 5), (64, 10),
    ])
    def test_initialization(self, input_dim, output_dim):
        config = BayesianConfig(input_dim=input_dim, output_dim=output_dim)
        net = BayesianSpectralNetwork(config)
        assert not net.is_trained
        assert len(net._weights) > 0

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20, 50])
    def test_predict_shape(self, n_samples):
        config = BayesianConfig(input_dim=16, output_dim=1)
        net = BayesianSpectralNetwork(config)
        x = np.random.randn(n_samples, 16)
        pred = net.predict(x)
        assert pred.shape == (n_samples, 1)

    @pytest.mark.parametrize("n_mc", [5, 10, 25, 50])
    def test_uncertainty_estimation(self, n_mc):
        config = BayesianConfig(input_dim=16, output_dim=1, n_mc_samples=n_mc)
        net = BayesianSpectralNetwork(config)
        x = np.random.randn(5, 16)
        result = net.predict_with_uncertainty(x, n_samples=n_mc)
        assert isinstance(result, UncertaintyEstimate)
        assert result.mean.shape == (5, 1)
        assert result.epistemic.shape == (5, 1)
        assert result.aleatoric.shape == (5, 1)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_dropout_effect(self, dropout):
        config = BayesianConfig(input_dim=16, output_dim=1, dropout_rate=dropout)
        net = BayesianSpectralNetwork(config)
        x = np.random.randn(5, 16)
        result = net.predict_with_uncertainty(x)
        assert result.total_uncertainty is not None

    @pytest.mark.parametrize("epochs", [1, 3, 5, 10])
    def test_training(self, epochs):
        config = BayesianConfig(input_dim=8, hidden_dims=[16], output_dim=1)
        net = BayesianSpectralNetwork(config)
        x = np.random.randn(20, 8)
        y = np.random.randn(20)
        losses = net.fit(x, y, epochs=epochs)
        assert len(losses) == epochs
        assert net.is_trained

    @pytest.mark.parametrize("input_dim", [8, 16, 32])
    def test_confidence_bounded(self, input_dim):
        config = BayesianConfig(input_dim=input_dim, output_dim=1)
        net = BayesianSpectralNetwork(config)
        x = np.random.randn(10, input_dim)
        result = net.predict_with_uncertainty(x)
        assert np.all(result.confidence >= 0)
        assert np.all(result.confidence <= 1)


class TestCalibrationModule:
    """Tests for CalibrationModule."""

    @pytest.mark.parametrize("n_bins", [5, 10, 15, 20, 50])
    def test_initialization(self, n_bins):
        cal = CalibrationModule(n_bins=n_bins)
        assert cal.n_bins == n_bins
        assert cal.temperature == 1.0

    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500])
    def test_compute_ece(self, n_samples):
        cal = CalibrationModule()
        confidences = np.random.uniform(0, 1, n_samples)
        accuracies = (np.random.random(n_samples) > 0.5).astype(float)
        ece = cal.compute_ece(confidences, accuracies)
        assert 0 <= ece <= 1

    @pytest.mark.parametrize("n_classes", [2, 3, 5, 10])
    def test_temperature_scaling(self, n_classes):
        cal = CalibrationModule()
        logits = np.random.randn(100, n_classes)
        labels = np.random.randint(0, n_classes, 100)
        temp = cal.temperature_scale(logits, labels)
        assert temp > 0
        assert cal.is_calibrated

    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_calibrate_predictions(self, n_classes):
        cal = CalibrationModule()
        logits = np.random.randn(50, n_classes)
        labels = np.random.randint(0, n_classes, 50)
        cal.temperature_scale(logits, labels)
        probs = cal.calibrate_predictions(logits)
        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""

    @pytest.mark.parametrize("n_models", [2, 3, 5, 7, 10])
    def test_initialization(self, n_models):
        ensemble = EnsemblePredictor(n_models=n_models)
        assert len(ensemble.models) == n_models

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20])
    def test_predict_ensemble(self, n_samples):
        config = BayesianConfig(input_dim=16, output_dim=1)
        ensemble = EnsemblePredictor(n_models=3, config=config)
        x = np.random.randn(n_samples, 16)
        result = ensemble.predict_ensemble(x)
        assert isinstance(result, UncertaintyEstimate)
        assert result.mean.shape == (n_samples, 1)

    @pytest.mark.parametrize("epochs", [1, 3, 5])
    def test_fit_ensemble(self, epochs):
        config = BayesianConfig(input_dim=8, hidden_dims=[16], output_dim=1)
        ensemble = EnsemblePredictor(n_models=3, config=config)
        x = np.random.randn(30, 8)
        y = np.random.randn(30)
        all_losses = ensemble.fit_ensemble(x, y, epochs=epochs)
        assert len(all_losses) == 3
        for losses in all_losses:
            assert len(losses) == epochs


# ============================================================
# GENERATIVE MODEL TESTS
# ============================================================


class TestVAEConfig:
    """Tests for VAEConfig."""

    @pytest.mark.parametrize("input_dim", [16, 32, 64, 128, 256])
    def test_input_dim(self, input_dim):
        config = VAEConfig(input_dim=input_dim)
        assert config.input_dim == input_dim

    @pytest.mark.parametrize("latent_dim", [2, 4, 8, 16, 32])
    def test_latent_dim(self, latent_dim):
        config = VAEConfig(latent_dim=latent_dim)
        assert config.latent_dim == latent_dim

    @pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_beta(self, beta):
        config = VAEConfig(beta=beta)
        assert config.beta == beta


class TestSpectralVAE:
    """Tests for SpectralVAE."""

    @pytest.mark.parametrize("input_dim,latent_dim", [
        (16, 4), (32, 8), (64, 16), (128, 32), (256, 64),
    ])
    def test_initialization(self, input_dim, latent_dim):
        config = VAEConfig(input_dim=input_dim, latent_dim=latent_dim)
        vae = SpectralVAE(config)
        assert not vae.is_trained

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20, 50])
    def test_encode_shape(self, n_samples):
        config = VAEConfig(input_dim=32, latent_dim=8)
        vae = SpectralVAE(config)
        x = np.random.randn(n_samples, 32)
        mu, logvar = vae.encode(x)
        assert mu.shape == (n_samples, 8)
        assert logvar.shape == (n_samples, 8)

    @pytest.mark.parametrize("latent_dim", [2, 4, 8, 16, 32])
    def test_reparameterize(self, latent_dim):
        config = VAEConfig(input_dim=32, latent_dim=latent_dim)
        vae = SpectralVAE(config)
        mu = np.zeros((5, latent_dim))
        logvar = np.zeros((5, latent_dim))
        z = vae.reparameterize(mu, logvar)
        assert z.shape == (5, latent_dim)

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20])
    def test_decode_shape(self, n_samples):
        config = VAEConfig(input_dim=32, latent_dim=8)
        vae = SpectralVAE(config)
        z = np.random.randn(n_samples, 8)
        output = vae.decode(z)
        assert output.shape == (n_samples, 32)

    @pytest.mark.parametrize("n_samples", [1, 5, 10])
    def test_forward_pass(self, n_samples):
        config = VAEConfig(input_dim=32, latent_dim=8)
        vae = SpectralVAE(config)
        x = np.random.randn(n_samples, 32)
        reconstruction, mu, logvar = vae.forward(x)
        assert reconstruction.shape == (n_samples, 32)
        assert mu.shape == (n_samples, 8)
        assert logvar.shape == (n_samples, 8)

    @pytest.mark.parametrize("beta", [0.1, 1.0, 5.0])
    def test_loss_computation(self, beta):
        config = VAEConfig(input_dim=32, latent_dim=8, beta=beta)
        vae = SpectralVAE(config)
        x = np.random.randn(10, 32)
        recon, mu, logvar = vae.forward(x)
        total, recon_loss, kl = vae.compute_loss(x, recon, mu, logvar)
        assert total >= 0
        assert recon_loss >= 0
        assert kl >= 0

    @pytest.mark.parametrize("n_generated", [1, 5, 10, 20, 50, 100])
    def test_generate(self, n_generated):
        config = VAEConfig(input_dim=32, latent_dim=8)
        vae = SpectralVAE(config)
        result = vae.generate(n_generated)
        assert isinstance(result, GenerationResult)
        assert result.samples.shape == (n_generated, 32)
        assert result.latent_codes.shape == (n_generated, 8)

    @pytest.mark.parametrize("epochs", [1, 3, 5, 10])
    def test_training(self, epochs):
        config = VAEConfig(input_dim=16, latent_dim=4, hidden_dims=[8])
        vae = SpectralVAE(config)
        x = np.random.randn(30, 16)
        losses = vae.fit(x, epochs=epochs)
        assert len(losses) == epochs
        assert vae.is_trained

    @pytest.mark.parametrize("n_samples", [5, 10, 20])
    def test_reconstruct(self, n_samples):
        config = VAEConfig(input_dim=32, latent_dim=8)
        vae = SpectralVAE(config)
        x = np.random.randn(n_samples, 32)
        result = vae.reconstruct(x)
        assert result.samples.shape == (n_samples, 32)
        assert result.reconstruction_error is not None


class TestSpectralDiffusion:
    """Tests for SpectralDiffusion."""

    @pytest.mark.parametrize("n_timesteps", [10, 25, 50, 100])
    def test_initialization(self, n_timesteps):
        config = DiffusionConfig(n_timesteps=n_timesteps)
        model = SpectralDiffusion(config)
        assert len(model._betas) == n_timesteps
        assert len(model._alphas) == n_timesteps

    @pytest.mark.parametrize("t", [0, 5, 10, 25, 49])
    def test_add_noise(self, t):
        config = DiffusionConfig(input_dim=32, n_timesteps=50)
        model = SpectralDiffusion(config)
        x = np.random.randn(5, 32)
        noisy, noise = model.add_noise(x, t)
        assert noisy.shape == x.shape
        assert noise.shape == x.shape

    @pytest.mark.parametrize("t", [0, 10, 25, 49])
    def test_predict_noise(self, t):
        config = DiffusionConfig(input_dim=32, n_timesteps=50)
        model = SpectralDiffusion(config)
        x = np.random.randn(5, 32)
        predicted = model.predict_noise(x, t)
        assert predicted.shape == x.shape

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20])
    def test_generate(self, n_samples):
        config = DiffusionConfig(input_dim=16, n_timesteps=20)
        model = SpectralDiffusion(config)
        result = model.generate(n_samples)
        assert isinstance(result, GenerationResult)
        assert result.samples.shape == (n_samples, 16)

    @pytest.mark.parametrize("epochs", [1, 3, 5])
    def test_training(self, epochs):
        config = DiffusionConfig(input_dim=16, n_timesteps=20)
        model = SpectralDiffusion(config)
        x = np.random.randn(20, 16)
        losses = model.fit(x, epochs=epochs)
        assert len(losses) == epochs
        assert model.is_trained

    @pytest.mark.parametrize("input_dim", [8, 16, 32, 64])
    def test_different_dimensions(self, input_dim):
        config = DiffusionConfig(input_dim=input_dim, n_timesteps=20)
        model = SpectralDiffusion(config)
        result = model.generate(5)
        assert result.samples.shape == (5, input_dim)


class TestSpectralGAN:
    """Tests for SpectralGAN."""

    @pytest.mark.parametrize("input_dim,latent_dim", [
        (16, 4), (32, 8), (64, 16), (128, 32),
    ])
    def test_initialization(self, input_dim, latent_dim):
        gan = SpectralGAN(input_dim=input_dim, latent_dim=latent_dim)
        assert not gan.is_trained

    @pytest.mark.parametrize("n_samples", [1, 5, 10, 20, 50])
    def test_generate(self, n_samples):
        gan = SpectralGAN(input_dim=32, latent_dim=8)
        result = gan.generate(n_samples)
        assert result.samples.shape == (n_samples, 32)
        assert result.latent_codes.shape == (n_samples, 8)

    @pytest.mark.parametrize("n_samples", [5, 10, 20])
    def test_discriminate(self, n_samples):
        gan = SpectralGAN(input_dim=32, latent_dim=8)
        x = np.random.randn(n_samples, 32)
        scores = gan.discriminate(x)
        assert scores.shape == (n_samples, 1)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    @pytest.mark.parametrize("epochs", [1, 3, 5])
    def test_training(self, epochs):
        gan = SpectralGAN(input_dim=16, latent_dim=4)
        real_data = np.random.randn(30, 16)
        losses = gan.fit(real_data, epochs=epochs)
        assert "generator_losses" in losses
        assert "discriminator_losses" in losses
        assert len(losses["generator_losses"]) == epochs
        assert gan.is_trained

    @pytest.mark.parametrize("input_dim", [8, 16, 32, 64])
    def test_output_bounded(self, input_dim):
        gan = SpectralGAN(input_dim=input_dim, latent_dim=8)
        result = gan.generate(10)
        # tanh output should be in [-1, 1]
        assert np.all(result.samples >= -1)
        assert np.all(result.samples <= 1)
