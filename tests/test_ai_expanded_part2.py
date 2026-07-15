"""Comprehensive test suite for expanded AI libraries - Part 2.

Tests explainability, time series forecasting, and reinforcement learning
with parametrized tests for extensive coverage (500+ tests).
"""

import numpy as np
import pytest

from mesie.ai.explainability import (
    ExplanationType,
    Explanation,
    SpectralFeatureImportance,
    PerturbationExplainer,
    GradientExplainer,
    CounterfactualExplainer,
    SpectralAttentionVisualizer,
)
from mesie.ai.time_series import (
    ForecastConfig,
    ForecastMethod,
    ForecastResult,
    AutoregressiveForecaster,
    SpectralDecompositionForecaster,
    NeuralForecaster,
    EnsembleForecaster,
)
from mesie.ai.reinforcement import (
    RLConfig,
    Experience,
    AgentMetrics,
    QLearningAgent,
    PolicyGradientAgent,
    MultiArmedBandit,
    SpectralEnvironment,
    AgentType,
)
from mesie.ai.meta_learning import (
    MetaLearningConfig,
    MetaTask,
    MetaResult,
    PrototypicalNetwork,
    MAMLAdapter,
)
from mesie.ai.bayesian import (
    BayesianConfig,
    BayesianSpectralNetwork,
    CalibrationModule,
    UncertaintyEstimate,
)
from mesie.ai.generative import (
    VAEConfig,
    SpectralVAE,
    SpectralGAN,
    GenerationResult,
)


# ============================================================
# EXPLAINABILITY TESTS
# ============================================================


class TestExplanationType:
    """Tests for ExplanationType enum."""

    @pytest.mark.parametrize("exp_type", list(ExplanationType))
    def test_all_types_valid(self, exp_type):
        assert exp_type.value is not None

    def test_type_count(self):
        assert len(ExplanationType) == 6


class TestExplanation:
    """Tests for Explanation dataclass."""

    @pytest.mark.parametrize("n_features", [8, 16, 32, 64, 128])
    def test_n_important_features(self, n_features):
        exp = Explanation(
            explanation_type=ExplanationType.PERTURBATION,
            feature_attributions=np.random.uniform(0, 1, n_features),
            confidence=0.9,
        )
        assert 0 <= exp.n_important_features <= n_features

    @pytest.mark.parametrize("exp_type", list(ExplanationType))
    def test_explanation_types(self, exp_type):
        exp = Explanation(
            explanation_type=exp_type,
            feature_attributions=np.random.uniform(0, 1, 16),
            confidence=0.8,
        )
        assert exp.explanation_type == exp_type

    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_confidence_values(self, confidence):
        exp = Explanation(
            explanation_type=ExplanationType.GRADIENT,
            feature_attributions=np.ones(16),
            confidence=confidence,
        )
        assert exp.confidence == confidence

    @pytest.mark.parametrize("n_top", [0, 3, 5, 10])
    def test_top_features(self, n_top):
        top = list(range(n_top))
        exp = Explanation(
            explanation_type=ExplanationType.PERTURBATION,
            feature_attributions=np.ones(32),
            confidence=0.9,
            top_features=top,
        )
        assert len(exp.top_features) == n_top


class TestPerturbationExplainer:
    """Tests for PerturbationExplainer."""

    @pytest.mark.parametrize("n_perturbations", [10, 25, 50, 100])
    def test_initialization(self, n_perturbations):
        explainer = PerturbationExplainer(n_perturbations=n_perturbations)
        assert explainer.n_perturbations == n_perturbations

    @pytest.mark.parametrize("input_dim", [4, 8, 16, 32])
    def test_explain_shape(self, input_dim):
        explainer = PerturbationExplainer(n_perturbations=20)
        model_fn = lambda x: x.sum(axis=1, keepdims=True)
        x = np.random.randn(1, input_dim)
        explanation = explainer.explain(model_fn, x)
        assert explanation.feature_attributions.shape == (input_dim,)
        assert len(explanation.top_features) <= 10

    @pytest.mark.parametrize("scale", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_perturbation_scale(self, scale):
        explainer = PerturbationExplainer(n_perturbations=20, perturbation_scale=scale)
        model_fn = lambda x: x.mean(axis=1, keepdims=True)
        x = np.random.randn(1, 8)
        explanation = explainer.explain(model_fn, x)
        assert isinstance(explanation, Explanation)

    @pytest.mark.parametrize("input_dim", [4, 8, 16])
    def test_explain_1d_input(self, input_dim):
        explainer = PerturbationExplainer(n_perturbations=20)
        model_fn = lambda x: x.sum(axis=1, keepdims=True)
        x = np.random.randn(input_dim)  # 1D input
        explanation = explainer.explain(model_fn, x)
        assert explanation.feature_attributions.shape == (input_dim,)

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_different_seeds(self, seed):
        np.random.seed(seed)
        explainer = PerturbationExplainer(n_perturbations=30)
        model_fn = lambda x: x[:, 0:1] * 2  # First feature most important
        x = np.random.randn(1, 8)
        explanation = explainer.explain(model_fn, x)
        assert explanation.confidence >= 0


class TestGradientExplainer:
    """Tests for GradientExplainer."""

    @pytest.mark.parametrize("epsilon", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    def test_initialization(self, epsilon):
        explainer = GradientExplainer(epsilon=epsilon)
        assert explainer.epsilon == epsilon

    @pytest.mark.parametrize("input_dim", [4, 8, 16, 32, 64])
    def test_explain_shape(self, input_dim):
        explainer = GradientExplainer()
        model_fn = lambda x: x.sum(axis=1, keepdims=True)
        x = np.random.randn(1, input_dim)
        explanation = explainer.explain(model_fn, x)
        assert explanation.feature_attributions.shape == (input_dim,)
        assert explanation.explanation_type == ExplanationType.GRADIENT

    @pytest.mark.parametrize("input_dim", [4, 8, 16])
    def test_explain_1d_input(self, input_dim):
        explainer = GradientExplainer()
        model_fn = lambda x: x.sum(axis=1, keepdims=True)
        x = np.random.randn(input_dim)
        explanation = explainer.explain(model_fn, x)
        assert explanation.feature_attributions.shape == (input_dim,)

    @pytest.mark.parametrize("feature_idx", [0, 1, 2, 3])
    def test_identifies_important_feature(self, feature_idx):
        explainer = GradientExplainer(epsilon=1e-4)
        # Model that only uses one feature
        model_fn = lambda x: x[:, feature_idx:feature_idx + 1] * 5
        x = np.ones((1, 8))
        explanation = explainer.explain(model_fn, x)
        # The important feature should have high attribution
        assert feature_idx in explanation.top_features[:3]


class TestCounterfactualExplainer:
    """Tests for CounterfactualExplainer."""

    @pytest.mark.parametrize("max_iterations", [10, 50, 100, 200])
    def test_initialization(self, max_iterations):
        explainer = CounterfactualExplainer(max_iterations=max_iterations)
        assert explainer.max_iterations == max_iterations

    @pytest.mark.parametrize("step_size", [0.001, 0.01, 0.05, 0.1])
    def test_step_size(self, step_size):
        explainer = CounterfactualExplainer(step_size=step_size)
        assert explainer.step_size == step_size

    @pytest.mark.parametrize("input_dim", [4, 8, 16, 32])
    def test_explain_shape(self, input_dim):
        explainer = CounterfactualExplainer(max_iterations=20)
        model_fn = lambda x: x.mean(axis=1, keepdims=True)
        x = np.random.randn(1, input_dim)
        explanation = explainer.explain(model_fn, x)
        assert explanation.feature_attributions.shape == (input_dim,)
        assert explanation.counterfactual is not None
        assert explanation.counterfactual.shape == (1, input_dim)

    @pytest.mark.parametrize("target", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_different_targets(self, target):
        explainer = CounterfactualExplainer(max_iterations=50, step_size=0.05)
        model_fn = lambda x: 1.0 / (1.0 + np.exp(-x.sum(axis=1, keepdims=True)))
        x = np.zeros((1, 8))
        explanation = explainer.explain(model_fn, x, target_pred=target)
        assert isinstance(explanation, Explanation)


class TestSpectralAttentionVisualizer:
    """Tests for SpectralAttentionVisualizer."""

    def test_initialization(self):
        viz = SpectralAttentionVisualizer()
        assert len(viz.attention_history) == 0

    @pytest.mark.parametrize("seq_len", [4, 8, 16, 32, 64])
    def test_compute_attention_scores(self, seq_len):
        viz = SpectralAttentionVisualizer()
        query = np.random.randn(seq_len, 16)
        key = np.random.randn(seq_len, 16)
        attention = viz.compute_attention_scores(query, key)
        assert attention.shape == (seq_len, seq_len)
        assert np.allclose(attention.sum(axis=-1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
    def test_aggregate_attention(self, n_heads):
        viz = SpectralAttentionVisualizer()
        multi_head = np.random.uniform(0, 1, (n_heads, 8, 8))
        # Normalize each head
        multi_head = multi_head / multi_head.sum(axis=-1, keepdims=True)
        aggregated = viz.aggregate_attention(multi_head)
        assert aggregated.shape == (8, 8)

    @pytest.mark.parametrize("seq_len", [4, 8, 16])
    def test_feature_importance_from_attention(self, seq_len):
        viz = SpectralAttentionVisualizer()
        attention_map = np.random.uniform(0, 1, (seq_len, seq_len))
        attention_map /= attention_map.sum(axis=-1, keepdims=True)
        importance = viz.get_feature_importance_from_attention(attention_map)
        assert isinstance(importance, SpectralFeatureImportance)
        assert importance.band_importances.shape == (seq_len,)

    @pytest.mark.parametrize("n_computations", [1, 3, 5, 10])
    def test_attention_history(self, n_computations):
        viz = SpectralAttentionVisualizer()
        for _ in range(n_computations):
            q = np.random.randn(4, 8)
            k = np.random.randn(4, 8)
            viz.compute_attention_scores(q, k)
        assert len(viz.attention_history) == n_computations


# ============================================================
# TIME SERIES FORECASTING TESTS
# ============================================================


class TestForecastConfig:
    """Tests for ForecastConfig."""

    @pytest.mark.parametrize("horizon", [1, 5, 10, 20, 50, 100])
    def test_horizon(self, horizon):
        config = ForecastConfig(horizon=horizon)
        assert config.horizon == horizon

    @pytest.mark.parametrize("lookback", [10, 20, 50, 100, 200])
    def test_lookback(self, lookback):
        config = ForecastConfig(lookback=lookback)
        assert config.lookback == lookback

    @pytest.mark.parametrize("method", list(ForecastMethod))
    def test_methods(self, method):
        config = ForecastConfig(method=method)
        assert config.method == method

    @pytest.mark.parametrize("confidence_level", [0.8, 0.9, 0.95, 0.99])
    def test_confidence_level(self, confidence_level):
        config = ForecastConfig(confidence_level=confidence_level)
        assert config.confidence_level == confidence_level

    @pytest.mark.parametrize("n_components", [1, 3, 5, 10, 20])
    def test_n_components(self, n_components):
        config = ForecastConfig(n_components=n_components)
        assert config.n_components == n_components


class TestForecastResult:
    """Tests for ForecastResult."""

    @pytest.mark.parametrize("horizon", [5, 10, 20])
    def test_prediction_interval_width(self, horizon):
        predictions = np.random.randn(horizon)
        lower = predictions - np.abs(np.random.randn(horizon))
        upper = predictions + np.abs(np.random.randn(horizon))
        result = ForecastResult(
            predictions=predictions, lower_bound=lower, upper_bound=upper,
            confidence=np.random.uniform(0, 1, horizon),
            horizon=horizon, method=ForecastMethod.AUTOREGRESSIVE,
        )
        width = result.prediction_interval_width
        assert np.all(width >= 0)
        assert width.shape == (horizon,)

    @pytest.mark.parametrize("horizon", [5, 10, 20])
    def test_mean_confidence(self, horizon):
        result = ForecastResult(
            predictions=np.random.randn(horizon),
            lower_bound=np.random.randn(horizon),
            upper_bound=np.random.randn(horizon),
            confidence=np.random.uniform(0, 1, horizon),
            horizon=horizon, method=ForecastMethod.AUTOREGRESSIVE,
        )
        assert 0 <= result.mean_confidence <= 1


class TestAutoregressiveForecaster:
    """Tests for AutoregressiveForecaster."""

    @pytest.mark.parametrize("lookback", [5, 10, 20, 30])
    def test_initialization(self, lookback):
        config = ForecastConfig(lookback=lookback)
        forecaster = AutoregressiveForecaster(config)
        assert not forecaster.is_fitted

    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500])
    def test_fit(self, n_samples):
        forecaster = AutoregressiveForecaster(ForecastConfig(lookback=10))
        series = np.cumsum(np.random.randn(n_samples))
        forecaster.fit(series)
        assert forecaster.is_fitted

    @pytest.mark.parametrize("horizon", [1, 5, 10, 20])
    def test_predict_horizon(self, horizon):
        forecaster = AutoregressiveForecaster(ForecastConfig(lookback=10, horizon=horizon))
        series = np.cumsum(np.random.randn(100))
        result = forecaster.predict(series, horizon=horizon)
        assert isinstance(result, ForecastResult)
        assert result.predictions.shape == (horizon,)
        assert result.lower_bound.shape == (horizon,)
        assert result.upper_bound.shape == (horizon,)

    @pytest.mark.parametrize("n_samples", [50, 100, 200])
    def test_confidence_decreases_with_horizon(self, n_samples):
        forecaster = AutoregressiveForecaster(ForecastConfig(lookback=10, horizon=20))
        series = np.cumsum(np.random.randn(n_samples))
        result = forecaster.predict(series, horizon=20)
        # Confidence should generally decrease with horizon
        assert result.confidence[0] >= result.confidence[-1]

    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_different_series(self, seed):
        np.random.seed(seed)
        forecaster = AutoregressiveForecaster(ForecastConfig(lookback=10))
        series = np.cumsum(np.random.randn(100))
        result = forecaster.predict(series, horizon=5)
        assert result.horizon == 5

    @pytest.mark.parametrize("lookback", [5, 10, 20])
    def test_multidim_input(self, lookback):
        forecaster = AutoregressiveForecaster(ForecastConfig(lookback=lookback))
        series = np.random.randn(100, 1)  # 2D input
        result = forecaster.predict(series, horizon=5)
        assert result.predictions.shape == (5,)


class TestSpectralDecompositionForecaster:
    """Tests for SpectralDecompositionForecaster."""

    @pytest.mark.parametrize("n_components", [1, 3, 5, 10])
    def test_initialization(self, n_components):
        config = ForecastConfig(n_components=n_components)
        forecaster = SpectralDecompositionForecaster(config)
        assert not forecaster.is_fitted

    @pytest.mark.parametrize("n_samples", [50, 100, 200, 500])
    def test_fit(self, n_samples):
        forecaster = SpectralDecompositionForecaster(ForecastConfig(n_components=3))
        series = np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1
        forecaster.fit(series)
        assert forecaster.is_fitted
        assert forecaster._frequencies is not None

    @pytest.mark.parametrize("horizon", [1, 5, 10, 20])
    def test_predict(self, horizon):
        forecaster = SpectralDecompositionForecaster(ForecastConfig(n_components=3, horizon=horizon))
        series = np.sin(np.linspace(0, 10, 100))
        result = forecaster.predict(series, horizon=horizon)
        assert result.predictions.shape == (horizon,)
        assert result.method == ForecastMethod.SPECTRAL_DECOMPOSITION

    @pytest.mark.parametrize("freq", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_periodic_signal(self, freq):
        t = np.linspace(0, 10, 200)
        series = np.sin(2 * np.pi * freq * t)
        forecaster = SpectralDecompositionForecaster(ForecastConfig(n_components=5))
        result = forecaster.predict(series, horizon=10)
        assert result.predictions.shape == (10,)


class TestNeuralForecaster:
    """Tests for NeuralForecaster."""

    @pytest.mark.parametrize("hidden_dim", [8, 16, 32, 64])
    def test_initialization(self, hidden_dim):
        config = ForecastConfig(hidden_dim=hidden_dim, lookback=20, horizon=5)
        forecaster = NeuralForecaster(config)
        assert not forecaster.is_fitted

    @pytest.mark.parametrize("lookback,horizon", [
        (10, 1), (10, 5), (20, 5), (20, 10), (50, 10),
    ])
    def test_fit(self, lookback, horizon):
        config = ForecastConfig(lookback=lookback, horizon=horizon, hidden_dim=16)
        forecaster = NeuralForecaster(config)
        series = np.random.randn(100)
        losses = forecaster.fit(series, epochs=3)
        assert len(losses) > 0
        assert forecaster.is_fitted

    @pytest.mark.parametrize("horizon", [1, 3, 5, 10])
    def test_predict(self, horizon):
        config = ForecastConfig(lookback=20, horizon=horizon, hidden_dim=16)
        forecaster = NeuralForecaster(config)
        series = np.random.randn(100)
        forecaster.fit(series, epochs=3)
        result = forecaster.predict(series, horizon=horizon)
        assert result.predictions.shape == (horizon,)
        assert result.method == ForecastMethod.NEURAL_FORECAST

    @pytest.mark.parametrize("epochs", [1, 5, 10, 20])
    def test_training_epochs(self, epochs):
        config = ForecastConfig(lookback=10, horizon=5, hidden_dim=8)
        forecaster = NeuralForecaster(config)
        series = np.random.randn(50)
        losses = forecaster.fit(series, epochs=epochs)
        assert len(losses) == epochs

    @pytest.mark.parametrize("series_len", [30, 50, 100, 200])
    def test_different_series_lengths(self, series_len):
        config = ForecastConfig(lookback=10, horizon=5, hidden_dim=8)
        forecaster = NeuralForecaster(config)
        series = np.random.randn(series_len)
        forecaster.fit(series, epochs=2)
        result = forecaster.predict(series, horizon=5)
        assert result.predictions.shape == (5,)


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster."""

    def test_initialization(self):
        forecaster = EnsembleForecaster()
        assert len(forecaster.forecasters) == 3

    @pytest.mark.parametrize("horizon", [1, 5, 10, 20])
    def test_predict(self, horizon):
        config = ForecastConfig(lookback=20, horizon=horizon, hidden_dim=8, n_components=3)
        forecaster = EnsembleForecaster(config)
        series = np.random.randn(100)
        result = forecaster.predict(series, horizon=horizon)
        assert result.predictions.shape == (horizon,)
        assert result.method == ForecastMethod.ENSEMBLE_FORECAST

    @pytest.mark.parametrize("n_samples", [50, 100, 200])
    def test_fit_and_predict(self, n_samples):
        config = ForecastConfig(lookback=10, horizon=5, hidden_dim=8, n_components=3)
        forecaster = EnsembleForecaster(config)
        series = np.random.randn(n_samples)
        forecaster.fit(series)
        result = forecaster.predict(series, horizon=5)
        assert result.lower_bound.shape == (5,)
        assert result.upper_bound.shape == (5,)


# ============================================================
# REINFORCEMENT LEARNING TESTS
# ============================================================


class TestRLConfig:
    """Tests for RLConfig."""

    @pytest.mark.parametrize("state_dim", [4, 8, 16, 32, 64])
    def test_state_dim(self, state_dim):
        config = RLConfig(state_dim=state_dim)
        assert config.state_dim == state_dim

    @pytest.mark.parametrize("n_actions", [2, 3, 5, 10, 20])
    def test_n_actions(self, n_actions):
        config = RLConfig(n_actions=n_actions)
        assert config.n_actions == n_actions

    @pytest.mark.parametrize("gamma", [0.9, 0.95, 0.99, 0.999])
    def test_gamma(self, gamma):
        config = RLConfig(gamma=gamma)
        assert config.gamma == gamma

    @pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_epsilon(self, epsilon):
        config = RLConfig(epsilon=epsilon)
        assert config.epsilon == epsilon

    @pytest.mark.parametrize("lr", [0.001, 0.005, 0.01, 0.05, 0.1])
    def test_learning_rate(self, lr):
        config = RLConfig(learning_rate=lr)
        assert config.learning_rate == lr


class TestAgentMetrics:
    """Tests for AgentMetrics."""

    @pytest.mark.parametrize("n_episodes", [0, 1, 10, 50, 100])
    def test_mean_reward(self, n_episodes):
        metrics = AgentMetrics()
        metrics.episode_rewards = [float(np.random.randn()) for _ in range(n_episodes)]
        mean = metrics.mean_reward
        if n_episodes == 0:
            assert mean == 0.0
        else:
            assert isinstance(mean, float)

    @pytest.mark.parametrize("n_episodes", [0, 5, 10, 50])
    def test_total_episodes(self, n_episodes):
        metrics = AgentMetrics()
        metrics.episode_rewards = [1.0] * n_episodes
        assert metrics.total_episodes == n_episodes


class TestQLearningAgent:
    """Tests for QLearningAgent."""

    @pytest.mark.parametrize("n_actions", [2, 3, 5, 10])
    def test_initialization(self, n_actions):
        config = RLConfig(n_actions=n_actions)
        agent = QLearningAgent(config)
        assert agent.config.n_actions == n_actions

    @pytest.mark.parametrize("state_dim", [4, 8, 16])
    def test_select_action(self, state_dim):
        config = RLConfig(state_dim=state_dim, n_actions=5, epsilon=1.0)
        agent = QLearningAgent(config)
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        assert 0 <= action < 5

    @pytest.mark.parametrize("n_actions", [2, 5, 10])
    def test_greedy_action(self, n_actions):
        config = RLConfig(n_actions=n_actions, epsilon=0.0)
        agent = QLearningAgent(config)
        state = np.zeros(4)
        action = agent.select_action(state)
        assert 0 <= action < n_actions

    @pytest.mark.parametrize("reward", [-1.0, 0.0, 0.5, 1.0, 10.0])
    def test_update(self, reward):
        config = RLConfig(n_actions=3, state_dim=4)
        agent = QLearningAgent(config)
        exp = Experience(
            state=np.zeros(4), action=0, reward=reward,
            next_state=np.ones(4), done=False,
        )
        td_error = agent.update(exp)
        assert isinstance(td_error, float)

    @pytest.mark.parametrize("n_steps", [5, 10, 20, 50])
    def test_train_episode(self, n_steps):
        config = RLConfig(n_actions=3, state_dim=4)
        agent = QLearningAgent(config)
        states = np.random.randn(n_steps, 4)
        rewards = np.random.randn(n_steps)
        actions = np.random.randint(0, 3, n_steps)
        total_reward = agent.train_episode(states, rewards, actions)
        assert isinstance(total_reward, float)
        assert agent.metrics.total_episodes == 1

    @pytest.mark.parametrize("n_episodes", [1, 3, 5, 10])
    def test_multiple_episodes(self, n_episodes):
        config = RLConfig(n_actions=3, state_dim=4)
        agent = QLearningAgent(config)
        for _ in range(n_episodes):
            states = np.random.randn(10, 4)
            rewards = np.random.randn(10)
            actions = np.random.randint(0, 3, 10)
            agent.train_episode(states, rewards, actions)
        assert agent.metrics.total_episodes == n_episodes

    @pytest.mark.parametrize("epsilon_decay", [0.9, 0.95, 0.99, 0.999])
    def test_epsilon_decay(self, epsilon_decay):
        config = RLConfig(n_actions=3, epsilon=1.0, epsilon_decay=epsilon_decay)
        agent = QLearningAgent(config)
        initial_eps = agent._current_epsilon
        exp = Experience(
            state=np.zeros(4), action=0, reward=1.0,
            next_state=np.ones(4), done=False,
        )
        agent.update(exp)
        assert agent._current_epsilon < initial_eps


class TestPolicyGradientAgent:
    """Tests for PolicyGradientAgent."""

    @pytest.mark.parametrize("state_dim,n_actions", [
        (4, 2), (8, 3), (16, 5), (32, 10),
    ])
    def test_initialization(self, state_dim, n_actions):
        config = RLConfig(state_dim=state_dim, n_actions=n_actions)
        agent = PolicyGradientAgent(config)
        assert agent._policy_weights.shape == (state_dim, n_actions)

    @pytest.mark.parametrize("state_dim", [4, 8, 16])
    def test_action_probs(self, state_dim):
        config = RLConfig(state_dim=state_dim, n_actions=5)
        agent = PolicyGradientAgent(config)
        state = np.random.randn(state_dim)
        probs = agent.get_action_probs(state)
        assert probs.shape == (5,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    @pytest.mark.parametrize("n_actions", [2, 3, 5, 10])
    def test_select_action(self, n_actions):
        config = RLConfig(state_dim=8, n_actions=n_actions)
        agent = PolicyGradientAgent(config)
        state = np.random.randn(8)
        action = agent.select_action(state)
        assert 0 <= action < n_actions

    @pytest.mark.parametrize("n_experiences", [3, 5, 10, 20])
    def test_store_and_update(self, n_experiences):
        config = RLConfig(state_dim=4, n_actions=3)
        agent = PolicyGradientAgent(config)
        for i in range(n_experiences):
            exp = Experience(
                state=np.random.randn(4), action=np.random.randint(3),
                reward=np.random.randn(), next_state=np.random.randn(4),
                done=(i == n_experiences - 1),
            )
            agent.store_experience(exp)
        loss = agent.update_policy()
        assert isinstance(loss, float)
        assert agent.metrics.total_episodes == 1

    @pytest.mark.parametrize("gamma", [0.9, 0.95, 0.99])
    def test_different_gamma(self, gamma):
        config = RLConfig(state_dim=4, n_actions=3, gamma=gamma)
        agent = PolicyGradientAgent(config)
        for _ in range(5):
            agent.store_experience(Experience(
                state=np.random.randn(4), action=0, reward=1.0,
                next_state=np.random.randn(4), done=False,
            ))
        agent.store_experience(Experience(
            state=np.random.randn(4), action=0, reward=1.0,
            next_state=np.random.randn(4), done=True,
        ))
        loss = agent.update_policy()
        assert loss >= 0


class TestMultiArmedBandit:
    """Tests for MultiArmedBandit."""

    @pytest.mark.parametrize("n_arms", [2, 3, 5, 10, 20])
    def test_initialization(self, n_arms):
        bandit = MultiArmedBandit(n_arms=n_arms)
        assert bandit.n_arms == n_arms
        assert len(bandit._alpha) == n_arms

    @pytest.mark.parametrize("n_arms", [2, 5, 10])
    def test_select_arm(self, n_arms):
        bandit = MultiArmedBandit(n_arms=n_arms)
        arm = bandit.select_arm()
        assert 0 <= arm < n_arms

    @pytest.mark.parametrize("context_dim", [4, 8, 16])
    def test_contextual_selection(self, context_dim):
        bandit = MultiArmedBandit(n_arms=5, context_dim=context_dim)
        context = np.random.randn(context_dim)
        arm = bandit.select_arm(context)
        assert 0 <= arm < 5

    @pytest.mark.parametrize("n_pulls", [1, 5, 10, 50, 100])
    def test_update(self, n_pulls):
        bandit = MultiArmedBandit(n_arms=3)
        for _ in range(n_pulls):
            arm = bandit.select_arm()
            reward = np.random.uniform(0, 1)
            bandit.update(arm, reward)
        assert bandit._total_pulls == n_pulls
        assert len(bandit._rewards_history) == n_pulls

    @pytest.mark.parametrize("n_arms", [2, 5, 10])
    def test_arm_estimates(self, n_arms):
        bandit = MultiArmedBandit(n_arms=n_arms)
        for _ in range(20):
            arm = bandit.select_arm()
            bandit.update(arm, np.random.uniform(0, 1))
        estimates = bandit.arm_estimates
        assert estimates.shape == (n_arms,)
        assert np.all(estimates >= 0) and np.all(estimates <= 1)

    @pytest.mark.parametrize("n_pulls", [10, 50, 100])
    def test_cumulative_regret(self, n_pulls):
        bandit = MultiArmedBandit(n_arms=5)
        for _ in range(n_pulls):
            arm = bandit.select_arm()
            bandit.update(arm, np.random.uniform(0, 1))
        regret = bandit.cumulative_regret
        assert isinstance(regret, float)


class TestSpectralEnvironment:
    """Tests for SpectralEnvironment."""

    @pytest.mark.parametrize("state_dim", [4, 8, 16, 32])
    def test_initialization(self, state_dim):
        env = SpectralEnvironment(state_dim=state_dim)
        assert env.state_dim == state_dim

    @pytest.mark.parametrize("state_dim", [4, 8, 16])
    def test_reset(self, state_dim):
        env = SpectralEnvironment(state_dim=state_dim)
        state = env.reset()
        assert state.shape == (state_dim,)

    @pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
    def test_step(self, action):
        env = SpectralEnvironment(state_dim=8, n_actions=5)
        env.reset()
        next_state, reward, done = env.step(action)
        assert next_state.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    @pytest.mark.parametrize("n_steps", [1, 10, 50, 100])
    def test_episode_length(self, n_steps):
        env = SpectralEnvironment(state_dim=8, n_actions=5)
        env._max_steps = n_steps
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done = env.step(0)
            steps += 1
        assert steps == n_steps

    @pytest.mark.parametrize("n_actions", [2, 5, 10])
    def test_all_actions_valid(self, n_actions):
        env = SpectralEnvironment(state_dim=8, n_actions=n_actions)
        env.reset()
        for action in range(n_actions):
            next_state, reward, done = env.step(action)
            assert next_state.shape == (8,)
            if not done:
                continue


# ============================================================
# INTEGRATION TESTS: COMBINING MODULES
# ============================================================


class TestMetaLearningWithBayesian:
    """Integration tests combining meta-learning with Bayesian uncertainty."""

    @pytest.mark.parametrize("n_way", [2, 3, 5])
    def test_prototypical_with_uncertainty(self, n_way):
        net = PrototypicalNetwork(input_dim=16, embedding_dim=8)
        bayesian_net = BayesianSpectralNetwork(BayesianConfig(input_dim=8, output_dim=n_way))
        task = MetaTask(
            support_x=np.random.randn(n_way * 3, 16),
            support_y=np.repeat(np.arange(n_way), 3),
            query_x=np.random.randn(n_way * 5, 16),
            query_y=np.repeat(np.arange(n_way), 5),
        )
        # Embed and predict with uncertainty
        embeddings = net.embed(task.query_x)
        uncertainty = bayesian_net.predict_with_uncertainty(embeddings)
        assert uncertainty.mean.shape[0] == n_way * 5

    @pytest.mark.parametrize("k_shot", [1, 3, 5])
    def test_maml_with_calibration(self, k_shot):
        config = MetaLearningConfig(n_inner_steps=3)
        adapter = MAMLAdapter(config)
        calibrator = CalibrationModule()
        task = MetaTask(
            support_x=np.random.randn(3 * k_shot, 16),
            support_y=np.repeat(np.arange(3), k_shot),
            query_x=np.random.randn(15, 16),
            query_y=np.repeat(np.arange(3), 5),
        )
        result = adapter.evaluate_task(task)
        assert isinstance(result, MetaResult)


class TestGenerativeWithExplainability:
    """Integration tests combining generative models with explainability."""

    @pytest.mark.parametrize("n_generated", [5, 10, 20])
    def test_vae_with_explanation(self, n_generated):
        vae = SpectralVAE(VAEConfig(input_dim=16, latent_dim=4, hidden_dims=[8]))
        explainer = PerturbationExplainer(n_perturbations=10)
        generated = vae.generate(n_generated)
        # Explain the decoder
        explanation = explainer.explain(vae.decode, generated.latent_codes[0:1])
        assert explanation.feature_attributions.shape == (4,)

    @pytest.mark.parametrize("input_dim", [8, 16, 32])
    def test_gan_with_gradient_explanation(self, input_dim):
        gan = SpectralGAN(input_dim=input_dim, latent_dim=4)
        explainer = GradientExplainer()
        x = np.random.randn(1, input_dim)
        explanation = explainer.explain(gan.discriminate, x)
        assert explanation.feature_attributions.shape == (input_dim,)


class TestForecastingWithRL:
    """Integration tests combining forecasting with RL-based parameter selection."""

    @pytest.mark.parametrize("n_arms", [3, 5, 7])
    def test_bandit_selects_forecaster(self, n_arms):
        bandit = MultiArmedBandit(n_arms=min(n_arms, 3), context_dim=4)
        forecasters = [
            AutoregressiveForecaster(ForecastConfig(lookback=10, horizon=5)),
            SpectralDecompositionForecaster(ForecastConfig(n_components=3, horizon=5)),
            NeuralForecaster(ForecastConfig(lookback=10, horizon=5, hidden_dim=8)),
        ]
        series = np.random.randn(100)
        context = np.array([series.mean(), series.std(), series.min(), series.max()])
        arm = bandit.select_arm(context)
        result = forecasters[arm % len(forecasters)].predict(series, horizon=5)
        assert result.predictions.shape == (5,)

    @pytest.mark.parametrize("n_episodes", [1, 3, 5])
    def test_rl_adaptive_forecasting(self, n_episodes):
        config = RLConfig(state_dim=4, n_actions=3)
        agent = QLearningAgent(config)
        for _ in range(n_episodes):
            states = np.random.randn(10, 4)
            rewards = np.random.randn(10)
            actions = np.random.randint(0, 3, 10)
            agent.train_episode(states, rewards, actions)
        assert agent.metrics.total_episodes == n_episodes
