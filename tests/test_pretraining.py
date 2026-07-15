"""Tests for the spectral pretraining module."""

import numpy as np
import pytest

from mesie.pretraining.world_tasks import (
    CoherenceHead,
    HarmonicStructureHead,
    ResonanceHead,
    SpectralDriftHead,
    TemporalLineageHead,
    WorldTaskSuite,
)
from mesie.pretraining.digital_twin import (
    DigitalTwinEnvironment,
    EntityType,
    SpectralEntity,
    SpectralStream,
)
from mesie.pretraining.spectral_memory import (
    LineageQuery,
    MemoryEntry,
    SpectralMemoryStore,
)
from mesie.pretraining.training_recipe import (
    EnvironmentStage,
    FineTuningStage,
    PretrainingStage,
    StageStatus,
    TrainingRecipe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def frequencies():
    """Standard frequency axis for testing."""
    return np.linspace(0.1, 100.0, 256)


@pytest.fixture
def harmonic_signal(frequencies):
    """A signal with clear harmonic structure (f0=10 Hz)."""
    f0 = 10.0
    amplitudes = np.zeros_like(frequencies)
    for h in range(1, 6):
        idx = np.argmin(np.abs(frequencies - f0 * h))
        amplitudes[idx] = 1.0 / h
    # Add noise floor
    amplitudes += 0.01
    return amplitudes


@pytest.fixture
def resonant_signal(frequencies):
    """A signal near resonance (high peak-to-mean ratio)."""
    amplitudes = np.ones_like(frequencies) * 0.1
    # Add a strong resonance peak
    idx = np.argmin(np.abs(frequencies - 25.0))
    amplitudes[idx - 2: idx + 3] = 5.0
    return amplitudes


@pytest.fixture
def embedding_sequence():
    """Temporal sequence of embeddings with drift."""
    rng = np.random.default_rng(42)
    n_steps = 50
    embed_dim = 16
    # Create drifting sequence
    embeddings = np.zeros((n_steps, embed_dim))
    embeddings[0] = rng.standard_normal(embed_dim)
    for t in range(1, n_steps):
        embeddings[t] = embeddings[t - 1] + rng.standard_normal(embed_dim) * 0.1
        # Add sudden drift at step 30
        if t == 30:
            embeddings[t] += rng.standard_normal(embed_dim) * 2.0
    return embeddings


# ---------------------------------------------------------------------------
# Test ResonanceHead
# ---------------------------------------------------------------------------


class TestResonanceHead:
    def test_generates_labels_shape(self, frequencies, resonant_signal):
        head = ResonanceHead(threshold=5.0)
        labels = head.generate_labels(frequencies, resonant_signal)

        assert "classification" in labels
        assert "regression" in labels
        assert "resonance_score" in labels
        assert labels["classification"].shape == (1,)

    def test_detects_resonance(self, frequencies, resonant_signal):
        head = ResonanceHead(threshold=5.0)
        labels = head.generate_labels(frequencies, resonant_signal)
        # Signal has peak-to-mean > 5, should classify as resonant
        assert labels["classification"][0] == 1
        assert labels["resonance_score"][0] > 5.0

    def test_no_resonance_in_flat_signal(self, frequencies):
        head = ResonanceHead(threshold=5.0)
        flat = np.ones_like(frequencies)
        labels = head.generate_labels(frequencies, flat)
        assert labels["classification"][0] == 0

    def test_batch_processing(self, frequencies):
        head = ResonanceHead()
        batch = np.random.rand(10, len(frequencies))
        labels = head.generate_labels(frequencies, batch)
        assert labels["classification"].shape == (10,)
        assert labels["resonance_score"].shape == (10,)

    def test_with_natural_frequencies(self, frequencies, resonant_signal):
        head = ResonanceHead(
            natural_frequencies=np.array([25.0, 50.0, 75.0]),
            bandwidth=2.0,
        )
        labels = head.generate_labels(frequencies, resonant_signal)
        # Peak is near 25 Hz natural frequency
        assert labels["regression"][0] < 1.0

    def test_classification_loss(self):
        head = ResonanceHead()
        predictions = np.array([0.9, 0.1, 0.8])
        targets = np.array([1, 0, 1])
        loss = head.compute_loss(predictions, targets, task="classification")
        assert loss > 0
        assert loss < 1.0  # Good predictions should have low loss

    def test_regression_loss(self):
        head = ResonanceHead()
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 2.9])
        loss = head.compute_loss(predictions, targets, task="regression")
        assert loss > 0
        assert loss < 0.1  # Close predictions


# ---------------------------------------------------------------------------
# Test CoherenceHead
# ---------------------------------------------------------------------------


class TestCoherenceHead:
    def test_identical_signals_have_high_coherence(self, frequencies):
        head = CoherenceHead()
        signal = np.random.rand(len(frequencies))
        components = np.array([signal, signal, signal])
        labels = head.generate_labels(components)

        assert labels["coherence_matrix"].shape == (3, 3)
        # Identical signals should have coherence 1.0
        np.testing.assert_allclose(
            labels["coherence_matrix"], np.ones((3, 3)), atol=1e-10
        )

    def test_uncorrelated_signals(self):
        head = CoherenceHead()
        rng = np.random.default_rng(42)
        components = rng.standard_normal((3, 100))
        labels = head.generate_labels(components)

        # Off-diagonal should be closer to 0.5 (random correlation maps to ~0.5)
        off_diag = labels["coherence_matrix"][0, 1]
        assert 0.0 <= off_diag <= 1.0

    def test_single_component(self):
        head = CoherenceHead()
        components = np.random.rand(1, 50)
        labels = head.generate_labels(components)
        assert labels["coherence_matrix"].shape == (1, 1)
        assert labels["mean_coherence"][0] == 1.0

    def test_frobenius_loss(self):
        head = CoherenceHead()
        predicted = np.eye(3)
        target = np.ones((3, 3)) * 0.5
        np.fill_diagonal(target, 1.0)
        loss = head.compute_loss(predicted, target)
        assert loss > 0


# ---------------------------------------------------------------------------
# Test HarmonicStructureHead
# ---------------------------------------------------------------------------


class TestHarmonicStructureHead:
    def test_detects_harmonic_series(self, frequencies, harmonic_signal):
        head = HarmonicStructureHead(max_harmonics=8, tolerance=0.1)
        labels = head.generate_labels(frequencies, harmonic_signal)

        assert "harmonic_mask" in labels
        assert "fundamental_frequency" in labels
        assert "n_harmonics" in labels
        assert "harmonic_amplitudes" in labels
        assert labels["n_harmonics"][0] >= 2  # Should find at least 2 harmonics

    def test_fundamental_detection(self, frequencies, harmonic_signal):
        head = HarmonicStructureHead(tolerance=0.1)
        labels = head.generate_labels(frequencies, harmonic_signal)
        # Fundamental should be near 10 Hz
        assert 8.0 <= labels["fundamental_frequency"][0] <= 12.0

    def test_no_harmonics_in_noise(self, frequencies):
        head = HarmonicStructureHead()
        noise = np.random.rand(len(frequencies)) * 0.01
        labels = head.generate_labels(frequencies, noise)
        # Harmonic detection runs without error on noise
        assert labels["n_harmonics"][0] >= 0

    def test_harmonic_amplitudes_shape(self, frequencies, harmonic_signal):
        head = HarmonicStructureHead(max_harmonics=8)
        labels = head.generate_labels(frequencies, harmonic_signal)
        assert labels["harmonic_amplitudes"].shape == (8,)

    def test_reconstruction_loss(self):
        head = HarmonicStructureHead()
        predicted = np.array([1.0, 0.5, 0.33, 0.25, 0, 0, 0, 0])
        target = np.array([1.0, 0.5, 0.33, 0.25, 0.2, 0, 0, 0])
        loss = head.compute_loss(predicted, target, task="reconstruction")
        assert loss > 0
        assert loss < 0.01


# ---------------------------------------------------------------------------
# Test SpectralDriftHead
# ---------------------------------------------------------------------------


class TestSpectralDriftHead:
    def test_no_drift_in_stationary(self):
        head = SpectralDriftHead(baseline_window=10)
        # Stationary embeddings
        embeddings = np.ones((50, 8)) + np.random.randn(50, 8) * 0.01
        labels = head.generate_labels(embeddings)

        assert "drift_scores" in labels
        assert "drift_detected" in labels
        assert labels["drift_scores"].shape == (50,)
        # Mostly no drift detected for stationary
        assert np.sum(labels["drift_detected"]) < 10

    def test_detects_sudden_drift(self, embedding_sequence):
        head = SpectralDriftHead(baseline_window=10, drift_metrics=["l2"])
        labels = head.generate_labels(embedding_sequence)

        # Drift scores should be higher after the sudden shift at step 30
        mean_before = np.mean(labels["drift_scores"][10:30])
        mean_after = np.mean(labels["drift_scores"][31:])
        assert mean_after > mean_before

    def test_multiple_metrics(self, embedding_sequence):
        head = SpectralDriftHead(drift_metrics=["l2", "cosine", "kl", "wasserstein"])
        labels = head.generate_labels(embedding_sequence)

        assert "metric_decomposition" in labels
        assert "l2" in labels["metric_decomposition"]
        assert "cosine" in labels["metric_decomposition"]
        assert "kl" in labels["metric_decomposition"]
        assert "wasserstein" in labels["metric_decomposition"]

    def test_fit_baseline_explicit(self):
        head = SpectralDriftHead(baseline_window=5)
        baseline = np.ones((10, 4))
        head.fit_baseline(baseline)

        # Query with same data should have low drift
        labels = head.generate_labels(baseline)
        assert np.mean(labels["drift_scores"]) < 0.5

    def test_drift_loss(self):
        head = SpectralDriftHead()
        predicted = np.array([0.1, 0.2, 0.3])
        target = np.array([0.1, 0.2, 0.3])
        assert head.compute_loss(predicted, target) == 0.0


# ---------------------------------------------------------------------------
# Test TemporalLineageHead
# ---------------------------------------------------------------------------


class TestTemporalLineageHead:
    def test_generates_targets(self, embedding_sequence):
        head = TemporalLineageHead(window_size=5, compression_dim=16)
        labels = head.generate_labels(embedding_sequence)

        assert "targets" in labels
        assert "inputs" in labels
        assert "valid_indices" in labels
        assert labels["targets"].shape[0] == 50 - 5  # n_timesteps - window
        assert labels["targets"].shape[1] == 16

    def test_inputs_match_current(self, embedding_sequence):
        head = TemporalLineageHead(window_size=5, compression_dim=16)
        labels = head.generate_labels(embedding_sequence)

        # inputs[0] should be embedding at index window_size
        np.testing.assert_array_equal(labels["inputs"][0], embedding_sequence[5])

    def test_compression_methods(self, embedding_sequence):
        for method in ["mean", "last", "stats", "pca"]:
            head = TemporalLineageHead(
                window_size=5, compression_dim=16, compression_method=method
            )
            labels = head.generate_labels(embedding_sequence)
            assert labels["targets"].shape == (45, 16)

    def test_short_sequence(self):
        head = TemporalLineageHead(window_size=10, compression_dim=8)
        short_seq = np.random.rand(5, 8)
        labels = head.generate_labels(short_seq)
        assert labels["targets"].shape[0] == 0

    def test_cosine_loss(self):
        head = TemporalLineageHead()
        # Perfect predictions should give low loss
        targets = np.random.rand(10, 16)
        loss = head.compute_loss(targets, targets)
        assert loss < 0.01

        # Random predictions should give higher loss
        random_preds = np.random.rand(10, 16)
        random_loss = head.compute_loss(random_preds, targets)
        assert random_loss > loss


# ---------------------------------------------------------------------------
# Test WorldTaskSuite
# ---------------------------------------------------------------------------


class TestWorldTaskSuite:
    def test_evaluate_all_basic(self, frequencies, resonant_signal):
        suite = WorldTaskSuite()
        results = suite.evaluate_all(
            frequencies=frequencies,
            amplitudes=resonant_signal,
        )
        assert "resonance" in results
        assert "harmonic" in results

    def test_evaluate_all_with_components(self, frequencies):
        suite = WorldTaskSuite()
        amplitudes = np.random.rand(5, len(frequencies))
        components = np.random.rand(3, len(frequencies))
        embeddings = np.random.rand(5, 16)

        results = suite.evaluate_all(
            frequencies=frequencies,
            amplitudes=amplitudes,
            components=components,
            embeddings=embeddings,
        )
        assert "resonance" in results
        assert "harmonic" in results
        assert "coherence" in results
        assert "drift" in results
        assert "lineage" in results

    def test_total_loss_weighting(self):
        suite = WorldTaskSuite(task_weights={"resonance": 2.0, "drift": 0.5})
        losses = {"resonance": 1.0, "drift": 1.0}
        total = suite.compute_total_loss(losses)
        assert total == 2.5  # 2.0*1.0 + 0.5*1.0

    def test_custom_configs(self):
        suite = WorldTaskSuite(
            resonance_config={"threshold": 3.0},
            drift_config={"baseline_window": 20},
            lineage_config={"window_size": 5},
        )
        assert suite.resonance_head.threshold == 3.0
        assert suite.drift_head.baseline_window == 20
        assert suite.lineage_head.window_size == 5


# ---------------------------------------------------------------------------
# Test DigitalTwinEnvironment
# ---------------------------------------------------------------------------


class TestDigitalTwinEnvironment:
    def test_create_factory(self):
        env = DigitalTwinEnvironment.create_factory_environment(n_machines=3)
        assert len(env.entities) == 3
        assert all(
            e.entity_type == EntityType.ROTATING_MACHINERY for e in env.entities
        )

    def test_create_bridge(self):
        env = DigitalTwinEnvironment.create_bridge_environment(n_sensors=4)
        assert len(env.entities) == 4
        assert all(
            e.entity_type == EntityType.STRUCTURAL_ELEMENT for e in env.entities
        )

    def test_reset_and_step(self):
        env = DigitalTwinEnvironment.create_factory_environment(n_machines=2)
        obs = env.reset()
        assert len(obs) == 2
        assert all(isinstance(v, np.ndarray) for v in obs.values())

        obs, rewards, done, info = env.step()
        assert len(obs) == 2
        assert not done  # Should not be done after one step

    def test_episode_terminates(self):
        env = DigitalTwinEnvironment.create_factory_environment(n_machines=1)
        env.episode_length = 5
        env.reset()

        for _ in range(10):
            _, _, done, _ = env.step()
            if done:
                break
        assert done

    def test_generate_stream(self):
        env = DigitalTwinEnvironment.create_factory_environment(n_machines=2)
        stream = env.generate_stream("machine_0", n_timesteps=100)

        assert isinstance(stream, SpectralStream)
        assert stream.n_timesteps == 100
        assert stream.entity_id == "machine_0"
        assert stream.amplitudes.shape == (100, 256)

    def test_spectral_entity_generates_spectrum(self):
        entity = SpectralEntity(
            entity_id="test",
            entity_type=EntityType.ROTATING_MACHINERY,
            natural_frequencies=np.array([10.0, 30.0]),
            damping_ratios=np.array([0.05, 0.03]),
        )
        freqs = np.linspace(0.1, 100, 128)
        spectrum = entity.generate_spectrum(freqs, time_step=0)
        assert spectrum.shape == (128,)
        assert np.all(spectrum >= 0)

    def test_stream_window(self):
        env = DigitalTwinEnvironment.create_factory_environment(n_machines=1)
        stream = env.generate_stream("machine_0", n_timesteps=50)
        window = stream.get_window(10, 20)
        assert window.n_timesteps == 10


# ---------------------------------------------------------------------------
# Test SpectralMemoryStore
# ---------------------------------------------------------------------------


class TestSpectralMemoryStore:
    def test_store_and_query(self):
        store = SpectralMemoryStore(embed_dim=8)
        rng = np.random.default_rng(42)

        # Store some entries
        for i in range(20):
            store.store(
                timestamp=float(i),
                embedding=rng.standard_normal(8),
                event_type="normal",
            )

        assert store.size == 20

        # Query
        result = store.query_simple(rng.standard_normal(8), top_k=5)
        assert len(result.entries) == 5
        assert result.context_embedding is not None
        assert result.context_embedding.shape == (8,)

    def test_event_filtering(self):
        store = SpectralMemoryStore(embed_dim=4)
        rng = np.random.default_rng(42)

        for i in range(10):
            store.store(float(i), rng.standard_normal(4), "normal")
        for i in range(10, 15):
            store.store(float(i), rng.standard_normal(4), "resonance")

        result = store.query_simple(
            rng.standard_normal(4), top_k=10, event_filter="resonance"
        )
        assert all(e.event_type == "resonance" for e in result.entries)

    def test_time_range_filtering(self):
        store = SpectralMemoryStore(embed_dim=4)
        rng = np.random.default_rng(42)

        for i in range(50):
            store.store(float(i), rng.standard_normal(4), "normal")

        query = LineageQuery(
            query_embedding=rng.standard_normal(4),
            top_k=100,
            time_range=(10.0, 20.0),
        )
        result = store.query(query)
        assert all(10.0 <= e.timestamp <= 20.0 for e in result.entries)

    def test_capacity_consolidation(self):
        store = SpectralMemoryStore(capacity=50, consolidation_threshold=40, embed_dim=4)
        rng = np.random.default_rng(42)

        for i in range(60):
            store.store(float(i), rng.standard_normal(4), "normal")

        # Should have consolidated
        assert store.size <= 50

    def test_get_lineage(self):
        store = SpectralMemoryStore(embed_dim=8)
        rng = np.random.default_rng(42)

        for i in range(20):
            store.store(float(i), rng.standard_normal(8), "normal")

        current = rng.standard_normal(8)
        lineage = store.get_lineage(current, window=5)
        # Should be concatenation of current + context
        assert lineage.shape == (16,)

    def test_empty_store_query(self):
        store = SpectralMemoryStore(embed_dim=4)
        result = store.query_simple(np.zeros(4), top_k=5)
        assert len(result.entries) == 0
        assert result.context_embedding is None

    def test_importance_decay(self):
        store = SpectralMemoryStore(embed_dim=4, importance_decay=0.9)
        store.store(0.0, np.ones(4), "resonance", importance=10.0)
        initial_importance = store._entries[0].importance

        store.decay_importance()
        assert store._entries[0].importance < initial_importance

    def test_event_counts(self):
        store = SpectralMemoryStore(embed_dim=4)
        rng = np.random.default_rng(42)

        for i in range(5):
            store.store(float(i), rng.standard_normal(4), "normal")
        for i in range(3):
            store.store(float(i + 5), rng.standard_normal(4), "resonance")

        counts = store.get_event_counts()
        assert counts["normal"] == 5
        assert counts["resonance"] == 3

    def test_clear(self):
        store = SpectralMemoryStore(embed_dim=4)
        store.store(0.0, np.ones(4), "normal")
        assert store.size == 1
        store.clear()
        assert store.size == 0


# ---------------------------------------------------------------------------
# Test TrainingRecipe
# ---------------------------------------------------------------------------


class TestTrainingRecipe:
    def test_pretraining_stage_runs(self, frequencies):
        stage = PretrainingStage(n_epochs=3, batch_size=8)
        data = np.random.rand(20, len(frequencies))
        metrics = stage.run(frequencies, data)

        assert stage.status == StageStatus.COMPLETED
        assert len(metrics.losses) == 3
        assert metrics.best_loss < float("inf")

    def test_environment_stage_runs(self):
        env = DigitalTwinEnvironment.create_factory_environment(n_machines=2)
        env.episode_length = 10
        stage = EnvironmentStage(
            environment=env, n_episodes=3, max_steps_per_episode=10
        )
        metrics = stage.run()

        assert stage.status == StageStatus.COMPLETED
        assert len(metrics.rewards) == 3

    def test_finetuning_stage_runs(self, frequencies):
        stage = FineTuningStage(n_epochs=3, batch_size=8)
        data = np.random.rand(20, len(frequencies))
        metrics = stage.run(data, frequencies)

        assert stage.status == StageStatus.COMPLETED
        assert len(metrics.losses) == 3

    def test_full_recipe(self, frequencies):
        recipe = TrainingRecipe(
            stage1=PretrainingStage(n_epochs=2, batch_size=8),
            stage2=EnvironmentStage(n_episodes=2, max_steps_per_episode=5),
            stage3=FineTuningStage(n_epochs=2, batch_size=8),
        )

        # Set short episode length for testing
        recipe.stage2.environment.episode_length = 5

        data = np.random.rand(16, len(frequencies))
        results = recipe.run_all(frequencies, data)

        assert "stage1" in results
        assert "stage2" in results
        assert "stage3" in results
        assert recipe.stage1.status == StageStatus.COMPLETED
        assert recipe.stage2.status == StageStatus.COMPLETED
        assert recipe.stage3.status == StageStatus.COMPLETED

    def test_recipe_status(self):
        recipe = TrainingRecipe()
        status = recipe.status
        assert status["stage1_pretraining"] == StageStatus.PENDING
        assert status["stage2_environment"] == StageStatus.PENDING
        assert status["stage3_finetuning"] == StageStatus.PENDING
