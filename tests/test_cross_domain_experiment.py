"""Tests for cross-domain spectral transfer and experiment management."""

import numpy as np
import pytest

from mesie.cognitive.cross_domain_transfer import (
    CORALTransfer,
    CrossDomainTransferEngine,
    DomainDescriptor,
    DomainInvariantNormalizer,
    MMDTransfer,
    SpectralCorpus,
    SpectralDomain,
    SpectralDomainGenerator,
    TransferLearningPipeline,
)
from mesie.cognitive.experiment_management import (
    AblationStudyRunner,
    CrossValidationEngine,
    DataAugmentation,
    ExperimentConfig,
    ExperimentPipeline,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTracker,
    HyperparameterOptimizer,
    OptimizationStrategy,
    ReproducibilityManager,
    SpectralBenchmark,
    StatisticalTestSuite,
)


# ============ Cross-Domain Transfer Tests ============


class TestDomainDescriptor:
    def test_create(self):
        desc = DomainDescriptor(
            domain=SpectralDomain.SEISMIC,
            name="Test",
            frequency_range=(0.1, 20.0),
        )
        assert desc.bandwidth == pytest.approx(19.9)

    def test_similarity_same(self):
        desc = DomainDescriptor(domain=SpectralDomain.SEISMIC, name="A", frequency_range=(0, 100), key_features=["a", "b"])
        sim = desc.compute_similarity(desc)
        assert sim > 0.9

    def test_similarity_different(self):
        a = DomainDescriptor(domain=SpectralDomain.SEISMIC, name="A", frequency_range=(0, 10), typical_snr=10, key_features=["x"])
        b = DomainDescriptor(domain=SpectralDomain.AUDIO_ACOUSTIC, name="B", frequency_range=(1000, 20000), typical_snr=40, key_features=["y"])
        sim = a.compute_similarity(b)
        assert sim < 0.5


class TestSpectralCorpus:
    def test_create(self):
        data = np.random.randn(50, 64)
        desc = DomainDescriptor(domain=SpectralDomain.GENERIC, name="Test")
        corpus = SpectralCorpus(domain=SpectralDomain.GENERIC, descriptor=desc, data=data)
        assert corpus.n_samples == 50
        assert corpus.n_features == 64
        assert "mean" in corpus.statistics

    def test_statistics(self):
        data = np.random.randn(100, 32)
        desc = DomainDescriptor(domain=SpectralDomain.GENERIC, name="Test")
        corpus = SpectralCorpus(domain=SpectralDomain.GENERIC, descriptor=desc, data=data)
        assert corpus.statistics["n_samples"] == 100
        assert corpus.statistics["spectral_centroid"] > 0


class TestDomainInvariantNormalizer:
    def test_fit_and_normalize(self):
        data = np.random.randn(50, 32)
        desc = DomainDescriptor(domain=SpectralDomain.SEISMIC, name="S")
        corpus = SpectralCorpus(domain=SpectralDomain.SEISMIC, descriptor=desc, data=data)
        norm = DomainInvariantNormalizer(target_dim=16)
        norm.fit_domain(corpus)
        result = norm.normalize(data, SpectralDomain.SEISMIC)
        assert result.shape == (50, 16)

    def test_multi_domain(self):
        norm = DomainInvariantNormalizer(target_dim=16)
        for domain in [SpectralDomain.SEISMIC, SpectralDomain.AUDIO_ACOUSTIC]:
            data = np.random.randn(30, 32)
            desc = DomainDescriptor(domain=domain, name=domain.value)
            corpus = SpectralCorpus(domain=domain, descriptor=desc, data=data)
            norm.fit_domain(corpus)
        assert norm.n_domains == 2
        dist = norm.compute_domain_distance(SpectralDomain.SEISMIC, SpectralDomain.AUDIO_ACOUSTIC)
        assert dist >= 0


class TestCORALTransfer:
    def test_fit(self):
        source = np.random.randn(50, 16)
        target = np.random.randn(50, 16) * 2 + 1
        coral = CORALTransfer()
        loss = coral.fit(source, target)
        assert loss >= 0
        assert coral.is_fitted

    def test_transform(self):
        source = np.random.randn(40, 16)
        target = np.random.randn(40, 16) + 3
        coral = CORALTransfer()
        coral.fit(source, target)
        aligned = coral.transform(source)
        assert aligned.shape == source.shape

    def test_alignment_reduces_distance(self):
        np.random.seed(42)
        source = np.random.randn(60, 16)
        target = np.random.randn(60, 16) * 3 + 2
        coral = CORALTransfer()
        coral.fit(source, target)
        aligned = coral.transform(source)
        # Aligned mean should be closer to target mean
        dist_before = np.linalg.norm(np.mean(source, 0) - np.mean(target, 0))
        dist_after = np.linalg.norm(np.mean(aligned, 0) - np.mean(target, 0))
        assert dist_after < dist_before


class TestMMDTransfer:
    def test_compute_mmd(self):
        source = np.random.randn(30, 16)
        target = np.random.randn(30, 16) + 5
        mmd = MMDTransfer()
        val = mmd.compute_mmd(source, target)
        assert val > 0

    def test_same_distribution_low_mmd(self):
        np.random.seed(42)
        data = np.random.randn(50, 16)
        mmd = MMDTransfer()
        val = mmd.compute_mmd(data[:25], data[25:])
        # Same distribution should have relatively low MMD
        different_val = mmd.compute_mmd(data[:25], data[:25] + 10)
        assert val < different_val

    def test_fit(self):
        source = np.random.randn(30, 16)
        target = np.random.randn(30, 16)
        mmd = MMDTransfer(n_features=8)
        final_mmd = mmd.fit(source, target, n_iterations=20)
        assert mmd.is_fitted
        assert len(mmd.mmd_history) > 0


class TestSpectralDomainGenerator:
    def test_generate_seismic(self):
        gen = SpectralDomainGenerator()
        corpus = gen.generate_seismic(50, 128)
        assert corpus.n_samples == 50
        assert corpus.n_features == 128
        assert corpus.labels is not None

    def test_generate_eeg(self):
        gen = SpectralDomainGenerator()
        corpus = gen.generate_eeg(40, 64)
        assert corpus.domain == SpectralDomain.EEG_NEURAL

    def test_generate_audio(self):
        gen = SpectralDomainGenerator()
        corpus = gen.generate_audio(30, 128)
        assert corpus.domain == SpectralDomain.AUDIO_ACOUSTIC

    def test_generate_all(self):
        gen = SpectralDomainGenerator()
        all_corpora = gen.generate_all_domains(20, 64)
        assert len(all_corpora) == 5


class TestCrossDomainTransferEngine:
    def test_register_and_transfer(self):
        gen = SpectralDomainGenerator()
        engine = CrossDomainTransferEngine(shared_dim=32)
        seismic = gen.generate_seismic(30, 64)
        structural = gen.generate_structural_vibration(30, 64)
        engine.register_corpus(seismic)
        engine.register_corpus(structural)
        result = engine.transfer(SpectralDomain.SEISMIC, SpectralDomain.STRUCTURAL_VIBRATION)
        assert "transfer_efficiency" in result
        assert result["aligned_data"].shape[0] == 30

    def test_find_best_source(self):
        gen = SpectralDomainGenerator()
        engine = CrossDomainTransferEngine(shared_dim=32)
        for corpus in gen.generate_all_domains(20, 32).values():
            engine.register_corpus(corpus)
        best = engine.find_best_source(SpectralDomain.STRUCTURAL_VIBRATION)
        assert best is not None


class TestTransferLearningPipeline:
    def test_initialize(self):
        pipe = TransferLearningPipeline(shared_dim=32)
        result = pipe.initialize_with_synthetic(n_samples=20, n_features=32)
        assert result["n_domains"] == 5
        assert pipe.is_initialized

    def test_evaluate_transfer(self):
        pipe = TransferLearningPipeline(shared_dim=32)
        pipe.initialize_with_synthetic(20, 32)
        result = pipe.evaluate_transfer(SpectralDomain.SEISMIC, SpectralDomain.STRUCTURAL_VIBRATION)
        assert "transfer_efficiency" in result

    def test_find_optimal_strategy(self):
        pipe = TransferLearningPipeline(shared_dim=16)
        pipe.initialize_with_synthetic(15, 32)
        result = pipe.find_optimal_transfer_strategy(SpectralDomain.EEG_NEURAL, SpectralDomain.AUDIO_ACOUSTIC)
        assert result["best_method"] is not None


# ============ Experiment Management Tests ============


class TestExperimentTracker:
    def test_start_end(self):
        tracker = ExperimentTracker()
        config = ExperimentConfig(name="test", parameters={"lr": 0.01})
        exp_id = tracker.start_experiment(config)
        tracker.log_metrics({"accuracy": 0.9}, step=1)
        result = tracker.end_experiment()
        assert result.status == ExperimentStatus.COMPLETED
        assert tracker.n_experiments == 1

    def test_best_experiment(self):
        tracker = ExperimentTracker()
        for i, acc in enumerate([0.7, 0.9, 0.8]):
            config = ExperimentConfig(name=f"exp_{i}", parameters={"x": i}, seed=i)
            tracker.start_experiment(config)
            tracker.log_metrics({"primary": acc})
            tracker.end_experiment()
        best = tracker.get_best_experiment("primary")
        assert best is not None
        assert tracker.experiments[best].metrics["primary"] == 0.9

    def test_compare(self):
        tracker = ExperimentTracker()
        ids = []
        for i in range(3):
            config = ExperimentConfig(name=f"e{i}", parameters={"i": i}, seed=i)
            eid = tracker.start_experiment(config)
            tracker.log_metrics({"primary": i * 0.3, "loss": 1 - i * 0.2})
            tracker.end_experiment()
            ids.append(eid)
        comp = tracker.compare_experiments(ids, ["primary", "loss"])
        assert "rankings" in comp


class TestHyperparameterOptimizer:
    def test_random_search(self):
        space = {"lr": {"type": "float", "range": [0.001, 0.1], "log": True}, "layers": {"type": "int", "range": [1, 5]}}
        opt = HyperparameterOptimizer(space, strategy=OptimizationStrategy.RANDOM_SEARCH, n_trials=10)
        for _ in range(10):
            params = opt.suggest()
            score = 1.0 - abs(params["lr"] - 0.01) * 10
            opt.report(params, score)
        assert opt.best_params is not None
        assert opt.n_completed_trials == 10

    def test_bayesian(self):
        space = {"x": {"type": "float", "range": [-5, 5]}}
        opt = HyperparameterOptimizer(space, strategy=OptimizationStrategy.BAYESIAN)
        for i in range(10):
            params = opt.suggest(i)
            score = -(params["x"] ** 2)  # Maximize at x=0
            opt.report(params, score)
        assert opt.best_score > -25

    def test_importance(self):
        space = {"a": {"type": "float", "range": [0, 1]}, "b": {"type": "float", "range": [0, 1]}}
        opt = HyperparameterOptimizer(space)
        for i in range(10):
            params = opt.suggest(i)
            score = params["a"] * 2  # Only 'a' matters
            opt.report(params, score)
        imp = opt.get_importance()
        assert "a" in imp and "b" in imp


class TestCrossValidationEngine:
    def test_kfold(self):
        cv = CrossValidationEngine(n_folds=5, strategy="kfold")
        folds = cv.generate_folds(100)
        assert len(folds) == 5
        for train, test in folds:
            assert len(train) + len(test) == 100

    def test_stratified(self):
        cv = CrossValidationEngine(n_folds=3, strategy="stratified")
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        folds = cv.generate_folds(90, labels)
        assert len(folds) == 3

    def test_timeseries(self):
        cv = CrossValidationEngine(n_folds=4, strategy="timeseries")
        folds = cv.generate_folds(100)
        assert len(folds) >= 1
        # In time series, train always comes before test
        for train, test in folds:
            assert len(train) > 0 and len(test) > 0

    def test_summary(self):
        cv = CrossValidationEngine(n_folds=3)
        cv.evaluate_fold(0, {"accuracy": 0.8})
        cv.evaluate_fold(1, {"accuracy": 0.85})
        cv.evaluate_fold(2, {"accuracy": 0.82})
        summary = cv.get_summary()
        assert abs(summary["accuracy"]["mean"] - 0.8233) < 0.01


class TestStatisticalTestSuite:
    def test_paired_t(self):
        stats = StatisticalTestSuite()
        a = np.array([0.8, 0.82, 0.85, 0.79, 0.83])
        b = np.array([0.75, 0.78, 0.80, 0.74, 0.77])
        result = stats.paired_t_test(a, b)
        assert result["significant"]  # a clearly better than b
        assert result["mean_difference"] > 0

    def test_bootstrap_ci(self):
        stats = StatisticalTestSuite()
        scores = np.array([0.8, 0.82, 0.85, 0.79, 0.83, 0.81, 0.84])
        ci = stats.bootstrap_ci(scores)
        assert ci["ci_lower"] < ci["mean"] < ci["ci_upper"]

    def test_wilcoxon(self):
        stats = StatisticalTestSuite()
        a = np.array([10, 12, 14, 11, 13])
        b = np.array([8, 9, 10, 7, 11])
        result = stats.wilcoxon_signed_rank(a, b)
        assert "statistic" in result

    def test_multiple_correction(self):
        stats = StatisticalTestSuite(correction="bonferroni")
        p_values = [0.01, 0.03, 0.05, 0.1]
        adjusted = stats.multiple_comparison_correction(p_values)
        assert len(adjusted) == 4
        assert adjusted[0]["adjusted_p"] >= adjusted[0]["original_p"]

    def test_cohens_d(self):
        stats = StatisticalTestSuite()
        a = np.random.randn(50) + 1
        b = np.random.randn(50)
        d = stats.effect_size_cohens_d(a, b)
        assert d > 0  # a has higher mean


class TestAblationStudy:
    def test_ablation(self):
        runner = AblationStudyRunner(baseline_config={}, components=["A", "B", "C"])
        runner.set_baseline(0.9)
        runner.record_ablation("A", 0.7)  # Important
        runner.record_ablation("B", 0.895)  # Not important (<1% change)
        runner.record_ablation("C", 0.6)  # Very important
        report = runner.get_summary_report()
        assert "C" in report["important_components"]
        assert "B" in report["redundant_components"]

    def test_importance(self):
        runner = AblationStudyRunner(baseline_config={}, components=["X", "Y"])
        runner.set_baseline(1.0)
        runner.record_ablation("X", 0.5)
        runner.record_ablation("Y", 0.9)
        imp = runner.get_component_importance()
        assert imp["X"] > imp["Y"]


class TestSpectralBenchmark:
    def test_datasets(self):
        bench = SpectralBenchmark(n_samples=50, n_features=64)
        assert "harmonics" in bench.dataset_names
        assert "gaussians" in bench.dataset_names
        assert "mixed" in bench.dataset_names

    def test_evaluate(self):
        bench = SpectralBenchmark(n_samples=50, n_features=64)
        ds = bench.get_dataset("harmonics")
        preds = ds["y"].copy()  # Perfect predictions
        result = bench.evaluate("perfect", preds, "harmonics")
        assert result["accuracy"] == 1.0

    def test_leaderboard(self):
        bench = SpectralBenchmark(n_samples=50, n_features=64)
        ds = bench.get_dataset("harmonics")
        bench.evaluate("perfect", ds["y"], "harmonics")
        bench.evaluate("random", np.random.randint(0, 4, 50), "harmonics")
        lb = bench.get_leaderboard("harmonics")
        assert lb[0]["method"] == "perfect"


class TestDataAugmentation:
    def test_noise(self):
        aug = DataAugmentation()
        spectrum = np.sin(np.linspace(0, 10, 128))
        noisy = aug.add_noise(spectrum, snr_db=20)
        assert noisy.shape == spectrum.shape
        assert not np.allclose(noisy, spectrum)

    def test_frequency_shift(self):
        aug = DataAugmentation()
        spectrum = np.zeros(64)
        spectrum[20] = 1.0
        shifted = aug.frequency_shift(spectrum, shift_amount=5)
        assert shifted[25] == 1.0

    def test_mixup(self):
        aug = DataAugmentation()
        a = np.ones(32)
        b = np.zeros(32)
        mixed = aug.mixup(a, b)
        assert 0 <= np.mean(mixed) <= 1

    def test_batch(self):
        aug = DataAugmentation()
        spectra = np.random.randn(20, 64)
        batch = aug.generate_augmented_batch(spectra, ["noise", "scale", "mask"], n_augmented=10)
        assert batch.shape == (10, 64)
        assert aug.total_augmentations > 0


class TestReproducibilityManager:
    def test_deterministic_seeds(self):
        rm = ReproducibilityManager(master_seed=42)
        s1 = rm.get_seed("model")
        s2 = rm.get_seed("data")
        s1_again = rm.get_seed("model")
        assert s1 == s1_again
        assert s1 != s2

    def test_snapshot(self):
        rm = ReproducibilityManager()
        snap = rm.snapshot_environment()
        assert "numpy_version" in snap

    def test_hash(self):
        rm = ReproducibilityManager()
        data = np.array([1.0, 2.0, 3.0])
        h1 = rm.compute_hash(data)
        h2 = rm.compute_hash(data)
        assert h1 == h2


class TestExperimentPipeline:
    def test_run_experiment(self):
        pipe = ExperimentPipeline(name="test")
        data = np.random.randn(50, 16)
        labels = np.random.randint(0, 3, 50)
        config = ExperimentConfig(name="trial_1", parameters={"k": 3})
        result = pipe.run_experiment(config, data, labels)
        assert result.status == ExperimentStatus.COMPLETED
        assert "primary" in result.metrics

    def test_optimize(self):
        space = {"scale": {"type": "float", "range": [0.1, 10.0]}}
        pipe = ExperimentPipeline(name="opt_test", search_space=space)
        data = np.random.randn(40, 8)
        labels = np.random.randint(0, 2, 40)
        result = pipe.optimize(data, labels, n_trials=3)
        assert result["best_params"] is not None
        assert result["n_trials"] == 3
