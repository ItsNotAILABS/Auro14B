"""Tests for knowledge graph, signal processing, learning framework,
calibration, pattern library, and distributed processing modules."""

import numpy as np
import pytest

# =============================================================================
# Knowledge Graph Tests
# =============================================================================

from mesie.cognitive.knowledge_graph import (
    InferenceRule,
    KnowledgeDrivenClassifier,
    KnowledgeNode,
    KnowledgeRelation,
    NodeType,
    OntologicalReasoner,
    RelationType,
    SpectralKnowledgeGraph,
    SpectralOntology,
)


class TestSpectralOntology:
    def test_create_ontology(self):
        onto = SpectralOntology()
        assert onto.n_categories > 20

    def test_is_a_direct(self):
        onto = SpectralOntology()
        assert onto.is_a("resonance", "spectral_phenomenon")
        assert onto.is_a("harmonic", "spectral_phenomenon")
        assert not onto.is_a("resonance", "harmonic")

    def test_is_a_transitive(self):
        onto = SpectralOntology()
        assert onto.is_a("natural_frequency", "spectral_phenomenon")
        assert onto.is_a("overtone", "spectral_entity")

    def test_is_a_reflexive(self):
        onto = SpectralOntology()
        assert onto.is_a("resonance", "resonance")

    def test_get_ancestors(self):
        onto = SpectralOntology()
        ancestors = onto.get_ancestors("natural_frequency")
        assert "resonance" in ancestors
        assert "spectral_phenomenon" in ancestors

    def test_get_descendants(self):
        onto = SpectralOntology()
        desc = onto.get_descendants("resonance")
        assert "natural_frequency" in desc
        assert "forced_resonance" in desc

    def test_get_siblings(self):
        onto = SpectralOntology()
        siblings = onto.get_siblings("resonance")
        assert "harmonic" in siblings

    def test_classify_features(self):
        onto = SpectralOntology()
        features = {"n_peaks": 1, "spectral_flatness": 0.2, "bandwidth": 0.02}
        classes = onto.classify(features)
        assert len(classes) > 0
        assert all(0 <= c[1] <= 1 for c in classes)

    def test_to_dict(self):
        onto = SpectralOntology()
        d = onto.to_dict()
        assert "n_categories" in d
        assert d["n_categories"] > 0


class TestSpectralKnowledgeGraph:
    def test_create_graph(self):
        g = SpectralKnowledgeGraph()
        assert g.n_nodes == 0
        assert g.n_relations == 0

    def test_add_node(self):
        g = SpectralKnowledgeGraph()
        node = KnowledgeNode(node_id="n1", name="Test", node_type=NodeType.PHENOMENON)
        g.add_node(node)
        assert g.n_nodes == 1
        assert g.get_node("n1") is not None

    def test_add_relation(self):
        g = SpectralKnowledgeGraph()
        g.add_node(KnowledgeNode(node_id="a", name="A", node_type=NodeType.PHENOMENON))
        g.add_node(KnowledgeNode(node_id="b", name="B", node_type=NodeType.PROPERTY))
        g.add_relation(KnowledgeRelation(
            source_id="a", target_id="b", relation_type=RelationType.HAS_PROPERTY
        ))
        assert g.n_relations == 1

    def test_get_neighbors(self):
        g = SpectralKnowledgeGraph()
        g.add_node(KnowledgeNode(node_id="a", name="A", node_type=NodeType.PHENOMENON))
        g.add_node(KnowledgeNode(node_id="b", name="B", node_type=NodeType.PROPERTY))
        g.add_relation(KnowledgeRelation(
            source_id="a", target_id="b", relation_type=RelationType.CAUSES
        ))
        neighbors = g.get_neighbors("a", direction="outgoing")
        assert len(neighbors) == 1
        assert neighbors[0][0].node_id == "b"

    def test_find_path(self):
        g = SpectralKnowledgeGraph()
        for i in range(4):
            g.add_node(KnowledgeNode(node_id=f"n{i}", name=f"N{i}", node_type=NodeType.PHENOMENON))
        g.add_relation(KnowledgeRelation(source_id="n0", target_id="n1", relation_type=RelationType.CAUSES))
        g.add_relation(KnowledgeRelation(source_id="n1", target_id="n2", relation_type=RelationType.CAUSES))
        g.add_relation(KnowledgeRelation(source_id="n2", target_id="n3", relation_type=RelationType.CAUSES))
        path = g.find_path("n0", "n3")
        assert path is not None
        assert len(path) == 3

    def test_semantic_search(self):
        g = SpectralKnowledgeGraph()
        emb = np.random.randn(64)
        node = KnowledgeNode(node_id="x", name="X", node_type=NodeType.PATTERN, embedding=emb)
        g.add_node(node)
        results = g.semantic_search(emb, top_k=5)
        assert len(results) == 1
        assert results[0][1] > 0.99  # Same vector

    def test_compute_centrality(self):
        g = SpectralKnowledgeGraph()
        g.add_node(KnowledgeNode(node_id="hub", name="Hub", node_type=NodeType.PHENOMENON))
        for i in range(5):
            g.add_node(KnowledgeNode(node_id=f"s{i}", name=f"S{i}", node_type=NodeType.PROPERTY))
            g.add_relation(KnowledgeRelation(source_id="hub", target_id=f"s{i}", relation_type=RelationType.CAUSES))
        centrality = g.compute_centrality()
        assert centrality["hub"] > centrality["s0"]

    def test_get_subgraph(self):
        g = SpectralKnowledgeGraph()
        g.add_node(KnowledgeNode(node_id="center", name="C", node_type=NodeType.PHENOMENON))
        g.add_node(KnowledgeNode(node_id="neighbor", name="N", node_type=NodeType.PROPERTY))
        g.add_relation(KnowledgeRelation(source_id="center", target_id="neighbor", relation_type=RelationType.CAUSES))
        nodes, rels = g.get_subgraph("center", radius=1)
        assert len(nodes) == 2


class TestOntologicalReasoner:
    def test_transitive_inference(self):
        g = SpectralKnowledgeGraph()
        for i in range(4):
            g.add_node(KnowledgeNode(node_id=f"n{i}", name=f"N{i}", node_type=NodeType.PHENOMENON))
        g.add_relation(KnowledgeRelation(source_id="n0", target_id="n1", relation_type=RelationType.IS_A))
        g.add_relation(KnowledgeRelation(source_id="n1", target_id="n2", relation_type=RelationType.IS_A))
        g.add_relation(KnowledgeRelation(source_id="n2", target_id="n3", relation_type=RelationType.IS_A))

        reasoner = OntologicalReasoner(g)
        results = reasoner.infer_transitive("n0", RelationType.IS_A)
        assert len(results) > 0
        assert any("n3" in r.conclusion for r in results)

    def test_find_analogies(self):
        g = SpectralKnowledgeGraph()
        # Create two structurally similar nodes
        g.add_node(KnowledgeNode(node_id="a", name="A", node_type=NodeType.PHENOMENON))
        g.add_node(KnowledgeNode(node_id="b", name="B", node_type=NodeType.PHENOMENON))
        g.add_node(KnowledgeNode(node_id="p1", name="P1", node_type=NodeType.PROPERTY))
        g.add_node(KnowledgeNode(node_id="p2", name="P2", node_type=NodeType.PROPERTY))
        g.add_relation(KnowledgeRelation(source_id="a", target_id="p1", relation_type=RelationType.HAS_PROPERTY))
        g.add_relation(KnowledgeRelation(source_id="b", target_id="p2", relation_type=RelationType.HAS_PROPERTY))

        reasoner = OntologicalReasoner(g)
        analogies = reasoner.find_analogies("a")
        assert len(analogies) > 0


class TestKnowledgeDrivenClassifier:
    def test_classify(self):
        g = SpectralKnowledgeGraph()
        classifier = KnowledgeDrivenClassifier(g, confidence_threshold=0.5)
        spectrum = np.random.randn(128)
        results = classifier.classify(spectrum)
        assert isinstance(results, list)

    def test_classify_with_features(self):
        g = SpectralKnowledgeGraph()
        classifier = KnowledgeDrivenClassifier(g, confidence_threshold=0.3)
        spectrum = np.random.randn(128)
        features = {"n_peaks": 1, "spectral_flatness": 0.2, "bandwidth": 0.02}
        results = classifier.classify(spectrum, features=features)
        assert len(results) > 0


# =============================================================================
# Signal Processing Tests
# =============================================================================

from mesie.cognitive.signal_processing import (
    AdaptiveFilter,
    AdvancedFeatureExtractor,
    DecompositionMethod,
    FilterType,
    SignalSynthesizer,
    SpectralAnomalyDetector,
    SpectralDecomposer,
    SpectralQualityAssessor,
)


class TestAdaptiveFilter:
    def test_lms_adapt(self):
        f = AdaptiveFilter(filter_length=16, filter_type=FilterType.LMS)
        input_sig = np.random.randn(200)
        desired = np.convolve(input_sig, [1, 0.5, 0.25], mode="same")
        error = f.adapt(input_sig, desired)
        assert len(error) == 200

    def test_nlms_convergence(self):
        f = AdaptiveFilter(filter_length=16, filter_type=FilterType.NLMS, learning_rate=0.5)
        np.random.seed(42)
        h_true = np.array([1.0, 0.5, 0.25, 0.1])
        input_sig = np.random.randn(500)
        desired = np.convolve(input_sig, h_true, mode="same")
        error = f.adapt(input_sig, desired)
        # Error should be finite and filter should have adapted
        assert np.all(np.isfinite(error))
        state = f.get_state()
        assert state.iterations > 0

    def test_rls_adapt(self):
        f = AdaptiveFilter(filter_length=8, filter_type=FilterType.RLS)
        input_sig = np.random.randn(100)
        desired = input_sig * 0.5
        error = f.adapt(input_sig, desired)
        assert len(error) == 100

    def test_predict(self):
        f = AdaptiveFilter(filter_length=8, filter_type=FilterType.NLMS)
        input_sig = np.random.randn(200)
        desired = np.convolve(input_sig, [1, 0.5], mode="same")
        f.adapt(input_sig, desired)
        output = f.predict(input_sig)
        assert len(output) == 200

    def test_get_state(self):
        f = AdaptiveFilter(filter_length=8)
        state = f.get_state()
        assert state.iterations == 0


class TestSpectralDecomposer:
    def test_emd_decompose(self):
        d = SpectralDecomposer(max_components=5)
        signal = np.sin(np.linspace(0, 10 * np.pi, 200)) + 0.5 * np.sin(np.linspace(0, 50 * np.pi, 200))
        result = d.decompose(signal, DecompositionMethod.EMD)
        assert result.n_components > 0
        assert len(result.residual) == 200

    def test_svd_decompose(self):
        d = SpectralDecomposer(max_components=3)
        signal = np.random.randn(128)
        result = d.decompose(signal, DecompositionMethod.SVD)
        assert result.n_components > 0

    def test_pca_decompose(self):
        d = SpectralDecomposer(max_components=3)
        signal = np.sin(np.linspace(0, 4 * np.pi, 128))
        result = d.decompose(signal, DecompositionMethod.PCA)
        assert result.n_components >= 0

    def test_wavelet_decompose(self):
        d = SpectralDecomposer(max_components=5)
        signal = np.random.randn(256)
        result = d.decompose(signal, DecompositionMethod.WAVELET)
        assert result.n_components > 0

    def test_nmf_decompose(self):
        d = SpectralDecomposer(max_components=3)
        signal = np.abs(np.random.randn(64))
        result = d.decompose(signal, DecompositionMethod.NMF)
        assert isinstance(result.residual, np.ndarray)


class TestAdvancedFeatureExtractor:
    def test_extract_basic(self):
        ext = AdvancedFeatureExtractor()
        spectrum = np.abs(np.random.randn(256))
        features = ext.extract(spectrum)
        assert features.centroid > 0
        assert features.entropy > 0

    def test_extract_temporal(self):
        ext = AdvancedFeatureExtractor()
        spectra = [np.abs(np.random.randn(128)) for _ in range(5)]
        features = ext.extract_temporal(spectra)
        assert len(features) == 5
        assert features[1].flux > 0  # Should have flux after first frame

    def test_feature_vector(self):
        ext = AdvancedFeatureExtractor()
        spectrum = np.abs(np.random.randn(128))
        features = ext.extract(spectrum)
        vec = features.to_vector()
        assert len(vec) == 13


class TestSignalSynthesizer:
    def test_harmonic_synthesis(self):
        synth = SignalSynthesizer(default_length=1024, sample_rate=44100)
        signal = synth.synthesize_harmonic(440, n_harmonics=5)
        assert len(signal) == 1024
        assert np.max(np.abs(signal)) > 0

    def test_from_peaks(self):
        synth = SignalSynthesizer(default_length=512)
        peaks = [(440.0, 1.0, 0.0), (880.0, 0.5, 0.0)]
        signal = synth.synthesize_from_peaks(peaks)
        assert len(signal) == 512

    def test_noise_synthesis(self):
        synth = SignalSynthesizer(default_length=1024)
        for noise_type in ["white", "pink", "brown", "blue"]:
            signal = synth.synthesize_noise(noise_type)
            assert len(signal) == 1024

    def test_chirp_synthesis(self):
        synth = SignalSynthesizer(default_length=512)
        signal = synth.synthesize_chirp(100, 5000)
        assert len(signal) == 512

    def test_add_noise(self):
        synth = SignalSynthesizer()
        clean = np.sin(np.linspace(0, 10, 100))
        noisy = synth.add_noise(clean, snr_db=20)
        assert len(noisy) == 100
        assert not np.allclose(clean, noisy)


class TestSpectralQualityAssessor:
    def test_assess_good_signal(self):
        assessor = SpectralQualityAssessor()
        # Good signal: clear peak + low noise
        signal = np.zeros(256)
        signal[50:60] = np.linspace(0, 1, 10)
        signal += 0.01 * np.random.randn(256)
        result = assessor.assess(signal)
        assert "quality_score" in result
        assert "snr_db" in result

    def test_assess_noisy_signal(self):
        assessor = SpectralQualityAssessor(min_snr=50)
        signal = np.random.randn(256)  # Pure noise
        result = assessor.assess(signal)
        assert result["quality_score"] < 0.8


class TestSpectralAnomalyDetector:
    def test_baseline_update(self):
        det = SpectralAnomalyDetector()
        for _ in range(5):
            det.update_baseline(np.ones(100) + 0.01 * np.random.randn(100))
        assert det.has_baseline
        assert det.baseline_samples == 5

    def test_detect_anomaly(self):
        det = SpectralAnomalyDetector(sensitivity=0.9)
        baseline = np.ones(100)
        for _ in range(10):
            det.update_baseline(baseline + 0.01 * np.random.randn(100))
        # Create anomalous spectrum
        anomalous = baseline.copy()
        anomalous[40:50] = 10.0  # Big spike
        anomalies = det.detect(anomalous)
        assert len(anomalies) > 0

    def test_no_anomaly(self):
        det = SpectralAnomalyDetector(sensitivity=0.3)  # Low sensitivity
        baseline = np.ones(100)
        for _ in range(10):
            det.update_baseline(baseline + 0.01 * np.random.randn(100))
        normal = baseline + 0.01 * np.random.randn(100)
        anomalies = det.detect(normal)
        # With low sensitivity, small noise shouldn't trigger many anomalies
        assert len(anomalies) < 50  # Reasonable upper bound


# =============================================================================
# Learning Framework Tests
# =============================================================================

from mesie.cognitive.learning_framework import (
    ActiveLearner,
    ContinualLearner,
    EnsemblePredictor,
    MetaLearner,
    OnlineLearner,
    SpectralOptimizer,
)


class TestOnlineLearner:
    def test_partial_fit(self):
        learner = OnlineLearner(input_dim=32, n_classes=5)
        x = np.random.randn(32)
        loss = learner.partial_fit(x, 2)
        assert loss > 0

    def test_predict(self):
        learner = OnlineLearner(input_dim=32, n_classes=5)
        x = np.random.randn(32)
        pred, probs = learner.predict(x)
        assert 0 <= pred < 5
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_learning_improves(self):
        np.random.seed(42)
        learner = OnlineLearner(input_dim=16, n_classes=3, learning_rate=0.1)
        # Simple linearly separable data
        for _ in range(200):
            cls = np.random.randint(3)
            x = np.zeros(16)
            x[cls * 5:(cls + 1) * 5] = 1.0 + 0.1 * np.random.randn(5)
            learner.partial_fit(x, cls)
        assert learner.accuracy > 0.5


class TestMetaLearner:
    def test_register_class(self):
        ml = MetaLearner(embedding_dim=32)
        support = np.random.randn(5, 128)
        ml.register_class(0, support)
        assert ml.n_classes == 1

    def test_classify(self):
        ml = MetaLearner(embedding_dim=32)
        ml.register_class(0, np.random.randn(5, 128))
        ml.register_class(1, np.random.randn(5, 128))
        pred, probs = ml.classify(np.random.randn(128))
        assert pred in (0, 1)
        assert len(probs) == 2

    def test_meta_train_episode(self):
        ml = MetaLearner(embedding_dim=32)
        support_X = np.random.randn(10, 128)
        support_y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        query_X = np.random.randn(4, 128)
        query_y = np.array([0, 0, 1, 1])
        loss = ml.meta_train_episode(support_X, support_y, query_X, query_y)
        assert loss > 0


class TestContinualLearner:
    def test_train_task(self):
        cl = ContinualLearner(input_dim=32, n_classes=5)
        X = np.random.randn(50, 32)
        y = np.random.randint(0, 5, 50)
        result = cl.train_task(X, y, task_id=0, n_epochs=3)
        assert "accuracy" in result
        assert result["accuracy"] >= 0

    def test_predict(self):
        cl = ContinualLearner(input_dim=32, n_classes=5)
        X = np.random.randn(50, 32)
        y = np.random.randint(0, 5, 50)
        cl.train_task(X, y, task_id=0, n_epochs=3)
        pred, probs = cl.predict(np.random.randn(32))
        assert 0 <= pred < 5

    def test_multiple_tasks(self):
        cl = ContinualLearner(input_dim=16, n_classes=4, ewc_lambda=100)
        for task_id in range(3):
            X = np.random.randn(30, 16)
            y = np.random.randint(0, 4, 30)
            cl.train_task(X, y, task_id=task_id, n_epochs=5)
        assert cl.n_tasks_learned == 3


class TestSpectralOptimizer:
    def test_suggest(self):
        opt = SpectralOptimizer(
            param_space={"lr": (0.001, 0.1), "n_layers": (1, 10)},
            n_initial=3,
        )
        params = opt.suggest()
        assert "lr" in params
        assert 0.001 <= params["lr"] <= 0.1

    def test_optimization_loop(self):
        opt = SpectralOptimizer(
            param_space={"x": (-5, 5), "y": (-5, 5)},
            n_initial=5,
        )
        for _ in range(20):
            params = opt.suggest()
            # Objective: minimize distance from (1, 1)
            score = -((params["x"] - 1) ** 2 + (params["y"] - 1) ** 2)
            opt.report(params, score)
        assert opt.best_score > -50  # Should find something reasonable
        assert opt.n_trials == 20


class TestEnsemblePredictor:
    def test_train(self):
        ens = EnsemblePredictor(n_estimators=5, input_dim=32, n_classes=3)
        X = np.random.randn(100, 32)
        y = np.random.randint(0, 3, 100)
        metrics = ens.train(X, y, n_epochs=5)
        assert "mean_accuracy" in metrics
        assert ens.is_trained

    def test_predict_with_uncertainty(self):
        ens = EnsemblePredictor(n_estimators=5, input_dim=32, n_classes=3)
        X = np.random.randn(50, 32)
        y = np.random.randint(0, 3, 50)
        ens.train(X, y, n_epochs=3)
        pred, conf, unc = ens.predict_with_uncertainty(np.random.randn(32))
        assert 0 <= pred < 3
        assert 0 <= conf <= 1
        assert 0 <= unc <= 1


class TestActiveLearner:
    def test_select_samples(self):
        al = ActiveLearner(input_dim=32, n_classes=5, batch_size=5)
        pool = np.random.randn(50, 32)
        samples = al.select_samples(pool)
        assert len(samples) == 5

    def test_label_and_learn(self):
        al = ActiveLearner(input_dim=32, n_classes=5)
        pool = np.random.randn(50, 32)
        samples = al.select_samples(pool)
        for s in samples:
            loss = al.label_sample(pool[s.index], np.random.randint(5), s.index)
            assert loss > 0
        assert al.n_labeled == len(samples)


# =============================================================================
# Calibration Tests
# =============================================================================

from mesie.cognitive.calibration import (
    CalibrationPoint,
    CalibrationTransferEngine,
    ConfidenceEstimator,
    DriftDetector,
    MeasurementValidator,
    SpectralCalibrator,
    UncertaintyQuantifier,
    ValidationStatus,
)


class TestSpectralCalibrator:
    def test_linear_calibration(self):
        cal = SpectralCalibrator()
        references = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        measured = references * 1.1 + 0.5  # Linear relation
        cal.add_points(references, measured)
        model = cal.calibrate()
        assert model.r_squared > 0.99

    def test_apply_calibration(self):
        cal = SpectralCalibrator()
        references = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        measured = references * 2.0
        cal.add_points(references, measured)
        cal.calibrate()
        result = cal.apply(np.array([2.0, 4.0, 6.0]))
        # Should map back close to 1.0, 2.0, 3.0
        assert abs(result[0] - 1.0) < 0.1

    def test_get_uncertainty(self):
        cal = SpectralCalibrator()
        cal.add_points(np.arange(10, dtype=float), np.arange(10, dtype=float) * 1.05)
        cal.calibrate()
        unc = cal.get_uncertainty(5.0)
        assert unc.std >= 0


class TestUncertaintyQuantifier:
    def test_linear_propagation(self):
        uq = UncertaintyQuantifier()
        values = np.array([1.0, 2.0, 3.0])
        uncertainties = np.array([0.1, 0.1, 0.1])
        transform = np.eye(3) * 2.0
        out_vals, out_unc = uq.propagate_linear(values, uncertainties, transform)
        assert np.allclose(out_vals, values * 2)
        assert np.allclose(out_unc, uncertainties * 2)

    def test_monte_carlo_propagation(self):
        uq = UncertaintyQuantifier(n_monte_carlo=500)
        values = np.array([1.0, 2.0])
        uncertainties = np.array([0.1, 0.1])
        mean_out, std_out = uq.propagate_monte_carlo(
            values, uncertainties, lambda x: x ** 2
        )
        assert len(mean_out) == 2

    def test_confidence_interval(self):
        uq = UncertaintyQuantifier(confidence_level=0.95)
        samples = np.random.randn(1000)
        mean, lower, upper = uq.compute_confidence_interval(samples)
        assert lower < mean < upper


class TestConfidenceEstimator:
    def test_calibrate(self):
        ce = ConfidenceEstimator()
        probs = np.random.uniform(0.5, 1.0, 100)
        labels = np.random.randint(0, 5, 100)
        preds = np.random.randint(0, 5, 100)
        ece = ce.calibrate(probs, labels, preds)
        assert 0 <= ece <= 1

    def test_estimate_confidence(self):
        ce = ConfidenceEstimator()
        probs = np.array([0.1, 0.2, 0.7])
        conf = ce.estimate_confidence(probs)
        assert 0 <= conf <= 1


class TestMeasurementValidator:
    def test_valid_measurement(self):
        val = MeasurementValidator(snr_threshold=10)
        spectrum = np.zeros(256)
        spectrum[100:110] = 5.0
        spectrum += 0.01 * np.random.randn(256)
        result = val.validate(spectrum)
        assert result.status in (ValidationStatus.VALID, ValidationStatus.SUSPECT)
        assert result.checks_total > 0

    def test_invalid_nan(self):
        val = MeasurementValidator()
        spectrum = np.ones(100)
        spectrum[50] = np.nan
        result = val.validate(spectrum)
        assert len(result.issues) > 0

    def test_empty_spectrum(self):
        val = MeasurementValidator()
        result = val.validate(np.array([]))
        assert result.status == ValidationStatus.INVALID


class TestCalibrationTransferEngine:
    def test_fit_transfer(self):
        engine = CalibrationTransferEngine()
        for _ in range(5):
            source = np.random.randn(64)
            target = source * 1.1 + 0.2 + 0.01 * np.random.randn(64)
            engine.add_standard(source, target)
        metrics = engine.fit()
        assert "rmse" in metrics
        assert engine.is_fitted

    def test_transfer(self):
        engine = CalibrationTransferEngine()
        for _ in range(5):
            source = np.random.randn(32)
            target = source * 2.0
            engine.add_standard(source, target)
        engine.fit()
        result = engine.transfer(np.ones(32))
        assert len(result) == 32


class TestDriftDetector:
    def test_no_drift(self):
        ref = np.ones(64)
        det = DriftDetector(reference_spectrum=ref, sensitivity=0.3)  # Low sensitivity
        report = det.add_measurement(ref)  # Exactly the same
        assert report is None

    def test_detect_offset_drift(self):
        ref = np.ones(64)
        det = DriftDetector(reference_spectrum=ref, sensitivity=0.9)
        # Add large offset
        report = det.add_measurement(ref + 5.0)
        assert report is not None
        assert report.drift_type.value == "offset"


# =============================================================================
# Pattern Library Tests
# =============================================================================

from mesie.cognitive.pattern_library import (
    MatchingMetric,
    PatternCategory,
    PatternEvolutionTracker,
    PatternGenerator,
    PatternLibrary,
    SpectralFingerprinter,
    SpectralTemplate,
)


class TestPatternLibrary:
    def test_add_template(self):
        lib = PatternLibrary()
        t = SpectralTemplate(
            template_id="t1", name="Test",
            pattern=np.random.randn(128),
            category=PatternCategory.HARMONIC,
        )
        lib.add_template(t)
        assert lib.n_templates == 1

    def test_search(self):
        lib = PatternLibrary()
        pattern = np.sin(np.linspace(0, 10, 128))
        lib.add_template(SpectralTemplate(
            template_id="sine", name="Sine",
            pattern=pattern, category=PatternCategory.HARMONIC,
        ))
        results = lib.search(pattern, top_k=5)
        assert len(results) == 1
        assert results[0].similarity > 0.9

    def test_search_by_category(self):
        lib = PatternLibrary()
        lib.add_template(SpectralTemplate(
            template_id="h1", name="H1",
            pattern=np.random.randn(64),
            category=PatternCategory.HARMONIC,
        ))
        lib.add_template(SpectralTemplate(
            template_id="n1", name="N1",
            pattern=np.random.randn(64),
            category=PatternCategory.NOISE,
        ))
        results = lib.search(np.random.randn(64), category=PatternCategory.HARMONIC)
        assert all(r.template_id == "h1" for r in results)


class TestPatternGenerator:
    def test_generate_all_categories(self):
        gen = PatternGenerator(length=256)
        for cat in PatternCategory:
            patterns = gen.generate(cat, n_patterns=2)
            assert len(patterns) == 2
            assert len(patterns[0]) == 256

    def test_augment(self):
        gen = PatternGenerator()
        pattern = np.sin(np.linspace(0, 10, 512))
        augmented = gen.augment(pattern, n_augmented=5)
        assert len(augmented) == 5


class TestSpectralFingerprinter:
    def test_fingerprint(self):
        fp = SpectralFingerprinter()
        spectrum = np.random.randn(256)
        result = fp.fingerprint(spectrum)
        assert len(result.features) == 32
        assert result.hash_code >= 0

    def test_compare_same(self):
        fp = SpectralFingerprinter()
        spectrum = np.random.randn(256)
        f1 = fp.fingerprint(spectrum)
        f2 = fp.fingerprint(spectrum)
        similarity = fp.compare(f1, f2)
        assert similarity > 0.99

    def test_compare_different(self):
        np.random.seed(123)
        fp = SpectralFingerprinter()
        f1 = fp.fingerprint(np.sin(np.linspace(0, 100, 256)))
        f2 = fp.fingerprint(np.cos(np.linspace(0, 50, 256)) * 5)
        similarity = fp.compare(f1, f2)
        # Different signals should have < 1.0 similarity
        assert similarity < 1.0


class TestPatternEvolutionTracker:
    def test_record_and_classify(self):
        tracker = PatternEvolutionTracker()
        for i in range(10):
            tracker.record("p1", np.ones(64) * (1 + 0.01 * i))
        evo_type = tracker.classify_evolution("p1")
        assert evo_type.value in ("stable", "drifting", "degrading", "emerging", "oscillating")

    def test_get_trend(self):
        tracker = PatternEvolutionTracker()
        for i in range(5):
            tracker.record("p1", np.random.randn(32))
        trend = tracker.get_trend("p1")
        assert "change_rate" in trend
        assert "stability" in trend


# =============================================================================
# Distributed Processing Tests
# =============================================================================

from mesie.cognitive.distributed_processing import (
    AggregationMethod,
    BatchItem,
    BatchProcessor,
    ResultAggregator,
    SpectralPipelineStage,
    SpectralProcessingPipeline,
    StreamProcessor,
    WorkflowOrchestrator,
    WorkflowStep,
)


class TestSpectralPipelineStage:
    def test_execute(self):
        stage = SpectralPipelineStage(name="double", process_fn=lambda x: x * 2)
        result = stage.execute(np.ones(10))
        assert result.status.value == "completed"
        assert np.allclose(result.output, 2.0)

    def test_execute_with_validation(self):
        stage = SpectralPipelineStage(
            name="checked",
            process_fn=lambda x: x,
            validate_input=lambda x: len(x) > 5,
        )
        # Should pass
        r1 = stage.execute(np.ones(10))
        assert r1.status.value == "completed"
        # Should fail
        r2 = stage.execute(np.ones(3))
        assert r2.status.value == "failed"


class TestSpectralProcessingPipeline:
    def test_sequential_pipeline(self):
        pipe = SpectralProcessingPipeline()
        pipe.add_function("normalize", lambda x: x / (np.max(np.abs(x)) + 1e-12))
        pipe.add_function("square", lambda x: x ** 2)
        result = pipe.execute(np.array([1.0, 2.0, 3.0]))
        assert result["status"] == "completed"
        assert result["output"] is not None

    def test_pipeline_failure(self):
        pipe = SpectralProcessingPipeline()

        def bad_fn(x):
            raise ValueError("Intentional failure")

        pipe.add_function("fail", bad_fn)
        result = pipe.execute(np.ones(5))
        assert result["status"] == "failed"


class TestBatchProcessor:
    def test_enqueue_and_process(self):
        pipe = SpectralProcessingPipeline()
        pipe.add_function("identity", lambda x: x)
        bp = BatchProcessor(pipeline=pipe, batch_size=5)

        for i in range(10):
            bp.enqueue(BatchItem(item_id=f"item_{i}", data=np.ones(32) * i))
        assert bp.queue_size == 10

        results = bp.process_batch()
        assert len(results) == 5
        assert bp.queue_size == 5

    def test_process_all(self):
        pipe = SpectralProcessingPipeline()
        pipe.add_function("double", lambda x: x * 2)
        bp = BatchProcessor(pipeline=pipe, batch_size=3)

        for i in range(7):
            bp.enqueue(BatchItem(item_id=f"i{i}", data=np.ones(16)))
        results = bp.process_all()
        assert len(results) == 7
        assert bp.processed_count == 7


class TestStreamProcessor:
    def test_stream_processing(self):
        pipe = SpectralProcessingPipeline()
        pipe.add_function("scale", lambda x: x * 0.5)
        sp = StreamProcessor(window_size=64, hop_size=32, pipeline=pipe)
        sp.start()

        outputs = sp.feed(np.ones(128))
        assert len(outputs) > 0
        assert sp.frame_count > 0

    def test_start_stop(self):
        sp = StreamProcessor()
        sp.start()
        assert sp.is_running
        sp.stop()
        assert not sp.is_running


class TestWorkflowOrchestrator:
    def test_simple_workflow(self):
        wf = WorkflowOrchestrator()
        wf.add_step(WorkflowStep(
            step_id="step1", name="Init",
            function=lambda ctx: {"initialized": True},
        ))
        wf.add_step(WorkflowStep(
            step_id="step2", name="Process",
            function=lambda ctx: {"processed": True},
            dependencies=["step1"],
        ))
        result = wf.execute()
        assert result["status"] == "completed"
        assert result["steps_executed"] == 2

    def test_conditional_workflow(self):
        wf = WorkflowOrchestrator()
        wf.add_step(WorkflowStep(
            step_id="s1", name="S1",
            function=lambda ctx: "done",
        ))
        wf.add_step(WorkflowStep(
            step_id="s2", name="S2 (skipped)",
            function=lambda ctx: "should not run",
            condition=lambda ctx: False,  # Never execute
        ))
        result = wf.execute()
        assert result["status"] == "completed"


class TestResultAggregator:
    def test_mean_aggregation(self):
        agg = ResultAggregator(method=AggregationMethod.MEAN)
        agg.add_result(np.array([1.0, 2.0, 3.0]))
        agg.add_result(np.array([3.0, 4.0, 5.0]))
        result = agg.aggregate()
        assert np.allclose(result, [2.0, 3.0, 4.0])

    def test_median_aggregation(self):
        agg = ResultAggregator(method=AggregationMethod.MEDIAN)
        agg.add_result(np.array([1.0, 10.0]))
        agg.add_result(np.array([2.0, 20.0]))
        agg.add_result(np.array([3.0, 30.0]))
        result = agg.aggregate()
        assert np.allclose(result, [2.0, 20.0])

    def test_statistics(self):
        agg = ResultAggregator()
        agg.add_result(np.ones(10))
        agg.add_result(np.ones(10) * 2)
        stats = agg.get_statistics()
        assert stats["n_results"] == 2
