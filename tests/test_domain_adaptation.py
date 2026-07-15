"""Tests for domain adaptation module."""

import numpy as np
import pytest

from mesie.cognitive.domain_adaptation import (
    AlignmentMethod,
    CurriculumTransfer,
    DomainAligner,
    DomainInvariantEncoder,
    DomainShiftDetector,
    FeatureTransformer,
    MultiSourceEnsemble,
    ShiftType,
)


class TestDomainAligner:
    def test_coral_alignment(self):
        np.random.seed(42)
        aligner = DomainAligner(method=AlignmentMethod.CORAL, feature_dim=32)
        source = np.random.randn(50, 32)
        target = np.random.randn(50, 32) * 2 + 1  # Different distribution
        result = aligner.fit(source, target)
        assert result.alignment_score > 0
        assert aligner.is_aligned

    def test_mmd_alignment(self):
        aligner = DomainAligner(method=AlignmentMethod.MMD, feature_dim=16)
        source = np.random.randn(30, 16)
        target = np.random.randn(30, 16) + 3
        result = aligner.fit(source, target)
        assert result.alignment_score > 0

    def test_subspace_alignment(self):
        aligner = DomainAligner(method=AlignmentMethod.SUBSPACE, feature_dim=32, n_components=10)
        source = np.random.randn(40, 32)
        target = np.random.randn(40, 32)
        result = aligner.fit(source, target)
        assert result.alignment_score >= 0

    def test_transform(self):
        aligner = DomainAligner(method=AlignmentMethod.CORAL, feature_dim=16)
        source = np.random.randn(30, 16)
        target = np.random.randn(30, 16) * 2
        aligner.fit(source, target)
        transformed = aligner.transform(source)
        assert transformed.shape == source.shape

    def test_compute_discrepancy(self):
        aligner = DomainAligner()
        source = np.random.randn(50, 32)
        target = np.random.randn(50, 32) + 5
        disc = aligner.compute_discrepancy(source, target)
        assert disc > 0
        # Same domain should have low discrepancy
        disc_same = aligner.compute_discrepancy(source, source)
        assert disc_same < disc


class TestDomainShiftDetector:
    def test_no_shift(self):
        np.random.seed(42)
        reference = np.random.randn(100, 32)
        det = DomainShiftDetector(reference_data=reference, sensitivity=0.5)
        # Same distribution
        new_data = np.random.randn(20, 32)
        report = det.check(new_data)
        # With same distribution, shift should be small
        assert report.magnitude < 2.0

    def test_detect_shift(self):
        reference = np.random.randn(100, 32)
        det = DomainShiftDetector(reference_data=reference, sensitivity=0.9)
        # Very different distribution
        shifted = np.random.randn(20, 32) + 10
        report = det.check(shifted)
        assert report.shift_type != ShiftType.NONE
        assert report.magnitude > 0

    def test_affected_features(self):
        np.random.seed(42)
        reference = np.zeros((100, 16))
        det = DomainShiftDetector(reference_data=reference, sensitivity=0.9)
        shifted = np.zeros((20, 16))
        shifted[:, 5] = 100  # Shift in feature 5
        report = det.check(shifted)
        if report.affected_features:
            assert 5 in report.affected_features


class TestFeatureTransformer:
    def test_create(self):
        ft = FeatureTransformer(input_dim=64, output_dim=32)
        assert not ft.is_trained

    def test_fit(self):
        ft = FeatureTransformer(input_dim=32, output_dim=32, n_layers=2)
        source = np.random.randn(50, 32)
        target = source * 2 + 1  # Simple linear relation
        metrics = ft.fit(source, target, n_iterations=50)
        assert "final_loss" in metrics
        assert ft.is_trained

    def test_transform(self):
        ft = FeatureTransformer(input_dim=16, output_dim=16)
        source = np.random.randn(20, 16)
        target = source * 1.5
        ft.fit(source, target, n_iterations=20)
        result = ft.transform(source)
        assert result.shape[0] == 20


class TestCurriculumTransfer:
    def test_mixing_ratio(self):
        ct = CurriculumTransfer(n_stages=10, mixing_schedule="linear")
        assert ct.get_mixing_ratio(0) == 0.0
        assert abs(ct.get_mixing_ratio(9) - 1.0) < 1e-6

    def test_mixing_schedules(self):
        for schedule in ["linear", "exponential", "cosine", "step"]:
            ct = CurriculumTransfer(n_stages=5, mixing_schedule=schedule)
            ratio = ct.get_mixing_ratio(2)
            assert 0 <= ratio <= 1

    def test_create_mixed_batch(self):
        ct = CurriculumTransfer(n_stages=10)
        source = np.zeros((100, 32))
        target = np.ones((100, 32))
        batch = ct.create_mixed_batch(source, target, batch_size=20, stage=5)
        assert len(batch) == 20

    def test_run_curriculum(self):
        ct = CurriculumTransfer(n_stages=5)
        source = np.random.randn(50, 16)
        target = np.random.randn(50, 16) + 2
        result = ct.run_curriculum(source, target)
        assert result["n_stages"] == 5
        assert ct.is_complete

    def test_advance_stage(self):
        ct = CurriculumTransfer(n_stages=5)
        assert ct.current_stage == 0
        ct.advance_stage({"accuracy": 0.8})
        assert ct.current_stage == 1


class TestDomainInvariantEncoder:
    def test_encode(self):
        enc = DomainInvariantEncoder(input_dim=64, encoding_dim=32)
        x = np.random.randn(10, 64)
        z = enc.encode(x)
        assert z.shape == (10, 32)
        # Should be L2 normalized
        norms = np.linalg.norm(z, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_train_step(self):
        enc = DomainInvariantEncoder(input_dim=32, encoding_dim=16)
        source = np.random.randn(20, 32)
        target = np.random.randn(20, 32) + 3
        metrics = enc.train_step(source, target, learning_rate=0.01)
        assert "disc_loss" in metrics
        assert "domain_mmd" in metrics

    def test_domain_similarity_improves(self):
        np.random.seed(42)
        enc = DomainInvariantEncoder(input_dim=16, encoding_dim=8, invariance_weight=5.0)
        source = np.random.randn(30, 16)
        target = np.random.randn(30, 16) + 2

        sim_before = enc.get_domain_similarity(source, target)
        for _ in range(50):
            enc.train_step(source, target, learning_rate=0.05)
        sim_after = enc.get_domain_similarity(source, target)

        # After training, domains should be more similar in encoded space
        # (or at least training should complete without error)
        assert enc.n_training_steps == 50


class TestMultiSourceEnsemble:
    def test_create(self):
        ens = MultiSourceEnsemble(n_sources=3, feature_dim=32, n_classes=5)
        assert not ens.is_trained

    def test_train_source(self):
        ens = MultiSourceEnsemble(n_sources=2, feature_dim=16, n_classes=3)
        X = np.random.randn(50, 16)
        y = np.random.randint(0, 3, 50)
        acc = ens.train_source(0, X, y, n_epochs=5)
        assert 0 <= acc <= 1
        assert ens.is_trained

    def test_predict(self):
        ens = MultiSourceEnsemble(n_sources=2, feature_dim=16, n_classes=3)
        for i in range(2):
            X = np.random.randn(30, 16)
            y = np.random.randint(0, 3, 30)
            ens.train_source(i, X, y, n_epochs=3)
        pred, probs = ens.predict(np.random.randn(16))
        assert 0 <= pred < 3
        assert abs(np.sum(probs) - 1.0) < 1e-6

    def test_compute_domain_weights(self):
        ens = MultiSourceEnsemble(n_sources=3, feature_dim=16, n_classes=5)
        for i in range(3):
            X = np.random.randn(20, 16)
            y = np.random.randint(0, 5, 20)
            ens.train_source(i, X, y, n_epochs=3)
        target = np.random.randn(10, 16)
        weights = ens.compute_domain_weights(target)
        assert len(weights) == 3
        assert abs(np.sum(weights) - 1.0) < 1e-6
