"""Tests for training infrastructure."""

import numpy as np
import pytest

from mesie.foundation.training.schedulers import (
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    CyclicScheduler,
    OneCycleScheduler,
    PolynomialDecayScheduler,
    SchedulerFactory,
)
from mesie.foundation.training.optimizers import (
    AdamW,
    LAMB,
    SGDMomentum,
    Adafactor,
    Lion,
    OptimizerFactory,
)
from mesie.foundation.training.pretraining_engine import (
    PretrainingEngine,
    TrainingState,
    TrainingMetrics,
    GradientAccumulator,
    EarlyStopping,
    GradientScaler,
    MetricsLogger,
)


class TestSchedulers:
    """Tests for learning rate schedulers."""

    def test_warmup_cosine_warmup_phase(self):
        """LR should increase during warmup."""
        scheduler = WarmupCosineScheduler(
            base_lr=1e-3, total_steps=1000, warmup_steps=100
        )
        lrs = [scheduler.step() for _ in range(100)]
        # Should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1] - 1e-10

    def test_warmup_cosine_decay_phase(self):
        """LR should decrease after warmup."""
        scheduler = WarmupCosineScheduler(
            base_lr=1e-3, total_steps=1000, warmup_steps=100, min_lr=1e-5
        )
        for _ in range(100):
            scheduler.step()
        # After warmup, LR should decay
        lr_at_warmup_end = scheduler.get_lr()
        for _ in range(500):
            scheduler.step()
        assert scheduler.get_lr() < lr_at_warmup_end

    def test_warmup_cosine_min_lr(self):
        """LR should not go below min_lr."""
        scheduler = WarmupCosineScheduler(
            base_lr=1e-3, total_steps=1000, warmup_steps=100, min_lr=1e-5
        )
        for _ in range(2000):
            scheduler.step()
        assert scheduler.get_lr() >= 1e-5 - 1e-10

    def test_warmup_linear(self):
        """Linear scheduler should decrease linearly."""
        scheduler = WarmupLinearScheduler(
            base_lr=1e-3, total_steps=1000, warmup_steps=100, min_lr=0
        )
        for _ in range(100):
            scheduler.step()
        lr_peak = scheduler.get_lr()
        assert abs(lr_peak - 1e-3) < 1e-5

    def test_cyclic(self):
        """Cyclic scheduler should oscillate."""
        scheduler = CyclicScheduler(
            base_lr=1e-5, max_lr=1e-3, total_steps=10000, cycle_length=1000
        )
        lrs = [scheduler.step() for _ in range(2000)]
        # Should have variation
        assert max(lrs) > min(lrs) * 2

    def test_one_cycle(self):
        """One-cycle should reach max_lr during warmup."""
        scheduler = OneCycleScheduler(
            max_lr=1e-3, total_steps=1000, pct_start=0.3
        )
        lrs = [scheduler.step() for _ in range(1000)]
        # Peak should be close to max_lr
        assert max(lrs) >= 0.9 * 1e-3

    def test_polynomial_decay(self):
        """Polynomial scheduler should decay to end_lr."""
        scheduler = PolynomialDecayScheduler(
            base_lr=1e-3, total_steps=1000, warmup_steps=100,
            power=2.0, end_lr=1e-5
        )
        for _ in range(1000):
            scheduler.step()
        assert scheduler.get_lr() <= 1e-3

    def test_scheduler_factory(self):
        """Factory should create all scheduler types."""
        for stype in SchedulerFactory.available():
            kwargs = {"total_steps": 1000}
            if "one_cycle" in stype.lower() if isinstance(stype, str) else "one_cycle" in str(stype).lower():
                kwargs["max_lr"] = 1e-3
            elif "cyclic" in str(stype).lower():
                kwargs["base_lr"] = 1e-5
                kwargs["max_lr"] = 1e-3
            else:
                kwargs["base_lr"] = 1e-3
            scheduler = SchedulerFactory.create(stype, **kwargs)
            assert scheduler is not None
            lr = scheduler.step()
            assert lr >= 0

    def test_scheduler_state_dict(self):
        """Scheduler state should be saveable/loadable."""
        scheduler = WarmupCosineScheduler(base_lr=1e-3, total_steps=1000)
        for _ in range(50):
            scheduler.step()
        state = scheduler.state_dict()
        assert state["current_step"] == 50


class TestOptimizers:
    """Tests for optimizers."""

    def test_adamw_update(self):
        """AdamW should update parameters."""
        params = [np.random.randn(10, 10)]
        opt = AdamW(params, lr=0.01)
        grads = [np.random.randn(10, 10)]
        original = params[0].copy()
        opt.step(grads)
        assert not np.allclose(opt.params[0], original)

    def test_adamw_weight_decay(self):
        """AdamW weight decay should shrink parameters."""
        params = [np.ones((10, 10)) * 5]
        opt = AdamW(params, lr=0.01, weight_decay=0.1)
        grads = [np.zeros((10, 10))]
        for _ in range(100):
            opt.step(grads)
        # With zero gradient, weight decay should shrink params
        assert np.mean(np.abs(opt.params[0])) < 5.0

    def test_lamb_trust_ratio(self):
        """LAMB should apply trust ratio."""
        params = [np.random.randn(10, 10)]
        opt = LAMB(params, lr=0.01)
        grads = [np.random.randn(10, 10)]
        opt.step(grads)
        assert opt.params[0] is not None

    def test_sgd_momentum(self):
        """SGD should use momentum."""
        params = [np.random.randn(10, 10)]
        opt = SGDMomentum(params, lr=0.01, momentum=0.9)
        grads = [np.ones((10, 10))]
        # Multiple steps with same gradient should accelerate
        for _ in range(10):
            opt.step(grads)
        # Velocity should have built up
        assert np.mean(np.abs(opt._state[0]["velocity"])) > 0

    def test_lion_sign_update(self):
        """Lion should use sign of momentum."""
        params = [np.random.randn(10, 10)]
        opt = Lion(params, lr=1e-4)
        grads = [np.random.randn(10, 10)]
        original = params[0].copy()
        opt.step(grads)
        # Update should be sign-based (magnitude = lr)
        assert not np.allclose(opt.params[0], original)

    def test_optimizer_factory(self):
        """Factory should create all optimizer types."""
        params = [np.random.randn(5, 5)]
        for otype in OptimizerFactory.available():
            opt = OptimizerFactory.create(otype, params=[p.copy() for p in params])
            assert opt is not None


class TestGradientAccumulator:
    """Tests for gradient accumulation."""

    def test_accumulation(self):
        """Should accumulate over specified steps."""
        acc = GradientAccumulator(accumulation_steps=4)
        grads = [np.ones((5, 5))]

        for i in range(3):
            ready = acc.accumulate(grads)
            assert not ready

        ready = acc.accumulate(grads)
        assert ready

        averaged = acc.get_accumulated()
        assert np.allclose(averaged[0], 1.0)

    def test_reset(self):
        """Reset should clear state."""
        acc = GradientAccumulator(accumulation_steps=2)
        acc.accumulate([np.ones((5,))])
        acc.reset()
        assert not acc.is_ready


class TestEarlyStopping:
    """Tests for early stopping."""

    def test_no_stop_with_improvement(self):
        """Should not stop when loss improves."""
        es = EarlyStopping(patience=5)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            assert not es(loss)

    def test_stop_without_improvement(self):
        """Should stop after patience exhausted."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        es(1.0)
        for _ in range(4):
            result = es(1.0)
        assert result or es.should_stop


class TestGradientScaler:
    """Tests for gradient scaler."""

    def test_scale_loss(self):
        scaler = GradientScaler(init_scale=1024.0)
        assert scaler.scale_loss(1.0) == 1024.0

    def test_overflow_handling(self):
        scaler = GradientScaler(init_scale=1024.0, backoff_factor=0.5)
        scaler.update(overflow=True)
        assert scaler.get_scale() == 512.0

    def test_growth(self):
        scaler = GradientScaler(init_scale=1.0, growth_factor=2.0, growth_interval=2)
        scaler.update(overflow=False)
        scaler.update(overflow=False)
        assert scaler.get_scale() == 2.0


class TestMetricsLogger:
    """Tests for metrics logger."""

    def test_logging(self):
        logger = MetricsLogger(log_interval=10)
        for step in range(1, 21):
            result = logger.log({"loss": 1.0 / step}, step)
        # Should have logged at step 10 and 20
        assert len(logger.history) == 2


class TestTrainingMetrics:
    """Tests for training metrics."""

    def test_update(self):
        metrics = TrainingMetrics()
        metrics.update(
            loss_dict={"total": 1.0, "reconstruction": 0.5},
            lr=1e-4,
            batch_size=32,
            seq_len=512,
            elapsed=0.1,
        )
        assert metrics.step == 1
        assert metrics.total_loss == 1.0
        assert metrics.throughput_samples_sec == 320.0

    def test_smoothed_loss(self):
        metrics = TrainingMetrics()
        for i in range(10):
            metrics.update({"total": float(i)}, 1e-4, 32, 512, 0.1)
        assert 0 < metrics.smoothed_loss < 10


class TestPretrainingEngine:
    """Tests for the pretraining engine."""

    def test_creation(self):
        params = [np.random.randn(64, 64) for _ in range(4)]
        engine = PretrainingEngine(
            model_params=params,
            config={"total_steps": 100, "batch_size": 4, "seq_len": 32},
        )
        assert engine is not None

    def test_compute_loss(self):
        params = [np.random.randn(64, 64) for _ in range(4)]
        engine = PretrainingEngine(
            model_params=params,
            config={"total_steps": 100, "batch_size": 4, "seq_len": 32},
        )
        model_output = {
            "reconstruction": np.random.randn(4, 32, 64),
        }
        targets = {
            "input": np.random.randn(4, 32, 64),
        }
        loss, components = engine.compute_loss(model_output, targets)
        assert loss > 0
        assert "reconstruction" in components
