"""Training infrastructure for spectral foundation model pretraining.

Provides full pretraining engine with:
- Distributed training support
- Mixed-precision simulation
- Gradient accumulation
- Learning rate scheduling
- Checkpoint management
- Metrics logging
"""

from mesie.foundation.training.pretraining_engine import (
    PretrainingEngine,
    TrainingState,
    TrainingMetrics,
    GradientAccumulator,
)
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
    OptimizerFactory,
)

__all__ = [
    "PretrainingEngine",
    "TrainingState",
    "TrainingMetrics",
    "GradientAccumulator",
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "CyclicScheduler",
    "OneCycleScheduler",
    "PolynomialDecayScheduler",
    "SchedulerFactory",
    "AdamW",
    "LAMB",
    "SGDMomentum",
    "Adafactor",
    "OptimizerFactory",
]
