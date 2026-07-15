"""Learning rate schedulers for spectral pretraining.

Implements various scheduling strategies optimized for
large-scale spectral foundation model training.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseScheduler:
    """Base class for learning rate schedulers.

    Attributes:
        base_lr: Initial learning rate.
        current_step: Current training step.
        current_lr: Current learning rate value.
    """

    def __init__(self, base_lr: float = 1e-4, total_steps: int = 100000):
        """Initialize scheduler.

        Args:
            base_lr: Base learning rate.
            total_steps: Total number of training steps.
        """
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.current_lr = base_lr
        self._lr_history: List[float] = []

    def step(self, step: Optional[int] = None) -> float:
        """Advance scheduler and return new learning rate.

        Args:
            step: Explicit step number (auto-increments if None).

        Returns:
            Current learning rate.
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        self.current_lr = self._compute_lr()
        self._lr_history.append(self.current_lr)
        return self.current_lr

    def _compute_lr(self) -> float:
        """Compute learning rate for current step."""
        return self.base_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def get_history(self) -> List[float]:
        """Get learning rate history."""
        return self._lr_history

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.base_lr = state["base_lr"]
        self.total_steps = state["total_steps"]
        self.current_step = state["current_step"]
        self.current_lr = state["current_lr"]


class WarmupCosineScheduler(BaseScheduler):
    """Cosine annealing with linear warmup.

    Learning rate schedule:
    - Warmup: linear from 0 to base_lr
    - Decay: cosine annealing to min_lr
    - Optional restarts for cyclic cosine

    Attributes:
        warmup_steps: Number of warmup steps.
        min_lr: Minimum learning rate.
        num_cycles: Number of cosine cycles.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        total_steps: int = 100000,
        warmup_steps: int = 10000,
        min_lr: float = 1e-6,
        num_cycles: int = 1,
        warmup_init_lr: float = 0.0,
    ):
        """Initialize warmup cosine scheduler.

        Args:
            base_lr: Peak learning rate.
            total_steps: Total training steps.
            warmup_steps: Linear warmup duration.
            min_lr: Minimum learning rate at end of decay.
            num_cycles: Number of cosine cycles (1 = standard, >1 = restart).
            warmup_init_lr: Initial LR before warmup.
        """
        super().__init__(base_lr, total_steps)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        self.warmup_init_lr = warmup_init_lr

    def _compute_lr(self) -> float:
        """Compute LR with warmup + cosine decay."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.warmup_init_lr + (self.base_lr - self.warmup_init_lr) * progress

        # Cosine decay
        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)
        decay_progress = min(decay_progress, 1.0)

        # Apply cycles
        cycle_progress = decay_progress * self.num_cycles
        cosine_value = math.cos(math.pi * (cycle_progress % 1.0))

        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cosine_value)


class WarmupLinearScheduler(BaseScheduler):
    """Linear decay with warmup.

    Learning rate schedule:
    - Warmup: linear from 0 to base_lr
    - Decay: linear from base_lr to min_lr

    Attributes:
        warmup_steps: Number of warmup steps.
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        total_steps: int = 100000,
        warmup_steps: int = 10000,
        min_lr: float = 0.0,
    ):
        """Initialize linear decay scheduler.

        Args:
            base_lr: Peak learning rate.
            total_steps: Total training steps.
            warmup_steps: Warmup duration.
            min_lr: Final learning rate.
        """
        super().__init__(base_lr, total_steps)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def _compute_lr(self) -> float:
        """Compute LR with warmup + linear decay."""
        if self.current_step < self.warmup_steps:
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.base_lr * progress

        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)
        decay_progress = min(decay_progress, 1.0)

        return self.base_lr - (self.base_lr - self.min_lr) * decay_progress


class CyclicScheduler(BaseScheduler):
    """Cyclic learning rate scheduler (triangular).

    Cycles between min_lr and max_lr with configurable
    cycle length and decay.

    Attributes:
        max_lr: Maximum learning rate in cycle.
        min_lr: Minimum learning rate in cycle.
        cycle_length: Steps per cycle.
        gamma: Decay factor per cycle.
    """

    def __init__(
        self,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        total_steps: int = 100000,
        cycle_length: int = 5000,
        gamma: float = 0.95,
        mode: str = "triangular2",
    ):
        """Initialize cyclic scheduler.

        Args:
            base_lr: Minimum LR (base).
            max_lr: Maximum LR (peak).
            total_steps: Total steps.
            cycle_length: Steps per full cycle.
            gamma: Per-cycle decay factor.
            mode: Cycling mode ('triangular', 'triangular2', 'exp_range').
        """
        super().__init__(base_lr, total_steps)
        self.max_lr = max_lr
        self.min_lr = base_lr
        self.cycle_length = cycle_length
        self.gamma = gamma
        self.mode = mode

    def _compute_lr(self) -> float:
        """Compute cyclic LR."""
        cycle = self.current_step // self.cycle_length
        cycle_progress = (self.current_step % self.cycle_length) / self.cycle_length

        # Triangular position within cycle
        if cycle_progress <= 0.5:
            x = 2 * cycle_progress
        else:
            x = 2 * (1 - cycle_progress)

        # Apply mode-specific scaling
        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** cycle)
        elif self.mode == "exp_range":
            scale = self.gamma ** self.current_step
        else:
            scale = 1.0

        amplitude = (self.max_lr - self.min_lr) * scale
        return self.min_lr + amplitude * x


class OneCycleScheduler(BaseScheduler):
    """One-cycle learning rate policy (Smith & Topin 2018).

    Phases:
    1. Linear warmup to max_lr (30% of training)
    2. Cosine annealing from max_lr to min_lr (70% of training)
    3. Optional final annealing phase

    Also implements momentum cycling (inverse of LR).

    Attributes:
        max_lr: Peak learning rate.
        min_lr: Final minimum learning rate.
        pct_start: Fraction of steps for warmup phase.
        div_factor: Initial LR divisor.
        final_div_factor: Final LR divisor.
    """

    def __init__(
        self,
        max_lr: float = 1e-3,
        total_steps: int = 100000,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: str = "cos",
    ):
        """Initialize one-cycle scheduler.

        Args:
            max_lr: Maximum learning rate.
            total_steps: Total training steps.
            pct_start: Percentage of steps for warmup.
            div_factor: Determines initial LR (max_lr / div_factor).
            final_div_factor: Determines final LR (max_lr / final_div_factor).
            anneal_strategy: Annealing strategy ('cos' or 'linear').
        """
        initial_lr = max_lr / div_factor
        super().__init__(initial_lr, total_steps)
        self.max_lr = max_lr
        self.min_lr = max_lr / final_div_factor
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        self.warmup_steps = int(total_steps * pct_start)

        # Momentum parameters (inverse of LR)
        self.max_momentum = 0.95
        self.min_momentum = 0.85
        self.current_momentum = self.max_momentum

    def _compute_lr(self) -> float:
        """Compute one-cycle LR."""
        if self.current_step < self.warmup_steps:
            # Phase 1: warmup
            progress = self.current_step / max(self.warmup_steps, 1)
            lr = self.base_lr + (self.max_lr - self.base_lr) * progress
            # Momentum decreases during warmup
            self.current_momentum = self.max_momentum - \
                (self.max_momentum - self.min_momentum) * progress
        else:
            # Phase 2: annealing
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)
            progress = min(progress, 1.0)

            if self.anneal_strategy == "cos":
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                    (1 + math.cos(math.pi * progress))
            else:
                lr = self.max_lr - (self.max_lr - self.min_lr) * progress

            # Momentum increases during annealing
            self.current_momentum = self.min_momentum + \
                (self.max_momentum - self.min_momentum) * progress

        return lr

    def get_momentum(self) -> float:
        """Get current momentum value."""
        return self.current_momentum


class PolynomialDecayScheduler(BaseScheduler):
    """Polynomial learning rate decay with warmup.

    LR decays as: base_lr * (1 - progress)^power

    Useful for BERT-style pretraining.

    Attributes:
        power: Polynomial power.
        warmup_steps: Warmup duration.
        end_lr: Final learning rate.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        total_steps: int = 100000,
        warmup_steps: int = 10000,
        power: float = 1.0,
        end_lr: float = 0.0,
    ):
        """Initialize polynomial decay scheduler.

        Args:
            base_lr: Peak learning rate.
            total_steps: Total training steps.
            warmup_steps: Warmup duration.
            power: Decay power (1=linear, 2=quadratic).
            end_lr: Final learning rate.
        """
        super().__init__(base_lr, total_steps)
        self.warmup_steps = warmup_steps
        self.power = power
        self.end_lr = end_lr

    def _compute_lr(self) -> float:
        """Compute polynomial decay LR."""
        if self.current_step < self.warmup_steps:
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.base_lr * progress

        decay_steps = self.total_steps - self.warmup_steps
        progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)
        progress = min(progress, 1.0)

        decay = (1 - progress) ** self.power
        return self.end_lr + (self.base_lr - self.end_lr) * decay


class WarmupInverseSquareRootScheduler(BaseScheduler):
    """Inverse square root decay (Vaswani et al., 2017).

    LR = base_lr * min(step^-0.5, step * warmup^-1.5)

    Standard Transformer schedule.

    Attributes:
        warmup_steps: Warmup duration.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        total_steps: int = 100000,
        warmup_steps: int = 4000,
    ):
        """Initialize inverse sqrt scheduler.

        Args:
            base_lr: Base learning rate (scaled by model dim).
            total_steps: Total steps.
            warmup_steps: Warmup steps.
        """
        super().__init__(base_lr, total_steps)
        self.warmup_steps = warmup_steps

    def _compute_lr(self) -> float:
        """Compute inverse sqrt LR."""
        step = max(self.current_step, 1)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return self.base_lr * min(arg1, arg2)


class MultiStepScheduler(BaseScheduler):
    """Multi-step learning rate decay.

    Reduces LR by gamma at specified milestones.

    Attributes:
        milestones: Steps at which to reduce LR.
        gamma: Multiplicative factor for reduction.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        total_steps: int = 100000,
        milestones: Optional[List[int]] = None,
        gamma: float = 0.1,
        warmup_steps: int = 1000,
    ):
        """Initialize multi-step scheduler.

        Args:
            base_lr: Initial learning rate.
            total_steps: Total steps.
            milestones: Steps to decay at.
            gamma: Decay factor.
            warmup_steps: Warmup steps.
        """
        super().__init__(base_lr, total_steps)
        self.milestones = sorted(milestones or [30000, 60000, 80000])
        self.gamma = gamma
        self.warmup_steps = warmup_steps

    def _compute_lr(self) -> float:
        """Compute multi-step LR."""
        if self.current_step < self.warmup_steps:
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.base_lr * progress

        lr = self.base_lr
        for milestone in self.milestones:
            if self.current_step >= milestone:
                lr *= self.gamma

        return lr


class ExponentialDecayScheduler(BaseScheduler):
    """Exponential learning rate decay.

    LR = base_lr * decay_rate^(step / decay_steps)

    Attributes:
        decay_rate: Rate of exponential decay.
        decay_steps: Steps per decay period.
    """

    def __init__(
        self,
        base_lr: float = 1e-4,
        total_steps: int = 100000,
        decay_rate: float = 0.96,
        decay_steps: int = 1000,
        warmup_steps: int = 5000,
        min_lr: float = 1e-7,
    ):
        """Initialize exponential decay scheduler.

        Args:
            base_lr: Initial LR.
            total_steps: Total steps.
            decay_rate: Decay rate per period.
            decay_steps: Steps per decay period.
            warmup_steps: Warmup steps.
            min_lr: Minimum LR.
        """
        super().__init__(base_lr, total_steps)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def _compute_lr(self) -> float:
        """Compute exponential decay LR."""
        if self.current_step < self.warmup_steps:
            progress = self.current_step / max(self.warmup_steps, 1)
            return self.base_lr * progress

        effective_step = self.current_step - self.warmup_steps
        lr = self.base_lr * (self.decay_rate ** (effective_step / self.decay_steps))
        return max(lr, self.min_lr)


class SchedulerFactory:
    """Factory for creating learning rate schedulers.

    Supports creating schedulers from configuration dictionaries.
    """

    _registry = {
        "warmup_cosine": WarmupCosineScheduler,
        "warmup_linear": WarmupLinearScheduler,
        "cyclic": CyclicScheduler,
        "one_cycle": OneCycleScheduler,
        "polynomial": PolynomialDecayScheduler,
        "inverse_sqrt": WarmupInverseSquareRootScheduler,
        "multi_step": MultiStepScheduler,
        "exponential": ExponentialDecayScheduler,
    }

    @classmethod
    def create(cls, scheduler_type: str, **kwargs) -> BaseScheduler:
        """Create a scheduler from type string.

        Args:
            scheduler_type: Type of scheduler.
            **kwargs: Scheduler parameters.

        Returns:
            Initialized scheduler.

        Raises:
            ValueError: If scheduler_type not recognized.
        """
        if scheduler_type not in cls._registry:
            raise ValueError(
                f"Unknown scheduler: {scheduler_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[scheduler_type](**kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseScheduler:
        """Create scheduler from config dictionary.

        Args:
            config: Must contain 'type' key and scheduler params.

        Returns:
            Initialized scheduler.
        """
        config = dict(config)
        scheduler_type = config.pop("type", "warmup_cosine")
        return cls.create(scheduler_type, **config)

    @classmethod
    def available(cls) -> List[str]:
        """List available scheduler types."""
        return list(cls._registry.keys())
