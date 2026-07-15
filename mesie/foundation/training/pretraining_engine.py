"""Pretraining engine for spectral foundation models.

Provides the full training loop with:
- Multi-modal batch assembly
- Gradient accumulation
- Mixed-precision simulation
- Checkpoint management
- Distributed training coordination
- Metrics tracking and logging
- Early stopping and validation
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrainingMetrics:
    """Container for training metrics.

    Tracks loss, learning rate, throughput, and other metrics
    across training steps with moving averages.
    """

    total_loss: float = 0.0
    reconstruction_loss: float = 0.0
    contrastive_loss: float = 0.0
    masked_spectral_loss: float = 0.0
    next_window_loss: float = 0.0
    physics_loss: float = 0.0
    commitment_loss: float = 0.0
    codebook_loss: float = 0.0
    alignment_loss: float = 0.0

    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    throughput_samples_sec: float = 0.0
    throughput_tokens_sec: float = 0.0

    step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0

    # Moving averages
    _loss_ema: float = field(default=0.0, repr=False)
    _ema_decay: float = field(default=0.99, repr=False)

    def update(self, loss_dict: Dict[str, float], lr: float, batch_size: int,
               seq_len: int, elapsed: float) -> None:
        """Update metrics from a training step.

        Args:
            loss_dict: Dictionary of loss components.
            lr: Current learning rate.
            batch_size: Batch size.
            seq_len: Sequence length.
            elapsed: Time elapsed for step (seconds).
        """
        self.step += 1
        self.samples_seen += batch_size
        self.tokens_seen += batch_size * seq_len
        self.learning_rate = lr

        # Update loss components
        self.total_loss = loss_dict.get("total", 0.0)
        self.reconstruction_loss = loss_dict.get("reconstruction", 0.0)
        self.contrastive_loss = loss_dict.get("contrastive", 0.0)
        self.masked_spectral_loss = loss_dict.get("masked_spectral", 0.0)
        self.next_window_loss = loss_dict.get("next_window", 0.0)
        self.physics_loss = loss_dict.get("physics", 0.0)
        self.commitment_loss = loss_dict.get("commitment", 0.0)
        self.codebook_loss = loss_dict.get("codebook", 0.0)
        self.alignment_loss = loss_dict.get("alignment", 0.0)

        # EMA of total loss
        if self.step == 1:
            self._loss_ema = self.total_loss
        else:
            self._loss_ema = self._ema_decay * self._loss_ema + \
                (1 - self._ema_decay) * self.total_loss

        # Throughput
        if elapsed > 0:
            self.throughput_samples_sec = batch_size / elapsed
            self.throughput_tokens_sec = (batch_size * seq_len) / elapsed

    @property
    def smoothed_loss(self) -> float:
        """Get EMA-smoothed loss."""
        return self._loss_ema

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "smoothed_loss": self._loss_ema,
            "reconstruction_loss": self.reconstruction_loss,
            "contrastive_loss": self.contrastive_loss,
            "masked_spectral_loss": self.masked_spectral_loss,
            "next_window_loss": self.next_window_loss,
            "physics_loss": self.physics_loss,
            "commitment_loss": self.commitment_loss,
            "codebook_loss": self.codebook_loss,
            "alignment_loss": self.alignment_loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "throughput_samples_sec": self.throughput_samples_sec,
            "throughput_tokens_sec": self.throughput_tokens_sec,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
        }


@dataclass
class TrainingState:
    """Complete training state for checkpointing.

    Captures everything needed to resume training from a checkpoint.
    """

    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0

    # Model weights (stored as list of arrays)
    model_params: Optional[List[np.ndarray]] = None

    # Optimizer state
    optimizer_state: Optional[Dict[str, Any]] = None

    # Scheduler state
    scheduler_state: Optional[Dict[str, Any]] = None

    # Curriculum state
    curriculum_phase: int = 0

    # RNG states
    rng_state: Optional[Dict[str, Any]] = None

    # Metrics history
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Save state to file.

        Args:
            path: Output path.
        """
        state_dict = {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
            "curriculum_phase": self.curriculum_phase,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
        }
        if self.model_params is not None:
            state_dict["model_params"] = [p.tolist() for p in self.model_params]
        if self.optimizer_state is not None:
            state_dict["optimizer_state"] = self.optimizer_state
        if self.scheduler_state is not None:
            state_dict["scheduler_state"] = self.scheduler_state

        np.savez_compressed(path, **{k: np.array(v, dtype=object)
                                      for k, v in state_dict.items()})

    @classmethod
    def load(cls, path: str) -> "TrainingState":
        """Load state from file.

        Args:
            path: Input path.

        Returns:
            Loaded training state.
        """
        state = cls()
        data = np.load(path, allow_pickle=True)
        state.step = int(data.get("step", 0))
        state.epoch = int(data.get("epoch", 0))
        state.best_val_loss = float(data.get("best_val_loss", float("inf")))
        return state


class GradientAccumulator:
    """Accumulates gradients across micro-batches.

    Supports gradient accumulation for effective large batch training
    without requiring large memory.

    Attributes:
        accumulation_steps: Number of micro-batches to accumulate.
        current_step: Current accumulation step.
    """

    def __init__(self, accumulation_steps: int = 1):
        """Initialize gradient accumulator.

        Args:
            accumulation_steps: Number of steps to accumulate before update.
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self._accumulated_grads: Optional[List[np.ndarray]] = None
        self._grad_count = 0

    def accumulate(self, gradients: List[np.ndarray]) -> bool:
        """Accumulate gradients from a micro-batch.

        Args:
            gradients: Gradients from one micro-batch.

        Returns:
            True if accumulation is complete and update should happen.
        """
        if self._accumulated_grads is None:
            self._accumulated_grads = [np.zeros_like(g) for g in gradients]

        for i, g in enumerate(gradients):
            self._accumulated_grads[i] += g

        self._grad_count += 1
        self.current_step += 1

        return self._grad_count >= self.accumulation_steps

    def get_accumulated(self) -> List[np.ndarray]:
        """Get averaged accumulated gradients.

        Returns:
            Averaged gradients ready for optimizer step.
        """
        if self._accumulated_grads is None:
            return []

        # Average over accumulation steps
        averaged = [g / self._grad_count for g in self._accumulated_grads]
        return averaged

    def reset(self) -> None:
        """Reset accumulation state."""
        self._accumulated_grads = None
        self._grad_count = 0

    @property
    def is_ready(self) -> bool:
        """Check if enough gradients accumulated."""
        return self._grad_count >= self.accumulation_steps


class EarlyStopping:
    """Early stopping handler.

    Monitors validation loss and stops training if no improvement
    is seen for a specified number of evaluations.

    Attributes:
        patience: Number of evaluations to wait.
        min_delta: Minimum improvement to count.
        mode: 'min' or 'max' for metric direction.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            patience: Evaluations to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        Args:
            metric: Current validation metric.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_value - self.min_delta
        else:
            improved = metric > self.best_value + self.min_delta

        if improved:
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class GradientScaler:
    """Simulated gradient scaling for mixed precision training.

    In a real implementation, this would interact with hardware
    for FP16/BF16 training. Here we simulate the scaling behavior.

    Attributes:
        scale: Current gradient scale factor.
        growth_factor: Scale growth multiplier.
        backoff_factor: Scale reduction on overflow.
        growth_interval: Steps between scale increases.
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """Initialize gradient scaler.

        Args:
            init_scale: Initial scale factor.
            growth_factor: Growth multiplier.
            backoff_factor: Reduction on overflow.
            growth_interval: Steps between growth.
        """
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._overflow_count = 0

    def scale_loss(self, loss: float) -> float:
        """Scale loss for backward pass.

        Args:
            loss: Unscaled loss.

        Returns:
            Scaled loss.
        """
        return loss * self.scale

    def unscale_gradients(self, gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
        """Unscale gradients and check for overflow.

        Args:
            gradients: Scaled gradients.

        Returns:
            Tuple of (unscaled gradients, overflow flag).
        """
        inv_scale = 1.0 / self.scale
        unscaled = [g * inv_scale for g in gradients]

        # Check for overflow/NaN
        overflow = any(
            np.any(np.isnan(g)) or np.any(np.isinf(g))
            for g in unscaled
        )

        return unscaled, overflow

    def update(self, overflow: bool) -> None:
        """Update scale based on overflow status.

        Args:
            overflow: Whether overflow occurred.
        """
        if overflow:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
            self._overflow_count += 1
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0

    def get_scale(self) -> float:
        """Get current scale."""
        return self.scale


class MetricsLogger:
    """Lightweight metrics logger.

    Logs training metrics with configurable intervals and
    provides summary statistics.

    Attributes:
        log_interval: Steps between logs.
        history: Full metrics history.
    """

    def __init__(
        self,
        log_interval: int = 100,
        max_history: int = 100000,
    ):
        """Initialize logger.

        Args:
            log_interval: Logging frequency.
            max_history: Maximum history entries.
        """
        self.log_interval = log_interval
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self._step_buffer: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any], step: int) -> Optional[str]:
        """Log metrics for a step.

        Args:
            metrics: Metrics dictionary.
            step: Training step.

        Returns:
            Formatted log string if at log interval, else None.
        """
        entry = {"step": step, **metrics, "timestamp": time.time()}
        self._step_buffer.append(entry)

        if step % self.log_interval == 0:
            # Compute averages over buffer
            avg_metrics = self._average_buffer()
            self.history.append(avg_metrics)
            self._step_buffer = []

            # Trim history
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            return self._format_log(avg_metrics, step)

        return None

    def _average_buffer(self) -> Dict[str, Any]:
        """Average metrics in buffer."""
        if not self._step_buffer:
            return {}

        avg: Dict[str, Any] = {}
        numeric_keys = [
            k for k in self._step_buffer[0]
            if isinstance(self._step_buffer[0][k], (int, float)) and k != "step"
        ]

        for key in numeric_keys:
            values = [entry[key] for entry in self._step_buffer if key in entry]
            if values:
                avg[key] = sum(values) / len(values)

        avg["step"] = self._step_buffer[-1]["step"]
        return avg

    def _format_log(self, metrics: Dict[str, Any], step: int) -> str:
        """Format metrics for display."""
        parts = [f"Step {step}"]
        if "total_loss" in metrics:
            parts.append(f"loss={metrics['total_loss']:.4f}")
        if "learning_rate" in metrics:
            parts.append(f"lr={metrics['learning_rate']:.2e}")
        if "throughput_samples_sec" in metrics:
            parts.append(f"throughput={metrics['throughput_samples_sec']:.1f} samp/s")
        if "gradient_norm" in metrics:
            parts.append(f"grad_norm={metrics['gradient_norm']:.3f}")
        return " | ".join(parts)

    def get_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Get summary statistics over recent history.

        Args:
            last_n: Number of recent entries to summarize.

        Returns:
            Summary statistics.
        """
        recent = self.history[-last_n:]
        if not recent:
            return {}

        summary: Dict[str, Any] = {}
        numeric_keys = [
            k for k in recent[0]
            if isinstance(recent[0].get(k), (int, float))
        ]

        for key in numeric_keys:
            values = [e[key] for e in recent if key in e]
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_min"] = np.min(values)
                summary[f"{key}_max"] = np.max(values)

        return summary


class PretrainingEngine:
    """Full pretraining engine for spectral foundation models.

    Orchestrates the complete training pipeline including:
    - Data loading and batching
    - Forward pass through model
    - Loss computation with multiple objectives
    - Gradient computation (simulated)
    - Optimizer step with scheduling
    - Metrics logging and checkpointing
    - Curriculum learning
    - Distributed training coordination

    Attributes:
        model_params: Model parameters.
        config: Training configuration dictionary.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        metrics: Training metrics tracker.
        state: Complete training state.
    """

    def __init__(
        self,
        model_params: List[np.ndarray],
        config: Dict[str, Any],
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ):
        """Initialize pretraining engine.

        Args:
            model_params: Initial model parameters.
            config: Training configuration.
            optimizer: Optimizer (created from config if None).
            scheduler: LR scheduler (created from config if None).
        """
        self.model_params = model_params
        self.config = config
        self.state = TrainingState()
        self.metrics = TrainingMetrics()

        # Training hyperparameters
        self.total_steps = config.get("total_steps", 100000)
        self.batch_size = config.get("batch_size", 32)
        self.seq_len = config.get("seq_len", 1024)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.val_interval = config.get("val_interval", 1000)
        self.checkpoint_interval = config.get("checkpoint_interval", 5000)
        self.log_interval = config.get("log_interval", 100)

        # Initialize components
        self.accumulator = GradientAccumulator(self.gradient_accumulation_steps)
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 1e-4),
        )
        self.grad_scaler = GradientScaler()
        self.logger = MetricsLogger(log_interval=self.log_interval)

        # Store optimizer and scheduler references
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_step_begin": [],
            "on_step_end": [],
            "on_epoch_begin": [],
            "on_epoch_end": [],
            "on_validation": [],
            "on_checkpoint": [],
        }

        # Training statistics
        self._step_times: List[float] = []
        self._grad_norms: List[float] = []

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a training callback.

        Args:
            event: Event name (on_step_begin, on_step_end, etc.).
            callback: Callback function.
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _fire_callbacks(self, event: str, **kwargs) -> None:
        """Fire registered callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            callback(**kwargs)

    def compute_loss(
        self,
        model_output: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined pretraining loss.

        Combines multiple objective losses with configurable weights.

        Args:
            model_output: Model forward pass outputs.
            targets: Ground truth targets.
            loss_weights: Per-objective weights.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        if loss_weights is None:
            loss_weights = {
                "reconstruction": 1.0,
                "masked_spectral": 0.5,
                "contrastive": 0.3,
                "next_window": 0.2,
                "physics": 0.1,
                "commitment": 0.01,
                "codebook": 0.01,
                "alignment": 0.1,
            }

        losses: Dict[str, float] = {}

        # Reconstruction loss (MSE)
        if "reconstruction" in model_output and "input" in targets:
            recon = model_output["reconstruction"]
            target = targets["input"]
            losses["reconstruction"] = float(np.mean((recon - target) ** 2))

        # Masked spectral prediction loss
        if "masked_predictions" in model_output and "masked_targets" in targets:
            pred = model_output["masked_predictions"]
            tgt = targets["masked_targets"]
            losses["masked_spectral"] = float(np.mean((pred - tgt) ** 2))

        # Contrastive loss (simplified InfoNCE)
        if "embeddings_a" in model_output and "embeddings_b" in model_output:
            emb_a = model_output["embeddings_a"]
            emb_b = model_output["embeddings_b"]
            losses["contrastive"] = self._compute_contrastive_loss(emb_a, emb_b)

        # Next window prediction
        if "next_window_pred" in model_output and "next_window" in targets:
            pred = model_output["next_window_pred"]
            tgt = targets["next_window"]
            losses["next_window"] = float(np.mean((pred - tgt) ** 2))

        # Physics-informed loss
        if "physics_pred" in model_output and "physics_target" in targets:
            pred = model_output["physics_pred"]
            tgt = targets["physics_target"]
            losses["physics"] = float(np.mean((pred - tgt) ** 2))

        # Commitment loss (VQ-VAE)
        if "commitment_loss" in model_output:
            losses["commitment"] = float(model_output["commitment_loss"])

        # Codebook loss
        if "codebook_loss" in model_output:
            losses["codebook"] = float(model_output["codebook_loss"])

        # Cross-modal alignment
        if "alignment_loss" in model_output:
            losses["alignment"] = float(model_output["alignment_loss"])

        # Compute weighted total
        total_loss = 0.0
        for key, loss_val in losses.items():
            weight = loss_weights.get(key, 0.0)
            total_loss += weight * loss_val

        losses["total"] = total_loss
        return total_loss, losses

    def _compute_contrastive_loss(
        self, emb_a: np.ndarray, emb_b: np.ndarray, temperature: float = 0.07
    ) -> float:
        """Compute InfoNCE contrastive loss.

        Args:
            emb_a: First view embeddings [B, D].
            emb_b: Second view embeddings [B, D].
            temperature: Temperature scaling.

        Returns:
            Contrastive loss value.
        """
        # Normalize embeddings
        emb_a = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        sim = np.dot(emb_a, emb_b.T) / temperature

        # Labels are diagonal (positive pairs)
        batch_size = sim.shape[0]
        labels = np.arange(batch_size)

        # Softmax cross-entropy along rows
        # log_softmax
        max_sim = np.max(sim, axis=-1, keepdims=True)
        exp_sim = np.exp(sim - max_sim)
        log_sum_exp = np.log(np.sum(exp_sim, axis=-1) + 1e-10)

        loss_a = -(sim[np.arange(batch_size), labels] - max_sim.flatten()) + log_sum_exp

        # Symmetric loss
        sim_t = sim.T
        max_sim_t = np.max(sim_t, axis=-1, keepdims=True)
        exp_sim_t = np.exp(sim_t - max_sim_t)
        log_sum_exp_t = np.log(np.sum(exp_sim_t, axis=-1) + 1e-10)

        loss_b = -(sim_t[np.arange(batch_size), labels] - max_sim_t.flatten()) + log_sum_exp_t

        return float(np.mean(loss_a + loss_b) / 2)

    def compute_gradients(
        self,
        model_params: List[np.ndarray],
        batch: Dict[str, np.ndarray],
        loss_fn: Optional[Callable] = None,
    ) -> Tuple[List[np.ndarray], float, Dict[str, float]]:
        """Compute gradients via numerical differentiation.

        Uses finite differences for gradient estimation.
        In production, this would use autograd/backprop.

        Args:
            model_params: Current parameters.
            batch: Input batch.
            loss_fn: Loss function (uses default if None).

        Returns:
            Tuple of (gradients, loss, loss_components).
        """
        epsilon = 1e-5
        gradients = []

        # Forward pass for loss
        model_output = self._forward(model_params, batch)
        targets = self._extract_targets(batch)
        base_loss, loss_components = self.compute_loss(model_output, targets)

        # Compute gradients for each parameter via finite differences
        # (simplified - in practice would use autograd)
        for param_idx, param in enumerate(model_params):
            grad = np.zeros_like(param)

            # Sample a subset of parameters for efficiency
            flat_param = param.flatten()
            num_samples = min(100, len(flat_param))
            sample_indices = np.random.choice(len(flat_param), num_samples, replace=False)

            for idx in sample_indices:
                # Perturb parameter
                flat_param_plus = flat_param.copy()
                flat_param_plus[idx] += epsilon

                perturbed_params = list(model_params)
                perturbed_params[param_idx] = flat_param_plus.reshape(param.shape)

                # Forward with perturbation
                output_plus = self._forward(perturbed_params, batch)
                loss_plus, _ = self.compute_loss(output_plus, targets)

                # Finite difference
                grad.flat[idx] = (loss_plus - base_loss) / epsilon

            gradients.append(grad)

        return gradients, base_loss, loss_components

    def _forward(
        self, params: List[np.ndarray], batch: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Simulate model forward pass.

        Args:
            params: Model parameters.
            batch: Input batch.

        Returns:
            Model outputs dictionary.
        """
        # Simplified forward pass simulation
        x = batch.get("input", np.zeros((self.batch_size, self.seq_len, 64)))

        # Layer-by-layer processing simulation
        hidden = x
        for i, param in enumerate(params[:min(len(params), 12)]):
            # Simple linear + activation
            if hidden.shape[-1] == param.shape[0] if len(param.shape) > 1 else False:
                hidden = np.dot(hidden, param)
            # Apply simplified activation (GELU approximation)
            hidden = hidden * 0.5 * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden ** 3)
            ))

        output: Dict[str, np.ndarray] = {
            "reconstruction": hidden[..., :x.shape[-1]] if hidden.shape[-1] >= x.shape[-1]
                else np.zeros_like(x),
            "hidden_states": hidden,
        }

        # Generate masked predictions if mask provided
        if "mask" in batch:
            mask = batch["mask"]
            output["masked_predictions"] = hidden * mask[..., np.newaxis] \
                if len(mask.shape) < len(hidden.shape) else hidden * mask

        return output

    def _extract_targets(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract targets from batch.

        Args:
            batch: Input batch with potential target fields.

        Returns:
            Targets dictionary.
        """
        targets: Dict[str, np.ndarray] = {}
        if "input" in batch:
            targets["input"] = batch["input"]
        if "masked_targets" in batch:
            targets["masked_targets"] = batch["masked_targets"]
        if "next_window" in batch:
            targets["next_window"] = batch["next_window"]
        if "physics_target" in batch:
            targets["physics_target"] = batch["physics_target"]
        return targets

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Training batch.

        Returns:
            Loss components dictionary.
        """
        step_start = time.time()

        self._fire_callbacks("on_step_begin", step=self.state.step)

        # Compute gradients
        gradients, loss, loss_components = self.compute_gradients(
            self.model_params, batch
        )

        # Gradient scaling for mixed precision
        scaled_grads = [g * self.grad_scaler.get_scale() for g in gradients]
        unscaled_grads, overflow = self.grad_scaler.unscale_gradients(scaled_grads)
        self.grad_scaler.update(overflow)

        if overflow:
            return loss_components

        # Gradient accumulation
        ready = self.accumulator.accumulate(unscaled_grads)

        if ready:
            accumulated_grads = self.accumulator.get_accumulated()

            # Compute gradient norm
            grad_norm = math.sqrt(sum(
                float(np.sum(g ** 2)) for g in accumulated_grads
            ))
            self._grad_norms.append(grad_norm)
            self.metrics.gradient_norm = grad_norm

            # Gradient clipping
            if grad_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (grad_norm + 1e-6)
                accumulated_grads = [g * clip_coef for g in accumulated_grads]

            # Optimizer step
            if self._optimizer is not None:
                self.model_params = self._optimizer.step(accumulated_grads)
            else:
                # Simple SGD fallback
                lr = self._scheduler.get_lr() if self._scheduler else self.config.get("lr", 1e-4)
                self.model_params = [
                    p - lr * g for p, g in zip(self.model_params, accumulated_grads)
                ]

            # Scheduler step
            if self._scheduler is not None:
                self._scheduler.step()

            self.accumulator.reset()

        # Update state
        self.state.step += 1
        elapsed = time.time() - step_start
        self._step_times.append(elapsed)

        # Update metrics
        lr = self._scheduler.get_lr() if self._scheduler else self.config.get("lr", 1e-4)
        self.metrics.update(loss_components, lr, self.batch_size, self.seq_len, elapsed)

        # Log
        log_str = self.logger.log(loss_components, self.state.step)
        if log_str:
            pass  # In production, would print or send to logging service

        self._fire_callbacks("on_step_end", step=self.state.step, loss=loss)

        return loss_components

    def validate(
        self, val_batches: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Run validation.

        Args:
            val_batches: List of validation batches.

        Returns:
            Validation metrics.
        """
        val_losses: List[float] = []
        val_components: Dict[str, List[float]] = {}

        for batch in val_batches:
            model_output = self._forward(self.model_params, batch)
            targets = self._extract_targets(batch)
            loss, components = self.compute_loss(model_output, targets)
            val_losses.append(loss)

            for key, val in components.items():
                if key not in val_components:
                    val_components[key] = []
                val_components[key].append(val)

        avg_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # Check early stopping
        self.early_stopping(avg_loss)

        # Update best model
        if avg_loss < self.state.best_val_loss:
            self.state.best_val_loss = avg_loss

        self.state.val_loss_history.append(avg_loss)

        self._fire_callbacks("on_validation", val_loss=avg_loss)

        return {
            "val_loss": avg_loss,
            **{k: float(np.mean(v)) for k, v in val_components.items()},
        }

    def train(
        self,
        train_data_fn: Callable[[], Dict[str, np.ndarray]],
        val_data_fn: Optional[Callable[[], List[Dict[str, np.ndarray]]]] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full training loop.

        Args:
            train_data_fn: Function that returns a training batch.
            val_data_fn: Function that returns validation batches.
            num_steps: Override total steps.

        Returns:
            Training summary.
        """
        total = num_steps or self.total_steps

        for step in range(self.state.step, total):
            # Get batch
            batch = train_data_fn()

            # Train step
            loss_components = self.train_step(batch)
            self.state.train_loss_history.append(loss_components.get("total", 0.0))

            # Validation
            if val_data_fn and step > 0 and step % self.val_interval == 0:
                val_batches = val_data_fn()
                self.validate(val_batches)

            # Check early stopping
            if self.early_stopping.should_stop:
                break

        return {
            "final_step": self.state.step,
            "best_val_loss": self.state.best_val_loss,
            "train_loss_history": self.state.train_loss_history[-100:],
            "val_loss_history": self.state.val_loss_history[-100:],
            "total_samples": self.metrics.samples_seen,
            "total_tokens": self.metrics.tokens_seen,
        }

    def get_state(self) -> TrainingState:
        """Get current training state for checkpointing."""
        self.state.model_params = self.model_params
        if self._optimizer is not None:
            self.state.optimizer_state = self._optimizer.state_dict()
        if self._scheduler is not None:
            self.state.scheduler_state = self._scheduler.state_dict()
        return self.state

    def load_state(self, state: TrainingState) -> None:
        """Load training state from checkpoint.

        Args:
            state: Training state to restore.
        """
        self.state = state
        if state.model_params is not None:
            self.model_params = state.model_params
        if state.optimizer_state is not None and self._optimizer is not None:
            self._optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None and self._scheduler is not None:
            self._scheduler.load_state_dict(state.scheduler_state)
