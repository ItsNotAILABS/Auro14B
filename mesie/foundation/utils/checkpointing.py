"""Checkpoint management for spectral pretraining.

Handles saving, loading, and managing model checkpoints
with support for best-model tracking and automatic cleanup.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelCheckpoint:
    """Single model checkpoint.

    Attributes:
        step: Training step.
        epoch: Training epoch.
        path: File path.
        metrics: Associated metrics.
        timestamp: Creation time.
    """

    step: int = 0
    epoch: int = 0
    path: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "path": self.path,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "is_best": self.is_best,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelCheckpoint":
        """Create from dictionary."""
        return cls(
            step=d.get("step", 0),
            epoch=d.get("epoch", 0),
            path=d.get("path", ""),
            metrics=d.get("metrics", {}),
            timestamp=d.get("timestamp", 0),
            is_best=d.get("is_best", False),
        )


class CheckpointManager:
    """Manages model checkpoints with automatic retention policies.

    Features:
    - Keep top-k checkpoints by metric
    - Periodic checkpointing
    - Best model tracking
    - Automatic cleanup of old checkpoints
    - Checkpoint metadata tracking

    Attributes:
        checkpoint_dir: Directory for checkpoints.
        max_to_keep: Maximum checkpoints to retain.
        metric_name: Metric to track for best model.
        mode: 'min' or 'max' for metric comparison.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_to_keep: int = 5,
        metric_name: str = "val_loss",
        mode: str = "min",
        save_interval: int = 1000,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Output directory.
            max_to_keep: Maximum checkpoints to keep.
            metric_name: Metric for best model selection.
            mode: 'min' = lower is better, 'max' = higher is better.
            save_interval: Steps between automatic saves.
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.metric_name = metric_name
        self.mode = mode
        self.save_interval = save_interval

        self.checkpoints: List[ModelCheckpoint] = []
        self.best_checkpoint: Optional[ModelCheckpoint] = None
        self.best_metric: Optional[float] = None

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step.

        Args:
            step: Current training step.

        Returns:
            Whether to save.
        """
        return step > 0 and step % self.save_interval == 0

    def save(
        self,
        model_params: List[np.ndarray],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> ModelCheckpoint:
        """Save a checkpoint.

        Args:
            model_params: Model parameters.
            optimizer_state: Optimizer state.
            scheduler_state: Scheduler state.
            step: Training step.
            epoch: Training epoch.
            metrics: Associated metrics.

        Returns:
            Created checkpoint object.
        """
        metrics = metrics or {}

        # Determine if this is the best
        is_best = False
        metric_value = metrics.get(self.metric_name)
        if metric_value is not None:
            if self.best_metric is None:
                is_best = True
            elif self.mode == "min" and metric_value < self.best_metric:
                is_best = True
            elif self.mode == "max" and metric_value > self.best_metric:
                is_best = True

        if is_best and metric_value is not None:
            self.best_metric = metric_value

        # Create checkpoint path
        filename = f"checkpoint_step_{step}_epoch_{epoch}.npz"
        path = os.path.join(self.checkpoint_dir, filename)

        # Save data
        save_data = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }
        for i, param in enumerate(model_params):
            save_data[f"param_{i}"] = param

        if optimizer_state:
            save_data["optimizer_state"] = optimizer_state
        if scheduler_state:
            save_data["scheduler_state"] = scheduler_state

        # Create checkpoint object
        checkpoint = ModelCheckpoint(
            step=step,
            epoch=epoch,
            path=path,
            metrics=metrics,
            timestamp=time.time(),
            is_best=is_best,
        )

        if is_best:
            self.best_checkpoint = checkpoint

        self.checkpoints.append(checkpoint)

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint

    def _cleanup(self) -> None:
        """Remove old checkpoints exceeding max_to_keep."""
        if len(self.checkpoints) <= self.max_to_keep:
            return

        # Sort by metric (keep best ones)
        if self.mode == "min":
            self.checkpoints.sort(
                key=lambda c: c.metrics.get(self.metric_name, float("inf"))
            )
        else:
            self.checkpoints.sort(
                key=lambda c: c.metrics.get(self.metric_name, float("-inf")),
                reverse=True,
            )

        # Keep top-k
        to_remove = self.checkpoints[self.max_to_keep:]
        self.checkpoints = self.checkpoints[:self.max_to_keep]

        # Always keep best
        if self.best_checkpoint and self.best_checkpoint not in self.checkpoints:
            self.checkpoints.append(self.best_checkpoint)

    def get_best(self) -> Optional[ModelCheckpoint]:
        """Get best checkpoint."""
        return self.best_checkpoint

    def get_latest(self) -> Optional[ModelCheckpoint]:
        """Get most recent checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.step)

    def list_checkpoints(self) -> List[ModelCheckpoint]:
        """List all managed checkpoints."""
        return sorted(self.checkpoints, key=lambda c: c.step)


def save_checkpoint(
    path: str,
    model_params: List[np.ndarray],
    step: int = 0,
    **kwargs: Any,
) -> None:
    """Save a model checkpoint to disk.

    Args:
        path: Output file path.
        model_params: Model parameter arrays.
        step: Training step.
        **kwargs: Additional data to save.
    """
    save_dict: Dict[str, Any] = {"step": np.array(step)}

    for i, param in enumerate(model_params):
        save_dict[f"param_{i}"] = param

    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float)):
            save_dict[key] = np.array(value)

    np.savez_compressed(path, **save_dict)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a model checkpoint from disk.

    Args:
        path: Checkpoint file path.

    Returns:
        Dictionary with model parameters and metadata.
    """
    data = np.load(path, allow_pickle=True)

    result: Dict[str, Any] = {}
    params: List[np.ndarray] = []

    for key in sorted(data.files):
        if key.startswith("param_"):
            params.append(data[key])
        else:
            result[key] = data[key]

    result["model_params"] = params
    return result
