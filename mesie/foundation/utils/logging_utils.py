"""Logging utilities for spectral pretraining.

Provides structured logging, metrics aggregation, and
experiment tracking for pretraining runs.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TrainingLogger:
    """Structured training logger.

    Provides hierarchical logging with configurable verbosity,
    output formatting, and integration points for external
    logging systems.

    Attributes:
        log_dir: Directory for log files.
        log_interval: Steps between logs.
        verbosity: Logging verbosity level.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        log_interval: int = 100,
        verbosity: int = 1,
        log_to_file: bool = True,
        log_to_console: bool = True,
    ):
        """Initialize training logger.

        Args:
            log_dir: Log output directory.
            log_interval: Steps between log entries.
            verbosity: Verbosity level (0=minimal, 1=standard, 2=verbose).
            log_to_file: Whether to write to file.
            log_to_console: Whether to print to console.
        """
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.verbosity = verbosity
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console

        self._entries: List[Dict[str, Any]] = []
        self._start_time = time.time()

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = "train",
    ) -> Optional[str]:
        """Log a training step.

        Args:
            step: Training step.
            metrics: Step metrics.
            prefix: Log prefix (train/val/test).

        Returns:
            Formatted log string if at log interval.
        """
        entry = {
            "step": step,
            "prefix": prefix,
            "timestamp": time.time(),
            "elapsed": time.time() - self._start_time,
            **{f"{prefix}/{k}": v for k, v in metrics.items()},
        }
        self._entries.append(entry)

        if step % self.log_interval == 0:
            msg = self._format_entry(entry)
            return msg

        return None

    def log_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Log a training event.

        Args:
            event: Event description.
            details: Optional event details.

        Returns:
            Formatted event string.
        """
        entry = {
            "event": event,
            "timestamp": time.time(),
            "elapsed": time.time() - self._start_time,
            "details": details or {},
        }
        self._entries.append(entry)

        msg = f"[{entry['elapsed']:.1f}s] EVENT: {event}"
        if details and self.verbosity >= 2:
            for k, v in details.items():
                msg += f"\n  {k}: {v}"

        return msg

    def _format_entry(self, entry: Dict[str, Any]) -> str:
        """Format a log entry for display."""
        elapsed = entry.get("elapsed", 0)
        step = entry.get("step", 0)
        prefix = entry.get("prefix", "")

        parts = [f"[{elapsed:.0f}s]", f"Step {step}", f"({prefix})"]

        # Add key metrics
        for key, value in entry.items():
            if key.startswith(f"{prefix}/") and isinstance(value, (int, float)):
                short_key = key.split("/")[-1]
                if "loss" in short_key:
                    parts.append(f"{short_key}={value:.4f}")
                elif "lr" in short_key or "learning_rate" in short_key:
                    parts.append(f"lr={value:.2e}")
                elif "throughput" in short_key:
                    parts.append(f"tput={value:.0f}")

        return " | ".join(parts)

    def get_history(
        self, prefix: Optional[str] = None, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get log history.

        Args:
            prefix: Filter by prefix.
            last_n: Only return last n entries.

        Returns:
            Filtered log entries.
        """
        entries = self._entries
        if prefix:
            entries = [e for e in entries if e.get("prefix") == prefix]
        if last_n:
            entries = entries[-last_n:]
        return entries


class MetricsAggregator:
    """Aggregates metrics across steps and epochs.

    Provides windowed statistics, percentiles, and trend
    detection for training metrics.

    Attributes:
        window_size: Size of rolling window.
        metrics_history: Per-metric history.
    """

    def __init__(self, window_size: int = 100):
        """Initialize metrics aggregator.

        Args:
            window_size: Rolling window size.
        """
        self.window_size = window_size
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self._step_count = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """Add new metrics.

        Args:
            metrics: Metric values.
        """
        self._step_count += 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.metrics_history[key].append(float(value))

    def get_windowed_stats(
        self, metric_name: str, window: Optional[int] = None
    ) -> Dict[str, float]:
        """Get windowed statistics for a metric.

        Args:
            metric_name: Metric name.
            window: Window size (defaults to self.window_size).

        Returns:
            Statistics dictionary.
        """
        window = window or self.window_size
        history = self.metrics_history.get(metric_name, [])

        if not history:
            return {}

        recent = history[-window:]
        arr = np.array(recent)

        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "trend": self._compute_trend(recent),
            "count": len(recent),
        }

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) of values.

        Args:
            values: Time series values.

        Returns:
            Slope (positive = increasing, negative = decreasing).
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2) + 1e-10

        return float(numerator / denominator)

    def detect_plateau(
        self, metric_name: str, threshold: float = 0.001, window: int = 50
    ) -> bool:
        """Detect if a metric has plateaued.

        Args:
            metric_name: Metric to check.
            threshold: Maximum allowed relative change.
            window: Window to check.

        Returns:
            True if metric has plateaued.
        """
        history = self.metrics_history.get(metric_name, [])
        if len(history) < window:
            return False

        recent = history[-window:]
        std = np.std(recent)
        mean = np.abs(np.mean(recent)) + 1e-10

        relative_variation = std / mean
        return relative_variation < threshold

    def detect_divergence(
        self, metric_name: str, threshold: float = 10.0
    ) -> bool:
        """Detect if training is diverging.

        Args:
            metric_name: Metric to check.
            threshold: Divergence threshold.

        Returns:
            True if diverging.
        """
        history = self.metrics_history.get(metric_name, [])
        if len(history) < 10:
            return False

        recent = history[-10:]
        if recent[-1] > threshold * recent[0] and recent[0] > 0:
            return True

        if np.any(np.isnan(recent)) or np.any(np.isinf(recent)):
            return True

        return False

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all tracked metrics."""
        summary = {}
        for name in self.metrics_history:
            summary[name] = self.get_windowed_stats(name)
        return summary


class ExperimentTracker:
    """Tracks experiment configurations and results.

    Provides experiment comparison and hyperparameter tracking
    for spectral pretraining runs.

    Attributes:
        experiment_name: Name of this experiment.
        config: Experiment configuration.
        runs: List of tracked runs.
    """

    def __init__(
        self,
        experiment_name: str = "spectral_pretraining",
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./experiments",
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Experiment name.
            config: Configuration dictionary.
            output_dir: Output directory.
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        self.output_dir = output_dir
        self.start_time = time.time()

        self.runs: List[Dict[str, Any]] = []
        self._current_run: Dict[str, Any] = {
            "name": experiment_name,
            "config": self.config,
            "start_time": self.start_time,
            "metrics": {},
            "checkpoints": [],
            "events": [],
        }

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration to log.
        """
        self._current_run["config"].update(config)

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = "train"
    ) -> None:
        """Log metrics for current run.

        Args:
            metrics: Metric values.
            step: Training step.
            prefix: Metric prefix.
        """
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}"
            if full_key not in self._current_run["metrics"]:
                self._current_run["metrics"][full_key] = []
            self._current_run["metrics"][full_key].append({
                "step": step,
                "value": value,
                "timestamp": time.time(),
            })

    def log_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """Log checkpoint creation.

        Args:
            checkpoint_info: Checkpoint metadata.
        """
        self._current_run["checkpoints"].append({
            **checkpoint_info,
            "timestamp": time.time(),
        })

    def log_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log significant event.

        Args:
            event: Event name.
            details: Event details.
        """
        self._current_run["events"].append({
            "event": event,
            "details": details or {},
            "timestamp": time.time(),
        })

    def finish_run(self, final_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Finish current run and store results.

        Args:
            final_metrics: Final evaluation metrics.

        Returns:
            Run summary.
        """
        self._current_run["end_time"] = time.time()
        self._current_run["duration_seconds"] = time.time() - self.start_time

        if final_metrics:
            self._current_run["final_metrics"] = final_metrics

        self.runs.append(self._current_run)

        # Summary
        summary = {
            "name": self.experiment_name,
            "duration": self._current_run["duration_seconds"],
            "num_events": len(self._current_run["events"]),
            "num_checkpoints": len(self._current_run["checkpoints"]),
        }

        if final_metrics:
            summary["final_metrics"] = final_metrics

        return summary

    def compare_runs(self, metric_name: str) -> List[Dict[str, Any]]:
        """Compare runs by a specific metric.

        Args:
            metric_name: Metric to compare.

        Returns:
            Sorted list of run summaries.
        """
        comparisons = []
        for run in self.runs:
            final = run.get("final_metrics", {})
            if metric_name in final:
                comparisons.append({
                    "name": run["name"],
                    "value": final[metric_name],
                    "config": run.get("config", {}),
                    "duration": run.get("duration_seconds", 0),
                })

        comparisons.sort(key=lambda x: x["value"])
        return comparisons
