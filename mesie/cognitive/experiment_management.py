"""
Spectral Experiment Management System.

Comprehensive experiment tracking, hyperparameter optimization, and reproducibility
framework for spectral analysis workflows.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategy."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    SUCCESSIVE_HALVING = "successive_halving"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    seed: int = 42
    max_iterations: int = 1000
    early_stopping_patience: int = 10
    checkpoint_frequency: int = 100

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "tags": self.tags,
            "seed": self.seed,
            "max_iterations": self.max_iterations,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_frequency": self.checkpoint_frequency,
        }

    def get_hash(self) -> str:
        """Get deterministic hash for this config."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ExperimentResult:
    """Result of an experiment run."""
    config_hash: str
    metrics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    status: ExperimentStatus = ExperimentStatus.COMPLETED
    error_message: str = ""
    iteration_history: list = field(default_factory=list)

    @property
    def best_metric(self) -> float:
        """Get the best primary metric value."""
        if "primary" in self.metrics:
            return self.metrics["primary"]
        if self.metrics:
            return list(self.metrics.values())[0]
        return float("nan")


@dataclass
class Checkpoint:
    """Model checkpoint during training."""
    iteration: int
    metrics: dict
    state: dict
    timestamp: float = field(default_factory=time.time)


class ExperimentTracker:
    """
    Track and manage spectral analysis experiments.

    Provides experiment versioning, metric logging, artifact storage,
    and comparison utilities for reproducible spectral research.
    """

    def __init__(self, project_name: str = "spectral_experiments"):
        self.project_name = project_name
        self.experiments: dict[str, ExperimentResult] = {}
        self.active_experiment: Optional[str] = None
        self._metric_history: dict[str, list[dict]] = {}
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._start_time: float = 0
        self._configs: dict[str, ExperimentConfig] = {}
        self._comparison_cache: dict[str, Any] = {}

    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment run."""
        exp_id = config.get_hash()
        self._configs[exp_id] = config
        self._metric_history[exp_id] = []
        self._checkpoints[exp_id] = []
        self.active_experiment = exp_id
        self._start_time = time.time()
        np.random.seed(config.seed)
        return exp_id

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics for the active experiment."""
        if self.active_experiment is None:
            raise RuntimeError("No active experiment")
        entry = {"metrics": metrics.copy(), "step": step, "timestamp": time.time()}
        self._metric_history[self.active_experiment].append(entry)

    def log_artifact(self, name: str, data: Any) -> None:
        """Log an artifact (array, model state, etc.) for the active experiment."""
        if self.active_experiment is None:
            raise RuntimeError("No active experiment")
        exp_id = self.active_experiment
        if exp_id not in self.experiments:
            self.experiments[exp_id] = ExperimentResult(config_hash=exp_id)
        self.experiments[exp_id].artifacts[name] = data

    def save_checkpoint(self, iteration: int, metrics: dict, state: dict) -> None:
        """Save a training checkpoint."""
        if self.active_experiment is None:
            raise RuntimeError("No active experiment")
        cp = Checkpoint(iteration=iteration, metrics=metrics.copy(), state=state.copy())
        self._checkpoints[self.active_experiment].append(cp)

    def end_experiment(
        self, status: ExperimentStatus = ExperimentStatus.COMPLETED, error: str = ""
    ) -> ExperimentResult:
        """End the active experiment and return results."""
        if self.active_experiment is None:
            raise RuntimeError("No active experiment")
        exp_id = self.active_experiment
        duration = time.time() - self._start_time

        # Compute final metrics from history
        final_metrics = {}
        if self._metric_history[exp_id]:
            last_entry = self._metric_history[exp_id][-1]
            final_metrics = last_entry["metrics"]

        result = ExperimentResult(
            config_hash=exp_id,
            metrics=final_metrics,
            duration_seconds=duration,
            status=status,
            error_message=error,
            iteration_history=[e["metrics"] for e in self._metric_history[exp_id]],
        )
        if exp_id in self.experiments:
            result.artifacts = self.experiments[exp_id].artifacts
        self.experiments[exp_id] = result
        self.active_experiment = None
        return result

    def get_best_experiment(self, metric_name: str = "primary", minimize: bool = False) -> Optional[str]:
        """Get the experiment ID with the best metric value."""
        best_id = None
        best_val = float("inf") if minimize else float("-inf")
        for exp_id, result in self.experiments.items():
            if result.status != ExperimentStatus.COMPLETED:
                continue
            val = result.metrics.get(metric_name, float("nan"))
            if np.isnan(val):
                continue
            if minimize and val < best_val:
                best_val = val
                best_id = exp_id
            elif not minimize and val > best_val:
                best_val = val
                best_id = exp_id
        return best_id

    def compare_experiments(self, exp_ids: list[str], metric_names: list[str]) -> dict:
        """Compare multiple experiments across specified metrics."""
        comparison = {"experiments": {}, "rankings": {}}
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                continue
            result = self.experiments[exp_id]
            comparison["experiments"][exp_id] = {
                m: result.metrics.get(m, None) for m in metric_names
            }
        # Rank by each metric
        for m in metric_names:
            values = []
            for exp_id in exp_ids:
                if exp_id in comparison["experiments"]:
                    val = comparison["experiments"][exp_id].get(m)
                    if val is not None:
                        values.append((exp_id, val))
            values.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"][m] = [v[0] for v in values]
        return comparison

    def get_metric_history(self, exp_id: Optional[str] = None) -> list[dict]:
        """Get the metric history for an experiment."""
        eid = exp_id or self.active_experiment
        if eid is None:
            return []
        return self._metric_history.get(eid, [])

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.experiments.values() if r.status == ExperimentStatus.COMPLETED)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for spectral models.

    Supports grid search, random search, Bayesian optimization,
    evolutionary strategies, and successive halving.
    """

    def __init__(
        self,
        search_space: dict[str, Any],
        strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH,
        n_trials: int = 50,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.strategy = strategy
        self.n_trials = n_trials
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.trial_results: list[dict] = []
        self._best_params: Optional[dict] = None
        self._best_score: float = float("-inf")
        self._surrogate_model: Optional[dict] = None
        self._population: list[dict] = []

    def _sample_random(self) -> dict:
        """Sample random parameters from search space."""
        params = {}
        for name, spec in self.search_space.items():
            if isinstance(spec, list):
                params[name] = spec[self.rng.integers(len(spec))]
            elif isinstance(spec, dict):
                if spec.get("type") == "float":
                    low, high = spec["range"]
                    if spec.get("log", False):
                        params[name] = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
                    else:
                        params[name] = float(self.rng.uniform(low, high))
                elif spec.get("type") == "int":
                    low, high = spec["range"]
                    params[name] = int(self.rng.integers(low, high + 1))
                elif spec.get("type") == "categorical":
                    choices = spec["choices"]
                    params[name] = choices[self.rng.integers(len(choices))]
                else:
                    params[name] = spec.get("default", 0)
            else:
                params[name] = spec
        return params

    def _sample_grid(self, trial_idx: int) -> dict:
        """Sample parameters from grid based on trial index."""
        # Build grid
        grid_dims = []
        dim_names = []
        for name, spec in self.search_space.items():
            dim_names.append(name)
            if isinstance(spec, list):
                grid_dims.append(spec)
            elif isinstance(spec, dict):
                n_points = spec.get("n_grid_points", 5)
                if spec.get("type") == "float":
                    low, high = spec["range"]
                    if spec.get("log", False):
                        grid_dims.append(list(np.exp(np.linspace(np.log(low), np.log(high), n_points))))
                    else:
                        grid_dims.append(list(np.linspace(low, high, n_points)))
                elif spec.get("type") == "int":
                    low, high = spec["range"]
                    grid_dims.append(list(range(low, min(high + 1, low + n_points))))
                elif spec.get("type") == "categorical":
                    grid_dims.append(spec["choices"])
                else:
                    grid_dims.append([spec.get("default", 0)])
            else:
                grid_dims.append([spec])

        # Index into grid
        params = {}
        idx = trial_idx
        for i, (name, dim) in enumerate(zip(dim_names, grid_dims)):
            dim_size = len(dim)
            params[name] = dim[idx % dim_size]
            idx //= dim_size
        return params

    def _sample_bayesian(self) -> dict:
        """Sample parameters using surrogate model."""
        if len(self.trial_results) < 5:
            return self._sample_random()

        # Simple GP-like surrogate: weighted random near best
        best_params = self._best_params or self._sample_random()
        params = {}
        for name, spec in self.search_space.items():
            if isinstance(spec, list):
                # With 70% prob use best, 30% explore
                if self.rng.random() < 0.7:
                    params[name] = best_params.get(name, spec[0])
                else:
                    params[name] = spec[self.rng.integers(len(spec))]
            elif isinstance(spec, dict):
                if spec.get("type") in ("float", "int"):
                    low, high = spec["range"]
                    best_val = best_params.get(name, (low + high) / 2)
                    # Perturb around best
                    spread = (high - low) * 0.2
                    new_val = best_val + self.rng.normal(0, spread)
                    new_val = np.clip(new_val, low, high)
                    if spec.get("type") == "int":
                        params[name] = int(round(new_val))
                    else:
                        params[name] = float(new_val)
                elif spec.get("type") == "categorical":
                    choices = spec["choices"]
                    if self.rng.random() < 0.7:
                        params[name] = best_params.get(name, choices[0])
                    else:
                        params[name] = choices[self.rng.integers(len(choices))]
                else:
                    params[name] = spec.get("default", 0)
            else:
                params[name] = spec
        return params

    def _sample_evolutionary(self) -> dict:
        """Sample parameters using evolutionary strategy."""
        pop_size = max(10, self.n_trials // 5)

        if len(self._population) < pop_size:
            # Initialize population
            params = self._sample_random()
            self._population.append(params)
            return params

        # Tournament selection + mutation
        tournament_size = 3
        tournament = self.rng.choice(len(self.trial_results), size=min(tournament_size, len(self.trial_results)), replace=False)
        best_idx = tournament[np.argmax([self.trial_results[i]["score"] for i in tournament])]
        parent = self.trial_results[best_idx]["params"]

        # Mutate
        params = {}
        mutation_rate = 0.3
        for name, spec in self.search_space.items():
            if self.rng.random() < mutation_rate:
                params[name] = self._sample_random()[name]
            else:
                params[name] = parent.get(name, self._sample_random()[name])
        return params

    def suggest(self, trial_idx: Optional[int] = None) -> dict:
        """Suggest next hyperparameter configuration."""
        idx = trial_idx if trial_idx is not None else len(self.trial_results)
        if self.strategy == OptimizationStrategy.GRID_SEARCH:
            return self._sample_grid(idx)
        elif self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._sample_random()
        elif self.strategy == OptimizationStrategy.BAYESIAN:
            return self._sample_bayesian()
        elif self.strategy == OptimizationStrategy.EVOLUTIONARY:
            return self._sample_evolutionary()
        elif self.strategy == OptimizationStrategy.SUCCESSIVE_HALVING:
            return self._sample_random()
        return self._sample_random()

    def report(self, params: dict, score: float, extra_metrics: Optional[dict] = None) -> None:
        """Report trial result."""
        result = {"params": params, "score": score, "extra_metrics": extra_metrics or {}}
        self.trial_results.append(result)
        if score > self._best_score:
            self._best_score = score
            self._best_params = params.copy()

    @property
    def best_params(self) -> Optional[dict]:
        return self._best_params

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def n_completed_trials(self) -> int:
        return len(self.trial_results)

    def get_importance(self) -> dict[str, float]:
        """Estimate parameter importance based on correlation with score."""
        if len(self.trial_results) < 5:
            return {name: 1.0 / len(self.search_space) for name in self.search_space}

        scores = np.array([r["score"] for r in self.trial_results])
        importances = {}
        for name in self.search_space:
            values = []
            for r in self.trial_results:
                val = r["params"].get(name, 0)
                if isinstance(val, (int, float)):
                    values.append(val)
                else:
                    values.append(hash(str(val)) % 1000)
            values = np.array(values, dtype=float)
            if np.std(values) > 0 and np.std(scores) > 0:
                corr = abs(np.corrcoef(values, scores)[0, 1])
                importances[name] = float(corr) if not np.isnan(corr) else 0.0
            else:
                importances[name] = 0.0

        # Normalize
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        return importances


class ReproducibilityManager:
    """
    Ensure reproducibility of spectral experiments.

    Manages seeds, environment snapshots, and deterministic execution
    to guarantee identical results across runs.
    """

    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self._rng = np.random.default_rng(master_seed)
        self._seed_registry: dict[str, int] = {}
        self._environment_snapshots: list[dict] = []
        self._execution_log: list[dict] = []
        self._deterministic_mode: bool = True

    def get_seed(self, component_name: str) -> int:
        """Get a deterministic seed for a component."""
        if component_name not in self._seed_registry:
            self._seed_registry[component_name] = int(self._rng.integers(0, 2**31))
        return self._seed_registry[component_name]

    def snapshot_environment(self) -> dict:
        """Capture current environment state."""
        snapshot = {
            "timestamp": time.time(),
            "master_seed": self.master_seed,
            "seed_registry": self._seed_registry.copy(),
            "numpy_version": np.__version__,
            "deterministic_mode": self._deterministic_mode,
        }
        self._environment_snapshots.append(snapshot)
        return snapshot

    def log_execution(self, operation: str, inputs_hash: str, outputs_hash: str) -> None:
        """Log an execution step for audit trail."""
        entry = {
            "operation": operation,
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "timestamp": time.time(),
        }
        self._execution_log.append(entry)

    def verify_reproducibility(self, func, args: tuple, kwargs: dict, expected_hash: str) -> bool:
        """Verify that a function produces the same output given the same inputs."""
        np.random.seed(self.master_seed)
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            actual_hash = hashlib.sha256(result.tobytes()).hexdigest()[:16]
        else:
            actual_hash = hashlib.sha256(str(result).encode()).hexdigest()[:16]
        return actual_hash == expected_hash

    def compute_hash(self, data: Any) -> str:
        """Compute deterministic hash for any data."""
        if isinstance(data, np.ndarray):
            return hashlib.sha256(data.tobytes()).hexdigest()[:16]
        return hashlib.sha256(str(data).encode()).hexdigest()[:16]

    def create_reproducible_context(self, name: str) -> "ReproducibleContext":
        """Create a reproducible execution context."""
        seed = self.get_seed(name)
        return ReproducibleContext(name=name, seed=seed, manager=self)

    @property
    def execution_log(self) -> list[dict]:
        return self._execution_log.copy()

    @property
    def n_logged_operations(self) -> int:
        return len(self._execution_log)


class ReproducibleContext:
    """Context manager for reproducible execution blocks."""

    def __init__(self, name: str, seed: int, manager: ReproducibilityManager):
        self.name = name
        self.seed = seed
        self.manager = manager
        self._prev_state: Optional[dict] = None
        self.rng = np.random.default_rng(seed)

    def __enter__(self):
        self._prev_state = np.random.get_state()
        np.random.seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_state is not None:
            np.random.set_state(self._prev_state)
        return False


class CrossValidationEngine:
    """
    Cross-validation engine for spectral data.

    Implements k-fold, stratified, time-series, and spectral-aware
    cross-validation strategies.
    """

    def __init__(
        self,
        n_folds: int = 5,
        strategy: str = "kfold",
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.n_folds = n_folds
        self.strategy = strategy
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fold_results: list[dict] = []
        self._fold_indices: list[tuple] = []

    def generate_folds(self, n_samples: int, labels: Optional[np.ndarray] = None) -> list[tuple]:
        """Generate train/test fold indices."""
        indices = np.arange(n_samples)
        if self.shuffle:
            self.rng.shuffle(indices)

        folds = []
        if self.strategy == "kfold":
            fold_size = n_samples // self.n_folds
            for i in range(self.n_folds):
                start = i * fold_size
                end = start + fold_size if i < self.n_folds - 1 else n_samples
                test_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])
                folds.append((train_idx, test_idx))

        elif self.strategy == "stratified" and labels is not None:
            unique_labels = np.unique(labels)
            fold_indices = [[] for _ in range(self.n_folds)]
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                if self.shuffle:
                    self.rng.shuffle(label_indices)
                fold_size = len(label_indices) // self.n_folds
                for i in range(self.n_folds):
                    start = i * fold_size
                    end = start + fold_size if i < self.n_folds - 1 else len(label_indices)
                    fold_indices[i].extend(label_indices[start:end])
            for i in range(self.n_folds):
                test_idx = np.array(fold_indices[i])
                train_idx = np.concatenate([np.array(fold_indices[j]) for j in range(self.n_folds) if j != i])
                folds.append((train_idx, test_idx))

        elif self.strategy == "timeseries":
            # Expanding window
            min_train = max(n_samples // (self.n_folds + 1), 10)
            for i in range(self.n_folds):
                train_end = min_train + i * (n_samples - min_train) // self.n_folds
                test_end = min(train_end + (n_samples - min_train) // self.n_folds, n_samples)
                train_idx = indices[:train_end]
                test_idx = indices[train_end:test_end]
                if len(test_idx) > 0:
                    folds.append((train_idx, test_idx))

        elif self.strategy == "spectral_aware":
            # Group similar spectra together using L2 norm buckets
            folds = self._spectral_aware_folds(indices, n_samples)

        else:
            # Default to kfold
            return self.generate_folds(n_samples, labels)

        self._fold_indices = folds
        return folds

    def _spectral_aware_folds(self, indices: np.ndarray, n_samples: int) -> list[tuple]:
        """Generate folds that ensure spectral diversity in each fold."""
        fold_size = n_samples // self.n_folds
        folds = []
        for i in range(self.n_folds):
            # Interleave to ensure diversity
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[i::self.n_folds] = True
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
            folds.append((train_idx, test_idx))
        return folds

    def evaluate_fold(self, fold_idx: int, metrics: dict) -> None:
        """Record metrics for a fold."""
        self.fold_results.append({"fold": fold_idx, "metrics": metrics})

    def get_summary(self) -> dict:
        """Get summary statistics across all folds."""
        if not self.fold_results:
            return {}
        all_metrics = {}
        for result in self.fold_results:
            for key, val in result["metrics"].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(val)

        summary = {}
        for key, values in all_metrics.items():
            arr = np.array(values)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "ci_95": float(1.96 * np.std(arr) / np.sqrt(len(arr))),
            }
        return summary

    @property
    def n_completed_folds(self) -> int:
        return len(self.fold_results)


class StatisticalTestSuite:
    """
    Statistical tests for comparing spectral models and methods.

    Implements paired tests, bootstrap confidence intervals, and
    multiple comparison corrections.
    """

    def __init__(self, alpha: float = 0.05, correction: str = "bonferroni"):
        self.alpha = alpha
        self.correction = correction
        self._test_results: list[dict] = []

    def paired_t_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
        """Perform paired t-test between two sets of scores."""
        diff = scores_a - scores_b
        n = len(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        if std_diff == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            # Approximate p-value using normal distribution for large n
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        result = {
            "test": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "significant": p_value < self.alpha,
            "effect_size": float(mean_diff / std_diff) if std_diff > 0 else 0.0,
        }
        self._test_results.append(result)
        return result

    def bootstrap_ci(
        self, scores: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95, seed: int = 42
    ) -> dict:
        """Compute bootstrap confidence interval."""
        rng = np.random.default_rng(seed)
        n = len(scores)
        boot_means = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = scores[rng.integers(0, n, size=n)]
            boot_means[i] = np.mean(sample)

        alpha = 1 - ci_level
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

        return {
            "mean": float(np.mean(scores)),
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_level": ci_level,
            "std_error": float(np.std(boot_means)),
        }

    def wilcoxon_signed_rank(self, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
        """Perform Wilcoxon signed-rank test (non-parametric)."""
        diff = scores_a - scores_b
        # Remove zeros
        nonzero = diff[diff != 0]
        n = len(nonzero)

        if n == 0:
            return {"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0, "significant": False}

        # Rank absolute differences
        abs_diff = np.abs(nonzero)
        ranks = np.argsort(np.argsort(abs_diff)) + 1.0

        # Sum of positive ranks
        w_plus = np.sum(ranks[nonzero > 0])
        w_minus = np.sum(ranks[nonzero < 0])
        w = min(w_plus, w_minus)

        # Normal approximation for large n
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        if std_w > 0:
            z = (w - mean_w) / std_w
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            p_value = 1.0

        result = {
            "test": "wilcoxon_signed_rank",
            "statistic": float(w),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }
        self._test_results.append(result)
        return result

    def multiple_comparison_correction(self, p_values: list[float]) -> list[dict]:
        """Apply multiple comparison correction to p-values."""
        n_tests = len(p_values)
        adjusted = []

        if self.correction == "bonferroni":
            for p in p_values:
                adj_p = min(p * n_tests, 1.0)
                adjusted.append({"original_p": p, "adjusted_p": adj_p, "significant": adj_p < self.alpha})

        elif self.correction == "holm":
            # Holm-Bonferroni
            sorted_idx = np.argsort(p_values)
            for rank, idx in enumerate(sorted_idx):
                adj_p = min(p_values[idx] * (n_tests - rank), 1.0)
                adjusted.append({"original_p": p_values[idx], "adjusted_p": adj_p, "rank": rank})
            # Re-sort by original order
            adjusted_ordered = [None] * n_tests
            for rank, idx in enumerate(sorted_idx):
                adjusted_ordered[idx] = adjusted[rank]
                adjusted_ordered[idx]["significant"] = adjusted[rank]["adjusted_p"] < self.alpha
            adjusted = adjusted_ordered

        elif self.correction == "fdr":
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_values)
            for rank, idx in enumerate(sorted_idx):
                adj_p = min(p_values[idx] * n_tests / (rank + 1), 1.0)
                adjusted.append({"original_p": p_values[idx], "adjusted_p": adj_p})
            adjusted_ordered = [None] * n_tests
            for rank, idx in enumerate(sorted_idx):
                adjusted_ordered[idx] = adjusted[rank]
                adjusted_ordered[idx]["significant"] = adjusted[rank]["adjusted_p"] < self.alpha
            adjusted = adjusted_ordered
        else:
            for p in p_values:
                adjusted.append({"original_p": p, "adjusted_p": p, "significant": p < self.alpha})

        return adjusted

    def effect_size_cohens_d(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n_a, n_b = len(group_a), len(group_b)
        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    @property
    def all_results(self) -> list[dict]:
        return self._test_results.copy()


class AblationStudyRunner:
    """
    Run ablation studies on spectral processing pipelines.

    Systematically removes or varies components to measure their
    individual contributions to overall performance.
    """

    def __init__(self, baseline_config: dict, components: list[str]):
        self.baseline_config = baseline_config
        self.components = components
        self.ablation_results: dict[str, dict] = {}
        self._baseline_score: Optional[float] = None

    def set_baseline(self, score: float, metrics: Optional[dict] = None) -> None:
        """Set the baseline performance."""
        self._baseline_score = score
        self.ablation_results["baseline"] = {
            "score": score,
            "metrics": metrics or {},
            "removed_component": None,
        }

    def record_ablation(self, component: str, score: float, metrics: Optional[dict] = None) -> dict:
        """Record the result of removing a component."""
        if self._baseline_score is None:
            raise RuntimeError("Set baseline first")

        delta = score - self._baseline_score
        relative_change = delta / abs(self._baseline_score) if self._baseline_score != 0 else 0.0

        result = {
            "score": score,
            "metrics": metrics or {},
            "removed_component": component,
            "absolute_change": float(delta),
            "relative_change": float(relative_change),
            "is_important": abs(relative_change) > 0.01,  # >1% change
        }
        self.ablation_results[component] = result
        return result

    def get_component_importance(self) -> dict[str, float]:
        """Rank components by their importance (larger drop = more important)."""
        importances = {}
        for comp in self.components:
            if comp in self.ablation_results:
                # Negative change means score drops without component = important
                importances[comp] = -self.ablation_results[comp]["absolute_change"]
            else:
                importances[comp] = 0.0
        # Normalize
        total = sum(abs(v) for v in importances.values())
        if total > 0:
            importances = {k: abs(v) / total for k, v in importances.items()}
        return importances

    def get_summary_report(self) -> dict:
        """Generate ablation study summary."""
        if self._baseline_score is None:
            return {"error": "No baseline set"}

        important = []
        redundant = []
        for comp in self.components:
            if comp in self.ablation_results:
                result = self.ablation_results[comp]
                if result["is_important"]:
                    important.append(comp)
                else:
                    redundant.append(comp)

        return {
            "baseline_score": self._baseline_score,
            "n_components": len(self.components),
            "n_ablated": len(self.ablation_results) - 1,  # minus baseline
            "important_components": important,
            "redundant_components": redundant,
            "component_importance": self.get_component_importance(),
        }


class SpectralBenchmark:
    """
    Benchmark suite for evaluating spectral processing methods.

    Provides standard datasets, metrics, and comparison utilities
    for fair evaluation of spectral analysis algorithms.
    """

    def __init__(self, name: str = "spectral_benchmark", n_samples: int = 1000, n_features: int = 256):
        self.name = name
        self.n_samples = n_samples
        self.n_features = n_features
        self._datasets: dict[str, dict] = {}
        self._results: dict[str, list[dict]] = {}
        self._generate_synthetic_datasets()

    def _generate_synthetic_datasets(self) -> None:
        """Generate standard synthetic benchmark datasets."""
        rng = np.random.default_rng(42)

        # Dataset 1: Pure harmonics (easy)
        t = np.linspace(0, 10, self.n_features)
        harmonics = np.zeros((self.n_samples, self.n_features))
        labels = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            n_harmonics = rng.integers(1, 5)
            label = n_harmonics - 1
            labels[i] = label
            for h in range(n_harmonics):
                freq = rng.uniform(0.5, 5.0)
                harmonics[i] += np.sin(2 * np.pi * freq * t * (h + 1))
            harmonics[i] += rng.normal(0, 0.1, self.n_features)
        self._datasets["harmonics"] = {"X": harmonics, "y": labels, "difficulty": "easy"}

        # Dataset 2: Overlapping Gaussians (medium)
        n_classes = 5
        gaussians = np.zeros((self.n_samples, self.n_features))
        g_labels = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            cls = i % n_classes
            g_labels[i] = cls
            center = self.n_features * (cls + 1) // (n_classes + 1)
            width = self.n_features // 10
            gaussians[i] = np.exp(-0.5 * ((np.arange(self.n_features) - center) / width) ** 2)
            gaussians[i] += rng.normal(0, 0.2, self.n_features)
        self._datasets["gaussians"] = {"X": gaussians, "y": g_labels, "difficulty": "medium"}

        # Dataset 3: Mixed signals (hard)
        mixed = np.zeros((self.n_samples, self.n_features))
        m_labels = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            cls = i % 3
            m_labels[i] = cls
            if cls == 0:
                mixed[i] = np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.5, self.n_features)
            elif cls == 1:
                mixed[i] = np.exp(-t / 3) * np.cos(2 * np.pi * t) + rng.normal(0, 0.5, self.n_features)
            else:
                mixed[i] = rng.normal(0, 1, self.n_features)
                mixed[i] = np.cumsum(mixed[i]) / np.sqrt(self.n_features)
        self._datasets["mixed"] = {"X": mixed, "y": m_labels, "difficulty": "hard"}

    def get_dataset(self, name: str) -> Optional[dict]:
        """Get a benchmark dataset by name."""
        return self._datasets.get(name)

    @property
    def dataset_names(self) -> list[str]:
        return list(self._datasets.keys())

    def evaluate(self, method_name: str, predictions: np.ndarray, dataset_name: str) -> dict:
        """Evaluate predictions against ground truth."""
        dataset = self._datasets.get(dataset_name)
        if dataset is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        y_true = dataset["y"]
        accuracy = float(np.mean(predictions == y_true))

        # Per-class accuracy
        unique_classes = np.unique(y_true)
        per_class = {}
        for cls in unique_classes:
            mask = y_true == cls
            per_class[int(cls)] = float(np.mean(predictions[mask] == cls))

        result = {
            "method": method_name,
            "dataset": dataset_name,
            "accuracy": accuracy,
            "per_class_accuracy": per_class,
            "n_samples": len(y_true),
            "n_classes": len(unique_classes),
        }

        if dataset_name not in self._results:
            self._results[dataset_name] = []
        self._results[dataset_name].append(result)
        return result

    def get_leaderboard(self, dataset_name: str) -> list[dict]:
        """Get method rankings for a dataset."""
        results = self._results.get(dataset_name, [])
        return sorted(results, key=lambda r: r["accuracy"], reverse=True)

    def get_overall_ranking(self) -> dict[str, float]:
        """Get average ranking across all datasets."""
        method_scores: dict[str, list[float]] = {}
        for dataset_name, results in self._results.items():
            for r in results:
                name = r["method"]
                if name not in method_scores:
                    method_scores[name] = []
                method_scores[name].append(r["accuracy"])
        return {name: float(np.mean(scores)) for name, scores in method_scores.items()}


class DataAugmentation:
    """
    Data augmentation strategies for spectral data.

    Implements spectral-specific augmentation techniques including
    noise injection, frequency shifting, amplitude scaling, and
    synthetic spectra generation.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._augmentation_count = 0

    def add_noise(self, spectrum: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Gaussian noise at specified SNR."""
        signal_power = np.mean(spectrum**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), spectrum.shape)
        self._augmentation_count += 1
        return spectrum + noise

    def frequency_shift(self, spectrum: np.ndarray, shift_amount: int = 5) -> np.ndarray:
        """Shift spectrum in frequency domain."""
        result = np.zeros_like(spectrum)
        if shift_amount > 0:
            result[shift_amount:] = spectrum[:-shift_amount]
        elif shift_amount < 0:
            result[:shift_amount] = spectrum[-shift_amount:]
        else:
            result = spectrum.copy()
        self._augmentation_count += 1
        return result

    def amplitude_scale(self, spectrum: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """Random amplitude scaling."""
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        self._augmentation_count += 1
        return spectrum * scale

    def time_warp(self, spectrum: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
        """Time-domain warping via interpolation."""
        n = len(spectrum)
        # Create warped time axis
        original_t = np.arange(n)
        warp = 1.0 + self.rng.uniform(-warp_factor, warp_factor, n)
        warped_t = np.cumsum(warp)
        warped_t = warped_t * (n - 1) / warped_t[-1]  # Normalize to same range
        # Interpolate
        result = np.interp(original_t, warped_t, spectrum)
        self._augmentation_count += 1
        return result

    def spectral_masking(self, spectrum: np.ndarray, mask_fraction: float = 0.1) -> np.ndarray:
        """Randomly mask portions of the spectrum."""
        n = len(spectrum)
        mask_len = max(1, int(n * mask_fraction))
        start = self.rng.integers(0, n - mask_len)
        result = spectrum.copy()
        result[start:start + mask_len] = 0
        self._augmentation_count += 1
        return result

    def mixup(self, spectrum_a: np.ndarray, spectrum_b: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Mixup augmentation between two spectra."""
        lam = self.rng.beta(alpha, alpha) if alpha > 0 else 0.5
        self._augmentation_count += 1
        return lam * spectrum_a + (1 - lam) * spectrum_b

    def cutmix(self, spectrum_a: np.ndarray, spectrum_b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """CutMix augmentation."""
        n = len(spectrum_a)
        lam = self.rng.beta(alpha, alpha)
        cut_len = int(n * (1 - lam))
        start = self.rng.integers(0, n - cut_len + 1)
        result = spectrum_a.copy()
        result[start:start + cut_len] = spectrum_b[start:start + cut_len]
        self._augmentation_count += 1
        return result

    def generate_augmented_batch(
        self, spectra: np.ndarray, augmentations: list[str], n_augmented: int = 0
    ) -> np.ndarray:
        """Generate an augmented batch of spectra."""
        if n_augmented <= 0:
            n_augmented = len(spectra)
        results = []
        for i in range(n_augmented):
            idx = self.rng.integers(len(spectra))
            spectrum = spectra[idx].copy()
            aug = augmentations[self.rng.integers(len(augmentations))]
            if aug == "noise":
                spectrum = self.add_noise(spectrum)
            elif aug == "shift":
                spectrum = self.frequency_shift(spectrum, self.rng.integers(-10, 11))
            elif aug == "scale":
                spectrum = self.amplitude_scale(spectrum)
            elif aug == "warp":
                spectrum = self.time_warp(spectrum)
            elif aug == "mask":
                spectrum = self.spectral_masking(spectrum)
            results.append(spectrum)
        return np.array(results)

    @property
    def total_augmentations(self) -> int:
        return self._augmentation_count


class ExperimentPipeline:
    """
    End-to-end experiment pipeline combining tracking, optimization, and evaluation.

    Orchestrates the full lifecycle: design, execution, analysis, and reporting.
    """

    def __init__(self, name: str, search_space: Optional[dict] = None):
        self.name = name
        self.tracker = ExperimentTracker(project_name=name)
        self.reproducibility = ReproducibilityManager()
        self.cv_engine = CrossValidationEngine()
        self.stats = StatisticalTestSuite()
        self.augmentation = DataAugmentation()
        self.optimizer = (
            HyperparameterOptimizer(search_space=search_space)
            if search_space
            else None
        )
        self._pipeline_runs: list[dict] = []

    def run_experiment(
        self,
        config: ExperimentConfig,
        data: np.ndarray,
        labels: np.ndarray,
        model_fn: Any = None,
    ) -> ExperimentResult:
        """Run a full experiment with cross-validation."""
        exp_id = self.tracker.start_experiment(config)

        try:
            # Cross-validate
            folds = self.cv_engine.generate_folds(len(data), labels)
            fold_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                X_train, X_test = data[train_idx], data[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # Simple nearest-centroid classifier as default
                if model_fn is None:
                    unique_labels = np.unique(y_train)
                    centroids = np.array([X_train[y_train == l].mean(axis=0) for l in unique_labels])
                    dists = np.array([np.linalg.norm(X_test - c, axis=1) for c in centroids])
                    preds = unique_labels[np.argmin(dists, axis=0)]
                    accuracy = float(np.mean(preds == y_test))
                else:
                    accuracy = model_fn(X_train, y_train, X_test, y_test)

                fold_scores.append(accuracy)
                self.cv_engine.evaluate_fold(fold_idx, {"accuracy": accuracy})
                self.tracker.log_metrics({"accuracy": accuracy, "fold": fold_idx}, step=fold_idx)

            # Final metrics
            mean_acc = float(np.mean(fold_scores))
            std_acc = float(np.std(fold_scores))
            self.tracker.log_metrics(
                {"primary": mean_acc, "accuracy_mean": mean_acc, "accuracy_std": std_acc},
                step=len(folds),
            )

            result = self.tracker.end_experiment()
            self._pipeline_runs.append({"config": config.to_dict(), "result_metrics": result.metrics})
            return result

        except Exception as e:
            return self.tracker.end_experiment(status=ExperimentStatus.FAILED, error=str(e))

    def optimize(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        n_trials: int = 10,
        model_fn: Any = None,
    ) -> dict:
        """Run hyperparameter optimization."""
        if self.optimizer is None:
            raise RuntimeError("No search space defined")

        for trial in range(n_trials):
            params = self.optimizer.suggest(trial)
            config = ExperimentConfig(name=f"{self.name}_trial_{trial}", parameters=params, seed=42 + trial)
            result = self.run_experiment(config, data, labels, model_fn)
            score = result.metrics.get("primary", 0.0)
            self.optimizer.report(params, score)

        return {
            "best_params": self.optimizer.best_params,
            "best_score": self.optimizer.best_score,
            "n_trials": n_trials,
            "importance": self.optimizer.get_importance(),
        }

    @property
    def n_runs(self) -> int:
        return len(self._pipeline_runs)
