"""Data sampling strategies for multi-modal pretraining.

Provides balanced, curriculum-based, and difficulty-aware sampling
across multiple spectral modalities.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ModalityBalancedSampler:
    """Sampler that balances samples across modalities.

    Ensures each modality is represented according to configured
    weights, supporting both proportional and temperature-based
    sampling strategies.

    Attributes:
        modality_weights: Per-modality sampling weights.
        temperature: Temperature for weight sharpening.
        strategy: Sampling strategy.
    """

    def __init__(
        self,
        modality_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        strategy: str = "proportional",
        seed: int = 42,
    ):
        """Initialize balanced sampler.

        Args:
            modality_weights: Weights per modality.
            temperature: Temperature (>1 = more uniform, <1 = more peaked).
            strategy: Sampling strategy ('proportional', 'uniform', 'sqrt').
            seed: Random seed.
        """
        self.modality_weights = modality_weights or {
            "seismic": 1.0,
            "vibration": 1.0,
            "eeg": 1.0,
            "ecg": 1.0,
            "audio": 1.0,
            "rf": 1.0,
            "synthetic": 1.0,
        }
        self.temperature = temperature
        self.strategy = strategy
        self._rng = np.random.RandomState(seed)

        self._compute_probabilities()

    def _compute_probabilities(self) -> None:
        """Compute sampling probabilities from weights."""
        weights = np.array(list(self.modality_weights.values()))

        if self.strategy == "uniform":
            probs = np.ones_like(weights) / len(weights)
        elif self.strategy == "sqrt":
            probs = np.sqrt(weights)
        elif self.strategy == "proportional":
            probs = weights
        else:
            probs = weights

        # Apply temperature
        if self.temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / self.temperature
            probs = np.exp(log_probs)

        # Normalize
        self.probabilities = probs / (np.sum(probs) + 1e-10)
        self.modality_names = list(self.modality_weights.keys())

    def sample_modality(self, n: int = 1) -> List[str]:
        """Sample modality names.

        Args:
            n: Number of samples.

        Returns:
            List of sampled modality names.
        """
        indices = self._rng.choice(
            len(self.modality_names), size=n, p=self.probabilities
        )
        return [self.modality_names[i] for i in indices]

    def sample_batch_composition(self, batch_size: int) -> Dict[str, int]:
        """Determine how many samples from each modality in a batch.

        Args:
            batch_size: Total batch size.

        Returns:
            Dictionary mapping modality -> count.
        """
        modalities = self.sample_modality(batch_size)
        composition: Dict[str, int] = {}
        for m in modalities:
            composition[m] = composition.get(m, 0) + 1
        return composition

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update modality weights.

        Args:
            new_weights: New weight values.
        """
        self.modality_weights.update(new_weights)
        self._compute_probabilities()

    def get_statistics(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "modality_weights": self.modality_weights,
            "probabilities": dict(zip(self.modality_names, self.probabilities.tolist())),
            "temperature": self.temperature,
            "strategy": self.strategy,
        }


class CurriculumSampler:
    """Curriculum learning sampler for progressive training.

    Gradually increases training difficulty by:
    1. Starting with simpler/synthetic data
    2. Progressively adding more complex modalities
    3. Increasing sequence length over time
    4. Adjusting noise levels

    Attributes:
        phases: List of curriculum phases.
        current_phase: Current phase index.
        phase_transitions: When to transition between phases.
    """

    def __init__(
        self,
        phases: Optional[List[Dict[str, Any]]] = None,
        total_steps: int = 100000,
        warmup_fraction: float = 0.1,
        seed: int = 42,
    ):
        """Initialize curriculum sampler.

        Args:
            phases: Curriculum phase definitions.
            total_steps: Total training steps.
            warmup_fraction: Fraction of steps for warmup.
            seed: Random seed.
        """
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction
        self._rng = np.random.RandomState(seed)
        self.current_step = 0
        self.current_phase = 0

        # Default curriculum phases
        if phases is None:
            self.phases = [
                {
                    "name": "warmup",
                    "start_fraction": 0.0,
                    "end_fraction": 0.1,
                    "modalities": ["synthetic"],
                    "max_seq_len": 512,
                    "difficulty": "easy",
                    "noise_level": 0.0,
                    "mask_ratio": 0.15,
                },
                {
                    "name": "foundation",
                    "start_fraction": 0.1,
                    "end_fraction": 0.3,
                    "modalities": ["synthetic", "seismic", "vibration"],
                    "max_seq_len": 1024,
                    "difficulty": "medium",
                    "noise_level": 0.1,
                    "mask_ratio": 0.25,
                },
                {
                    "name": "expansion",
                    "start_fraction": 0.3,
                    "end_fraction": 0.6,
                    "modalities": ["synthetic", "seismic", "vibration", "audio", "rf"],
                    "max_seq_len": 2048,
                    "difficulty": "medium-hard",
                    "noise_level": 0.2,
                    "mask_ratio": 0.3,
                },
                {
                    "name": "full",
                    "start_fraction": 0.6,
                    "end_fraction": 0.85,
                    "modalities": "all",
                    "max_seq_len": 4096,
                    "difficulty": "hard",
                    "noise_level": 0.3,
                    "mask_ratio": 0.35,
                },
                {
                    "name": "mastery",
                    "start_fraction": 0.85,
                    "end_fraction": 1.0,
                    "modalities": "all",
                    "max_seq_len": 8192,
                    "difficulty": "expert",
                    "noise_level": 0.4,
                    "mask_ratio": 0.4,
                },
            ]
        else:
            self.phases = phases

    def step(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Advance curriculum and return current phase config.

        Args:
            step: Current training step (auto-increments if None).

        Returns:
            Current phase configuration.
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Determine current phase
        progress = self.current_step / max(self.total_steps, 1)

        for i, phase in enumerate(self.phases):
            if phase["start_fraction"] <= progress < phase["end_fraction"]:
                self.current_phase = i
                break

        phase = self.phases[self.current_phase]

        # Compute interpolated difficulty within phase
        phase_progress = (progress - phase["start_fraction"]) / \
            (phase["end_fraction"] - phase["start_fraction"] + 1e-10)
        phase_progress = min(1.0, max(0.0, phase_progress))

        return {
            "phase_name": phase["name"],
            "phase_idx": self.current_phase,
            "phase_progress": phase_progress,
            "modalities": phase["modalities"],
            "max_seq_len": phase["max_seq_len"],
            "difficulty": phase["difficulty"],
            "noise_level": phase["noise_level"],
            "mask_ratio": phase["mask_ratio"],
            "global_progress": progress,
        }

    def get_modality_weights(self) -> Dict[str, float]:
        """Get current modality weights based on curriculum phase.

        Returns:
            Modality weight dictionary.
        """
        phase = self.phases[self.current_phase]
        modalities = phase["modalities"]

        all_modalities = ["seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"]

        if modalities == "all":
            return {m: 1.0 for m in all_modalities}

        weights = {}
        for m in all_modalities:
            if m in modalities:
                weights[m] = 1.0
            else:
                weights[m] = 0.0

        return weights


class DifficultyAwareSampler:
    """Sampler that adjusts based on sample difficulty.

    Tracks per-sample difficulty (via loss) and adjusts sampling
    to focus on samples at the appropriate difficulty level for
    the current training stage.

    Attributes:
        difficulty_scores: Per-sample difficulty estimates.
        target_difficulty: Target difficulty percentile.
        smoothing: EMA smoothing for difficulty updates.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        target_difficulty: float = 0.5,
        smoothing: float = 0.9,
        min_difficulty: float = 0.1,
        max_difficulty: float = 0.9,
        seed: int = 42,
    ):
        """Initialize difficulty-aware sampler.

        Args:
            num_samples: Number of samples to track.
            target_difficulty: Target difficulty (0=easy, 1=hard).
            smoothing: EMA smoothing factor.
            min_difficulty: Minimum difficulty threshold.
            max_difficulty: Maximum difficulty threshold.
            seed: Random seed.
        """
        self.num_samples = num_samples
        self.target_difficulty = target_difficulty
        self.smoothing = smoothing
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self._rng = np.random.RandomState(seed)

        # Initialize difficulty scores uniformly
        self.difficulty_scores = np.ones(num_samples) * 0.5
        self.loss_history = np.zeros(num_samples)
        self.sample_counts = np.zeros(num_samples, dtype=np.int64)

    def update_difficulty(self, sample_indices: np.ndarray, losses: np.ndarray) -> None:
        """Update difficulty scores based on observed losses.

        Args:
            sample_indices: Indices of samples in the batch.
            losses: Per-sample losses.
        """
        for idx, loss in zip(sample_indices.flatten(), losses.flatten()):
            idx = int(idx) % self.num_samples
            # EMA update
            self.difficulty_scores[idx] = (
                self.smoothing * self.difficulty_scores[idx] +
                (1 - self.smoothing) * min(loss, 10.0) / 10.0
            )
            self.loss_history[idx] = loss
            self.sample_counts[idx] += 1

    def sample(self, n: int) -> np.ndarray:
        """Sample indices based on difficulty.

        Samples are weighted to focus on the target difficulty range.

        Args:
            n: Number of indices to sample.

        Returns:
            Array of sample indices.
        """
        # Compute sampling weights based on distance to target difficulty
        distance_to_target = np.abs(self.difficulty_scores - self.target_difficulty)

        # Gaussian weighting centered on target difficulty
        sigma = 0.2
        weights = np.exp(-0.5 * (distance_to_target / sigma) ** 2)

        # Also boost rarely-seen samples
        frequency_bonus = 1.0 / (self.sample_counts + 1)
        weights = weights + 0.1 * frequency_bonus

        # Filter by difficulty range
        mask = (self.difficulty_scores >= self.min_difficulty) & \
               (self.difficulty_scores <= self.max_difficulty)
        weights = weights * mask

        # Normalize
        weights = weights / (np.sum(weights) + 1e-10)

        indices = self._rng.choice(self.num_samples, size=n, p=weights)
        return indices

    def adjust_target(self, new_target: float) -> None:
        """Adjust target difficulty.

        Args:
            new_target: New target difficulty (0-1).
        """
        self.target_difficulty = np.clip(new_target, 0.0, 1.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "target_difficulty": self.target_difficulty,
            "mean_difficulty": float(np.mean(self.difficulty_scores)),
            "std_difficulty": float(np.std(self.difficulty_scores)),
            "difficulty_histogram": np.histogram(
                self.difficulty_scores, bins=10, range=(0, 1)
            )[0].tolist(),
            "total_samples_seen": int(np.sum(self.sample_counts)),
            "never_seen": int(np.sum(self.sample_counts == 0)),
        }
