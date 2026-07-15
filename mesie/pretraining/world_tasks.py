"""Self-supervised world tasks for spectral reasoning pretraining.

Each task is implemented as an auxiliary head that operates on top of the
MESIE backbone (spectral embeddings). Tasks produce labels from the spectral
data itself or from physics-based rules, enabling self-supervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Task result containers
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Result from a single world task evaluation."""

    task_name: str
    prediction: np.ndarray
    target: np.ndarray
    loss: float
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resonance Head
# ---------------------------------------------------------------------------


class ResonanceHead:
    """Predict whether a system is near resonance given past spectra.

    Operates as both a classifier (near/not-near resonance) and a regressor
    (distance to nearest resonance frequency). Labels are derived from
    physics-based rules: peak-to-mean ratio thresholds, mode shape analysis,
    and natural frequency proximity.

    Parameters
    ----------
    threshold : float
        Peak-to-mean ratio above which the system is considered near resonance.
    natural_frequencies : array-like or None
        Known natural frequencies of the system. If provided, proximity to
        these frequencies is used for regression targets.
    bandwidth : float
        Half-power bandwidth (Hz) for resonance region definition.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        natural_frequencies: Optional[np.ndarray] = None,
        bandwidth: float = 1.0,
    ):
        self.threshold = threshold
        self.natural_frequencies = (
            np.asarray(natural_frequencies) if natural_frequencies is not None else None
        )
        self.bandwidth = bandwidth

    def generate_labels(
        self, frequencies: np.ndarray, amplitudes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate self-supervised resonance labels from spectral data.

        Parameters
        ----------
        frequencies : ndarray, shape (n_freq,)
            Frequency axis.
        amplitudes : ndarray, shape (n_samples, n_freq) or (n_freq,)
            Amplitude spectra.

        Returns
        -------
        dict with keys:
            'classification' : ndarray of {0, 1}
            'regression' : ndarray of float (distance to resonance)
            'resonance_score' : ndarray of float (peak-to-mean ratio)
        """
        amplitudes = np.atleast_2d(amplitudes)
        n_samples = amplitudes.shape[0]

        scores = np.zeros(n_samples)
        classifications = np.zeros(n_samples, dtype=np.int64)
        regressions = np.zeros(n_samples)

        for i in range(n_samples):
            amp = amplitudes[i]
            mean_amp = np.mean(amp)
            if mean_amp > 0:
                peak_to_mean = np.max(amp) / mean_amp
            else:
                peak_to_mean = 0.0

            scores[i] = peak_to_mean
            classifications[i] = int(peak_to_mean >= self.threshold)

            # Regression: distance to nearest natural frequency
            if self.natural_frequencies is not None and len(self.natural_frequencies) > 0:
                peak_freq = frequencies[np.argmax(amp)]
                min_dist = np.min(np.abs(self.natural_frequencies - peak_freq))
                regressions[i] = min_dist / self.bandwidth
            else:
                # Use inverse of peak-to-mean as proxy
                regressions[i] = 1.0 / max(peak_to_mean, 1e-8)

        return {
            "classification": classifications,
            "regression": regressions,
            "resonance_score": scores,
        }

    def compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray, task: str = "classification"
    ) -> float:
        """Compute loss for resonance prediction.

        Parameters
        ----------
        predictions : ndarray
            Model predictions.
        targets : ndarray
            Ground-truth labels.
        task : str
            One of 'classification' or 'regression'.

        Returns
        -------
        float
            Loss value.
        """
        if task == "classification":
            # Binary cross-entropy
            preds = np.clip(predictions, 1e-7, 1 - 1e-7)
            return float(
                -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
            )
        else:
            # Mean squared error
            return float(np.mean((predictions - targets) ** 2))


# ---------------------------------------------------------------------------
# Coherence Head
# ---------------------------------------------------------------------------


class CoherenceHead:
    """Estimate coherence between spectral channels/components.

    Produces a coherence matrix (0-1) between all pairs of components in a
    multi-element record. Self-supervised targets are derived from cross-spectral
    density estimation.

    Parameters
    ----------
    n_segments : int
        Number of segments for Welch-based coherence estimation.
    overlap : float
        Fractional overlap between segments (0 to 1).
    """

    def __init__(self, n_segments: int = 8, overlap: float = 0.5):
        self.n_segments = n_segments
        self.overlap = overlap

    def generate_labels(
        self, components: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate coherence targets from multi-component spectra.

        Parameters
        ----------
        components : ndarray, shape (n_components, n_freq)
            Amplitude spectra of multiple components.

        Returns
        -------
        dict with keys:
            'coherence_matrix' : ndarray, shape (n_components, n_components)
            'mean_coherence' : float
        """
        n_comp = components.shape[0]
        coherence_matrix = np.zeros((n_comp, n_comp))

        for i in range(n_comp):
            for j in range(n_comp):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    coherence_matrix[i, j] = self._estimate_coherence(
                        components[i], components[j]
                    )

        mean_coherence = np.mean(
            coherence_matrix[np.triu_indices(n_comp, k=1)]
        ) if n_comp > 1 else 1.0

        return {
            "coherence_matrix": coherence_matrix,
            "mean_coherence": np.array([mean_coherence]),
        }

    def _estimate_coherence(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate magnitude-squared coherence between two spectra.

        Uses normalized cross-correlation as a proxy for coherence when
        only amplitude spectra are available.
        """
        x_norm = x - np.mean(x)
        y_norm = y - np.mean(y)

        x_std = np.std(x_norm)
        y_std = np.std(y_norm)

        if x_std < 1e-10 or y_std < 1e-10:
            return 0.0

        correlation = np.mean(x_norm * y_norm) / (x_std * y_std)
        # Map correlation [-1,1] to coherence [0,1]
        return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))

    def compute_loss(
        self, predicted_matrix: np.ndarray, target_matrix: np.ndarray
    ) -> float:
        """Compute coherence estimation loss (Frobenius norm).

        Parameters
        ----------
        predicted_matrix : ndarray, shape (n, n)
            Predicted coherence matrix.
        target_matrix : ndarray, shape (n, n)
            Target coherence matrix.

        Returns
        -------
        float
            Normalized Frobenius distance.
        """
        diff = predicted_matrix - target_matrix
        n = target_matrix.shape[0]
        return float(np.sqrt(np.sum(diff ** 2)) / max(n, 1))


# ---------------------------------------------------------------------------
# Harmonic Structure Head
# ---------------------------------------------------------------------------


class HarmonicStructureHead:
    """Classify or reconstruct harmonic families in spectral data.

    Identifies harmonic series (f0, 2*f0, 3*f0, ...) and trains the model to
    either classify which harmonic family is present or reconstruct the full
    harmonic pattern from partial observations.

    Parameters
    ----------
    max_harmonics : int
        Maximum number of harmonics to consider per fundamental.
    tolerance : float
        Frequency tolerance for harmonic matching (fraction of f0).
    min_amplitude_ratio : float
        Minimum amplitude ratio for a peak to be considered part of a harmonic.
    """

    def __init__(
        self,
        max_harmonics: int = 8,
        tolerance: float = 0.05,
        min_amplitude_ratio: float = 0.1,
    ):
        self.max_harmonics = max_harmonics
        self.tolerance = tolerance
        self.min_amplitude_ratio = min_amplitude_ratio

    def generate_labels(
        self, frequencies: np.ndarray, amplitudes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate harmonic structure labels from spectral data.

        Parameters
        ----------
        frequencies : ndarray, shape (n_freq,)
            Frequency axis.
        amplitudes : ndarray, shape (n_freq,)
            Amplitude spectrum.

        Returns
        -------
        dict with keys:
            'harmonic_mask' : ndarray, shape (n_freq,) binary mask of harmonic bins
            'fundamental_frequency' : float, detected fundamental
            'n_harmonics' : int, number of detected harmonics
            'harmonic_amplitudes' : ndarray, shape (max_harmonics,)
        """
        # Find peaks
        peak_indices = self._find_peaks(amplitudes)

        if len(peak_indices) == 0:
            return {
                "harmonic_mask": np.zeros_like(amplitudes),
                "fundamental_frequency": np.array([0.0]),
                "n_harmonics": np.array([0]),
                "harmonic_amplitudes": np.zeros(self.max_harmonics),
            }

        # Try each peak as potential fundamental
        best_f0 = 0.0
        best_count = 0
        best_mask = np.zeros_like(amplitudes)

        for idx in peak_indices:
            f0 = frequencies[idx]
            if f0 <= 0:
                continue
            mask, count = self._check_harmonics(frequencies, amplitudes, f0)
            if count > best_count:
                best_count = count
                best_f0 = f0
                best_mask = mask

        # Extract harmonic amplitudes
        harmonic_amps = np.zeros(self.max_harmonics)
        for h in range(self.max_harmonics):
            target_freq = best_f0 * (h + 1)
            if target_freq > 0 and target_freq <= frequencies[-1]:
                idx = np.argmin(np.abs(frequencies - target_freq))
                if np.abs(frequencies[idx] - target_freq) / best_f0 <= self.tolerance:
                    harmonic_amps[h] = amplitudes[idx]

        return {
            "harmonic_mask": best_mask,
            "fundamental_frequency": np.array([best_f0]),
            "n_harmonics": np.array([best_count]),
            "harmonic_amplitudes": harmonic_amps,
        }

    def _find_peaks(self, amplitudes: np.ndarray) -> np.ndarray:
        """Find local maxima in amplitude spectrum."""
        peaks = []
        threshold = np.max(amplitudes) * self.min_amplitude_ratio
        for i in range(1, len(amplitudes) - 1):
            if (
                amplitudes[i] > amplitudes[i - 1]
                and amplitudes[i] > amplitudes[i + 1]
                and amplitudes[i] >= threshold
            ):
                peaks.append(i)
        return np.array(peaks, dtype=np.int64)

    def _check_harmonics(
        self, frequencies: np.ndarray, amplitudes: np.ndarray, f0: float
    ) -> Tuple[np.ndarray, int]:
        """Check how many harmonics are present for a given fundamental."""
        mask = np.zeros_like(amplitudes)
        count = 0

        for h in range(1, self.max_harmonics + 1):
            target_freq = f0 * h
            if target_freq > frequencies[-1]:
                break
            idx = np.argmin(np.abs(frequencies - target_freq))
            if np.abs(frequencies[idx] - target_freq) / f0 <= self.tolerance:
                if amplitudes[idx] >= np.max(amplitudes) * self.min_amplitude_ratio:
                    mask[idx] = 1.0
                    count += 1

        return mask, count

    def compute_loss(
        self, predicted: np.ndarray, target: np.ndarray, task: str = "reconstruction"
    ) -> float:
        """Compute harmonic structure loss.

        Parameters
        ----------
        predicted : ndarray
            Predicted harmonic representation.
        target : ndarray
            Target harmonic representation.
        task : str
            'reconstruction' for MSE on harmonic amplitudes,
            'classification' for binary cross-entropy on harmonic mask.

        Returns
        -------
        float
            Loss value.
        """
        if task == "classification":
            preds = np.clip(predicted, 1e-7, 1 - 1e-7)
            return float(
                -np.mean(target * np.log(preds) + (1 - target) * np.log(1 - preds))
            )
        else:
            return float(np.mean((predicted - target) ** 2))


# ---------------------------------------------------------------------------
# Spectral Drift Head
# ---------------------------------------------------------------------------


class SpectralDriftHead:
    """Detect and quantify slow distributional shifts in spectral data.

    Computes a drift score between current embedding z_t and a canonical
    baseline z_ref. Uses multiple distance metrics for robustness.

    Parameters
    ----------
    baseline_window : int
        Number of initial samples to use as the canonical baseline.
    drift_metrics : list of str
        Distance metrics to use. Options: 'l2', 'cosine', 'kl', 'wasserstein'.
    ewma_alpha : float
        Exponential weighted moving average smoothing factor for drift tracking.
    """

    def __init__(
        self,
        baseline_window: int = 50,
        drift_metrics: Optional[List[str]] = None,
        ewma_alpha: float = 0.1,
    ):
        self.baseline_window = baseline_window
        self.drift_metrics = drift_metrics or ["l2", "cosine"]
        self.ewma_alpha = ewma_alpha
        self._baseline: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._ewma_drift: float = 0.0

    def fit_baseline(self, embeddings: np.ndarray) -> None:
        """Fit the canonical baseline from initial embeddings.

        Parameters
        ----------
        embeddings : ndarray, shape (n_samples, embed_dim)
            Initial embedding sequence to establish baseline.
        """
        embeddings = np.atleast_2d(embeddings)
        n = min(self.baseline_window, embeddings.shape[0])
        self._baseline = np.mean(embeddings[:n], axis=0)
        self._baseline_std = np.std(embeddings[:n], axis=0) + 1e-8
        self._ewma_drift = 0.0

    def generate_labels(
        self, embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate drift scores for a sequence of embeddings.

        Parameters
        ----------
        embeddings : ndarray, shape (n_samples, embed_dim)
            Embedding sequence to evaluate for drift.

        Returns
        -------
        dict with keys:
            'drift_scores' : ndarray, shape (n_samples,)
            'drift_detected' : ndarray, shape (n_samples,) binary
            'metric_decomposition' : dict of metric -> scores
        """
        embeddings = np.atleast_2d(embeddings)
        n_samples = embeddings.shape[0]

        if self._baseline is None:
            self.fit_baseline(embeddings)

        metric_scores: Dict[str, np.ndarray] = {}
        for metric in self.drift_metrics:
            metric_scores[metric] = np.zeros(n_samples)
            for i in range(n_samples):
                metric_scores[metric][i] = self._compute_distance(
                    embeddings[i], metric
                )

        # Aggregate drift score (mean across metrics)
        drift_scores = np.mean(
            np.stack(list(metric_scores.values()), axis=0), axis=0
        )

        # Adaptive threshold via EWMA
        drift_detected = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            self._ewma_drift = (
                self.ewma_alpha * drift_scores[i]
                + (1 - self.ewma_alpha) * self._ewma_drift
            )
            # Drift detected if score exceeds 2x the running average
            if drift_scores[i] > 2.0 * max(self._ewma_drift, 0.1):
                drift_detected[i] = 1

        return {
            "drift_scores": drift_scores,
            "drift_detected": drift_detected,
            "metric_decomposition": metric_scores,
        }

    def _compute_distance(self, embedding: np.ndarray, metric: str) -> float:
        """Compute distance between embedding and baseline."""
        if self._baseline is None:
            return 0.0

        if metric == "l2":
            return float(np.sqrt(np.sum((embedding - self._baseline) ** 2)))
        elif metric == "cosine":
            dot = np.dot(embedding, self._baseline)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(self._baseline)
            if norm_a < 1e-10 or norm_b < 1e-10:
                return 1.0
            return float(1.0 - dot / (norm_a * norm_b))
        elif metric == "kl":
            # Symmetric KL divergence on normalized distributions
            p = np.abs(embedding) + 1e-10
            q = np.abs(self._baseline) + 1e-10
            p = p / np.sum(p)
            q = q / np.sum(q)
            return float(
                0.5 * np.sum(p * np.log(p / q)) + 0.5 * np.sum(q * np.log(q / p))
            )
        elif metric == "wasserstein":
            # 1D Wasserstein (earth mover) approximation via sorted values
            sorted_a = np.sort(embedding)
            sorted_b = np.sort(self._baseline)
            return float(np.mean(np.abs(sorted_a - sorted_b)))
        else:
            return float(np.sqrt(np.sum((embedding - self._baseline) ** 2)))

    def compute_loss(
        self, predicted_scores: np.ndarray, target_scores: np.ndarray
    ) -> float:
        """Compute drift prediction loss (MSE).

        Parameters
        ----------
        predicted_scores : ndarray
            Predicted drift scores.
        target_scores : ndarray
            Target drift scores.

        Returns
        -------
        float
            Mean squared error.
        """
        return float(np.mean((predicted_scores - target_scores) ** 2))


# ---------------------------------------------------------------------------
# Temporal Lineage Head
# ---------------------------------------------------------------------------


class TemporalLineageHead:
    """Infer temporal lineage from current spectra.

    Given current embedding z_t, predict a compressed representation of the
    past window z_{t-k:t-1}. This forces the model to encode temporal history
    in its representations.

    Parameters
    ----------
    window_size : int
        Number of past time steps to predict.
    compression_dim : int
        Dimensionality of the compressed past representation.
    compression_method : str
        Method for compressing past window: 'pca', 'mean', 'last', 'stats'.
    """

    def __init__(
        self,
        window_size: int = 10,
        compression_dim: int = 16,
        compression_method: str = "stats",
    ):
        self.window_size = window_size
        self.compression_dim = compression_dim
        self.compression_method = compression_method
        self._pca_components: Optional[np.ndarray] = None

    def generate_labels(
        self, embedding_sequence: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate temporal lineage targets from an embedding sequence.

        Parameters
        ----------
        embedding_sequence : ndarray, shape (n_timesteps, embed_dim)
            Temporal sequence of embeddings.

        Returns
        -------
        dict with keys:
            'targets' : ndarray, shape (n_valid, compression_dim)
            'inputs' : ndarray, shape (n_valid, embed_dim)
            'valid_indices' : ndarray, indices where targets are available
        """
        embedding_sequence = np.atleast_2d(embedding_sequence)
        n_timesteps, embed_dim = embedding_sequence.shape

        if n_timesteps <= self.window_size:
            return {
                "targets": np.zeros((0, self.compression_dim)),
                "inputs": np.zeros((0, embed_dim)),
                "valid_indices": np.array([], dtype=np.int64),
            }

        valid_indices = np.arange(self.window_size, n_timesteps)
        n_valid = len(valid_indices)

        targets = np.zeros((n_valid, self.compression_dim))
        inputs = np.zeros((n_valid, embed_dim))

        for i, t in enumerate(valid_indices):
            past_window = embedding_sequence[t - self.window_size: t]
            targets[i] = self._compress_window(past_window)
            inputs[i] = embedding_sequence[t]

        return {
            "targets": targets,
            "inputs": inputs,
            "valid_indices": valid_indices,
        }

    def _compress_window(self, window: np.ndarray) -> np.ndarray:
        """Compress a past window into a fixed-size representation."""
        if self.compression_method == "mean":
            compressed = np.mean(window, axis=0)
        elif self.compression_method == "last":
            compressed = window[-1]
        elif self.compression_method == "stats":
            # Concatenate mean, std, min, max trends
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            trend = window[-1] - window[0]  # temporal gradient
            compressed = np.concatenate([mean, std, trend])
        elif self.compression_method == "pca":
            # Simple PCA via SVD
            centered = window - np.mean(window, axis=0)
            if centered.shape[0] > 1:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                n_comp = min(self.compression_dim, vt.shape[0])
                compressed = vt[:n_comp].flatten()
            else:
                compressed = centered.flatten()
        else:
            compressed = np.mean(window, axis=0)

        # Pad or truncate to compression_dim
        if len(compressed) >= self.compression_dim:
            return compressed[: self.compression_dim]
        else:
            padded = np.zeros(self.compression_dim)
            padded[: len(compressed)] = compressed
            return padded

    def compute_loss(
        self, predicted: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute lineage prediction loss.

        Parameters
        ----------
        predicted : ndarray, shape (n, compression_dim)
            Predicted compressed past.
        targets : ndarray, shape (n, compression_dim)
            Target compressed past.

        Returns
        -------
        float
            Cosine similarity loss (1 - mean cosine similarity).
        """
        predicted = np.atleast_2d(predicted)
        targets = np.atleast_2d(targets)

        # Cosine similarity loss
        dot_products = np.sum(predicted * targets, axis=1)
        pred_norms = np.linalg.norm(predicted, axis=1) + 1e-8
        target_norms = np.linalg.norm(targets, axis=1) + 1e-8
        cosine_sim = dot_products / (pred_norms * target_norms)

        return float(1.0 - np.mean(cosine_sim))


# ---------------------------------------------------------------------------
# World Task Suite (Orchestrator)
# ---------------------------------------------------------------------------


class WorldTaskSuite:
    """Orchestrates all self-supervised world tasks for pretraining.

    Combines resonance, coherence, harmonic structure, drift, and lineage
    tasks into a unified pretraining objective with configurable loss weights.

    Parameters
    ----------
    task_weights : dict or None
        Mapping of task name to loss weight. Defaults to uniform weighting.
    resonance_config : dict or None
        Configuration for ResonanceHead.
    coherence_config : dict or None
        Configuration for CoherenceHead.
    harmonic_config : dict or None
        Configuration for HarmonicStructureHead.
    drift_config : dict or None
        Configuration for SpectralDriftHead.
    lineage_config : dict or None
        Configuration for TemporalLineageHead.
    """

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        resonance_config: Optional[Dict] = None,
        coherence_config: Optional[Dict] = None,
        harmonic_config: Optional[Dict] = None,
        drift_config: Optional[Dict] = None,
        lineage_config: Optional[Dict] = None,
    ):
        self.task_weights = task_weights or {
            "resonance": 1.0,
            "coherence": 1.0,
            "harmonic": 1.0,
            "drift": 1.0,
            "lineage": 1.0,
        }

        self.resonance_head = ResonanceHead(**(resonance_config or {}))
        self.coherence_head = CoherenceHead(**(coherence_config or {}))
        self.harmonic_head = HarmonicStructureHead(**(harmonic_config or {}))
        self.drift_head = SpectralDriftHead(**(drift_config or {}))
        self.lineage_head = TemporalLineageHead(**(lineage_config or {}))

    def evaluate_all(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        components: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Run all world tasks and generate labels.

        Parameters
        ----------
        frequencies : ndarray, shape (n_freq,)
            Frequency axis.
        amplitudes : ndarray, shape (n_samples, n_freq) or (n_freq,)
            Amplitude spectra.
        components : ndarray or None, shape (n_components, n_freq)
            Multi-component spectra for coherence task.
        embeddings : ndarray or None, shape (n_timesteps, embed_dim)
            Embedding sequence for drift and lineage tasks.

        Returns
        -------
        dict mapping task name -> label dict
        """
        results = {}

        # Resonance task
        results["resonance"] = self.resonance_head.generate_labels(
            frequencies, amplitudes
        )

        # Harmonic structure task
        amp_1d = amplitudes[0] if amplitudes.ndim > 1 else amplitudes
        results["harmonic"] = self.harmonic_head.generate_labels(
            frequencies, amp_1d
        )

        # Coherence task
        if components is not None:
            results["coherence"] = self.coherence_head.generate_labels(components)

        # Drift task
        if embeddings is not None:
            results["drift"] = self.drift_head.generate_labels(embeddings)

        # Lineage task
        if embeddings is not None:
            results["lineage"] = self.lineage_head.generate_labels(embeddings)

        return results

    def compute_total_loss(self, task_losses: Dict[str, float]) -> float:
        """Compute weighted total loss across all tasks.

        Parameters
        ----------
        task_losses : dict
            Mapping of task name to individual loss value.

        Returns
        -------
        float
            Weighted sum of task losses.
        """
        total = 0.0
        for task_name, loss in task_losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total += weight * loss
        return total
