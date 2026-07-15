"""Temporal Dynamics Processing Pipeline.

Provides temporal analysis capabilities for spectral data streams,
including time-frequency decomposition, temporal attention,
sequence modeling, change detection, and predictive analysis.

Key Components:
    - TemporalSpectralBuffer: Sliding window for spectral sequences
    - TimeFrequencyDecomposer: STFT-like decomposition
    - TemporalAttentionLayer: Attention over time steps
    - ChangePointDetector: Online change detection
    - SpectralPredictor: Next-step spectral prediction
    - TemporalDynamicsPipeline: Full temporal processing orchestrator
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TemporalConfig:
    """Configuration for the temporal dynamics pipeline.

    Args:
        window_size: Number of time steps in the analysis window.
        hop_size: Number of time steps between windows.
        n_frequency_bins: Number of frequency bins for decomposition.
        d_temporal: Temporal embedding dimension.
        n_attention_heads: Number of temporal attention heads.
        prediction_horizon: Number of future steps to predict.
        change_sensitivity: Sensitivity for change point detection.
        memory_length: Maximum memory length in time steps.
        decay_rate: Temporal decay rate for past observations.
    """

    window_size: int = 64
    hop_size: int = 16
    n_frequency_bins: int = 128
    d_temporal: int = 64
    n_attention_heads: int = 4
    prediction_horizon: int = 10
    change_sensitivity: float = 2.0
    memory_length: int = 500
    decay_rate: float = 0.01


@dataclass
class TemporalAnalysisResult:
    """Result from temporal dynamics analysis.

    Args:
        time_frequency_map: Time-frequency representation.
        temporal_attention: Attention weights over time.
        change_points: Detected change point indices.
        predictions: Predicted future spectral states.
        temporal_features: Extracted temporal features.
        stability_score: Overall temporal stability (0-1).
        metadata: Additional metadata.
    """

    time_frequency_map: np.ndarray
    temporal_attention: np.ndarray
    change_points: List[int]
    predictions: Optional[np.ndarray] = None
    temporal_features: Dict[str, float] = field(default_factory=dict)
    stability_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Temporal Buffer
# =============================================================================


class TemporalSpectralBuffer:
    """Sliding window buffer for temporal spectral sequences.

    Maintains a rolling window of spectral observations with
    associated timestamps and metadata. Supports efficient
    window extraction and temporal queries.

    Args:
        max_length: Maximum number of observations to store.
        d_spectral: Dimension of each spectral observation.
    """

    def __init__(self, max_length: int = 500, d_spectral: int = 128) -> None:
        self.max_length = max_length
        self.d_spectral = d_spectral
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_length)
        self._timestamps: Deque[float] = deque(maxlen=max_length)
        self._metadata: Deque[Dict[str, Any]] = deque(maxlen=max_length)
        self._total_observations: int = 0

    def push(
        self,
        observation: np.ndarray,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Push a new observation into the buffer.

        Args:
            observation: Spectral data.
            timestamp: Time of observation.
            metadata: Associated metadata.
        """
        obs = np.atleast_1d(observation).flatten()
        # Resize to standard dimension
        if len(obs) != self.d_spectral:
            obs = np.interp(
                np.linspace(0, 1, self.d_spectral),
                np.linspace(0, 1, len(obs)),
                obs,
            )

        self._buffer.append(obs)
        self._timestamps.append(timestamp or time.time())
        self._metadata.append(metadata or {})
        self._total_observations += 1

    def get_window(self, window_size: int, offset: int = 0) -> np.ndarray:
        """Get a window of recent observations.

        Args:
            window_size: Number of observations in the window.
            offset: Offset from the most recent observation.

        Returns:
            Array of shape (actual_size, d_spectral).
        """
        n = len(self._buffer)
        start = max(0, n - window_size - offset)
        end = n - offset
        if start >= end:
            return np.zeros((0, self.d_spectral))

        return np.array(list(self._buffer)[start:end])

    def get_recent(self, n: int = 1) -> np.ndarray:
        """Get the n most recent observations.

        Args:
            n: Number of recent observations.

        Returns:
            Array of shape (n, d_spectral) or fewer if buffer is smaller.
        """
        actual_n = min(n, len(self._buffer))
        if actual_n == 0:
            return np.zeros((0, self.d_spectral))
        return np.array(list(self._buffer)[-actual_n:])

    def get_temporal_diff(self, lag: int = 1) -> Optional[np.ndarray]:
        """Compute temporal difference between consecutive observations.

        Args:
            lag: Time lag for difference computation.

        Returns:
            Array of temporal differences or None if insufficient data.
        """
        if len(self._buffer) <= lag:
            return None
        data = np.array(list(self._buffer))
        return data[lag:] - data[:-lag]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute buffer statistics.

        Returns:
            Dictionary of temporal statistics.
        """
        if not self._buffer:
            return {"size": 0, "total_observations": 0}

        data = np.array(list(self._buffer))
        timestamps = np.array(list(self._timestamps))

        return {
            "size": len(self._buffer),
            "total_observations": self._total_observations,
            "mean_spectrum": float(np.mean(data)),
            "std_spectrum": float(np.std(data)),
            "max_value": float(np.max(data)),
            "min_value": float(np.min(data)),
            "time_span": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
            "mean_interval": float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 0.0,
        }

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether the buffer is at capacity."""
        return len(self._buffer) >= self.max_length

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._timestamps.clear()
        self._metadata.clear()


# =============================================================================
# Time-Frequency Decomposition
# =============================================================================


class TimeFrequencyDecomposer:
    """Decompose spectral sequences into time-frequency representations.

    Implements a spectral-domain analogue of the Short-Time Fourier Transform,
    providing a 2D representation of how spectral content evolves over time.

    Args:
        window_size: Analysis window size.
        hop_size: Hop size between windows.
        n_frequency_bins: Number of output frequency bins.
        window_type: Window function type ('hann', 'hamming', 'rectangular').
    """

    def __init__(
        self,
        window_size: int = 64,
        hop_size: int = 16,
        n_frequency_bins: int = 128,
        window_type: str = "hann",
    ) -> None:
        self.window_size = window_size
        self.hop_size = hop_size
        self.n_frequency_bins = n_frequency_bins
        self.window_type = window_type

        # Pre-compute window function
        self._window = self._create_window(window_size)

    def _create_window(self, size: int) -> np.ndarray:
        """Create the analysis window function."""
        if self.window_type == "hann":
            return 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
        elif self.window_type == "hamming":
            return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        else:
            return np.ones(size)

    def decompose(self, temporal_sequence: np.ndarray) -> np.ndarray:
        """Decompose a temporal sequence into time-frequency representation.

        Args:
            temporal_sequence: Array of shape (n_time_steps, n_spectral_bins)
                              or (n_time_steps,) for single-channel.

        Returns:
            Time-frequency map of shape (n_frames, n_frequency_bins).
        """
        if temporal_sequence.ndim == 1:
            temporal_sequence = temporal_sequence[:, np.newaxis]

        n_time_steps, n_channels = temporal_sequence.shape
        n_frames = max(1, (n_time_steps - self.window_size) // self.hop_size + 1)

        tf_map = np.zeros((n_frames, self.n_frequency_bins))

        for frame_idx in range(n_frames):
            start = frame_idx * self.hop_size
            end = start + self.window_size

            if end > n_time_steps:
                break

            # Extract frame
            frame = temporal_sequence[start:end]

            # Apply window (broadcast across channels)
            windowed = frame * self._window[:, np.newaxis]

            # Compute spectral content per frame
            for ch in range(n_channels):
                # FFT-like spectral analysis
                spectrum = np.abs(np.fft.rfft(windowed[:, ch], n=self.n_frequency_bins * 2))
                tf_map[frame_idx] += spectrum[:self.n_frequency_bins]

            tf_map[frame_idx] /= max(1, n_channels)

        return tf_map

    def compute_spectral_flux(self, tf_map: np.ndarray) -> np.ndarray:
        """Compute spectral flux (rate of change) from time-frequency map.

        Args:
            tf_map: Time-frequency representation.

        Returns:
            Spectral flux over time.
        """
        if tf_map.shape[0] < 2:
            return np.zeros(1)

        # Half-wave rectified spectral difference
        diff = np.diff(tf_map, axis=0)
        flux = np.sum(np.maximum(0, diff), axis=1)
        return flux

    def compute_spectral_centroid_trajectory(self, tf_map: np.ndarray) -> np.ndarray:
        """Compute the spectral centroid over time.

        Args:
            tf_map: Time-frequency representation.

        Returns:
            Spectral centroid at each time frame.
        """
        freqs = np.arange(tf_map.shape[1])
        centroids = np.zeros(tf_map.shape[0])

        for t in range(tf_map.shape[0]):
            total = np.sum(tf_map[t]) + 1e-12
            centroids[t] = np.sum(freqs * tf_map[t]) / total

        return centroids


# =============================================================================
# Temporal Attention
# =============================================================================


class TemporalAttentionLayer:
    """Attention mechanism operating over the temporal dimension.

    Computes attention weights across time steps to identify
    the most relevant past observations for current analysis.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        causal: Whether to use causal (unidirectional) attention.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        causal: bool = True,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        scale = 0.02
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal attention over a sequence.

        Args:
            sequence: Input of shape (seq_len, d_model).

        Returns:
            Tuple of (output, attention_weights).
        """
        seq_len = sequence.shape[0]

        # Ensure d_model matches
        if sequence.shape[1] != self.d_model:
            # Project to d_model
            proj = np.random.randn(sequence.shape[1], self.d_model) * 0.02
            sequence = sequence @ proj

        Q = sequence @ self.W_q
        K = sequence @ self.W_k
        V = sequence @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        scores = np.einsum("qhd,khd->hqk", Q, K) / np.sqrt(self.head_dim)

        # Causal mask
        if self.causal:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores += mask[np.newaxis, :, :]

        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-10)

        # Apply attention
        output = np.einsum("hqk,khd->qhd", attn_weights, V)
        output = output.reshape(seq_len, self.d_model)
        output = output @ self.W_o

        avg_attn = np.mean(attn_weights, axis=0)
        return output, avg_attn


# =============================================================================
# Change Point Detection
# =============================================================================


class ChangePointDetector:
    """Online change point detection for spectral streams.

    Detects significant changes in spectral behavior over time using
    CUSUM (Cumulative Sum) and energy-based methods.

    Args:
        sensitivity: Detection sensitivity (lower = more sensitive).
        min_segment_length: Minimum samples between change points.
        method: Detection method ('cusum', 'energy', 'combined').
    """

    def __init__(
        self,
        sensitivity: float = 2.0,
        min_segment_length: int = 10,
        method: str = "combined",
    ) -> None:
        self.sensitivity = sensitivity
        self.min_segment_length = min_segment_length
        self.method = method
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._observations: List[float] = []
        self._change_points: List[int] = []
        self._last_change: int = 0

    def update(self, observation: np.ndarray) -> bool:
        """Process a new observation and check for change points.

        Args:
            observation: New spectral observation.

        Returns:
            True if a change point was detected.
        """
        # Use energy as scalar summary
        energy = float(np.sum(observation ** 2))
        self._observations.append(energy)
        current_idx = len(self._observations) - 1

        # Build baseline from initial observations
        if len(self._observations) < 20:
            self._baseline_mean = float(np.mean(self._observations))
            self._baseline_std = float(np.std(self._observations)) + 1e-12
            return False

        if self._baseline_mean is None:
            self._baseline_mean = float(np.mean(self._observations[:20]))
            self._baseline_std = float(np.std(self._observations[:20])) + 1e-12

        # Check minimum segment length
        if current_idx - self._last_change < self.min_segment_length:
            return False

        is_change = False

        if self.method in ("cusum", "combined"):
            is_change = is_change or self._cusum_detect(energy)

        if self.method in ("energy", "combined"):
            is_change = is_change or self._energy_detect(energy)

        if is_change:
            self._change_points.append(current_idx)
            self._last_change = current_idx
            # Update baseline
            recent = self._observations[-20:]
            self._baseline_mean = float(np.mean(recent))
            self._baseline_std = float(np.std(recent)) + 1e-12
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

        return is_change

    def _cusum_detect(self, value: float) -> bool:
        """CUSUM change detection."""
        normalized = (value - self._baseline_mean) / self._baseline_std

        # Update CUSUM statistics
        self._cusum_pos = max(0, self._cusum_pos + normalized - 0.5)
        self._cusum_neg = max(0, self._cusum_neg - normalized - 0.5)

        threshold = self.sensitivity * 3.0
        return self._cusum_pos > threshold or self._cusum_neg > threshold

    def _energy_detect(self, energy: float) -> bool:
        """Energy-based change detection."""
        z_score = abs(energy - self._baseline_mean) / self._baseline_std
        return z_score > self.sensitivity * 2.0

    @property
    def change_points(self) -> List[int]:
        """List of detected change point indices."""
        return self._change_points.copy()

    @property
    def n_changes(self) -> int:
        """Number of detected change points."""
        return len(self._change_points)

    def reset(self) -> None:
        """Reset the detector state."""
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._baseline_mean = None
        self._baseline_std = None
        self._observations.clear()
        self._change_points.clear()
        self._last_change = 0


# =============================================================================
# Spectral Predictor
# =============================================================================


class SpectralPredictor:
    """Predict future spectral states from past observations.

    Uses a simple autoregressive approach with exponential smoothing
    and trend extrapolation for spectral time series prediction.

    Args:
        d_spectral: Spectral dimension.
        horizon: Number of future steps to predict.
        alpha: Exponential smoothing parameter.
        use_trend: Whether to extrapolate trends.
    """

    def __init__(
        self,
        d_spectral: int = 128,
        horizon: int = 10,
        alpha: float = 0.3,
        use_trend: bool = True,
    ) -> None:
        self.d_spectral = d_spectral
        self.horizon = horizon
        self.alpha = alpha
        self.use_trend = use_trend
        self._level: Optional[np.ndarray] = None
        self._trend: Optional[np.ndarray] = None
        self._prediction_count: int = 0

    def fit(self, history: np.ndarray) -> None:
        """Fit the predictor to historical data.

        Args:
            history: Array of shape (n_time_steps, d_spectral).
        """
        if len(history) < 2:
            self._level = history[-1] if len(history) > 0 else np.zeros(self.d_spectral)
            self._trend = np.zeros(self.d_spectral)
            return

        # Initialize level and trend using Holt's method
        self._level = history[-1].copy()
        self._trend = (history[-1] - history[0]) / max(1, len(history) - 1)

        # Apply exponential smoothing
        beta = self.alpha * 0.5  # Trend smoothing

        level = history[0].copy()
        trend = history[1] - history[0] if len(history) > 1 else np.zeros(self.d_spectral)

        for t in range(1, len(history)):
            new_level = self.alpha * history[t] + (1 - self.alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            trend = new_trend

        self._level = level
        self._trend = trend if self.use_trend else np.zeros(self.d_spectral)

    def predict(self, n_steps: Optional[int] = None) -> np.ndarray:
        """Predict future spectral states.

        Args:
            n_steps: Number of steps to predict (defaults to horizon).

        Returns:
            Array of shape (n_steps, d_spectral) with predictions.
        """
        self._prediction_count += 1
        steps = n_steps or self.horizon

        if self._level is None:
            return np.zeros((steps, self.d_spectral))

        predictions = np.zeros((steps, self.d_spectral))
        for t in range(steps):
            predictions[t] = self._level + (t + 1) * self._trend

        return predictions

    def update(self, observation: np.ndarray) -> None:
        """Update the predictor with a new observation.

        Args:
            observation: New spectral observation.
        """
        obs = np.atleast_1d(observation).flatten()
        if len(obs) != self.d_spectral:
            obs = np.interp(
                np.linspace(0, 1, self.d_spectral),
                np.linspace(0, 1, len(obs)),
                obs,
            )

        if self._level is None:
            self._level = obs.copy()
            self._trend = np.zeros(self.d_spectral)
            return

        # Exponential smoothing update
        beta = self.alpha * 0.5
        new_level = self.alpha * obs + (1 - self.alpha) * (self._level + self._trend)
        new_trend = beta * (new_level - self._level) + (1 - beta) * self._trend
        self._level = new_level
        self._trend = new_trend if self.use_trend else np.zeros(self.d_spectral)

    @property
    def prediction_count(self) -> int:
        """Number of predictions made."""
        return self._prediction_count


# =============================================================================
# Temporal Dynamics Pipeline
# =============================================================================


class TemporalDynamicsPipeline:
    """Full temporal dynamics processing pipeline for spectral streams.

    Orchestrates temporal buffering, time-frequency analysis,
    temporal attention, change detection, and prediction into
    a unified processing pipeline.

    Args:
        config: Temporal processing configuration.
    """

    def __init__(self, config: Optional[TemporalConfig] = None) -> None:
        self.config = config or TemporalConfig()

        # Components
        self._buffer = TemporalSpectralBuffer(
            max_length=self.config.memory_length,
            d_spectral=self.config.n_frequency_bins,
        )
        self._decomposer = TimeFrequencyDecomposer(
            window_size=self.config.window_size,
            hop_size=self.config.hop_size,
            n_frequency_bins=self.config.n_frequency_bins,
        )
        self._temporal_attention = TemporalAttentionLayer(
            d_model=self.config.d_temporal,
            n_heads=self.config.n_attention_heads,
            causal=True,
        )
        self._change_detector = ChangePointDetector(
            sensitivity=self.config.change_sensitivity,
        )
        self._predictor = SpectralPredictor(
            d_spectral=self.config.n_frequency_bins,
            horizon=self.config.prediction_horizon,
        )

        # State
        self._processing_count: int = 0
        self._total_changes_detected: int = 0

    def process(
        self,
        spectrum: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> TemporalAnalysisResult:
        """Process a new spectral observation through the full pipeline.

        Args:
            spectrum: New spectral observation.
            timestamp: Time of observation.

        Returns:
            TemporalAnalysisResult with full temporal analysis.
        """
        self._processing_count += 1

        # Buffer the observation
        self._buffer.push(spectrum, timestamp=timestamp)

        # Update predictor
        self._predictor.update(spectrum)

        # Change detection
        is_change = self._change_detector.update(np.atleast_1d(spectrum).flatten())
        if is_change:
            self._total_changes_detected += 1

        # Get window for analysis
        window = self._buffer.get_window(self.config.window_size)

        # Time-frequency decomposition
        if window.shape[0] >= 4:
            tf_map = self._decomposer.decompose(window)
        else:
            tf_map = np.zeros((1, self.config.n_frequency_bins))

        # Temporal attention
        temporal_attention = np.ones((1, 1))
        if window.shape[0] >= 2:
            # Project to temporal dimension
            if window.shape[1] != self.config.d_temporal:
                proj = np.random.randn(window.shape[1], self.config.d_temporal) * 0.02
                projected = window @ proj
            else:
                projected = window
            _, temporal_attention = self._temporal_attention.forward(projected)

        # Predictions
        predictions = self._predictor.predict()

        # Compute temporal features
        temporal_features = self._compute_temporal_features(window, tf_map)

        # Stability score
        stability = self._compute_stability(window)

        return TemporalAnalysisResult(
            time_frequency_map=tf_map,
            temporal_attention=temporal_attention,
            change_points=self._change_detector.change_points,
            predictions=predictions,
            temporal_features=temporal_features,
            stability_score=stability,
            metadata={
                "processing_id": self._processing_count,
                "buffer_size": self._buffer.size,
                "total_changes": self._total_changes_detected,
                "is_current_change": is_change,
            },
        )

    def _compute_temporal_features(
        self,
        window: np.ndarray,
        tf_map: np.ndarray,
    ) -> Dict[str, float]:
        """Compute temporal feature summary."""
        features = {}

        if window.shape[0] < 2:
            return {"insufficient_data": 1.0}

        # Temporal variance
        features["temporal_variance"] = float(np.mean(np.var(window, axis=0)))

        # Rate of change
        diff = np.diff(window, axis=0)
        features["mean_rate_of_change"] = float(np.mean(np.abs(diff)))
        features["max_rate_of_change"] = float(np.max(np.abs(diff)))

        # Autocorrelation (lag-1)
        if window.shape[0] > 2:
            mean_spec = np.mean(window, axis=1)
            if np.std(mean_spec) > 1e-12:
                acf = np.corrcoef(mean_spec[:-1], mean_spec[1:])[0, 1]
                features["autocorrelation_lag1"] = float(acf)
            else:
                features["autocorrelation_lag1"] = 0.0

        # Spectral flux from TF map
        if tf_map.shape[0] > 1:
            flux = self._decomposer.compute_spectral_flux(tf_map)
            features["mean_spectral_flux"] = float(np.mean(flux))
            features["max_spectral_flux"] = float(np.max(flux))

        return features

    def _compute_stability(self, window: np.ndarray) -> float:
        """Compute temporal stability score (0=unstable, 1=stable)."""
        if window.shape[0] < 3:
            return 1.0

        # Based on coefficient of variation over time
        temporal_mean = np.mean(window, axis=1)
        if np.mean(np.abs(temporal_mean)) < 1e-12:
            return 1.0

        cv = np.std(temporal_mean) / (np.mean(np.abs(temporal_mean)) + 1e-12)
        stability = float(np.exp(-cv))
        return np.clip(stability, 0.0, 1.0)

    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get temporal buffer statistics."""
        return self._buffer.get_statistics()

    @property
    def processing_count(self) -> int:
        """Total observations processed."""
        return self._processing_count

    @property
    def n_change_points(self) -> int:
        """Total change points detected."""
        return self._total_changes_detected

    def reset(self) -> None:
        """Reset the pipeline state."""
        self._buffer.clear()
        self._change_detector.reset()
        self._processing_count = 0
        self._total_changes_detected = 0
