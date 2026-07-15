"""Streaming anomaly detection at sensor frequency.

Supports vibration monitoring, seismic/event classification,
material/process sensing, and equipment health monitoring.
Designed for edge deployment with minimal latency.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


class SensorDomain(Enum):
    """Sensor application domains for anomaly detection."""

    VIBRATION = "vibration"
    SEISMIC = "seismic"
    MATERIAL_PROCESS = "material_process"
    EQUIPMENT_HEALTH = "equipment_health"
    ACOUSTIC = "acoustic"
    THERMAL = "thermal"
    ELECTROMAGNETIC = "electromagnetic"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection.

    Args:
        domain: Sensor domain for tuned detection.
        window_size: Number of samples per analysis window.
        overlap: Overlap between consecutive windows.
        threshold_sigma: Standard deviations for anomaly threshold.
        min_anomaly_duration: Minimum consecutive anomalous windows to trigger.
        spectral_bands: Frequency bands to monitor (Hz pairs).
        adaptive_threshold: Whether to adapt thresholds over time.
        learning_rate: Rate of threshold adaptation (0-1).
        max_history: Maximum number of past windows to retain for baseline.
    """

    domain: SensorDomain = SensorDomain.VIBRATION
    window_size: int = 256
    overlap: int = 64
    threshold_sigma: float = 3.0
    min_anomaly_duration: int = 2
    spectral_bands: List[tuple] = field(default_factory=lambda: [
        (0.1, 10.0), (10.0, 100.0), (100.0, 1000.0)
    ])
    adaptive_threshold: bool = True
    learning_rate: float = 0.01
    max_history: int = 1000


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a signal window.

    Attributes:
        is_anomaly: Whether an anomaly was detected.
        severity: Severity level of the anomaly.
        anomaly_score: Normalized anomaly score (0-1).
        spectral_deviation: Per-band deviation from baseline.
        timestamp: When the anomaly was detected.
        event_id: Unique identifier for this detection event.
        metadata: Additional context about the detection.
    """

    is_anomaly: bool
    severity: AnomalySeverity
    anomaly_score: float
    spectral_deviation: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Real-time spectral anomaly detector for IoT sensor streams.

    Performs frequency-domain analysis at sensor rate with adaptive
    baseline tracking and multi-band deviation scoring. Designed for
    edge deployment with O(n log n) per-window complexity.

    Args:
        config: Detection configuration.
        sample_rate: Sensor sample rate in Hz.
    """

    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        sample_rate: float = 1000.0,
    ) -> None:
        self.config = config or AnomalyConfig()
        self.sample_rate = sample_rate
        self._baseline_psd: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._history: deque = deque(maxlen=self.config.max_history)
        self._anomaly_streak = 0
        self._total_windows = 0
        self._total_anomalies = 0
        self._callbacks: List[Callable[[AnomalyResult], None]] = []

    def on_anomaly(self, callback: Callable[[AnomalyResult], None]) -> None:
        """Register a callback for anomaly events.

        Args:
            callback: Function called with AnomalyResult on detection.
        """
        self._callbacks.append(callback)

    def ingest(self, samples: np.ndarray) -> List[AnomalyResult]:
        """Ingest raw sensor samples and detect anomalies.

        Processes samples through windowed FFT analysis,
        comparing against adaptive baseline.

        Args:
            samples: 1D array of sensor readings.

        Returns:
            List of anomaly results for each complete window.
        """
        samples = np.asarray(samples, dtype=np.float64).ravel()
        results: List[AnomalyResult] = []

        step = self.config.window_size - self.config.overlap
        n_windows = max(0, (len(samples) - self.config.window_size) // step + 1)

        for i in range(n_windows):
            start = i * step
            window = samples[start:start + self.config.window_size]
            result = self._analyze_window(window)
            results.append(result)

        return results

    def _analyze_window(self, window: np.ndarray) -> AnomalyResult:
        """Analyze a single signal window for anomalies.

        Args:
            window: Signal window of length window_size.

        Returns:
            AnomalyResult for this window.
        """
        self._total_windows += 1

        # Compute PSD via FFT
        psd = self._compute_psd(window)

        # Update or initialize baseline
        if self._baseline_psd is None:
            self._baseline_psd = psd.copy()
            self._baseline_std = np.abs(psd) * 0.1 + 1e-10
            self._history.append(psd)
            return AnomalyResult(
                is_anomaly=False,
                severity=AnomalySeverity.INFO,
                anomaly_score=0.0,
            )

        # Compute per-band deviations
        band_deviations = self._compute_band_deviations(psd)
        anomaly_score = self._compute_anomaly_score(band_deviations)

        # Determine if anomalous
        is_anomaly = anomaly_score > (1.0 / self.config.threshold_sigma)

        if is_anomaly:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = 0

        # Only trigger if streak meets minimum duration
        triggered = is_anomaly and self._anomaly_streak >= self.config.min_anomaly_duration
        severity = self._score_to_severity(anomaly_score) if triggered else AnomalySeverity.INFO

        if triggered:
            self._total_anomalies += 1

        # Adapt baseline
        if self.config.adaptive_threshold and not triggered:
            lr = self.config.learning_rate
            self._baseline_psd = (1 - lr) * self._baseline_psd + lr * psd
            if len(self._history) > 10:
                history_arr = np.array(list(self._history)[-50:])
                self._baseline_std = np.std(history_arr, axis=0) + 1e-10

        self._history.append(psd)

        result = AnomalyResult(
            is_anomaly=triggered,
            severity=severity,
            anomaly_score=float(anomaly_score),
            spectral_deviation={
                f"band_{k}": float(v) for k, v in band_deviations.items()
            },
            metadata={
                "domain": self.config.domain.value,
                "window_index": self._total_windows,
                "streak": self._anomaly_streak,
            },
        )

        if triggered:
            for cb in self._callbacks:
                cb(result)

        return result

    def _compute_psd(self, window: np.ndarray) -> np.ndarray:
        """Compute power spectral density of a window.

        Args:
            window: Time-domain signal window.

        Returns:
            PSD array (one-sided).
        """
        # Apply Hann window to reduce spectral leakage
        windowed = window * np.hanning(len(window))
        fft_vals = np.fft.rfft(windowed)
        psd = np.abs(fft_vals) ** 2 / len(window)
        return psd

    def _compute_band_deviations(self, psd: np.ndarray) -> Dict[int, float]:
        """Compute deviation from baseline per frequency band.

        Args:
            psd: Current window PSD.

        Returns:
            Dictionary mapping band index to deviation score.
        """
        freqs = np.fft.rfftfreq(self.config.window_size, d=1.0 / self.sample_rate)
        deviations: Dict[int, float] = {}

        for idx, (low, high) in enumerate(self.config.spectral_bands):
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                deviations[idx] = 0.0
                continue

            current_band = psd[mask]
            baseline_band = self._baseline_psd[mask]
            std_band = self._baseline_std[mask]

            deviation = np.mean(np.abs(current_band - baseline_band) / std_band)
            deviations[idx] = float(deviation)

        return deviations

    def _compute_anomaly_score(self, band_deviations: Dict[int, float]) -> float:
        """Compute overall anomaly score from band deviations.

        Args:
            band_deviations: Per-band deviation scores.

        Returns:
            Normalized anomaly score (0-1).
        """
        if not band_deviations:
            return 0.0
        max_dev = max(band_deviations.values())
        # Normalize using sigmoid-like mapping
        score = 1.0 - 1.0 / (1.0 + max_dev / self.config.threshold_sigma)
        return min(1.0, max(0.0, score))

    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Map anomaly score to severity level.

        Args:
            score: Normalized anomaly score (0-1).

        Returns:
            Corresponding severity level.
        """
        if score >= 0.9:
            return AnomalySeverity.EMERGENCY
        elif score >= 0.7:
            return AnomalySeverity.CRITICAL
        elif score >= 0.4:
            return AnomalySeverity.WARNING
        return AnomalySeverity.INFO

    @property
    def stats(self) -> Dict[str, Any]:
        """Return detection statistics.

        Returns:
            Dictionary of detection metrics.
        """
        return {
            "total_windows": self._total_windows,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate": (
                self._total_anomalies / self._total_windows
                if self._total_windows > 0 else 0.0
            ),
            "current_streak": self._anomaly_streak,
            "baseline_established": self._baseline_psd is not None,
        }

    def reset_baseline(self) -> None:
        """Reset the adaptive baseline, forcing re-learning."""
        self._baseline_psd = None
        self._baseline_std = None
        self._history.clear()
        self._anomaly_streak = 0
