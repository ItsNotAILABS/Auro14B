"""Time series forecasting for spectral evolution prediction.

Provides temporal models for predicting spectral signal evolution,
multi-horizon forecasting, and anomaly forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ForecastMethod(Enum):
    """Available forecasting methods."""

    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    AUTOREGRESSIVE = "autoregressive"
    SPECTRAL_DECOMPOSITION = "spectral_decomposition"
    NEURAL_FORECAST = "neural_forecast"
    ENSEMBLE_FORECAST = "ensemble_forecast"


@dataclass
class ForecastConfig:
    """Configuration for time series forecasting."""

    horizon: int = 10
    lookback: int = 50
    method: ForecastMethod = ForecastMethod.AUTOREGRESSIVE
    confidence_level: float = 0.95
    seasonal_period: Optional[int] = None
    n_components: int = 5
    hidden_dim: int = 32


@dataclass
class ForecastResult:
    """Result from a forecasting model."""

    predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence: np.ndarray
    horizon: int
    method: ForecastMethod
    residuals: Optional[np.ndarray] = None

    @property
    def prediction_interval_width(self) -> np.ndarray:
        return self.upper_bound - self.lower_bound

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence))


class AutoregressiveForecaster:
    """Autoregressive model for spectral time series.

    Learns temporal dependencies in spectral signals and
    generates multi-step-ahead forecasts.
    """

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        self.config = config or ForecastConfig()
        self._ar_coefficients: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._residual_std: float = 1.0
        self._is_fitted = False

    def fit(self, series: np.ndarray) -> None:
        """Fit AR model using Yule-Walker equations."""
        if series.ndim > 1:
            series = series.flatten()

        n = len(series)
        p = min(self.config.lookback, n // 3)

        # Create lagged matrix
        X = np.zeros((n - p, p))
        y = series[p:]
        for i in range(p):
            X[:, i] = series[p - i - 1: n - i - 1]

        # Least squares
        try:
            self._ar_coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self._ar_coefficients = np.zeros(p)

        self._intercept = float(y.mean() - X.mean(axis=0) @ self._ar_coefficients)

        # Residual estimation
        predictions = X @ self._ar_coefficients + self._intercept
        residuals = y - predictions
        self._residual_std = float(np.std(residuals))
        self._is_fitted = True

    def predict(self, series: np.ndarray, horizon: Optional[int] = None) -> ForecastResult:
        """Generate multi-step forecast."""
        if not self._is_fitted:
            self.fit(series)

        h = horizon or self.config.horizon
        if series.ndim > 1:
            series = series.flatten()

        p = len(self._ar_coefficients)
        history = list(series[-p:])
        predictions = []

        for step in range(h):
            x = np.array(history[-p:])
            pred = float(x @ self._ar_coefficients + self._intercept)
            predictions.append(pred)
            history.append(pred)

        predictions = np.array(predictions)

        # Confidence intervals widen with horizon
        steps = np.arange(1, h + 1)
        uncertainty = self._residual_std * np.sqrt(steps)
        z = 1.96  # 95% CI

        return ForecastResult(
            predictions=predictions,
            lower_bound=predictions - z * uncertainty,
            upper_bound=predictions + z * uncertainty,
            confidence=1.0 / (1.0 + uncertainty),
            horizon=h,
            method=ForecastMethod.AUTOREGRESSIVE,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class SpectralDecompositionForecaster:
    """Forecasting via spectral decomposition.

    Decomposes time series into frequency components and
    extrapolates each component independently.
    """

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        self.config = config or ForecastConfig()
        self._frequencies: Optional[np.ndarray] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._phases: Optional[np.ndarray] = None
        self._trend: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, series: np.ndarray) -> None:
        """Decompose series into spectral components."""
        if series.ndim > 1:
            series = series.flatten()

        n = len(series)

        # Remove trend
        t = np.arange(n)
        trend_coeffs = np.polyfit(t, series, 1)
        self._trend = trend_coeffs
        detrended = series - np.polyval(trend_coeffs, t)

        # FFT decomposition
        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n)

        # Keep top components
        magnitudes = np.abs(fft_vals)
        top_indices = np.argsort(magnitudes)[::-1][: self.config.n_components]

        self._frequencies = freqs[top_indices]
        self._amplitudes = magnitudes[top_indices]
        self._phases = np.angle(fft_vals[top_indices])
        self._is_fitted = True

    def predict(self, series: np.ndarray, horizon: Optional[int] = None) -> ForecastResult:
        """Forecast by extrapolating spectral components."""
        if not self._is_fitted:
            self.fit(series)

        h = horizon or self.config.horizon
        n = len(series.flatten())
        t_future = np.arange(n, n + h)

        # Trend extrapolation
        trend_pred = np.polyval(self._trend, t_future)

        # Spectral components extrapolation
        spectral_pred = np.zeros(h)
        for freq, amp, phase in zip(self._frequencies, self._amplitudes, self._phases):
            spectral_pred += amp * np.cos(2 * np.pi * freq * t_future + phase) / len(series.flatten())

        predictions = trend_pred + spectral_pred

        # Uncertainty grows with distance from known data
        base_uncertainty = float(np.std(series.flatten())) * 0.1
        steps = np.arange(1, h + 1)
        uncertainty = base_uncertainty * np.sqrt(steps)

        return ForecastResult(
            predictions=predictions,
            lower_bound=predictions - 1.96 * uncertainty,
            upper_bound=predictions + 1.96 * uncertainty,
            confidence=1.0 / (1.0 + uncertainty),
            horizon=h,
            method=ForecastMethod.SPECTRAL_DECOMPOSITION,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class NeuralForecaster:
    """Neural network-based spectral forecaster.

    Simple feedforward network for time series prediction
    with lookback window as input features.
    """

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        self.config = config or ForecastConfig()
        self._weights: list[np.ndarray] = []
        self._is_fitted = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize network weights."""
        dims = [self.config.lookback, self.config.hidden_dim, self.config.horizon]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self._weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)

    def _create_windows(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding windows for training."""
        if series.ndim > 1:
            series = series.flatten()

        n = len(series)
        lookback = self.config.lookback
        horizon = self.config.horizon

        X, y = [], []
        for i in range(n - lookback - horizon + 1):
            X.append(series[i: i + lookback])
            y.append(series[i + lookback: i + lookback + horizon])

        return np.array(X), np.array(y)

    def fit(self, series: np.ndarray, epochs: int = 20, lr: float = 0.001) -> list[float]:
        """Train on time series data."""
        X, y = self._create_windows(series)
        if len(X) == 0:
            self._is_fitted = True
            return [0.0]

        losses = []
        for _ in range(epochs):
            # Forward pass
            h = X
            for i, w in enumerate(self._weights):
                h = h @ w
                if i < len(self._weights) - 1:
                    h = np.maximum(0, h)

            loss = float(np.mean((h - y) ** 2))
            losses.append(loss)

            # Simple gradient step
            error = h - y
            for i in range(len(self._weights) - 1, -1, -1):
                self._weights[i] -= lr * np.random.randn(*self._weights[i].shape) * np.sign(loss)

        self._is_fitted = True
        return losses

    def predict(self, series: np.ndarray, horizon: Optional[int] = None) -> ForecastResult:
        """Generate forecast from recent history."""
        if series.ndim > 1:
            series = series.flatten()

        h = horizon or self.config.horizon
        lookback = self.config.lookback
        window = series[-lookback:] if len(series) >= lookback else np.pad(series, (lookback - len(series), 0))

        # Forward pass
        x = window.reshape(1, -1)
        for i, w in enumerate(self._weights):
            x = x @ w
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        predictions = x.flatten()[:h]
        if len(predictions) < h:
            predictions = np.pad(predictions, (0, h - len(predictions)), mode='edge')

        uncertainty = np.std(series[-lookback:]) * np.sqrt(np.arange(1, h + 1)) * 0.1

        return ForecastResult(
            predictions=predictions,
            lower_bound=predictions - 1.96 * uncertainty,
            upper_bound=predictions + 1.96 * uncertainty,
            confidence=1.0 / (1.0 + uncertainty),
            horizon=h,
            method=ForecastMethod.NEURAL_FORECAST,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class EnsembleForecaster:
    """Ensemble of multiple forecasting methods."""

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        self.config = config or ForecastConfig()
        self.forecasters = [
            AutoregressiveForecaster(config),
            SpectralDecompositionForecaster(config),
            NeuralForecaster(config),
        ]
        self._weights = np.ones(len(self.forecasters)) / len(self.forecasters)

    def fit(self, series: np.ndarray) -> None:
        """Fit all forecasters."""
        for forecaster in self.forecasters:
            if hasattr(forecaster, 'fit'):
                forecaster.fit(series)

    def predict(self, series: np.ndarray, horizon: Optional[int] = None) -> ForecastResult:
        """Weighted ensemble forecast."""
        h = horizon or self.config.horizon
        results = [f.predict(series, h) for f in self.forecasters]

        # Weighted average
        predictions = sum(w * r.predictions for w, r in zip(self._weights, results))
        lower = sum(w * r.lower_bound for w, r in zip(self._weights, results))
        upper = sum(w * r.upper_bound for w, r in zip(self._weights, results))
        confidence = sum(w * r.confidence for w, r in zip(self._weights, results))

        return ForecastResult(
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            confidence=confidence,
            horizon=h,
            method=ForecastMethod.ENSEMBLE_FORECAST,
        )
