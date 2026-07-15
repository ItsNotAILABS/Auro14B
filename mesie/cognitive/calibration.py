"""Spectral Calibration and Uncertainty Quantification.

Provides tools for spectral measurement calibration, uncertainty
propagation, confidence estimation, and measurement validation.

Key Components:
    - SpectralCalibrator: Multi-point calibration for spectra
    - UncertaintyQuantifier: Uncertainty propagation and estimation
    - ConfidenceEstimator: Prediction confidence computation
    - MeasurementValidator: Validate spectral measurements
    - CalibrationTransferEngine: Transfer calibrations between instruments
    - DriftDetector: Detect instrument drift over time
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class CalibrationMethod(Enum):
    """Calibration methods."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    SPLINE = "spline"
    PIECEWISE = "piecewise"
    NEURAL = "neural"


class UncertaintyType(Enum):
    """Types of uncertainty."""
    ALEATORIC = "aleatoric"      # Data inherent
    EPISTEMIC = "epistemic"       # Model/knowledge
    TOTAL = "total"              # Combined
    MEASUREMENT = "measurement"   # Instrument


class DriftType(Enum):
    """Types of instrument drift."""
    OFFSET = "offset"           # Constant shift
    GAIN = "gain"              # Scale change
    NONLINEAR = "nonlinear"    # Nonlinear warping
    SPECTRAL = "spectral"      # Frequency-dependent
    TEMPORAL = "temporal"      # Time-varying


class ValidationStatus(Enum):
    """Measurement validation status."""
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    UNCALIBRATED = "uncalibrated"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CalibrationPoint:
    """A calibration reference point.

    Args:
        reference_value: Known reference value.
        measured_value: Measured value.
        frequency: Frequency of measurement.
        uncertainty: Uncertainty in reference.
        timestamp: When measured.
    """
    reference_value: float
    measured_value: float
    frequency: float = 0.0
    uncertainty: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibrationModel:
    """Fitted calibration model.

    Args:
        coefficients: Model coefficients.
        method: Calibration method used.
        residuals: Fit residuals.
        r_squared: Coefficient of determination.
        valid_range: Valid frequency range.
    """
    coefficients: np.ndarray
    method: CalibrationMethod
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: float = 0.0
    valid_range: Tuple[float, float] = (0.0, float("inf"))


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a measurement.

    Args:
        value: Central value.
        lower: Lower bound.
        upper: Upper bound.
        std: Standard deviation.
        confidence_level: Confidence level (e.g., 0.95).
        type: Type of uncertainty.
    """
    value: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    std: float = 0.0
    confidence_level: float = 0.95
    type: UncertaintyType = UncertaintyType.TOTAL


@dataclass
class DriftReport:
    """Report of detected instrument drift.

    Args:
        drift_type: Type of drift.
        magnitude: Drift magnitude.
        rate: Drift rate per unit time.
        confidence: Confidence in detection.
        correction: Suggested correction.
        timestamp: When detected.
    """
    drift_type: DriftType
    magnitude: float
    rate: float = 0.0
    confidence: float = 0.0
    correction: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationResult:
    """Result of measurement validation.

    Args:
        status: Overall validation status.
        checks_passed: Number of checks passed.
        checks_total: Total number of checks.
        issues: List of issue descriptions.
        metrics: Validation metrics.
    """
    status: ValidationStatus
    checks_passed: int = 0
    checks_total: int = 0
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Spectral Calibrator
# =============================================================================


class SpectralCalibrator:
    """Multi-point calibration for spectral measurements.

    Fits calibration models to reference standards and applies
    corrections to new measurements.

    Args:
        method: Calibration method.
        polynomial_degree: Degree for polynomial calibration.
        n_segments: Number of segments for piecewise calibration.
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.POLYNOMIAL,
        polynomial_degree: int = 3,
        n_segments: int = 5,
    ) -> None:
        self.method = method
        self.polynomial_degree = polynomial_degree
        self.n_segments = n_segments

        self._points: List[CalibrationPoint] = []
        self._model: Optional[CalibrationModel] = None
        self._is_calibrated: bool = False

    def add_point(self, point: CalibrationPoint) -> None:
        """Add a calibration reference point.

        Args:
            point: Calibration reference point.
        """
        self._points.append(point)
        self._is_calibrated = False

    def add_points(self, references: np.ndarray, measured: np.ndarray) -> None:
        """Add multiple calibration points.

        Args:
            references: Reference values.
            measured: Measured values.
        """
        references = np.atleast_1d(references)
        measured = np.atleast_1d(measured)
        for ref, meas in zip(references, measured):
            self._points.append(CalibrationPoint(
                reference_value=float(ref),
                measured_value=float(meas),
            ))
        self._is_calibrated = False

    def calibrate(self) -> CalibrationModel:
        """Fit calibration model to reference points.

        Returns:
            Fitted CalibrationModel.
        """
        if len(self._points) < 2:
            raise ValueError("Need at least 2 calibration points")

        measured = np.array([p.measured_value for p in self._points])
        reference = np.array([p.reference_value for p in self._points])

        if self.method == CalibrationMethod.LINEAR:
            coeffs = np.polyfit(measured, reference, 1)
        elif self.method == CalibrationMethod.POLYNOMIAL:
            degree = min(self.polynomial_degree, len(self._points) - 1)
            coeffs = np.polyfit(measured, reference, degree)
        elif self.method == CalibrationMethod.PIECEWISE:
            coeffs = self._fit_piecewise(measured, reference)
        else:
            coeffs = np.polyfit(measured, reference, min(3, len(self._points) - 1))

        # Compute residuals and R²
        predicted = np.polyval(coeffs, measured)
        residuals = reference - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((reference - np.mean(reference)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

        self._model = CalibrationModel(
            coefficients=coeffs,
            method=self.method,
            residuals=residuals,
            r_squared=float(r_squared),
            valid_range=(float(np.min(measured)), float(np.max(measured))),
        )
        self._is_calibrated = True
        return self._model

    def apply(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply calibration to a spectrum.

        Args:
            spectrum: Raw measured spectrum.

        Returns:
            Calibrated spectrum.
        """
        if not self._is_calibrated or self._model is None:
            return spectrum.copy()

        spectrum = np.atleast_1d(spectrum).flatten()
        return np.polyval(self._model.coefficients, spectrum)

    def get_uncertainty(self, measured_value: float) -> UncertaintyEstimate:
        """Estimate calibration uncertainty at a point.

        Args:
            measured_value: Measured value.

        Returns:
            Uncertainty estimate.
        """
        if self._model is None or len(self._model.residuals) == 0:
            return UncertaintyEstimate(value=measured_value, std=float("inf"))

        residual_std = float(np.std(self._model.residuals))
        calibrated = float(np.polyval(self._model.coefficients, measured_value))

        return UncertaintyEstimate(
            value=calibrated,
            lower=calibrated - 2 * residual_std,
            upper=calibrated + 2 * residual_std,
            std=residual_std,
            confidence_level=0.95,
            type=UncertaintyType.MEASUREMENT,
        )

    def _fit_piecewise(self, measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Fit piecewise linear calibration."""
        # Sort by measured value
        order = np.argsort(measured)
        measured = measured[order]
        reference = reference[order]

        # Simple: fit linear to full range (approximation)
        return np.polyfit(measured, reference, 1)

    @property
    def is_calibrated(self) -> bool:
        """Whether calibration has been performed."""
        return self._is_calibrated

    @property
    def n_points(self) -> int:
        """Number of calibration points."""
        return len(self._points)

    @property
    def r_squared(self) -> float:
        """R² of calibration fit."""
        return self._model.r_squared if self._model else 0.0


# =============================================================================
# Uncertainty Quantifier
# =============================================================================


class UncertaintyQuantifier:
    """Quantify and propagate uncertainty in spectral measurements.

    Supports Monte Carlo propagation, analytical GUM methods,
    and ensemble-based uncertainty estimation.

    Args:
        n_monte_carlo: Number of Monte Carlo samples.
        confidence_level: Confidence level for intervals.
    """

    def __init__(
        self,
        n_monte_carlo: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        self.n_monte_carlo = n_monte_carlo
        self.confidence_level = confidence_level

    def propagate_linear(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate uncertainty through linear transformation.

        Args:
            values: Input values.
            uncertainties: Input uncertainties (std dev).
            transform_matrix: Linear transformation matrix.

        Returns:
            Tuple of (output_values, output_uncertainties).
        """
        values = np.atleast_1d(values)
        uncertainties = np.atleast_1d(uncertainties)

        output_values = transform_matrix @ values
        # GUM linear propagation
        output_var = transform_matrix ** 2 @ uncertainties ** 2
        output_uncertainties = np.sqrt(output_var)

        return output_values, output_uncertainties

    def propagate_monte_carlo(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
        transform_fn: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate uncertainty via Monte Carlo simulation.

        Args:
            values: Input values.
            uncertainties: Input standard deviations.
            transform_fn: Transformation function.

        Returns:
            Tuple of (mean_output, std_output).
        """
        values = np.atleast_1d(values)
        uncertainties = np.atleast_1d(uncertainties)

        # Generate Monte Carlo samples
        samples = []
        for _ in range(self.n_monte_carlo):
            perturbed = values + uncertainties * np.random.randn(len(values))
            output = transform_fn(perturbed)
            samples.append(output)

        samples = np.array(samples)
        mean_output = np.mean(samples, axis=0)
        std_output = np.std(samples, axis=0)

        return mean_output, std_output

    def compute_confidence_interval(
        self,
        samples: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute confidence interval from samples.

        Args:
            samples: Array of measurement samples.

        Returns:
            Tuple of (mean, lower_bound, upper_bound).
        """
        samples = np.atleast_1d(samples)
        mean = float(np.mean(samples))
        alpha = 1.0 - self.confidence_level
        lower = float(np.percentile(samples, 100 * alpha / 2))
        upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return mean, lower, upper

    def estimate_aleatoric(self, spectra: np.ndarray) -> np.ndarray:
        """Estimate aleatoric uncertainty from repeated measurements.

        Args:
            spectra: Multiple measurements (n_measurements x n_frequencies).

        Returns:
            Per-frequency aleatoric uncertainty.
        """
        spectra = np.atleast_2d(spectra)
        return np.std(spectra, axis=0)

    def estimate_epistemic(
        self,
        predictions: List[np.ndarray],
    ) -> np.ndarray:
        """Estimate epistemic uncertainty from model ensemble.

        Args:
            predictions: List of predictions from different models.

        Returns:
            Per-output epistemic uncertainty.
        """
        preds = np.array(predictions)
        return np.std(preds, axis=0)

    def combine_uncertainties(
        self,
        aleatoric: np.ndarray,
        epistemic: np.ndarray,
    ) -> np.ndarray:
        """Combine aleatoric and epistemic uncertainties.

        Args:
            aleatoric: Aleatoric uncertainty.
            epistemic: Epistemic uncertainty.

        Returns:
            Total uncertainty.
        """
        return np.sqrt(aleatoric ** 2 + epistemic ** 2)


# =============================================================================
# Confidence Estimator
# =============================================================================


class ConfidenceEstimator:
    """Estimate prediction confidence for spectral models.

    Uses calibrated probabilities, ensemble disagreement,
    and distance-based methods.

    Args:
        n_bins: Number of bins for calibration.
        method: Confidence calibration method.
    """

    def __init__(
        self,
        n_bins: int = 15,
        method: str = "temperature_scaling",
    ) -> None:
        self.n_bins = n_bins
        self.method = method

        self._temperature: float = 1.0
        self._calibration_data: List[Tuple[float, bool]] = []
        self._ece: float = 0.0  # Expected Calibration Error

    def calibrate(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> float:
        """Calibrate confidence estimates.

        Args:
            predicted_probs: Maximum predicted probabilities.
            true_labels: Ground truth labels.
            predicted_labels: Predicted labels.

        Returns:
            Expected Calibration Error (ECE).
        """
        predicted_probs = np.atleast_1d(predicted_probs)
        true_labels = np.atleast_1d(true_labels)
        predicted_labels = np.atleast_1d(predicted_labels)

        correct = predicted_labels == true_labels

        # Store calibration data
        for prob, corr in zip(predicted_probs, correct):
            self._calibration_data.append((float(prob), bool(corr)))

        # Compute ECE
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n_total = len(predicted_probs)

        for i in range(self.n_bins):
            mask = (predicted_probs >= bin_boundaries[i]) & (predicted_probs < bin_boundaries[i + 1])
            if not np.any(mask):
                continue
            bin_conf = np.mean(predicted_probs[mask])
            bin_acc = np.mean(correct[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / n_total) * abs(bin_acc - bin_conf)

        self._ece = float(ece)

        # Temperature scaling
        if self.method == "temperature_scaling":
            self._fit_temperature(predicted_probs, correct)

        return self._ece

    def estimate_confidence(self, probabilities: np.ndarray) -> float:
        """Estimate calibrated confidence for a prediction.

        Args:
            probabilities: Model output probabilities.

        Returns:
            Calibrated confidence score.
        """
        probabilities = np.atleast_1d(probabilities)
        max_prob = float(np.max(probabilities))

        # Apply temperature scaling
        scaled = max_prob ** (1.0 / self._temperature)
        return float(np.clip(scaled, 0, 1))

    def _fit_temperature(self, probs: np.ndarray, correct: np.ndarray) -> None:
        """Fit temperature parameter."""
        # Simple grid search
        best_ece = float("inf")
        best_temp = 1.0

        for temp in np.linspace(0.1, 5.0, 50):
            scaled_probs = probs ** (1.0 / temp)
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            ece = 0.0
            n = len(probs)

            for i in range(self.n_bins):
                mask = (scaled_probs >= bin_boundaries[i]) & (scaled_probs < bin_boundaries[i + 1])
                if not np.any(mask):
                    continue
                bin_conf = np.mean(scaled_probs[mask])
                bin_acc = np.mean(correct[mask])
                ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self._temperature = best_temp

    @property
    def expected_calibration_error(self) -> float:
        """ECE of the calibration."""
        return self._ece

    @property
    def temperature(self) -> float:
        """Calibrated temperature."""
        return self._temperature


# =============================================================================
# Measurement Validator
# =============================================================================


class MeasurementValidator:
    """Validate spectral measurement quality.

    Performs multiple checks on spectral data to ensure
    measurement validity and data quality.

    Args:
        snr_threshold: Minimum SNR in dB.
        saturation_threshold: Maximum allowed value fraction.
        noise_floor: Expected noise floor level.
    """

    def __init__(
        self,
        snr_threshold: float = 20.0,
        saturation_threshold: float = 0.99,
        noise_floor: float = 0.001,
    ) -> None:
        self.snr_threshold = snr_threshold
        self.saturation_threshold = saturation_threshold
        self.noise_floor = noise_floor

    def validate(self, spectrum: np.ndarray) -> ValidationResult:
        """Run full validation on a spectrum.

        Args:
            spectrum: Input spectrum.

        Returns:
            ValidationResult with all check outcomes.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        issues = []
        metrics = {}
        checks_passed = 0
        checks_total = 0

        # Check 1: Non-empty
        checks_total += 1
        if len(spectrum) == 0:
            issues.append("Empty spectrum")
            return ValidationResult(
                status=ValidationStatus.INVALID,
                checks_passed=0,
                checks_total=checks_total,
                issues=issues,
                metrics=metrics,
            )
        checks_passed += 1

        # Check 2: No NaN/Inf
        checks_total += 1
        if np.any(np.isnan(spectrum)) or np.any(np.isinf(spectrum)):
            issues.append("Contains NaN or Inf values")
        else:
            checks_passed += 1
        metrics["has_nan"] = float(np.any(np.isnan(spectrum)))

        # Check 3: Not all zeros
        checks_total += 1
        if np.all(spectrum == 0):
            issues.append("All-zero spectrum")
        else:
            checks_passed += 1
        metrics["is_zero"] = float(np.all(spectrum == 0))

        # Check 4: SNR check
        checks_total += 1
        snr = self._estimate_snr(spectrum)
        metrics["snr_db"] = snr
        if snr >= self.snr_threshold:
            checks_passed += 1
        else:
            issues.append(f"Low SNR: {snr:.1f} dB (threshold: {self.snr_threshold})")

        # Check 5: Saturation check
        checks_total += 1
        max_val = np.max(np.abs(spectrum))
        if max_val > 0:
            saturation = np.sum(np.abs(spectrum) > self.saturation_threshold * max_val) / len(spectrum)
        else:
            saturation = 0.0
        metrics["saturation_fraction"] = float(saturation)
        if saturation < 0.05:
            checks_passed += 1
        else:
            issues.append(f"Signal saturation detected: {saturation:.1%}")

        # Check 6: Dynamic range
        checks_total += 1
        dynamic_range = self._compute_dynamic_range(spectrum)
        metrics["dynamic_range_db"] = dynamic_range
        if dynamic_range > 10:
            checks_passed += 1
        else:
            issues.append(f"Low dynamic range: {dynamic_range:.1f} dB")

        # Check 7: Stationarity
        checks_total += 1
        stationarity = self._check_stationarity(spectrum)
        metrics["stationarity"] = stationarity
        if stationarity > 0.5:
            checks_passed += 1
        else:
            issues.append("Non-stationary signal detected")

        # Check 8: Spectral regularity
        checks_total += 1
        regularity = self._check_regularity(spectrum)
        metrics["regularity"] = regularity
        if regularity > 0.3:
            checks_passed += 1
        else:
            issues.append("Irregular spectral structure")

        # Determine status
        pass_rate = checks_passed / max(1, checks_total)
        if pass_rate >= 0.9:
            status = ValidationStatus.VALID
        elif pass_rate >= 0.6:
            status = ValidationStatus.SUSPECT
        else:
            status = ValidationStatus.INVALID

        return ValidationResult(
            status=status,
            checks_passed=checks_passed,
            checks_total=checks_total,
            issues=issues,
            metrics=metrics,
        )

    def _estimate_snr(self, spectrum: np.ndarray) -> float:
        """Estimate SNR from spectrum."""
        abs_spec = np.abs(spectrum)
        if len(abs_spec) < 4:
            return 0.0
        sorted_spec = np.sort(abs_spec)
        noise = np.mean(sorted_spec[:len(sorted_spec) // 4]) + 1e-12
        signal = np.mean(sorted_spec[-len(sorted_spec) // 4:])
        return float(20 * np.log10(signal / noise))

    def _compute_dynamic_range(self, spectrum: np.ndarray) -> float:
        """Compute dynamic range."""
        abs_spec = np.abs(spectrum)
        max_val = np.max(abs_spec)
        positive = abs_spec[abs_spec > 0]
        if len(positive) == 0:
            return 0.0
        min_val = np.min(positive)
        return float(20 * np.log10(max_val / (min_val + 1e-12)))

    def _check_stationarity(self, spectrum: np.ndarray) -> float:
        """Check spectral stationarity."""
        n = len(spectrum)
        if n < 8:
            return 1.0
        half = n // 2
        m1, s1 = np.mean(spectrum[:half]), np.std(spectrum[:half])
        m2, s2 = np.mean(spectrum[half:]), np.std(spectrum[half:])
        diff = abs(m1 - m2) / (abs(m1) + abs(m2) + 1e-12)
        return float(1.0 - min(1.0, diff))

    def _check_regularity(self, spectrum: np.ndarray) -> float:
        """Check spectral regularity (smoothness)."""
        if len(spectrum) < 3:
            return 1.0
        diffs = np.abs(np.diff(spectrum))
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        if max_diff < 1e-12:
            return 1.0
        return float(1.0 - mean_diff / max_diff)


# =============================================================================
# Calibration Transfer Engine
# =============================================================================


class CalibrationTransferEngine:
    """Transfer calibrations between instruments.

    Enables spectral measurement comparability across different
    instruments through standardization and transfer models.

    Args:
        n_standards: Number of transfer standards.
        transfer_method: Transfer model type.
    """

    def __init__(
        self,
        n_standards: int = 10,
        transfer_method: str = "piecewise_linear",
    ) -> None:
        self.n_standards = n_standards
        self.transfer_method = transfer_method

        self._source_spectra: List[np.ndarray] = []
        self._target_spectra: List[np.ndarray] = []
        self._transfer_model: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def add_standard(self, source_spectrum: np.ndarray, target_spectrum: np.ndarray) -> None:
        """Add a transfer standard measurement pair.

        Args:
            source_spectrum: Spectrum from source instrument.
            target_spectrum: Same sample on target instrument.
        """
        self._source_spectra.append(np.atleast_1d(source_spectrum).flatten())
        self._target_spectra.append(np.atleast_1d(target_spectrum).flatten())
        self._is_fitted = False

    def fit(self) -> Dict[str, float]:
        """Fit the transfer model.

        Returns:
            Fit metrics.
        """
        if len(self._source_spectra) < 2:
            raise ValueError("Need at least 2 transfer standards")

        # Align lengths
        max_len = max(
            max(len(s) for s in self._source_spectra),
            max(len(t) for t in self._target_spectra),
        )

        source_matrix = np.zeros((len(self._source_spectra), max_len))
        target_matrix = np.zeros((len(self._target_spectra), max_len))

        for i, (s, t) in enumerate(zip(self._source_spectra, self._target_spectra)):
            source_matrix[i, :len(s)] = s
            target_matrix[i, :len(t)] = t

        # Fit linear transfer model: target = source * A + b
        # Using least squares
        n_features = max_len
        X = np.column_stack([source_matrix, np.ones(len(source_matrix))])

        # Fit per-frequency
        self._transfer_model = np.zeros((n_features, 2))  # slope, intercept per freq
        residuals = []

        for freq_idx in range(n_features):
            y = target_matrix[:, freq_idx]
            x = source_matrix[:, freq_idx]

            if np.std(x) < 1e-12:
                self._transfer_model[freq_idx] = [1.0, 0.0]
                continue

            # Linear fit
            coeffs = np.polyfit(x, y, 1)
            self._transfer_model[freq_idx] = coeffs
            pred = np.polyval(coeffs, x)
            residuals.extend((y - pred).tolist())

        self._is_fitted = True
        residuals = np.array(residuals)

        return {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "max_error": float(np.max(np.abs(residuals))) if len(residuals) > 0 else 0.0,
            "n_standards": len(self._source_spectra),
        }

    def transfer(self, source_spectrum: np.ndarray) -> np.ndarray:
        """Transfer a spectrum from source to target instrument.

        Args:
            source_spectrum: Spectrum from source instrument.

        Returns:
            Spectrum as if measured on target instrument.
        """
        if not self._is_fitted or self._transfer_model is None:
            return source_spectrum.copy()

        source = np.atleast_1d(source_spectrum).flatten()
        n = min(len(source), len(self._transfer_model))
        result = np.zeros(len(source))

        for i in range(n):
            slope, intercept = self._transfer_model[i]
            result[i] = slope * source[i] + intercept

        # Copy remaining frequencies unchanged
        result[n:] = source[n:]
        return result

    @property
    def is_fitted(self) -> bool:
        """Whether transfer model is fitted."""
        return self._is_fitted


# =============================================================================
# Drift Detector
# =============================================================================


class DriftDetector:
    """Detect instrument drift over time.

    Monitors spectral measurements from reference standards
    to detect and quantify instrument drift.

    Args:
        reference_spectrum: Known reference spectrum.
        sensitivity: Detection sensitivity (0-1).
        window_size: Moving average window.
    """

    def __init__(
        self,
        reference_spectrum: Optional[np.ndarray] = None,
        sensitivity: float = 0.7,
        window_size: int = 20,
    ) -> None:
        self.sensitivity = sensitivity
        self.window_size = window_size

        self._reference = None
        if reference_spectrum is not None:
            self._reference = np.atleast_1d(reference_spectrum).flatten()

        self._measurements: List[Tuple[float, np.ndarray]] = []
        self._drift_history: List[DriftReport] = []

    def set_reference(self, spectrum: np.ndarray) -> None:
        """Set the reference spectrum.

        Args:
            spectrum: Reference spectrum for drift detection.
        """
        self._reference = np.atleast_1d(spectrum).flatten()

    def add_measurement(self, spectrum: np.ndarray, timestamp: Optional[float] = None) -> Optional[DriftReport]:
        """Add a new measurement and check for drift.

        Args:
            spectrum: New measurement of the reference.
            timestamp: Measurement timestamp.

        Returns:
            DriftReport if drift detected, else None.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        ts = timestamp or time.time()
        self._measurements.append((ts, spectrum))

        if self._reference is None:
            self._reference = spectrum.copy()
            return None

        # Check for drift
        n = min(len(spectrum), len(self._reference))
        diff = spectrum[:n] - self._reference[:n]

        # Offset drift
        offset = np.mean(diff)
        offset_threshold = (1.0 - self.sensitivity) * np.std(self._reference[:n])

        if abs(offset) > offset_threshold:
            report = DriftReport(
                drift_type=DriftType.OFFSET,
                magnitude=float(abs(offset)),
                rate=self._compute_drift_rate(),
                confidence=min(1.0, abs(offset) / (offset_threshold + 1e-12)),
                correction=-diff,
                timestamp=ts,
            )
            self._drift_history.append(report)
            return report

        # Gain drift
        if np.std(self._reference[:n]) > 1e-12:
            gain_ratio = np.std(spectrum[:n]) / np.std(self._reference[:n])
            if abs(gain_ratio - 1.0) > 0.1 * (1.0 - self.sensitivity):
                report = DriftReport(
                    drift_type=DriftType.GAIN,
                    magnitude=float(abs(gain_ratio - 1.0)),
                    rate=self._compute_drift_rate(),
                    confidence=min(1.0, abs(gain_ratio - 1.0) * 10),
                    timestamp=ts,
                )
                self._drift_history.append(report)
                return report

        return None

    def _compute_drift_rate(self) -> float:
        """Compute rate of drift from history."""
        if len(self._measurements) < 2:
            return 0.0

        # Use first and last measurements
        t1, s1 = self._measurements[0]
        t2, s2 = self._measurements[-1]
        dt = t2 - t1
        if dt < 1e-12:
            return 0.0

        n = min(len(s1), len(s2))
        mean_change = float(np.mean(np.abs(s2[:n] - s1[:n])))
        return mean_change / dt

    def get_correction(self) -> Optional[np.ndarray]:
        """Get current drift correction vector.

        Returns:
            Correction to apply, or None if no drift.
        """
        if not self._drift_history:
            return None
        latest = self._drift_history[-1]
        return latest.correction

    @property
    def n_measurements(self) -> int:
        """Number of monitoring measurements."""
        return len(self._measurements)

    @property
    def n_drift_events(self) -> int:
        """Number of drift events detected."""
        return len(self._drift_history)

    @property
    def has_drift(self) -> bool:
        """Whether drift has been detected."""
        return len(self._drift_history) > 0
