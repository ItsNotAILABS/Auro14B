"""Advanced Spectral Signal Processing Library.

Comprehensive signal processing toolkit for spectral analysis,
providing advanced filtering, decomposition, feature extraction,
and signal reconstruction capabilities.

Key Components:
    - AdaptiveFilter: LMS/RLS adaptive filtering
    - SpectralDecomposer: EMD, wavelet, SVD decomposition
    - AdvancedFeatureExtractor: Rich spectral feature computation
    - SignalSynthesizer: Spectral signal reconstruction/generation
    - SpectralQualityAssessor: Signal quality metrics
    - AnomalyDetector: Spectral anomaly detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class FilterType(Enum):
    """Adaptive filter types."""
    LMS = "lms"
    NLMS = "nlms"
    RLS = "rls"
    KALMAN = "kalman"


class DecompositionMethod(Enum):
    """Signal decomposition methods."""
    EMD = "emd"
    WAVELET = "wavelet"
    SVD = "svd"
    PCA = "pca"
    ICA = "ica"
    NMF = "nmf"


class WindowFunction(Enum):
    """Spectral window functions."""
    RECTANGULAR = "rectangular"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    FLAT_TOP = "flat_top"
    GAUSSIAN = "gaussian"


class AnomalyType(Enum):
    """Types of spectral anomalies."""
    PEAK_SHIFT = "peak_shift"
    AMPLITUDE_CHANGE = "amplitude_change"
    NEW_PEAK = "new_peak"
    MISSING_PEAK = "missing_peak"
    BANDWIDTH_CHANGE = "bandwidth_change"
    NOISE_INCREASE = "noise_increase"
    HARMONIC_DISTORTION = "harmonic_distortion"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FilterState:
    """State of an adaptive filter.

    Args:
        weights: Filter coefficients.
        error_history: Error signal history.
        convergence_rate: Current convergence rate.
        iterations: Number of iterations.
    """
    weights: np.ndarray
    error_history: List[float] = field(default_factory=list)
    convergence_rate: float = 0.0
    iterations: int = 0


@dataclass
class DecompositionResult:
    """Result of signal decomposition.

    Args:
        components: Decomposed signal components.
        residual: Residual after decomposition.
        explained_variance: Variance explained per component.
        metadata: Additional decomposition info.
    """
    components: List[np.ndarray]
    residual: np.ndarray
    explained_variance: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self.components)

    @property
    def total_variance_explained(self) -> float:
        """Total explained variance."""
        return sum(self.explained_variance)


@dataclass
class SpectralFeatures:
    """Comprehensive spectral feature set.

    Args:
        centroid: Spectral centroid frequency.
        bandwidth: Spectral bandwidth.
        rolloff: Spectral rolloff frequency.
        flatness: Spectral flatness (tonality).
        flux: Spectral flux (change rate).
        contrast: Spectral contrast.
        peaks: Peak frequencies and amplitudes.
        crest_factor: Crest factor.
        skewness: Spectral skewness.
        kurtosis: Spectral kurtosis.
        entropy: Spectral entropy.
        slope: Spectral slope.
        spread: Spectral spread.
        decrease: Spectral decrease.
    """
    centroid: float = 0.0
    bandwidth: float = 0.0
    rolloff: float = 0.0
    flatness: float = 0.0
    flux: float = 0.0
    contrast: float = 0.0
    peaks: List[Tuple[int, float]] = field(default_factory=list)
    crest_factor: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    entropy: float = 0.0
    slope: float = 0.0
    spread: float = 0.0
    decrease: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.centroid, self.bandwidth, self.rolloff,
            self.flatness, self.flux, self.contrast,
            self.crest_factor, self.skewness, self.kurtosis,
            self.entropy, self.slope, self.spread, self.decrease,
        ])


@dataclass
class AnomalyReport:
    """Report of a detected spectral anomaly.

    Args:
        anomaly_type: Type of anomaly detected.
        severity: Anomaly severity (0-1).
        location: Frequency index/range of anomaly.
        description: Human-readable description.
        confidence: Confidence in detection.
        reference_value: Expected value.
        observed_value: Actual value.
    """
    anomaly_type: AnomalyType
    severity: float
    location: Tuple[int, int]
    description: str
    confidence: float = 0.0
    reference_value: float = 0.0
    observed_value: float = 0.0


# =============================================================================
# Adaptive Filter
# =============================================================================


class AdaptiveFilter:
    """Adaptive filtering for spectral signal processing.

    Implements multiple adaptive filtering algorithms (LMS, NLMS, RLS)
    for noise cancellation, system identification, and signal prediction.

    Args:
        filter_length: Number of filter taps.
        filter_type: Algorithm type.
        learning_rate: Step size (mu) for LMS/NLMS.
        forgetting_factor: Lambda for RLS.
    """

    def __init__(
        self,
        filter_length: int = 32,
        filter_type: FilterType = FilterType.NLMS,
        learning_rate: float = 0.01,
        forgetting_factor: float = 0.99,
    ) -> None:
        self.filter_length = filter_length
        self.filter_type = filter_type
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor

        self._weights = np.zeros(filter_length)
        self._P: Optional[np.ndarray] = None  # RLS covariance
        self._error_history: List[float] = []
        self._iterations: int = 0

        if filter_type == FilterType.RLS:
            self._P = np.eye(filter_length) * 100.0

    def adapt(self, input_signal: np.ndarray, desired: np.ndarray) -> np.ndarray:
        """Run adaptive filter on signals.

        Args:
            input_signal: Input/reference signal.
            desired: Desired output signal.

        Returns:
            Error signal (desired - output).
        """
        input_signal = np.atleast_1d(input_signal).flatten()
        desired = np.atleast_1d(desired).flatten()
        n_samples = min(len(input_signal), len(desired))

        error = np.zeros(n_samples)

        for i in range(self.filter_length, n_samples):
            x = input_signal[i - self.filter_length: i][::-1]
            y = np.dot(self._weights, x)
            e = desired[i] - y
            error[i] = e
            self._error_history.append(float(e))
            self._iterations += 1

            if self.filter_type == FilterType.LMS:
                self._weights += 2 * self.learning_rate * e * x
            elif self.filter_type == FilterType.NLMS:
                norm = np.dot(x, x) + 1e-8
                self._weights += (2 * self.learning_rate * e * x) / norm
            elif self.filter_type == FilterType.RLS:
                self._rls_update(x, e)

        return error

    def _rls_update(self, x: np.ndarray, e: float) -> None:
        """RLS weight update."""
        if self._P is None:
            return
        Px = self._P @ x
        denominator = self.forgetting_factor + x @ Px
        gain = Px / (denominator + 1e-12)
        self._weights += gain * e
        self._P = (self._P - np.outer(gain, x @ self._P)) / self.forgetting_factor

    def predict(self, input_signal: np.ndarray) -> np.ndarray:
        """Apply learned filter to input.

        Args:
            input_signal: Signal to filter.

        Returns:
            Filtered output.
        """
        input_signal = np.atleast_1d(input_signal).flatten()
        output = np.zeros(len(input_signal))

        for i in range(self.filter_length, len(input_signal)):
            x = input_signal[i - self.filter_length: i][::-1]
            output[i] = np.dot(self._weights, x)

        return output

    def get_state(self) -> FilterState:
        """Get current filter state."""
        recent_errors = self._error_history[-100:] if self._error_history else []
        rate = 0.0
        if len(recent_errors) > 10:
            first_half = np.mean(np.abs(recent_errors[:len(recent_errors)//2]))
            second_half = np.mean(np.abs(recent_errors[len(recent_errors)//2:]))
            rate = float((first_half - second_half) / (first_half + 1e-12))

        return FilterState(
            weights=self._weights.copy(),
            error_history=recent_errors,
            convergence_rate=rate,
            iterations=self._iterations,
        )

    def reset(self) -> None:
        """Reset filter state."""
        self._weights = np.zeros(self.filter_length)
        self._error_history = []
        self._iterations = 0
        if self.filter_type == FilterType.RLS:
            self._P = np.eye(self.filter_length) * 100.0

    @property
    def weights(self) -> np.ndarray:
        """Current filter weights."""
        return self._weights.copy()

    @property
    def converged(self) -> bool:
        """Whether the filter has converged."""
        if len(self._error_history) < 50:
            return False
        recent = np.abs(self._error_history[-50:])
        return bool(np.std(recent) < 0.01 * (np.mean(recent) + 1e-12))


# =============================================================================
# Spectral Decomposer
# =============================================================================


class SpectralDecomposer:
    """Advanced spectral decomposition engine.

    Decomposes signals into meaningful components using various
    methods: EMD, wavelet packets, SVD, and PCA.

    Args:
        max_components: Maximum components to extract.
        energy_threshold: Minimum energy fraction to retain.
    """

    def __init__(
        self,
        max_components: int = 10,
        energy_threshold: float = 0.01,
    ) -> None:
        self.max_components = max_components
        self.energy_threshold = energy_threshold

    def decompose(
        self,
        signal: np.ndarray,
        method: DecompositionMethod = DecompositionMethod.EMD,
    ) -> DecompositionResult:
        """Decompose signal into components.

        Args:
            signal: Input signal.
            method: Decomposition method.

        Returns:
            DecompositionResult with components and metadata.
        """
        signal = np.atleast_1d(signal).flatten()

        if method == DecompositionMethod.EMD:
            return self._emd_decompose(signal)
        elif method == DecompositionMethod.SVD:
            return self._svd_decompose(signal)
        elif method == DecompositionMethod.PCA:
            return self._pca_decompose(signal)
        elif method == DecompositionMethod.WAVELET:
            return self._wavelet_decompose(signal)
        elif method == DecompositionMethod.NMF:
            return self._nmf_decompose(signal)
        else:
            return self._emd_decompose(signal)

    def _emd_decompose(self, signal: np.ndarray) -> DecompositionResult:
        """Empirical Mode Decomposition (simplified).

        Iteratively extracts intrinsic mode functions (IMFs).
        """
        components = []
        residual = signal.copy()
        total_energy = np.sum(signal ** 2)

        for _ in range(self.max_components):
            if np.sum(residual ** 2) < self.energy_threshold * total_energy:
                break

            # Sifting: extract IMF
            imf = self._sift(residual)
            if np.sum(imf ** 2) < self.energy_threshold * total_energy:
                break

            components.append(imf)
            residual = residual - imf

        # Compute explained variance
        explained = []
        for comp in components:
            explained.append(float(np.sum(comp ** 2) / (total_energy + 1e-12)))

        return DecompositionResult(
            components=components,
            residual=residual,
            explained_variance=explained,
            metadata={"method": "emd", "n_siftings": len(components)},
        )

    def _sift(self, signal: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """Sifting process for EMD."""
        h = signal.copy()
        n = len(signal)

        for _ in range(max_iter):
            # Find local maxima and minima
            maxima_idx = []
            minima_idx = []
            for i in range(1, n - 1):
                if h[i] > h[i - 1] and h[i] > h[i + 1]:
                    maxima_idx.append(i)
                elif h[i] < h[i - 1] and h[i] < h[i + 1]:
                    minima_idx.append(i)

            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break

            # Compute envelopes via linear interpolation
            upper = np.interp(np.arange(n), maxima_idx, h[maxima_idx])
            lower = np.interp(np.arange(n), minima_idx, h[minima_idx])
            mean_envelope = (upper + lower) / 2

            h = h - mean_envelope

            # Check if IMF criteria met (simplified)
            if np.sum(mean_envelope ** 2) < 0.01 * np.sum(h ** 2):
                break

        return h

    def _svd_decompose(self, signal: np.ndarray) -> DecompositionResult:
        """SVD-based decomposition using Hankel matrix."""
        n = len(signal)
        window = min(n // 2, 64)
        n_cols = n - window + 1

        # Build Hankel matrix
        H = np.zeros((window, n_cols))
        for i in range(window):
            H[i, :] = signal[i: i + n_cols]

        # SVD
        U, s, Vh = np.linalg.svd(H, full_matrices=False)

        # Extract components
        total_energy = np.sum(s ** 2)
        components = []
        explained = []

        for k in range(min(self.max_components, len(s))):
            if s[k] ** 2 < self.energy_threshold * total_energy:
                break

            # Reconstruct component
            H_k = s[k] * np.outer(U[:, k], Vh[k, :])
            # Average anti-diagonals to get time series
            comp = self._hankel_to_signal(H_k, n)
            components.append(comp)
            explained.append(float(s[k] ** 2 / total_energy))

        residual = signal - sum(components) if components else signal.copy()

        return DecompositionResult(
            components=components,
            residual=residual,
            explained_variance=explained,
            metadata={"method": "svd", "singular_values": s[:10].tolist()},
        )

    def _pca_decompose(self, signal: np.ndarray) -> DecompositionResult:
        """PCA-based decomposition using sliding windows."""
        n = len(signal)
        window = min(n // 4, 32)
        n_windows = n - window + 1

        # Build data matrix
        X = np.zeros((n_windows, window))
        for i in range(n_windows):
            X[i, :] = signal[i: i + window]

        # Center
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Covariance and eigendecomposition
        cov = (X_centered.T @ X_centered) / (n_windows - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract components
        total_var = np.sum(eigenvalues)
        components = []
        explained = []

        for k in range(min(self.max_components, window)):
            if eigenvalues[k] < self.energy_threshold * total_var:
                break

            # Project onto component
            scores = X_centered @ eigenvectors[:, k]
            # Reconstruct signal from this component
            reconstructed = np.zeros(n)
            counts = np.zeros(n)
            for i in range(n_windows):
                reconstructed[i: i + window] += scores[i] * eigenvectors[:, k]
                counts[i: i + window] += 1
            counts = np.maximum(counts, 1)
            reconstructed /= counts

            components.append(reconstructed)
            explained.append(float(eigenvalues[k] / total_var))

        residual = signal - sum(components) if components else signal.copy()

        return DecompositionResult(
            components=components,
            residual=residual,
            explained_variance=explained,
            metadata={"method": "pca", "eigenvalues": eigenvalues[:10].tolist()},
        )

    def _wavelet_decompose(self, signal: np.ndarray) -> DecompositionResult:
        """Wavelet packet decomposition (simplified Haar)."""
        components = []
        current = signal.copy()
        total_energy = np.sum(signal ** 2)

        for level in range(min(self.max_components, 8)):
            if len(current) < 4:
                break

            n = len(current) - (len(current) % 2)
            half = n // 2

            # Haar wavelet: approximation and detail
            approx = (current[:n:2] + current[1:n:2]) / np.sqrt(2)
            detail = (current[:n:2] - current[1:n:2]) / np.sqrt(2)

            # Upsample detail back to original length
            detail_full = np.zeros(len(signal))
            step = max(1, len(signal) // max(1, len(detail)))
            for i, d in enumerate(detail):
                idx = i * step
                if idx < len(detail_full):
                    detail_full[idx] = d

            components.append(detail_full)
            current = approx

        # Compute explained variance
        explained = []
        for comp in components:
            explained.append(float(np.sum(comp ** 2) / (total_energy + 1e-12)))

        # Residual is the final approximation
        residual_full = np.zeros(len(signal))
        step = max(1, len(signal) // max(1, len(current)))
        for i, val in enumerate(current):
            idx = i * step
            if idx < len(residual_full):
                residual_full[idx] = val

        return DecompositionResult(
            components=components,
            residual=residual_full,
            explained_variance=explained,
            metadata={"method": "wavelet", "levels": len(components)},
        )

    def _nmf_decompose(self, signal: np.ndarray) -> DecompositionResult:
        """Non-negative Matrix Factorization decomposition."""
        # Ensure non-negative
        signal_nn = np.maximum(signal, 0)
        n = len(signal_nn)
        window = min(n // 4, 32)
        n_windows = max(1, n - window + 1)

        # Build non-negative matrix
        X = np.zeros((window, n_windows))
        for i in range(n_windows):
            X[:, i] = signal_nn[i: i + window]
        X = np.maximum(X, 1e-10)

        # NMF via multiplicative updates
        n_comp = min(self.max_components, window, n_windows)
        W = np.abs(np.random.randn(window, n_comp)) + 0.1
        H = np.abs(np.random.randn(n_comp, n_windows)) + 0.1

        for _ in range(50):
            # Update H
            numerator = W.T @ X
            denominator = W.T @ W @ H + 1e-10
            H *= numerator / denominator
            # Update W
            numerator = X @ H.T
            denominator = W @ H @ H.T + 1e-10
            W *= numerator / denominator

        # Extract components
        total_energy = np.sum(signal ** 2) + 1e-12
        components = []
        explained = []

        for k in range(n_comp):
            # Reconstruct component
            H_k = np.outer(W[:, k], H[k, :])
            comp = self._hankel_to_signal(H_k, n)
            comp_energy = np.sum(comp ** 2) / total_energy
            if comp_energy < self.energy_threshold:
                continue
            components.append(comp)
            explained.append(float(comp_energy))

        residual = signal - sum(components) if components else signal.copy()

        return DecompositionResult(
            components=components,
            residual=residual,
            explained_variance=explained,
            metadata={"method": "nmf", "n_iterations": 50},
        )

    def _hankel_to_signal(self, H: np.ndarray, length: int) -> np.ndarray:
        """Convert Hankel matrix back to signal by averaging anti-diagonals."""
        rows, cols = H.shape
        signal = np.zeros(length)
        counts = np.zeros(length)
        for i in range(rows):
            for j in range(cols):
                idx = i + j
                if idx < length:
                    signal[idx] += H[i, j]
                    counts[idx] += 1
        counts = np.maximum(counts, 1)
        return signal / counts


# =============================================================================
# Advanced Feature Extractor
# =============================================================================


class AdvancedFeatureExtractor:
    """Comprehensive spectral feature extraction.

    Computes a rich set of spectral features for classification,
    monitoring, and analysis applications.

    Args:
        n_peaks: Maximum number of peaks to detect.
        sample_rate: Assumed sample rate for frequency conversion.
    """

    def __init__(
        self,
        n_peaks: int = 10,
        sample_rate: float = 1.0,
    ) -> None:
        self.n_peaks = n_peaks
        self.sample_rate = sample_rate

    def extract(self, spectrum: np.ndarray) -> SpectralFeatures:
        """Extract comprehensive features from a spectrum.

        Args:
            spectrum: Input spectrum (magnitude or power).

        Returns:
            SpectralFeatures dataclass with all computed features.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        abs_spec = np.abs(spectrum)
        n = len(abs_spec)

        if n == 0:
            return SpectralFeatures()

        # Frequency axis
        freqs = np.arange(n) * self.sample_rate / n

        # Ensure positive for certain calculations
        spec_pos = abs_spec + 1e-12
        total = np.sum(spec_pos)

        # Spectral centroid
        centroid = float(np.sum(freqs * spec_pos) / total)

        # Spectral spread (bandwidth)
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec_pos) / total))

        # Spectral bandwidth (another definition)
        bandwidth = spread

        # Spectral rolloff (85%)
        cumsum = np.cumsum(spec_pos)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
        rolloff = float(freqs[min(rolloff_idx, n - 1)])

        # Spectral flatness
        log_spec = np.log(spec_pos)
        geo_mean = np.exp(np.mean(log_spec))
        arith_mean = np.mean(spec_pos)
        flatness = float(geo_mean / (arith_mean + 1e-12))

        # Crest factor
        crest = float(np.max(abs_spec) / (np.sqrt(np.mean(abs_spec ** 2)) + 1e-12))

        # Spectral skewness
        if spread > 0:
            skew = float(np.sum(((freqs - centroid) ** 3) * spec_pos) / (total * spread ** 3 + 1e-12))
        else:
            skew = 0.0

        # Spectral kurtosis
        if spread > 0:
            kurt = float(
                np.sum(((freqs - centroid) ** 4) * spec_pos)
                / (total * spread ** 4 + 1e-12)
                - 3.0
            )
        else:
            kurt = 0.0

        # Spectral entropy
        p = spec_pos / total
        entropy = float(-np.sum(p * np.log2(p + 1e-12)))

        # Spectral slope
        if n > 1:
            slope = float(np.polyfit(np.arange(n), abs_spec, 1)[0])
        else:
            slope = 0.0

        # Spectral decrease
        if n > 1:
            decrease = float(
                np.sum((abs_spec[1:] - abs_spec[0]) / (np.arange(1, n) + 1e-12))
                / (total - abs_spec[0] + 1e-12)
            )
        else:
            decrease = 0.0

        # Peak detection
        peaks = self._detect_peaks(abs_spec)

        # Spectral contrast
        contrast = self._compute_contrast(abs_spec)

        return SpectralFeatures(
            centroid=centroid,
            bandwidth=bandwidth,
            rolloff=rolloff,
            flatness=flatness,
            flux=0.0,  # Requires previous frame
            contrast=contrast,
            peaks=peaks,
            crest_factor=crest,
            skewness=skew,
            kurtosis=kurt,
            entropy=entropy,
            slope=slope,
            spread=spread,
            decrease=decrease,
        )

    def extract_temporal(
        self,
        spectra: List[np.ndarray],
    ) -> List[SpectralFeatures]:
        """Extract features from a sequence of spectra.

        Args:
            spectra: List of spectral frames.

        Returns:
            List of SpectralFeatures with flux computed.
        """
        features_list = []
        prev_spec = None

        for spectrum in spectra:
            f = self.extract(spectrum)

            # Compute flux if we have previous frame
            if prev_spec is not None:
                flux = float(np.sum(
                    (np.abs(spectrum) - np.abs(prev_spec)) ** 2
                ))
                f = SpectralFeatures(
                    centroid=f.centroid, bandwidth=f.bandwidth,
                    rolloff=f.rolloff, flatness=f.flatness,
                    flux=flux, contrast=f.contrast, peaks=f.peaks,
                    crest_factor=f.crest_factor, skewness=f.skewness,
                    kurtosis=f.kurtosis, entropy=f.entropy,
                    slope=f.slope, spread=f.spread, decrease=f.decrease,
                )

            features_list.append(f)
            prev_spec = spectrum

        return features_list

    def _detect_peaks(self, spectrum: np.ndarray) -> List[Tuple[int, float]]:
        """Detect spectral peaks."""
        n = len(spectrum)
        peaks = []
        threshold = np.mean(spectrum) + 0.5 * np.std(spectrum)

        for i in range(1, n - 1):
            if (spectrum[i] > spectrum[i - 1] and
                spectrum[i] > spectrum[i + 1] and
                spectrum[i] > threshold):
                peaks.append((i, float(spectrum[i])))

        # Sort by amplitude and limit
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:self.n_peaks]

    def _compute_contrast(self, spectrum: np.ndarray) -> float:
        """Compute spectral contrast (peak-valley ratio)."""
        n = len(spectrum)
        if n < 4:
            return 0.0

        # Divide into subbands
        n_bands = min(6, n // 4)
        band_size = n // n_bands
        contrasts = []

        for i in range(n_bands):
            band = spectrum[i * band_size: (i + 1) * band_size]
            if len(band) > 0:
                peak = np.max(band)
                valley = np.min(band)
                contrasts.append(peak - valley)

        return float(np.mean(contrasts)) if contrasts else 0.0


# =============================================================================
# Signal Synthesizer
# =============================================================================


class SignalSynthesizer:
    """Generate and reconstruct spectral signals.

    Provides signal synthesis from spectral descriptions,
    inverse transform capabilities, and augmentation.

    Args:
        default_length: Default signal length.
        sample_rate: Sample rate.
    """

    def __init__(
        self,
        default_length: int = 1024,
        sample_rate: float = 44100.0,
    ) -> None:
        self.default_length = default_length
        self.sample_rate = sample_rate

    def synthesize_harmonic(
        self,
        fundamental: float,
        n_harmonics: int = 5,
        amplitudes: Optional[np.ndarray] = None,
        length: Optional[int] = None,
    ) -> np.ndarray:
        """Synthesize a harmonic signal.

        Args:
            fundamental: Fundamental frequency in Hz.
            n_harmonics: Number of harmonics.
            amplitudes: Amplitude per harmonic (default: 1/k decay).
            length: Signal length.

        Returns:
            Synthesized signal.
        """
        n = length or self.default_length
        t = np.arange(n) / self.sample_rate

        if amplitudes is None:
            amplitudes = 1.0 / (np.arange(1, n_harmonics + 1))

        signal = np.zeros(n)
        for k in range(n_harmonics):
            amp = amplitudes[k] if k < len(amplitudes) else 0
            freq = fundamental * (k + 1)
            signal += amp * np.sin(2 * np.pi * freq * t)

        return signal

    def synthesize_from_peaks(
        self,
        peaks: List[Tuple[float, float, float]],
        length: Optional[int] = None,
    ) -> np.ndarray:
        """Synthesize signal from spectral peak descriptions.

        Args:
            peaks: List of (frequency, amplitude, phase) tuples.
            length: Signal length.

        Returns:
            Synthesized signal.
        """
        n = length or self.default_length
        t = np.arange(n) / self.sample_rate
        signal = np.zeros(n)

        for freq, amp, phase in peaks:
            signal += amp * np.sin(2 * np.pi * freq * t + phase)

        return signal

    def synthesize_noise(
        self,
        noise_type: str = "white",
        length: Optional[int] = None,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Synthesize colored noise.

        Args:
            noise_type: 'white', 'pink', 'brown', or 'blue'.
            length: Signal length.
            amplitude: Noise amplitude.

        Returns:
            Noise signal.
        """
        n = length or self.default_length

        if noise_type == "white":
            return amplitude * np.random.randn(n)

        # Generate in frequency domain
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
        freqs[0] = 1e-10  # Avoid division by zero

        # Random complex spectrum
        phases = np.random.uniform(0, 2 * np.pi, len(freqs))
        magnitudes = np.ones(len(freqs))

        if noise_type == "pink":
            magnitudes = 1.0 / np.sqrt(freqs)
        elif noise_type == "brown":
            magnitudes = 1.0 / freqs
        elif noise_type == "blue":
            magnitudes = np.sqrt(freqs)

        spectrum = magnitudes * np.exp(1j * phases)
        signal = np.fft.irfft(spectrum, n=n)

        # Normalize
        signal = amplitude * signal / (np.std(signal) + 1e-12)
        return signal

    def synthesize_chirp(
        self,
        f_start: float,
        f_end: float,
        length: Optional[int] = None,
        method: str = "linear",
    ) -> np.ndarray:
        """Synthesize a frequency sweep (chirp) signal.

        Args:
            f_start: Starting frequency.
            f_end: Ending frequency.
            length: Signal length.
            method: 'linear' or 'logarithmic'.

        Returns:
            Chirp signal.
        """
        n = length or self.default_length
        t = np.arange(n) / self.sample_rate
        T = n / self.sample_rate

        if method == "linear":
            phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t ** 2 / (2 * T))
        else:  # logarithmic
            k = (f_end / (f_start + 1e-12)) ** (1.0 / T)
            phase = 2 * np.pi * f_start * (k ** t - 1) / (np.log(k) + 1e-12)

        return np.sin(phase)

    def add_noise(
        self,
        signal: np.ndarray,
        snr_db: float = 20.0,
    ) -> np.ndarray:
        """Add noise to a signal at specified SNR.

        Args:
            signal: Clean signal.
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Noisy signal.
        """
        signal_power = np.mean(signal ** 2) + 1e-12
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise


# =============================================================================
# Spectral Quality Assessor
# =============================================================================


class SpectralQualityAssessor:
    """Assess quality of spectral measurements.

    Evaluates signal quality based on SNR, dynamic range,
    spectral resolution, and other metrics.

    Args:
        min_snr: Minimum acceptable SNR in dB.
        min_dynamic_range: Minimum dynamic range in dB.
    """

    def __init__(
        self,
        min_snr: float = 20.0,
        min_dynamic_range: float = 40.0,
    ) -> None:
        self.min_snr = min_snr
        self.min_dynamic_range = min_dynamic_range

    def assess(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """Comprehensive quality assessment of a spectrum.

        Args:
            spectrum: Input spectrum.

        Returns:
            Dictionary of quality metrics.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        abs_spec = np.abs(spectrum)

        # SNR estimation
        snr = self._estimate_snr(abs_spec)

        # Dynamic range
        dynamic_range = self._compute_dynamic_range(abs_spec)

        # Spectral resolution
        resolution = self._estimate_resolution(abs_spec)

        # Stationarity check
        stationarity = self._check_stationarity(spectrum)

        # Clipping detection
        clipping = self._detect_clipping(spectrum)

        # Overall quality score (0-1)
        quality = self._compute_quality_score(
            snr, dynamic_range, resolution, stationarity, clipping
        )

        return {
            "quality_score": quality,
            "snr_db": snr,
            "dynamic_range_db": dynamic_range,
            "spectral_resolution": resolution,
            "stationarity": stationarity,
            "clipping_detected": clipping,
            "n_samples": len(spectrum),
            "passes_threshold": quality > 0.5,
        }

    def _estimate_snr(self, spectrum: np.ndarray) -> float:
        """Estimate SNR from spectrum."""
        if len(spectrum) < 4:
            return 0.0

        sorted_spec = np.sort(spectrum)
        noise_floor = np.mean(sorted_spec[:len(sorted_spec) // 4])
        signal_level = np.mean(sorted_spec[-len(sorted_spec) // 4:])

        if noise_floor < 1e-12:
            return 60.0  # Very high SNR
        return float(20 * np.log10(signal_level / noise_floor))

    def _compute_dynamic_range(self, spectrum: np.ndarray) -> float:
        """Compute dynamic range in dB."""
        max_val = np.max(spectrum)
        min_val = np.min(spectrum[spectrum > 0]) if np.any(spectrum > 0) else 1e-12
        if min_val < 1e-12:
            return 120.0
        return float(20 * np.log10(max_val / min_val))

    def _estimate_resolution(self, spectrum: np.ndarray) -> float:
        """Estimate spectral resolution from peak width."""
        n = len(spectrum)
        max_idx = np.argmax(spectrum)
        peak_val = spectrum[max_idx]
        half_power = peak_val / np.sqrt(2)

        # Find -3dB points
        left = max_idx
        right = max_idx
        while left > 0 and spectrum[left] > half_power:
            left -= 1
        while right < n - 1 and spectrum[right] > half_power:
            right += 1

        width = right - left
        return float(width) / n if n > 0 else 0.0

    def _check_stationarity(self, spectrum: np.ndarray) -> float:
        """Check signal stationarity (0 = non-stationary, 1 = stationary)."""
        n = len(spectrum)
        if n < 8:
            return 1.0

        # Compare first and second half statistics
        half = n // 2
        mean1, std1 = np.mean(spectrum[:half]), np.std(spectrum[:half])
        mean2, std2 = np.mean(spectrum[half:]), np.std(spectrum[half:])

        mean_diff = abs(mean1 - mean2) / (abs(mean1) + abs(mean2) + 1e-12)
        std_diff = abs(std1 - std2) / (std1 + std2 + 1e-12)

        stationarity = 1.0 - min(1.0, mean_diff + std_diff)
        return float(stationarity)

    def _detect_clipping(self, spectrum: np.ndarray) -> bool:
        """Detect signal clipping."""
        max_val = np.max(np.abs(spectrum))
        if max_val < 1e-12:
            return False

        # Count samples near maximum
        near_max = np.sum(np.abs(spectrum) > 0.99 * max_val)
        return bool(near_max > 3)

    def _compute_quality_score(
        self,
        snr: float,
        dynamic_range: float,
        resolution: float,
        stationarity: float,
        clipping: bool,
    ) -> float:
        """Compute overall quality score."""
        score = 0.0

        # SNR contribution (0-0.3)
        snr_score = min(1.0, max(0.0, snr / self.min_snr))
        score += 0.3 * snr_score

        # Dynamic range (0-0.2)
        dr_score = min(1.0, max(0.0, dynamic_range / self.min_dynamic_range))
        score += 0.2 * dr_score

        # Resolution (0-0.2) - lower is better
        res_score = 1.0 - min(1.0, resolution * 10)
        score += 0.2 * max(0.0, res_score)

        # Stationarity (0-0.2)
        score += 0.2 * stationarity

        # Clipping penalty (-0.1)
        if clipping:
            score -= 0.1

        return float(max(0.0, min(1.0, score)))


# =============================================================================
# Anomaly Detector
# =============================================================================


class SpectralAnomalyDetector:
    """Detect anomalies in spectral data.

    Uses statistical and pattern-based methods to identify
    unusual spectral behavior relative to a baseline.

    Args:
        sensitivity: Detection sensitivity (0-1).
        baseline_window: Number of spectra for baseline computation.
    """

    def __init__(
        self,
        sensitivity: float = 0.7,
        baseline_window: int = 50,
    ) -> None:
        self.sensitivity = sensitivity
        self.baseline_window = baseline_window
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._history: List[np.ndarray] = []
        self._peak_history: List[List[Tuple[int, float]]] = []

    def update_baseline(self, spectrum: np.ndarray) -> None:
        """Update the baseline with a new spectrum.

        Args:
            spectrum: New spectrum measurement.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        self._history.append(spectrum)

        if len(self._history) > self.baseline_window:
            self._history = self._history[-self.baseline_window:]

        if len(self._history) >= 3:
            # Pad/truncate to consistent length
            max_len = max(len(s) for s in self._history)
            padded = np.zeros((len(self._history), max_len))
            for i, s in enumerate(self._history):
                padded[i, :len(s)] = s

            self._baseline_mean = np.mean(padded, axis=0)
            self._baseline_std = np.std(padded, axis=0) + 1e-12

    def detect(self, spectrum: np.ndarray) -> List[AnomalyReport]:
        """Detect anomalies in a spectrum against baseline.

        Args:
            spectrum: Spectrum to check.

        Returns:
            List of detected anomalies.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        anomalies = []

        if self._baseline_mean is None:
            return anomalies

        # Match lengths
        n = min(len(spectrum), len(self._baseline_mean))
        spec = spectrum[:n]
        mean = self._baseline_mean[:n]
        std = self._baseline_std[:n]

        # Z-score based detection
        threshold = 3.0 * (1.0 - self.sensitivity) + 1.0
        z_scores = np.abs(spec - mean) / std

        # Find anomalous regions
        anomalous = z_scores > threshold
        if np.any(anomalous):
            # Group contiguous anomalous regions
            regions = self._find_regions(anomalous)
            for start, end in regions:
                max_z = float(np.max(z_scores[start:end]))
                severity = min(1.0, max_z / (threshold * 3))

                # Classify anomaly type
                atype = self._classify_anomaly(spec, mean, start, end)
                anomalies.append(AnomalyReport(
                    anomaly_type=atype,
                    severity=severity,
                    location=(start, end),
                    description=f"{atype.value} at frequency bins {start}-{end}",
                    confidence=min(1.0, max_z / threshold),
                    reference_value=float(np.mean(mean[start:end])),
                    observed_value=float(np.mean(spec[start:end])),
                ))

        return anomalies

    def _find_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous True regions in a boolean mask."""
        regions = []
        in_region = False
        start = 0

        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False

        if in_region:
            regions.append((start, len(mask)))

        return regions

    def _classify_anomaly(
        self,
        spectrum: np.ndarray,
        baseline: np.ndarray,
        start: int,
        end: int,
    ) -> AnomalyType:
        """Classify the type of spectral anomaly."""
        region_spec = spectrum[start:end]
        region_base = baseline[start:end]

        mean_diff = np.mean(region_spec) - np.mean(region_base)

        # Check for amplitude change
        if abs(mean_diff) > 2 * np.std(baseline):
            return AnomalyType.AMPLITUDE_CHANGE

        # Check for new peak
        spec_peak = np.max(region_spec)
        base_peak = np.max(region_base)
        if spec_peak > 3 * base_peak:
            return AnomalyType.NEW_PEAK

        # Check for missing peak
        if base_peak > 3 * spec_peak:
            return AnomalyType.MISSING_PEAK

        # Check bandwidth change
        spec_width = np.sum(region_spec > 0.5 * spec_peak)
        base_width = np.sum(region_base > 0.5 * base_peak)
        if abs(spec_width - base_width) > 0.3 * base_width:
            return AnomalyType.BANDWIDTH_CHANGE

        return AnomalyType.AMPLITUDE_CHANGE

    @property
    def has_baseline(self) -> bool:
        """Whether a baseline has been established."""
        return self._baseline_mean is not None

    @property
    def baseline_samples(self) -> int:
        """Number of samples in the baseline."""
        return len(self._history)
