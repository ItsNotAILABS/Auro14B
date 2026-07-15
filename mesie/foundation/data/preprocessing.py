"""Spectral data preprocessing pipelines.

Provides preprocessing transformations for converting raw time-domain
signals into spectral representations suitable for tokenization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class WindowExtractor:
    """Extract overlapping windows from continuous signals.

    Attributes:
        window_size: Size of each window in samples.
        hop_size: Hop between consecutive windows.
        window_function: Window function type.
        center: Whether to center windows.
        pad_mode: Padding mode for edge windows.
    """

    def __init__(
        self,
        window_size: int = 1024,
        hop_size: int = 256,
        window_function: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        """Initialize window extractor.

        Args:
            window_size: Window size.
            hop_size: Hop size.
            window_function: Window function name.
            center: Whether to center windows.
            pad_mode: Padding mode.
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_function = window_function
        self.center = center
        self.pad_mode = pad_mode

        # Precompute window
        self.window = self._create_window()

    def _create_window(self) -> np.ndarray:
        """Create window function."""
        n = self.window_size
        if self.window_function == "hann":
            return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
        elif self.window_function == "hamming":
            return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / n)
        elif self.window_function == "blackman":
            a0, a1, a2 = 0.42, 0.5, 0.08
            return a0 - a1 * np.cos(2 * np.pi * np.arange(n) / n) + \
                a2 * np.cos(4 * np.pi * np.arange(n) / n)
        elif self.window_function == "kaiser":
            beta = 8.6
            return np.kaiser(n, beta)
        elif self.window_function == "flat_top":
            a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
            w = np.zeros(n)
            for i, ai in enumerate(a):
                w += ai * np.cos(2 * np.pi * i * np.arange(n) / n) * ((-1) ** i)
            return w
        else:  # rectangular
            return np.ones(n)

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extract windowed frames from signal.

        Args:
            signal: Input signal [..., length].

        Returns:
            Windowed frames [..., num_frames, window_size].
        """
        length = signal.shape[-1]

        # Pad if centering
        if self.center:
            pad_len = self.window_size // 2
            if signal.ndim == 1:
                signal = np.pad(signal, (pad_len, pad_len), mode=self.pad_mode)
            else:
                pad_width = [(0, 0)] * (signal.ndim - 1) + [(pad_len, pad_len)]
                signal = np.pad(signal, pad_width, mode=self.pad_mode)
            length = signal.shape[-1]

        # Compute number of frames
        num_frames = (length - self.window_size) // self.hop_size + 1

        # Extract frames
        frames = []
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.window_size
            frame = signal[..., start:end] * self.window
            frames.append(frame)

        return np.stack(frames, axis=-2)

    def reconstruct(self, frames: np.ndarray, length: int) -> np.ndarray:
        """Reconstruct signal from windowed frames via overlap-add.

        Args:
            frames: Windowed frames [..., num_frames, window_size].
            length: Target output length.

        Returns:
            Reconstructed signal [..., length].
        """
        num_frames = frames.shape[-2]
        output_len = (num_frames - 1) * self.hop_size + self.window_size
        batch_shape = frames.shape[:-2]

        output = np.zeros((*batch_shape, output_len))
        window_sum = np.zeros(output_len)

        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.window_size
            output[..., start:end] += frames[..., i, :] * self.window
            window_sum[start:end] += self.window ** 2

        # Normalize
        window_sum = np.maximum(window_sum, 1e-10)
        output = output / window_sum

        if self.center:
            pad_len = self.window_size // 2
            output = output[..., pad_len:pad_len + length]

        return output[..., :length]


class FrequencyTransform:
    """Frequency-domain transformation for spectral analysis.

    Provides various spectral analysis methods including FFT, STFT,
    power spectral density, and mel-spectrogram computation.

    Attributes:
        n_fft: FFT size.
        hop_length: Hop length for STFT.
        window_size: Analysis window size.
        window_function: Window function.
        output_type: Type of spectral output.
        log_scale: Whether to apply log scaling.
        power: Power for power spectrum (1=amplitude, 2=power).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        window_size: Optional[int] = None,
        window_function: str = "hann",
        output_type: str = "magnitude",
        log_scale: bool = True,
        power: float = 2.0,
        normalize: bool = True,
        n_mels: int = 0,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        sampling_rate: float = 22050.0,
    ):
        """Initialize frequency transform.

        Args:
            n_fft: FFT size.
            hop_length: Hop length.
            window_size: Window size (default: n_fft).
            window_function: Window function name.
            output_type: Output type ('magnitude', 'power', 'complex', 'psd').
            log_scale: Whether to apply log scaling.
            power: Power exponent.
            normalize: Whether to normalize output.
            n_mels: Number of mel bands (0 = no mel).
            fmin: Minimum frequency for mel.
            fmax: Maximum frequency for mel.
            sampling_rate: Sampling rate for mel computation.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size or n_fft
        self.window_function = window_function
        self.output_type = output_type
        self.log_scale = log_scale
        self.power = power
        self.normalize = normalize
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sampling_rate / 2.0
        self.sampling_rate = sampling_rate

        # Window
        self.window_extractor = WindowExtractor(
            window_size=self.window_size,
            hop_size=hop_length,
            window_function=window_function,
        )

        # Mel filterbank (if using mel)
        if n_mels > 0:
            self.mel_filterbank = self._create_mel_filterbank()
        else:
            self.mel_filterbank = None

    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        """Convert Mel scale to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel-scale filterbank matrix."""
        n_freqs = self.n_fft // 2 + 1

        # Mel-spaced center frequencies
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)
        mel_centers = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_centers = np.array([self._mel_to_hz(m) for m in mel_centers])

        # Convert to FFT bin indices
        freq_bins = np.floor(hz_centers * self.n_fft / self.sampling_rate).astype(int)

        # Create triangular filters
        filterbank = np.zeros((self.n_mels, n_freqs))
        for i in range(self.n_mels):
            left = freq_bins[i]
            center = freq_bins[i + 1]
            right = freq_bins[i + 2]

            # Rising slope
            for j in range(left, center):
                if j < n_freqs and center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, right):
                if j < n_freqs and right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def stft(self, signal: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform.

        Args:
            signal: Input signal [..., length].

        Returns:
            Complex STFT [..., num_frames, n_fft//2 + 1].
        """
        frames = self.window_extractor.extract(signal)

        # Zero-pad to n_fft if needed
        if self.window_size < self.n_fft:
            pad_len = self.n_fft - self.window_size
            frames = np.pad(
                frames, [(0, 0)] * (frames.ndim - 1) + [(0, pad_len)]
            )

        # FFT
        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        return spectrum

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply full spectral transformation.

        Args:
            signal: Input time-domain signal [..., length].

        Returns:
            Spectral representation.
        """
        # Compute STFT
        spectrum = self.stft(signal)

        # Convert to requested output type
        if self.output_type == "complex":
            output = spectrum
        elif self.output_type == "magnitude":
            output = np.abs(spectrum)
        elif self.output_type == "power":
            output = np.abs(spectrum) ** self.power
        elif self.output_type == "psd":
            output = np.abs(spectrum) ** 2 / self.n_fft
        else:
            output = np.abs(spectrum)

        # Apply mel filterbank
        if self.mel_filterbank is not None and self.output_type != "complex":
            output = np.einsum("...f,mf->...m", output, self.mel_filterbank)

        # Log scale
        if self.log_scale and self.output_type != "complex":
            output = np.log(output + 1e-10)

        # Normalize
        if self.normalize and self.output_type != "complex":
            mean = np.mean(output, axis=-1, keepdims=True)
            std = np.std(output, axis=-1, keepdims=True) + 1e-10
            output = (output - mean) / std

        return output

    def inverse_transform(
        self, spectrum: np.ndarray, length: Optional[int] = None
    ) -> np.ndarray:
        """Inverse spectral transform (Griffin-Lim for magnitude).

        Args:
            spectrum: Spectral representation.
            length: Target output length.

        Returns:
            Reconstructed time-domain signal.
        """
        if self.output_type == "complex":
            # Direct inverse STFT
            frames = np.fft.irfft(spectrum, n=self.n_fft, axis=-1)
            frames = frames[..., :self.window_size]
            out_len = length or (frames.shape[-2] - 1) * self.hop_length + self.window_size
            return self.window_extractor.reconstruct(frames, out_len)

        # For magnitude: use Griffin-Lim algorithm
        if self.log_scale:
            magnitude = np.exp(spectrum)
        else:
            magnitude = spectrum

        if self.power > 1:
            magnitude = magnitude ** (1.0 / self.power)

        # Griffin-Lim iterations
        n_iter = 32
        phase = np.exp(2j * np.pi * np.random.random(magnitude.shape))
        complex_spec = magnitude * phase

        for _ in range(n_iter):
            frames = np.fft.irfft(complex_spec, n=self.n_fft, axis=-1)
            frames = frames[..., :self.window_size]
            recomputed = np.fft.rfft(frames, n=self.n_fft, axis=-1)
            phase = np.exp(1j * np.angle(recomputed))
            complex_spec = magnitude * phase

        frames = np.fft.irfft(complex_spec, n=self.n_fft, axis=-1)
        frames = frames[..., :self.window_size]
        out_len = length or (frames.shape[-2] - 1) * self.hop_length + self.window_size
        return self.window_extractor.reconstruct(frames, out_len)


class NormalizationPipeline:
    """Multi-stage normalization for spectral data.

    Provides various normalization strategies that can be composed
    into a pipeline for consistent data preparation.

    Attributes:
        methods: List of normalization methods to apply.
        per_channel: Whether to normalize per channel.
        clip_value: Maximum absolute value for clipping.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        per_channel: bool = True,
        clip_value: float = 10.0,
        eps: float = 1e-8,
    ):
        """Initialize normalization pipeline.

        Args:
            methods: List of methods ('zscore', 'minmax', 'rms', 'peak', 'percentile').
            per_channel: Whether to normalize channels independently.
            clip_value: Clip amplitude to this value.
            eps: Epsilon for numerical stability.
        """
        self.methods = methods or ["zscore"]
        self.per_channel = per_channel
        self.clip_value = clip_value
        self.eps = eps

        # Running statistics for online normalization
        self._running_mean = None
        self._running_var = None
        self._count = 0

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        if self.per_channel and x.ndim > 1:
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True) + self.eps
        else:
            mean = np.mean(x)
            std = np.std(x) + self.eps
        return (x - mean) / std

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        if self.per_channel and x.ndim > 1:
            x_min = np.min(x, axis=-1, keepdims=True)
            x_max = np.max(x, axis=-1, keepdims=True)
        else:
            x_min = np.min(x)
            x_max = np.max(x)
        return (x - x_min) / (x_max - x_min + self.eps)

    def _rms(self, x: np.ndarray) -> np.ndarray:
        """RMS normalization."""
        if self.per_channel and x.ndim > 1:
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        else:
            rms = np.sqrt(np.mean(x ** 2) + self.eps)
        return x / rms

    def _peak(self, x: np.ndarray) -> np.ndarray:
        """Peak normalization to [-1, 1]."""
        if self.per_channel and x.ndim > 1:
            peak = np.max(np.abs(x), axis=-1, keepdims=True) + self.eps
        else:
            peak = np.max(np.abs(x)) + self.eps
        return x / peak

    def _percentile(self, x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
        """Percentile-based normalization (robust to outliers)."""
        p_low = np.percentile(x, low)
        p_high = np.percentile(x, high)
        x_clipped = np.clip(x, p_low, p_high)
        return (x_clipped - p_low) / (p_high - p_low + self.eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization pipeline.

        Args:
            x: Input array.

        Returns:
            Normalized array.
        """
        result = x.copy()

        for method in self.methods:
            if method == "zscore":
                result = self._zscore(result)
            elif method == "minmax":
                result = self._minmax(result)
            elif method == "rms":
                result = self._rms(result)
            elif method == "peak":
                result = self._peak(result)
            elif method == "percentile":
                result = self._percentile(result)
            elif method == "clip":
                result = np.clip(result, -self.clip_value, self.clip_value)

        # Final clipping
        result = np.clip(result, -self.clip_value, self.clip_value)

        # Handle NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)

        return result

    def update_statistics(self, x: np.ndarray) -> None:
        """Update running statistics for online normalization.

        Args:
            x: New data batch.
        """
        batch_mean = np.mean(x)
        batch_var = np.var(x)

        if self._running_mean is None:
            self._running_mean = batch_mean
            self._running_var = batch_var
        else:
            momentum = 0.1
            self._running_mean = (1 - momentum) * self._running_mean + momentum * batch_mean
            self._running_var = (1 - momentum) * self._running_var + momentum * batch_var

        self._count += 1


class ArtifactRemoval:
    """Artifact detection and removal for spectral data.

    Identifies and removes or attenuates common artifacts in
    spectral data across different modalities.

    Attributes:
        methods: List of artifact removal methods.
        threshold: Detection threshold.
        replacement: How to replace detected artifacts.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        threshold: float = 5.0,
        replacement: str = "interpolate",
        max_artifact_ratio: float = 0.3,
    ):
        """Initialize artifact removal.

        Args:
            methods: Removal methods ('clip', 'interpolate', 'zero', 'median').
            threshold: Z-score threshold for detection.
            replacement: Replacement strategy.
            max_artifact_ratio: Maximum ratio of data that can be artifacts.
        """
        self.methods = methods or ["clip", "interpolate"]
        self.threshold = threshold
        self.replacement = replacement
        self.max_artifact_ratio = max_artifact_ratio

    def detect_artifacts(self, x: np.ndarray) -> np.ndarray:
        """Detect artifacts in signal.

        Args:
            x: Input signal.

        Returns:
            Boolean mask (True = artifact).
        """
        # Z-score based detection
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-10
        z_scores = np.abs((x - mean) / std)
        artifacts = z_scores > self.threshold

        # Limit artifact detection
        artifact_ratio = np.mean(artifacts)
        if artifact_ratio > self.max_artifact_ratio:
            # Raise threshold
            sorted_z = np.sort(z_scores.flatten())[::-1]
            max_artifacts = int(self.max_artifact_ratio * z_scores.size)
            new_threshold = sorted_z[max_artifacts] if max_artifacts < len(sorted_z) else self.threshold
            artifacts = z_scores > new_threshold

        return artifacts

    def remove_artifacts(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove detected artifacts.

        Args:
            x: Input signal.

        Returns:
            Tuple of (cleaned_signal, artifact_info).
        """
        artifacts = self.detect_artifacts(x)
        result = x.copy()
        artifact_count = int(np.sum(artifacts))

        if artifact_count == 0:
            return result, {"artifacts_detected": 0, "artifact_ratio": 0.0}

        if self.replacement == "zero":
            result[artifacts] = 0.0
        elif self.replacement == "clip":
            mean = np.mean(x[~artifacts]) if np.any(~artifacts) else 0.0
            std = np.std(x[~artifacts]) if np.any(~artifacts) else 1.0
            upper = mean + self.threshold * std
            lower = mean - self.threshold * std
            result = np.clip(result, lower, upper)
        elif self.replacement == "interpolate":
            # Linear interpolation over artifacts
            flat = result.flatten()
            artifact_flat = artifacts.flatten()
            clean_indices = np.where(~artifact_flat)[0]
            artifact_indices = np.where(artifact_flat)[0]

            if len(clean_indices) > 1 and len(artifact_indices) > 0:
                flat[artifact_indices] = np.interp(
                    artifact_indices, clean_indices, flat[clean_indices]
                )
            result = flat.reshape(x.shape)
        elif self.replacement == "median":
            # Replace with local median
            window = min(11, x.shape[-1])
            for i in range(x.shape[-1]):
                if artifacts.flatten()[i] if artifacts.ndim == 1 else artifacts[..., i].any():
                    start = max(0, i - window // 2)
                    end = min(x.shape[-1], i + window // 2 + 1)
                    local = x[..., start:end]
                    result[..., i] = np.median(local, axis=-1)

        info = {
            "artifacts_detected": artifact_count,
            "artifact_ratio": float(np.mean(artifacts)),
            "replacement_method": self.replacement,
        }

        return result, info


class SpectralPreprocessor:
    """Complete preprocessing pipeline for spectral data.

    Combines windowing, frequency transformation, normalization,
    and artifact removal into a single configurable pipeline.

    Attributes:
        window_extractor: Window extraction component.
        freq_transform: Frequency transform component.
        normalizer: Normalization component.
        artifact_removal: Artifact removal component.
        detrend: Whether to detrend signals.
        taper: Whether to apply tapering.
    """

    def __init__(
        self,
        sampling_rate: float = 100.0,
        window_size: int = 1024,
        hop_size: int = 256,
        n_fft: int = 1024,
        window_function: str = "hann",
        output_type: str = "magnitude",
        log_scale: bool = True,
        normalization: str = "zscore",
        detrend: bool = True,
        remove_artifacts: bool = True,
        artifact_threshold: float = 5.0,
        n_mels: int = 0,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        """Initialize preprocessor.

        Args:
            sampling_rate: Signal sampling rate.
            window_size: Analysis window size.
            hop_size: Hop size.
            n_fft: FFT size.
            window_function: Window function.
            output_type: Spectral output type.
            log_scale: Whether to use log scale.
            normalization: Normalization method.
            detrend: Whether to detrend.
            remove_artifacts: Whether to remove artifacts.
            artifact_threshold: Artifact detection threshold.
            n_mels: Number of mel bands.
            fmin: Minimum frequency.
            fmax: Maximum frequency.
        """
        self.sampling_rate = sampling_rate
        self.detrend = detrend
        self.remove_artifacts = remove_artifacts

        self.window_extractor = WindowExtractor(
            window_size=window_size,
            hop_size=hop_size,
            window_function=window_function,
        )

        self.freq_transform = FrequencyTransform(
            n_fft=n_fft,
            hop_length=hop_size,
            window_size=window_size,
            window_function=window_function,
            output_type=output_type,
            log_scale=log_scale,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            sampling_rate=sampling_rate,
        )

        self.normalizer = NormalizationPipeline(
            methods=[normalization, "clip"],
        )

        self.artifact_removal = ArtifactRemoval(
            threshold=artifact_threshold,
        )

    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply full preprocessing pipeline.

        Args:
            signal: Raw input signal.

        Returns:
            Tuple of (processed_spectrum, processing_info).
        """
        info: Dict[str, Any] = {"original_shape": signal.shape}

        # 1. Detrend
        if self.detrend:
            if signal.ndim == 1:
                mean = np.mean(signal)
                signal = signal - mean
                # Linear detrend
                x = np.arange(len(signal))
                slope = np.polyfit(x, signal, 1)[0]
                signal = signal - slope * x
            else:
                for ch in range(signal.shape[0]):
                    signal[ch] -= np.mean(signal[ch])

        # 2. Artifact removal
        if self.remove_artifacts:
            signal, artifact_info = self.artifact_removal.remove_artifacts(signal)
            info["artifact_info"] = artifact_info

        # 3. Frequency transform
        spectrum = self.freq_transform.transform(signal)
        info["spectrum_shape"] = spectrum.shape

        # 4. Normalization
        spectrum = self.normalizer.normalize(spectrum)

        info["output_range"] = (float(np.min(spectrum)), float(np.max(spectrum)))

        return spectrum, info
