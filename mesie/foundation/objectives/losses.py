"""Core loss functions for spectral pretraining.

Implements reconstruction, frequency-band, multi-scale,
and perceptual losses optimized for spectral data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SpectralReconstructionLoss:
    """Reconstruction loss for spectral signals.

    Combines time-domain and frequency-domain reconstruction
    errors with configurable weighting. Supports multiple
    error metrics optimized for spectral fidelity.

    Attributes:
        time_weight: Weight for time-domain loss.
        freq_weight: Weight for frequency-domain loss.
        phase_weight: Weight for phase error.
        metric: Error metric ('mse', 'mae', 'huber', 'log_cosh').
    """

    def __init__(
        self,
        time_weight: float = 1.0,
        freq_weight: float = 1.0,
        phase_weight: float = 0.5,
        metric: str = "mse",
        huber_delta: float = 1.0,
        weighted_frequencies: bool = True,
        perceptual_weighting: str = "a_weighting",
    ):
        """Initialize spectral reconstruction loss.

        Args:
            time_weight: Time-domain loss weight.
            freq_weight: Frequency-domain loss weight.
            phase_weight: Phase reconstruction weight.
            metric: Base error metric.
            huber_delta: Delta for Huber loss.
            weighted_frequencies: Use frequency-dependent weighting.
            perceptual_weighting: Perceptual weighting curve.
        """
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.phase_weight = phase_weight
        self.metric = metric
        self.huber_delta = huber_delta
        self.weighted_frequencies = weighted_frequencies
        self.perceptual_weighting = perceptual_weighting

    def compute(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        sample_rate: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute spectral reconstruction loss.

        Args:
            prediction: Predicted signal [B, T] or [B, T, C].
            target: Target signal (same shape).
            sample_rate: Sampling rate for frequency weighting.

        Returns:
            Total loss and component losses.
        """
        losses: Dict[str, float] = {}

        # Time-domain loss
        if self.time_weight > 0:
            time_loss = self._compute_error(prediction, target)
            losses["time_domain"] = time_loss

        # Frequency-domain loss
        if self.freq_weight > 0:
            freq_loss = self._frequency_loss(prediction, target, sample_rate)
            losses["frequency_domain"] = freq_loss

        # Phase loss
        if self.phase_weight > 0:
            phase_loss = self._phase_loss(prediction, target)
            losses["phase"] = phase_loss

        # Weighted total
        total = (
            self.time_weight * losses.get("time_domain", 0.0) +
            self.freq_weight * losses.get("frequency_domain", 0.0) +
            self.phase_weight * losses.get("phase", 0.0)
        )
        losses["total"] = total

        return total, losses

    def _compute_error(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute base error metric."""
        diff = pred - target

        if self.metric == "mse":
            return float(np.mean(diff ** 2))
        elif self.metric == "mae":
            return float(np.mean(np.abs(diff)))
        elif self.metric == "huber":
            abs_diff = np.abs(diff)
            quadratic = np.minimum(abs_diff, self.huber_delta)
            linear = abs_diff - quadratic
            return float(np.mean(0.5 * quadratic ** 2 + self.huber_delta * linear))
        elif self.metric == "log_cosh":
            return float(np.mean(np.log(np.cosh(diff + 1e-12))))
        else:
            return float(np.mean(diff ** 2))

    def _frequency_loss(
        self, pred: np.ndarray, target: np.ndarray, sample_rate: float
    ) -> float:
        """Compute frequency-domain loss."""
        # FFT of prediction and target
        pred_fft = np.fft.rfft(pred, axis=-1 if pred.ndim <= 2 else -2)
        target_fft = np.fft.rfft(target, axis=-1 if target.ndim <= 2 else -2)

        # Magnitude spectrum loss
        pred_mag = np.abs(pred_fft)
        target_mag = np.abs(target_fft)

        # Log-magnitude (more perceptually relevant)
        pred_log_mag = np.log(pred_mag + 1e-8)
        target_log_mag = np.log(target_mag + 1e-8)

        # Apply frequency weighting
        if self.weighted_frequencies:
            n_freqs = pred_log_mag.shape[-1]
            weights = self._get_frequency_weights(n_freqs, sample_rate)
            # Broadcast weights
            weight_shape = [1] * (pred_log_mag.ndim - 1) + [n_freqs]
            weights = weights.reshape(weight_shape)
            mag_loss = float(np.mean(weights * (pred_log_mag - target_log_mag) ** 2))
        else:
            mag_loss = float(np.mean((pred_log_mag - target_log_mag) ** 2))

        return mag_loss

    def _phase_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute phase reconstruction loss."""
        pred_fft = np.fft.rfft(pred, axis=-1 if pred.ndim <= 2 else -2)
        target_fft = np.fft.rfft(target, axis=-1 if target.ndim <= 2 else -2)

        pred_phase = np.angle(pred_fft)
        target_phase = np.angle(target_fft)

        # Phase difference (handle wraparound)
        phase_diff = pred_phase - target_phase
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Weight by magnitude (phase matters more for strong components)
        target_mag = np.abs(target_fft)
        weighted_phase_error = target_mag * np.abs(phase_diff)

        return float(np.mean(weighted_phase_error))

    def _get_frequency_weights(self, n_freqs: int, sample_rate: float) -> np.ndarray:
        """Get perceptual frequency weighting.

        Args:
            n_freqs: Number of frequency bins.
            sample_rate: Sampling rate.

        Returns:
            Frequency weights array.
        """
        freqs = np.linspace(0, sample_rate / 2, n_freqs)

        if self.perceptual_weighting == "a_weighting":
            # Simplified A-weighting (for audio-like signals)
            # Emphasizes mid-frequencies
            f_sq = freqs ** 2
            weights = (12194 ** 2 * f_sq ** 2) / (
                (f_sq + 20.6 ** 2) *
                np.sqrt((f_sq + 107.7 ** 2) * (f_sq + 737.9 ** 2)) *
                (f_sq + 12194 ** 2) + 1e-10
            )
            weights = weights / (np.max(weights) + 1e-10)
        elif self.perceptual_weighting == "mel":
            # Mel-scale weighting (lower freqs get more weight)
            mel = 2595 * np.log10(1 + freqs / 700)
            weights = mel / (np.max(mel) + 1e-10)
        elif self.perceptual_weighting == "bark":
            # Bark scale
            bark = 13 * np.arctan(0.00076 * freqs) + \
                3.5 * np.arctan((freqs / 7500) ** 2)
            weights = bark / (np.max(bark) + 1e-10)
        else:
            weights = np.ones(n_freqs)

        return weights + 0.1  # Ensure minimum weight


class FrequencyBandLoss:
    """Loss computed separately for frequency bands.

    Divides the spectrum into bands and computes independent
    losses for each, allowing fine-grained spectral supervision.

    Attributes:
        bands: List of (low_freq, high_freq) tuples.
        band_weights: Weight per band.
    """

    def __init__(
        self,
        num_bands: int = 8,
        band_weights: Optional[List[float]] = None,
        sample_rate: float = 1.0,
        scale: str = "linear",
        overlap: float = 0.0,
    ):
        """Initialize frequency band loss.

        Args:
            num_bands: Number of frequency bands.
            band_weights: Per-band weights.
            sample_rate: Sampling rate.
            scale: Band scale ('linear', 'log', 'mel', 'octave').
            overlap: Band overlap fraction.
        """
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.scale = scale
        self.overlap = overlap

        # Compute band edges
        self.bands = self._compute_bands()

        # Band weights
        if band_weights is not None:
            self.band_weights = np.array(band_weights)
        else:
            self.band_weights = np.ones(num_bands)
        self.band_weights = self.band_weights / np.sum(self.band_weights)

    def _compute_bands(self) -> List[Tuple[float, float]]:
        """Compute frequency band boundaries."""
        nyquist = self.sample_rate / 2

        if self.scale == "linear":
            edges = np.linspace(0, nyquist, self.num_bands + 1)
        elif self.scale == "log":
            edges = np.logspace(
                np.log10(max(1.0, nyquist / 1000)),
                np.log10(nyquist),
                self.num_bands + 1
            )
            edges[0] = 0
        elif self.scale == "mel":
            mel_low = 0
            mel_high = 2595 * np.log10(1 + nyquist / 700)
            mel_edges = np.linspace(mel_low, mel_high, self.num_bands + 1)
            edges = 700 * (10 ** (mel_edges / 2595) - 1)
        elif self.scale == "octave":
            base = max(20.0, nyquist / (2 ** self.num_bands))
            edges = [base * (2 ** i) for i in range(self.num_bands + 1)]
            edges = np.array(edges)
            edges = np.clip(edges, 0, nyquist)
        else:
            edges = np.linspace(0, nyquist, self.num_bands + 1)

        bands = []
        for i in range(self.num_bands):
            low = edges[i]
            high = edges[i + 1]
            # Apply overlap
            if self.overlap > 0 and i > 0:
                low -= (edges[i] - edges[i-1]) * self.overlap
            if self.overlap > 0 and i < self.num_bands - 1:
                high += (edges[i+1] - edges[i]) * self.overlap
            bands.append((max(0, low), min(nyquist, high)))

        return bands

    def compute(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute frequency-band loss.

        Args:
            prediction: Predicted signal [B, T].
            target: Target signal [B, T].

        Returns:
            Total loss and per-band losses.
        """
        # Compute FFT
        n_fft = prediction.shape[-1]
        pred_fft = np.fft.rfft(prediction, axis=-1)
        target_fft = np.fft.rfft(target, axis=-1)

        n_freqs = pred_fft.shape[-1]
        freqs = np.linspace(0, self.sample_rate / 2, n_freqs)

        total_loss = 0.0
        band_losses: Dict[str, float] = {}

        for i, (low, high) in enumerate(self.bands):
            # Create band mask
            band_mask = ((freqs >= low) & (freqs < high)).astype(np.float64)

            if np.sum(band_mask) == 0:
                band_losses[f"band_{i}_{low:.0f}_{high:.0f}Hz"] = 0.0
                continue

            # Extract band
            pred_band = pred_fft * band_mask
            target_band = target_fft * band_mask

            # Magnitude loss in this band
            pred_mag = np.abs(pred_band)
            target_mag = np.abs(target_band)

            band_loss = float(np.mean((pred_mag - target_mag) ** 2))
            band_losses[f"band_{i}_{low:.0f}_{high:.0f}Hz"] = band_loss

            total_loss += self.band_weights[i] * band_loss

        band_losses["total"] = total_loss
        return total_loss, band_losses


class MultiScaleLoss:
    """Multi-scale spectral loss.

    Computes loss at multiple time/frequency resolutions,
    similar to multi-resolution STFT loss used in audio synthesis.

    Attributes:
        fft_sizes: FFT sizes for each resolution.
        hop_sizes: Hop sizes for each resolution.
        window_sizes: Window sizes for each resolution.
    """

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        window_sizes: Optional[List[int]] = None,
        mag_weight: float = 1.0,
        phase_weight: float = 0.0,
        log_mag: bool = True,
    ):
        """Initialize multi-scale loss.

        Args:
            fft_sizes: FFT sizes at each scale.
            hop_sizes: Hop sizes at each scale.
            window_sizes: Window sizes at each scale.
            mag_weight: Magnitude loss weight.
            phase_weight: Phase loss weight.
            log_mag: Use log-magnitude.
        """
        self.fft_sizes = fft_sizes or [256, 512, 1024, 2048, 4096]
        self.hop_sizes = hop_sizes or [64, 128, 256, 512, 1024]
        self.window_sizes = window_sizes or [256, 512, 1024, 2048, 4096]
        self.mag_weight = mag_weight
        self.phase_weight = phase_weight
        self.log_mag = log_mag

        assert len(self.fft_sizes) == len(self.hop_sizes) == len(self.window_sizes)
        self.num_scales = len(self.fft_sizes)

    def compute(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute multi-scale loss.

        Args:
            prediction: Predicted signal [B, T].
            target: Target signal [B, T].

        Returns:
            Total loss and per-scale losses.
        """
        total_loss = 0.0
        scale_losses: Dict[str, float] = {}

        for i in range(self.num_scales):
            fft_size = self.fft_sizes[i]
            hop_size = self.hop_sizes[i]
            win_size = self.window_sizes[i]

            # Skip if signal too short
            if prediction.shape[-1] < win_size:
                continue

            # Compute STFT at this scale
            pred_stft = self._stft(prediction, fft_size, hop_size, win_size)
            target_stft = self._stft(target, fft_size, hop_size, win_size)

            # Magnitude loss
            pred_mag = np.abs(pred_stft)
            target_mag = np.abs(target_stft)

            if self.log_mag:
                pred_mag = np.log(pred_mag + 1e-8)
                target_mag = np.log(target_mag + 1e-8)

            mag_loss = float(np.mean((pred_mag - target_mag) ** 2))

            # Spectral convergence loss
            sc_loss = float(
                np.linalg.norm(target_mag - pred_mag) /
                (np.linalg.norm(target_mag) + 1e-8)
            )

            scale_loss = self.mag_weight * (mag_loss + sc_loss)

            # Phase loss
            if self.phase_weight > 0:
                pred_phase = np.angle(pred_stft)
                target_phase = np.angle(target_stft)
                phase_diff = np.abs(np.exp(1j * pred_phase) - np.exp(1j * target_phase))
                phase_loss = float(np.mean(phase_diff))
                scale_loss += self.phase_weight * phase_loss

            scale_losses[f"scale_{fft_size}"] = scale_loss
            total_loss += scale_loss

        total_loss /= max(self.num_scales, 1)
        scale_losses["total"] = total_loss

        return total_loss, scale_losses

    def _stft(
        self,
        x: np.ndarray,
        fft_size: int,
        hop_size: int,
        win_size: int,
    ) -> np.ndarray:
        """Compute short-time Fourier transform.

        Args:
            x: Input signal [B, T].
            fft_size: FFT size.
            hop_size: Hop size.
            win_size: Window size.

        Returns:
            STFT [B, n_frames, n_freq].
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]

        batch_size, signal_len = x.shape
        window = np.hanning(win_size)

        # Compute number of frames
        n_frames = (signal_len - win_size) // hop_size + 1
        n_freq = fft_size // 2 + 1

        stft_out = np.zeros((batch_size, n_frames, n_freq), dtype=np.complex128)

        for b in range(batch_size):
            for t in range(n_frames):
                start = t * hop_size
                end = start + win_size
                if end > signal_len:
                    break
                frame = x[b, start:end] * window
                # Zero-pad to fft_size
                padded = np.zeros(fft_size)
                padded[:win_size] = frame
                stft_out[b, t] = np.fft.rfft(padded)

        return stft_out


class PerceptualSpectralLoss:
    """Perceptual loss using spectral features.

    Computes loss in a learned feature space that captures
    perceptually relevant spectral characteristics.

    Similar to perceptual loss in image generation but
    adapted for spectral/frequency-domain signals.

    Attributes:
        feature_layers: Layers to extract features from.
        layer_weights: Weights per layer.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 4,
        layer_weights: Optional[List[float]] = None,
        style_weight: float = 0.0,
        content_weight: float = 1.0,
    ):
        """Initialize perceptual loss.

        Args:
            feature_dim: Feature dimension per layer.
            num_layers: Number of feature extraction layers.
            layer_weights: Per-layer weights.
            style_weight: Style loss weight.
            content_weight: Content loss weight.
        """
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.style_weight = style_weight
        self.content_weight = content_weight

        if layer_weights is not None:
            self.layer_weights = np.array(layer_weights)
        else:
            self.layer_weights = np.ones(num_layers) / num_layers

        # Initialize feature extraction layers (random but fixed)
        self.feature_layers: List[np.ndarray] = []
        dim = feature_dim
        for i in range(num_layers):
            out_dim = dim * 2 if i < num_layers - 1 else dim
            # Fixed random projection (not trained)
            np.random.seed(42 + i)  # Deterministic initialization
            self.feature_layers.append(
                np.random.randn(dim, out_dim) / np.sqrt(dim)
            )
            dim = out_dim

    def compute(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute perceptual spectral loss.

        Args:
            prediction: Predicted features [B, D].
            target: Target features [B, D].

        Returns:
            Total perceptual loss and components.
        """
        # Extract features at each layer
        pred_features = self._extract_features(prediction)
        target_features = self._extract_features(target)

        total_loss = 0.0
        layer_losses: Dict[str, float] = {}

        for i, (pred_feat, target_feat) in enumerate(
            zip(pred_features, target_features)
        ):
            # Content loss (MSE in feature space)
            content_loss = float(np.mean((pred_feat - target_feat) ** 2))

            # Style loss (Gram matrix difference)
            style_loss = 0.0
            if self.style_weight > 0:
                pred_gram = self._gram_matrix(pred_feat)
                target_gram = self._gram_matrix(target_feat)
                style_loss = float(np.mean((pred_gram - target_gram) ** 2))

            layer_loss = self.content_weight * content_loss + \
                self.style_weight * style_loss

            layer_losses[f"layer_{i}"] = layer_loss
            total_loss += self.layer_weights[i] * layer_loss

        layer_losses["total"] = total_loss
        return total_loss, layer_losses

    def _extract_features(self, x: np.ndarray) -> List[np.ndarray]:
        """Extract multi-layer features."""
        features = []
        h = x

        for i, layer in enumerate(self.feature_layers):
            # Adapt input dimension if needed
            if h.shape[-1] != layer.shape[0]:
                # Project to match
                adapt = np.random.randn(h.shape[-1], layer.shape[0]) / np.sqrt(h.shape[-1])
                h = np.dot(h, adapt)

            h = np.dot(h, layer)
            # ReLU activation
            h = np.maximum(0, h)
            features.append(h.copy())

        return features

    def _gram_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute Gram matrix for style loss.

        Args:
            features: Feature maps [B, D].

        Returns:
            Gram matrix [B, D, D] or [D, D].
        """
        if features.ndim == 2:
            return np.dot(features.T, features) / features.shape[0]
        elif features.ndim == 3:
            # [B, T, D] -> [B, D, D]
            return np.einsum("btd,bte->bde", features, features) / features.shape[1]
        return np.dot(features.T, features)
