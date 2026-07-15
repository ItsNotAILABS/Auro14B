"""Spectral data augmentation for contrastive pretraining.

Provides domain-specific augmentations for spectral data that create
positive pairs for contrastive learning while preserving semantic content.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TimeStretch:
    """Time-stretching augmentation for spectral sequences.

    Stretches or compresses the temporal axis of spectral data
    without changing frequency content.

    Attributes:
        min_rate: Minimum stretch rate (< 1 = compress).
        max_rate: Maximum stretch rate (> 1 = stretch).
        preserve_length: Whether to pad/crop to original length.
    """

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
        preserve_length: bool = True,
    ):
        """Initialize time stretch.

        Args:
            min_rate: Minimum rate.
            max_rate: Maximum rate.
            preserve_length: Whether to preserve output length.
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.preserve_length = preserve_length

    def __call__(self, x: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """Apply time stretching.

        Args:
            x: Input signal [..., length].
            rate: Optional specific rate. Random if None.

        Returns:
            Time-stretched signal.
        """
        if rate is None:
            rate = np.random.uniform(self.min_rate, self.max_rate)

        original_len = x.shape[-1]
        new_len = int(original_len * rate)

        if new_len == original_len:
            return x

        # Resample using linear interpolation
        original_indices = np.linspace(0, original_len - 1, new_len)
        floor_indices = np.floor(original_indices).astype(int)
        ceil_indices = np.minimum(floor_indices + 1, original_len - 1)
        fractions = original_indices - floor_indices

        stretched = x[..., floor_indices] * (1 - fractions) + x[..., ceil_indices] * fractions

        if self.preserve_length:
            if new_len > original_len:
                stretched = stretched[..., :original_len]
            elif new_len < original_len:
                pad_len = original_len - new_len
                padding = np.zeros((*x.shape[:-1], pad_len))
                stretched = np.concatenate([stretched, padding], axis=-1)

        return stretched


class FrequencyShift:
    """Frequency shifting augmentation.

    Shifts the frequency content up or down, simulating
    Doppler effects or transposition.

    Attributes:
        min_shift: Minimum shift in bins (negative = down).
        max_shift: Maximum shift in bins (positive = up).
        circular: Whether to wrap frequencies circularly.
    """

    def __init__(
        self,
        min_shift: float = -0.1,
        max_shift: float = 0.1,
        circular: bool = False,
    ):
        """Initialize frequency shift.

        Args:
            min_shift: Minimum relative shift.
            max_shift: Maximum relative shift.
            circular: Whether to use circular shifting.
        """
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.circular = circular

    def __call__(self, x: np.ndarray, shift: Optional[float] = None) -> np.ndarray:
        """Apply frequency shift.

        Args:
            x: Input spectrum [..., num_bins].
            shift: Optional specific shift. Random if None.

        Returns:
            Frequency-shifted spectrum.
        """
        if shift is None:
            shift = np.random.uniform(self.min_shift, self.max_shift)

        num_bins = x.shape[-1]
        shift_bins = int(shift * num_bins)

        if shift_bins == 0:
            return x

        if self.circular:
            return np.roll(x, shift_bins, axis=-1)
        else:
            result = np.zeros_like(x)
            if shift_bins > 0:
                result[..., shift_bins:] = x[..., :-shift_bins]
            else:
                result[..., :shift_bins] = x[..., -shift_bins:]
            return result


class SpectralMasking:
    """Spectral masking augmentation.

    Masks random frequency bands or time segments,
    similar to SpecAugment for audio.

    Attributes:
        num_freq_masks: Number of frequency band masks.
        num_time_masks: Number of time segment masks.
        freq_mask_width: Maximum width of frequency masks.
        time_mask_width: Maximum width of time masks.
        mask_value: Value to fill masked regions.
    """

    def __init__(
        self,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_width: int = 32,
        time_mask_width: int = 32,
        mask_value: float = 0.0,
    ):
        """Initialize spectral masking.

        Args:
            num_freq_masks: Number of frequency masks.
            num_time_masks: Number of time masks.
            freq_mask_width: Max frequency mask width.
            time_mask_width: Max time mask width.
            mask_value: Fill value for masks.
        """
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.mask_value = mask_value

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spectral masking.

        Args:
            x: Input spectrogram [..., time, freq] or [..., freq].

        Returns:
            Tuple of (masked_spectrogram, mask_indices).
        """
        result = x.copy()
        mask = np.ones_like(x, dtype=bool)

        if x.ndim >= 2:
            freq_dim = x.shape[-1]
            time_dim = x.shape[-2] if x.ndim >= 2 else 1

            # Frequency masks
            for _ in range(self.num_freq_masks):
                width = np.random.randint(1, min(self.freq_mask_width, freq_dim) + 1)
                start = np.random.randint(0, max(1, freq_dim - width))
                result[..., start:start + width] = self.mask_value
                mask[..., start:start + width] = False

            # Time masks (if 2D or higher)
            if x.ndim >= 2:
                for _ in range(self.num_time_masks):
                    width = np.random.randint(1, min(self.time_mask_width, time_dim) + 1)
                    start = np.random.randint(0, max(1, time_dim - width))
                    result[..., start:start + width, :] = self.mask_value
                    mask[..., start:start + width, :] = False
        else:
            # 1D: only frequency masking
            freq_dim = x.shape[-1]
            for _ in range(self.num_freq_masks):
                width = np.random.randint(1, min(self.freq_mask_width, freq_dim) + 1)
                start = np.random.randint(0, max(1, freq_dim - width))
                result[..., start:start + width] = self.mask_value
                mask[..., start:start + width] = False

        return result, mask


class NoiseInjection:
    """Noise injection augmentation for spectral data.

    Adds various types of noise to spectral data at controlled
    signal-to-noise ratios.

    Attributes:
        min_snr_db: Minimum SNR in dB.
        max_snr_db: Maximum SNR in dB.
        noise_type: Type of noise to inject.
    """

    def __init__(
        self,
        min_snr_db: float = 10.0,
        max_snr_db: float = 40.0,
        noise_type: str = "gaussian",
    ):
        """Initialize noise injection.

        Args:
            min_snr_db: Minimum SNR (more noise).
            max_snr_db: Maximum SNR (less noise).
            noise_type: Noise type ('gaussian', 'pink', 'brown', 'uniform').
        """
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.noise_type = noise_type

    def _generate_noise(self, shape: Tuple[int, ...], noise_type: str) -> np.ndarray:
        """Generate noise of specified type.

        Args:
            shape: Noise shape.
            noise_type: Type of noise.

        Returns:
            Generated noise array.
        """
        if noise_type == "gaussian":
            return np.random.randn(*shape)
        elif noise_type == "pink":
            # Pink noise (1/f)
            white = np.random.randn(*shape)
            # Simple approximation via frequency weighting
            n = shape[-1]
            freqs = np.fft.rfftfreq(n)
            freqs[0] = 1  # Avoid division by zero
            spectrum = np.fft.rfft(white, axis=-1)
            spectrum = spectrum / np.sqrt(freqs)
            return np.fft.irfft(spectrum, n=n, axis=-1)
        elif noise_type == "brown":
            # Brown noise (1/f^2)
            white = np.random.randn(*shape)
            return np.cumsum(white, axis=-1) / math.sqrt(shape[-1])
        elif noise_type == "uniform":
            return np.random.uniform(-1, 1, shape)
        else:
            return np.random.randn(*shape)

    def __call__(
        self, x: np.ndarray, snr_db: Optional[float] = None
    ) -> np.ndarray:
        """Add noise to signal.

        Args:
            x: Input signal.
            snr_db: Optional specific SNR. Random if None.

        Returns:
            Noisy signal.
        """
        if snr_db is None:
            snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)

        # Compute signal power
        signal_power = np.mean(x ** 2) + 1e-10

        # Compute required noise power
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate and scale noise
        noise = self._generate_noise(x.shape, self.noise_type)
        noise_actual_power = np.mean(noise ** 2) + 1e-10
        noise = noise * np.sqrt(noise_power / noise_actual_power)

        return x + noise


class PhaseRandomization:
    """Phase randomization augmentation.

    Randomizes the phase of the signal while preserving its
    power spectrum, creating time-domain variations with
    identical spectral content.

    Attributes:
        phase_std: Standard deviation of phase perturbation.
        full_randomize: Whether to fully randomize phase.
        preserve_symmetry: Whether to preserve conjugate symmetry.
    """

    def __init__(
        self,
        phase_std: float = 0.5,
        full_randomize: bool = False,
        preserve_symmetry: bool = True,
    ):
        """Initialize phase randomization.

        Args:
            phase_std: Phase perturbation std (radians).
            full_randomize: Whether to completely randomize phase.
            preserve_symmetry: Whether to maintain real-signal symmetry.
        """
        self.phase_std = phase_std
        self.full_randomize = full_randomize
        self.preserve_symmetry = preserve_symmetry

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply phase randomization.

        Args:
            x: Input signal [..., length].

        Returns:
            Phase-randomized signal.
        """
        n = x.shape[-1]

        # FFT
        spectrum = np.fft.rfft(x, axis=-1)
        magnitude = np.abs(spectrum)
        original_phase = np.angle(spectrum)

        if self.full_randomize:
            # Completely random phase
            new_phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        else:
            # Perturb phase
            phase_noise = np.random.randn(*magnitude.shape) * self.phase_std
            new_phase = original_phase + phase_noise

        # Reconstruct with new phase
        new_spectrum = magnitude * (np.cos(new_phase) + 1j * np.sin(new_phase))

        # Ensure DC and Nyquist are real (conjugate symmetry)
        if self.preserve_symmetry:
            new_spectrum[..., 0] = np.abs(new_spectrum[..., 0])
            if n % 2 == 0:
                new_spectrum[..., -1] = np.abs(new_spectrum[..., -1])

        # Inverse FFT
        result = np.fft.irfft(new_spectrum, n=n, axis=-1)

        return result


class AmplitudeScaling:
    """Random amplitude scaling augmentation.

    Scales the amplitude of spectral data by a random factor.

    Attributes:
        min_scale: Minimum scaling factor.
        max_scale: Maximum scaling factor.
        per_channel: Whether to scale channels independently.
    """

    def __init__(
        self,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        per_channel: bool = False,
    ):
        """Initialize amplitude scaling.

        Args:
            min_scale: Minimum scale.
            max_scale: Maximum scale.
            per_channel: Whether to scale each channel differently.
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.per_channel = per_channel

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply amplitude scaling.

        Args:
            x: Input signal.

        Returns:
            Scaled signal.
        """
        if self.per_channel and x.ndim > 1:
            num_channels = x.shape[0]
            scales = np.random.uniform(self.min_scale, self.max_scale, num_channels)
            return x * scales[:, np.newaxis]
        else:
            scale = np.random.uniform(self.min_scale, self.max_scale)
            return x * scale


class ChannelDropout:
    """Random channel dropout augmentation.

    Randomly zeroes out entire channels to encourage
    the model to not rely on any single channel.

    Attributes:
        dropout_prob: Probability of dropping each channel.
        min_channels: Minimum channels to keep.
    """

    def __init__(
        self,
        dropout_prob: float = 0.2,
        min_channels: int = 1,
    ):
        """Initialize channel dropout.

        Args:
            dropout_prob: Per-channel dropout probability.
            min_channels: Minimum channels to preserve.
        """
        self.dropout_prob = dropout_prob
        self.min_channels = min_channels

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply channel dropout.

        Args:
            x: Multi-channel input [channels, ...].

        Returns:
            Input with some channels zeroed.
        """
        if x.ndim < 2:
            return x

        num_channels = x.shape[0]
        mask = np.random.random(num_channels) > self.dropout_prob

        # Ensure minimum channels
        if np.sum(mask) < self.min_channels:
            keep_indices = np.random.choice(
                num_channels, self.min_channels, replace=False
            )
            mask[keep_indices] = True

        result = x.copy()
        result[~mask] = 0.0
        return result


class SpectralAugmentation:
    """Composite spectral augmentation pipeline.

    Combines multiple augmentation strategies into a configurable
    pipeline for creating augmented views of spectral data during
    contrastive pretraining.

    Attributes:
        augmentations: List of augmentation instances.
        probabilities: Per-augmentation application probability.
    """

    def __init__(
        self,
        time_stretch: bool = True,
        frequency_shift: bool = True,
        spectral_masking: bool = True,
        noise_injection: bool = True,
        phase_randomization: bool = True,
        amplitude_scaling: bool = True,
        channel_dropout: bool = True,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        freq_shift_range: Tuple[float, float] = (-0.1, 0.1),
        noise_snr_range: Tuple[float, float] = (10.0, 40.0),
        amplitude_range: Tuple[float, float] = (0.5, 2.0),
        augmentation_prob: float = 0.5,
    ):
        """Initialize augmentation pipeline.

        Args:
            time_stretch: Whether to include time stretching.
            frequency_shift: Whether to include frequency shifting.
            spectral_masking: Whether to include spectral masking.
            noise_injection: Whether to include noise injection.
            phase_randomization: Whether to include phase randomization.
            amplitude_scaling: Whether to include amplitude scaling.
            channel_dropout: Whether to include channel dropout.
            time_stretch_range: Range for time stretch rate.
            freq_shift_range: Range for frequency shift.
            noise_snr_range: Range for noise SNR.
            amplitude_range: Range for amplitude scaling.
            augmentation_prob: Probability of applying each augmentation.
        """
        self.augmentation_prob = augmentation_prob
        self.augmentations: List[Tuple[str, Any]] = []

        if time_stretch:
            self.augmentations.append((
                "time_stretch",
                TimeStretch(min_rate=time_stretch_range[0], max_rate=time_stretch_range[1]),
            ))
        if frequency_shift:
            self.augmentations.append((
                "frequency_shift",
                FrequencyShift(min_shift=freq_shift_range[0], max_shift=freq_shift_range[1]),
            ))
        if spectral_masking:
            self.augmentations.append((
                "spectral_masking",
                SpectralMasking(),
            ))
        if noise_injection:
            self.augmentations.append((
                "noise_injection",
                NoiseInjection(min_snr_db=noise_snr_range[0], max_snr_db=noise_snr_range[1]),
            ))
        if phase_randomization:
            self.augmentations.append((
                "phase_randomization",
                PhaseRandomization(),
            ))
        if amplitude_scaling:
            self.augmentations.append((
                "amplitude_scaling",
                AmplitudeScaling(min_scale=amplitude_range[0], max_scale=amplitude_range[1]),
            ))
        if channel_dropout:
            self.augmentations.append((
                "channel_dropout",
                ChannelDropout(),
            ))

    def __call__(
        self, x: np.ndarray, return_info: bool = False
    ) -> Any:
        """Apply augmentation pipeline.

        Args:
            x: Input spectral data.
            return_info: Whether to return augmentation info.

        Returns:
            Augmented data, or tuple of (augmented_data, info).
        """
        result = x.copy()
        applied: List[str] = []

        for name, augmentation in self.augmentations:
            if np.random.random() < self.augmentation_prob:
                if name == "spectral_masking":
                    result, _ = augmentation(result)
                else:
                    result = augmentation(result)
                applied.append(name)

        if return_info:
            return result, {"applied_augmentations": applied}
        return result

    def create_positive_pair(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Create a positive pair for contrastive learning.

        Applies different random augmentations to the same input
        to create two views that should be mapped close in latent space.

        Args:
            x: Input spectral data.

        Returns:
            Tuple of (view1, view2, augmentation_info).
        """
        view1, info1 = self(x, return_info=True)
        view2, info2 = self(x, return_info=True)

        info = {
            "view1_augmentations": info1["applied_augmentations"],
            "view2_augmentations": info2["applied_augmentations"],
        }

        return view1, view2, info
