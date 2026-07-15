"""Spectral input encoder with domain-specific processing layers.

This module provides the spectral-specific encoder that processes raw
spectral data before it enters the main transformer. It includes
learnable DFT layers, wavelet decomposition, harmonic attention,
and octave-band pooling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LearnableDFTLayer:
    """Learnable Discrete Fourier Transform layer.

    Implements a parameterized DFT where the frequency basis functions
    are learnable, allowing the model to adapt its spectral decomposition
    to the data distribution.

    Attributes:
        input_dim: Input signal dimension.
        dft_dim: Number of frequency components.
        learnable_freqs: Whether frequency components are learnable.
        learnable_phases: Whether phase offsets are learnable.
        use_inverse: Whether to include inverse DFT capability.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        dft_dim: int = 512,
        learnable_freqs: bool = True,
        learnable_phases: bool = True,
        use_inverse: bool = True,
    ):
        """Initialize learnable DFT layer.

        Args:
            input_dim: Input dimension.
            dft_dim: Number of DFT components.
            learnable_freqs: Whether frequencies are learnable.
            learnable_phases: Whether phases are learnable.
            use_inverse: Whether to support inverse transform.
        """
        self.input_dim = input_dim
        self.dft_dim = dft_dim
        self.learnable_freqs = learnable_freqs
        self.learnable_phases = learnable_phases
        self.use_inverse = use_inverse

        # Initialize frequency parameters
        if learnable_freqs:
            # Start with standard DFT frequencies
            self.frequencies = np.arange(dft_dim, dtype=np.float64) / dft_dim
            # Add learnable perturbation
            self.freq_perturbation = np.zeros(dft_dim)
        else:
            self.frequencies = np.arange(dft_dim, dtype=np.float64) / dft_dim

        # Phase offsets
        if learnable_phases:
            self.phase_offsets = np.zeros(dft_dim)
        else:
            self.phase_offsets = np.zeros(dft_dim)

        # Amplitude scaling per frequency
        self.amplitude_scale = np.ones(dft_dim)

        # Build DFT matrix
        self._dft_matrix = self._build_dft_matrix()

        # Inverse DFT matrix
        if use_inverse:
            self._idft_matrix = self._build_idft_matrix()

    def _build_dft_matrix(self) -> np.ndarray:
        """Build the DFT basis matrix.

        Returns:
            Complex DFT matrix [input_dim, dft_dim].
        """
        n = np.arange(self.input_dim)[:, np.newaxis]
        effective_freqs = self.frequencies
        if self.learnable_freqs:
            effective_freqs = effective_freqs + self.freq_perturbation

        # Complex exponentials with learnable parameters
        angles = 2 * np.pi * n * effective_freqs[np.newaxis, :] + self.phase_offsets
        real_part = np.cos(angles) * self.amplitude_scale
        imag_part = -np.sin(angles) * self.amplitude_scale

        return real_part + 1j * imag_part

    def _build_idft_matrix(self) -> np.ndarray:
        """Build the inverse DFT matrix.

        Returns:
            Complex inverse DFT matrix [dft_dim, input_dim].
        """
        k = np.arange(self.dft_dim)[:, np.newaxis]
        n = np.arange(self.input_dim)[np.newaxis, :]
        effective_freqs = self.frequencies
        if self.learnable_freqs:
            effective_freqs = effective_freqs + self.freq_perturbation

        angles = 2 * np.pi * effective_freqs[:, np.newaxis] * n / self.input_dim
        real_part = np.cos(angles) / self.input_dim
        imag_part = np.sin(angles) / self.input_dim

        return real_part + 1j * imag_part

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply learnable DFT.

        Args:
            x: Input signal [..., input_dim].

        Returns:
            Tuple of (magnitude, phase) arrays with shape [..., dft_dim].
        """
        # Rebuild matrix if parameters changed
        self._dft_matrix = self._build_dft_matrix()

        # Apply DFT
        spectrum = np.einsum("...n,nk->...k", x.astype(np.float64), self._dft_matrix)

        # Extract magnitude and phase
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        return magnitude.astype(np.float32), phase.astype(np.float32)

    def inverse(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Apply inverse DFT.

        Args:
            magnitude: Magnitude spectrum [..., dft_dim].
            phase: Phase spectrum [..., dft_dim].

        Returns:
            Reconstructed signal [..., input_dim].
        """
        if not self.use_inverse:
            raise RuntimeError("Inverse DFT not enabled")

        self._idft_matrix = self._build_idft_matrix()

        # Reconstruct complex spectrum
        spectrum = magnitude * (np.cos(phase) + 1j * np.sin(phase))

        # Apply inverse DFT
        signal = np.einsum("...k,kn->...n", spectrum, self._idft_matrix)

        return np.real(signal).astype(np.float32)


class WaveletDecompositionLayer:
    """Multi-resolution wavelet decomposition for spectral analysis.

    Implements a learnable wavelet decomposition that provides
    multi-resolution analysis of spectral signals. Uses a bank
    of parameterized wavelets for time-frequency decomposition.

    Attributes:
        input_dim: Input signal dimension.
        num_levels: Number of decomposition levels.
        wavelet_type: Base wavelet type.
        learnable: Whether wavelet parameters are learnable.
        output_dim: Output feature dimension per level.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_levels: int = 6,
        wavelet_type: str = "morlet",
        learnable: bool = True,
        output_dim: int = 64,
    ):
        """Initialize wavelet decomposition layer.

        Args:
            input_dim: Input signal dimension.
            num_levels: Number of wavelet levels.
            wavelet_type: Wavelet type ('morlet', 'mexican_hat', 'gaussian').
            learnable: Whether parameters are learnable.
            output_dim: Output features per level.
        """
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.wavelet_type = wavelet_type
        self.learnable = learnable
        self.output_dim = output_dim

        # Wavelet parameters per level
        self.scales = 2.0 ** np.arange(1, num_levels + 1)
        self.center_freqs = 1.0 / self.scales

        # Learnable parameters
        if learnable:
            self.scale_perturbation = np.zeros(num_levels)
            self.bandwidth = np.ones(num_levels)
            self.shape_params = np.zeros(num_levels)

        # Projection weights for each level
        self.level_projections = [
            np.random.randn(input_dim, output_dim) * 0.02
            for _ in range(num_levels)
        ]

        # Build wavelet filter bank
        self._filter_bank = self._build_filter_bank()

    def _build_filter_bank(self) -> List[np.ndarray]:
        """Build wavelet filter bank.

        Returns:
            List of wavelet filters, one per level.
        """
        filters = []
        t = np.linspace(-4, 4, self.input_dim)

        for level in range(self.num_levels):
            scale = self.scales[level]
            if self.learnable:
                scale = scale * (1.0 + self.scale_perturbation[level])

            if self.wavelet_type == "morlet":
                # Morlet wavelet
                bw = self.bandwidth[level] if self.learnable else 1.0
                sigma = scale * bw
                gaussian = np.exp(-0.5 * (t / sigma) ** 2)
                oscillation = np.cos(2 * np.pi * self.center_freqs[level] * t)
                wavelet = gaussian * oscillation
                # Admissibility correction
                wavelet = wavelet - np.mean(wavelet)

            elif self.wavelet_type == "mexican_hat":
                # Mexican hat (Ricker) wavelet
                sigma = scale
                normalized_t = t / sigma
                wavelet = (1 - normalized_t ** 2) * np.exp(-0.5 * normalized_t ** 2)

            elif self.wavelet_type == "gaussian":
                # Gaussian derivative wavelet
                sigma = scale
                normalized_t = t / sigma
                wavelet = -normalized_t * np.exp(-0.5 * normalized_t ** 2)

            else:
                # Default: Morlet
                sigma = scale
                gaussian = np.exp(-0.5 * (t / sigma) ** 2)
                wavelet = gaussian * np.cos(2 * np.pi * t / scale)

            # Normalize
            wavelet = wavelet / (np.sqrt(np.sum(wavelet ** 2)) + 1e-10)
            filters.append(wavelet)

        return filters

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply wavelet decomposition.

        Args:
            x: Input signal [..., input_dim].

        Returns:
            Tuple of (multi_resolution_features, level_info).
            Features shape: [..., num_levels * output_dim].
        """
        self._filter_bank = self._build_filter_bank()

        level_outputs = []
        level_info: Dict[str, Any] = {}

        for level in range(self.num_levels):
            # Convolve with wavelet filter (via element-wise multiplication in freq domain)
            # Simplified: direct correlation
            wavelet = self._filter_bank[level]

            # Compute wavelet coefficients
            # Using correlation (equivalent to convolution with conjugate)
            coefficients = np.einsum("...n,n->...n", x, wavelet)

            # Project to output dimension
            projected = np.einsum(
                "...n,no->...o", coefficients, self.level_projections[level]
            )
            level_outputs.append(projected)

            level_info[f"level_{level}"] = {
                "scale": float(self.scales[level]),
                "center_freq": float(self.center_freqs[level]),
                "energy": float(np.mean(coefficients ** 2)),
            }

        # Concatenate all levels
        output = np.concatenate(level_outputs, axis=-1)

        return output, level_info

    def get_scalogram(self, x: np.ndarray) -> np.ndarray:
        """Compute full scalogram (2D time-frequency representation).

        Args:
            x: Input signal [..., input_dim].

        Returns:
            Scalogram with shape [..., num_levels, input_dim].
        """
        self._filter_bank = self._build_filter_bank()
        scalogram_levels = []

        for level in range(self.num_levels):
            wavelet = self._filter_bank[level]
            # Element-wise multiplication gives point-by-point energy
            coefficients = x * wavelet
            scalogram_levels.append(coefficients)

        if x.ndim == 1:
            return np.stack(scalogram_levels, axis=0)
        else:
            return np.stack(scalogram_levels, axis=-2)


class HarmonicAttentionLayer:
    """Attention layer that explicitly attends to harmonic relationships.

    This layer identifies potential harmonic series in the input spectrum
    and uses attention to enhance related harmonic components, improving
    the model's ability to recognize overtone structures.

    Attributes:
        input_dim: Input feature dimension.
        max_harmonics: Maximum number of harmonics to track.
        num_heads: Number of harmonic attention heads.
        temperature: Softmax temperature.
        learnable_ratios: Whether harmonic ratios are learnable.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        max_harmonics: int = 16,
        num_heads: int = 4,
        temperature: float = 1.0,
        learnable_ratios: bool = True,
    ):
        """Initialize harmonic attention layer.

        Args:
            input_dim: Input dimension.
            max_harmonics: Maximum harmonics per fundamental.
            num_heads: Number of attention heads.
            temperature: Softmax temperature.
            learnable_ratios: Whether ratios are learnable.
        """
        self.input_dim = input_dim
        self.max_harmonics = max_harmonics
        self.num_heads = num_heads
        self.temperature = temperature
        self.learnable_ratios = learnable_ratios

        # Standard harmonic ratios (integer multiples)
        self.harmonic_ratios = np.arange(1, max_harmonics + 1, dtype=np.float64)

        # Learnable deviations from integer ratios (for inharmonicity)
        if learnable_ratios:
            self.ratio_deviations = np.zeros(max_harmonics)
            self.harmonic_weights = np.ones(max_harmonics) / max_harmonics

        # Attention projections
        self.query_proj = np.random.randn(input_dim, num_heads * 32) * 0.02
        self.key_proj = np.random.randn(input_dim, num_heads * 32) * 0.02
        self.value_proj = np.random.randn(input_dim, num_heads * 32) * 0.02
        self.output_proj = np.random.randn(num_heads * 32, input_dim) * 0.02

        # Harmonic position encoding
        self._harmonic_pe = self._build_harmonic_position_encoding()

    def _build_harmonic_position_encoding(self) -> np.ndarray:
        """Build harmonic-aware position encoding.

        Returns:
            Position encoding [input_dim, max_harmonics].
        """
        positions = np.arange(self.input_dim, dtype=np.float64)
        pe = np.zeros((self.input_dim, self.max_harmonics))

        for h in range(self.max_harmonics):
            ratio = self.harmonic_ratios[h]
            if self.learnable_ratios:
                ratio = ratio + self.ratio_deviations[h]

            # For each position, compute its harmonic alignment score
            harmonic_positions = positions * ratio
            # Wrap around
            harmonic_idx = (harmonic_positions % self.input_dim).astype(int)
            # Gaussian proximity
            for pos in range(self.input_dim):
                target = int(harmonic_idx[pos]) % self.input_dim
                distance = min(
                    abs(pos - target),
                    self.input_dim - abs(pos - target)
                )
                pe[pos, h] = np.exp(-0.5 * (distance / (self.input_dim * 0.01)) ** 2)

        return pe

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply harmonic attention.

        Args:
            x: Input [..., input_dim].

        Returns:
            Tuple of (output, harmonic_attention_weights).
        """
        # Build harmonic-enhanced attention mask
        self._harmonic_pe = self._build_harmonic_position_encoding()

        # Compute attention with harmonic bias
        seq_len = x.shape[-1]
        head_dim = 32

        # Project
        q = np.einsum("...d,dh->...h", x, self.query_proj)
        k = np.einsum("...d,dh->...h", x, self.key_proj)
        v = np.einsum("...d,dh->...h", x, self.value_proj)

        # Reshape for multi-head
        if x.ndim == 1:
            q = q.reshape(self.num_heads, head_dim)
            k = k.reshape(self.num_heads, head_dim)
            v = v.reshape(self.num_heads, head_dim)
        else:
            batch_dims = x.shape[:-1]
            q = q.reshape(*batch_dims, self.num_heads, head_dim)
            k = k.reshape(*batch_dims, self.num_heads, head_dim)
            v = v.reshape(*batch_dims, self.num_heads, head_dim)

        # Harmonic-weighted attention (simplified for element-wise case)
        # Weight harmonics
        harmonic_mask = np.sum(
            self._harmonic_pe[:seq_len, :] * self.harmonic_weights[np.newaxis, :],
            axis=-1,
        )

        # Apply harmonic weighting to value
        if x.ndim == 1:
            output = v * harmonic_mask[:self.num_heads * head_dim // seq_len + 1][:v.shape[-1]]
        else:
            output = v

        # Project output
        if x.ndim == 1:
            output_flat = output.reshape(-1)
            result = np.einsum("h,ho->o", output_flat, self.output_proj)
        else:
            output_flat = output.reshape(*batch_dims, -1)
            result = np.einsum("...h,ho->...o", output_flat, self.output_proj)

        return result, harmonic_mask


class OctaveBandPooling:
    """Octave-band pooling for multi-resolution spectral features.

    Pools spectral features into octave bands (logarithmically-spaced
    frequency groups), providing a perceptually-motivated dimensionality
    reduction that preserves important frequency structure.

    Attributes:
        input_dim: Input dimension (number of frequency bins).
        num_octaves: Number of octave bands.
        output_dim: Output dimension per octave.
        pooling_method: Pooling method ('mean', 'max', 'attention').
        overlap: Overlap between adjacent octave bands.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_octaves: int = 10,
        output_dim: int = 64,
        pooling_method: str = "attention",
        overlap: float = 0.1,
        min_freq: float = 1.0,
        max_freq: float = 20000.0,
    ):
        """Initialize octave-band pooling.

        Args:
            input_dim: Number of input frequency bins.
            num_octaves: Number of octave bands.
            output_dim: Features per octave band.
            pooling_method: How to pool within bands.
            overlap: Overlap ratio between bands.
            min_freq: Minimum frequency (Hz).
            max_freq: Maximum frequency (Hz).
        """
        self.input_dim = input_dim
        self.num_octaves = num_octaves
        self.output_dim = output_dim
        self.pooling_method = pooling_method
        self.overlap = overlap
        self.min_freq = min_freq
        self.max_freq = max_freq

        # Compute octave band boundaries
        self.band_boundaries = self._compute_band_boundaries()

        # Per-band projection weights
        self.band_projections = [
            np.random.randn(self._get_band_size(i), output_dim) * 0.02
            for i in range(num_octaves)
        ]

        # Attention weights for attention pooling
        if pooling_method == "attention":
            self.attention_weights = [
                np.random.randn(self._get_band_size(i), 1) * 0.02
                for i in range(num_octaves)
            ]

        # Inter-octave mixing weights
        self.mixing_weights = np.eye(num_octaves) + 0.1 * np.random.randn(
            num_octaves, num_octaves
        )

    def _compute_band_boundaries(self) -> List[Tuple[int, int]]:
        """Compute octave band boundaries in bin indices.

        Returns:
            List of (start_bin, end_bin) for each octave.
        """
        # Logarithmically-spaced band edges
        log_min = np.log2(max(self.min_freq, 1e-6))
        log_max = np.log2(self.max_freq)
        edges = np.logspace(
            log_min, log_max, self.num_octaves + 1, base=2
        )

        # Map to bin indices
        freq_per_bin = (self.max_freq - self.min_freq) / self.input_dim
        boundaries = []

        for i in range(self.num_octaves):
            start_bin = int((edges[i] - self.min_freq) / freq_per_bin)
            end_bin = int((edges[i + 1] - self.min_freq) / freq_per_bin)

            # Apply overlap
            overlap_bins = int((end_bin - start_bin) * self.overlap)
            start_bin = max(0, start_bin - overlap_bins)
            end_bin = min(self.input_dim, end_bin + overlap_bins)

            # Ensure minimum band size
            if end_bin - start_bin < 2:
                end_bin = min(self.input_dim, start_bin + 2)

            boundaries.append((start_bin, end_bin))

        return boundaries

    def _get_band_size(self, band_idx: int) -> int:
        """Get size of a specific band.

        Args:
            band_idx: Band index.

        Returns:
            Number of bins in the band.
        """
        start, end = self.band_boundaries[band_idx]
        return end - start

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply octave-band pooling.

        Args:
            x: Input [..., input_dim].

        Returns:
            Tuple of (pooled_features, band_info).
            Features shape: [..., num_octaves * output_dim].
        """
        band_outputs = []
        band_info: Dict[str, Any] = {}

        for i, (start, end) in enumerate(self.band_boundaries):
            # Ensure indices are valid
            start = min(start, x.shape[-1] - 1)
            end = min(end, x.shape[-1])
            if start >= end:
                end = start + 1

            # Extract band
            band = x[..., start:end]
            band_size = band.shape[-1]

            if self.pooling_method == "mean":
                pooled = np.mean(band, axis=-1, keepdims=True)
                pooled = np.repeat(pooled, self.output_dim, axis=-1)

            elif self.pooling_method == "max":
                pooled = np.max(band, axis=-1, keepdims=True)
                pooled = np.repeat(pooled, self.output_dim, axis=-1)

            elif self.pooling_method == "attention":
                # Attention-weighted pooling
                attn_w = self.attention_weights[i][:band_size]
                attn_scores = np.einsum("...n,na->...a", band, attn_w)
                attn_scores = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
                attn_probs = attn_scores / (np.sum(attn_scores, axis=-1, keepdims=True) + 1e-10)

                # Weighted sum
                pooled = band * attn_probs
                pooled = np.sum(pooled, axis=-1, keepdims=True)
                pooled = np.repeat(pooled, self.output_dim, axis=-1)

            else:
                pooled = np.mean(band, axis=-1, keepdims=True)
                pooled = np.repeat(pooled, self.output_dim, axis=-1)

            band_outputs.append(pooled)
            band_info[f"octave_{i}"] = {
                "start_bin": start,
                "end_bin": end,
                "energy": float(np.mean(band ** 2)),
            }

        # Concatenate all bands
        output = np.concatenate(band_outputs, axis=-1)

        return output, band_info


class SpectralInputEncoder:
    """Complete spectral input encoder pipeline.

    Combines all spectral-specific processing layers into a unified
    encoder that transforms raw spectral data into rich representations
    ready for the transformer backbone.

    Architecture:
        1. Input projection
        2. Learnable DFT (optional)
        3. Wavelet decomposition (optional)
        4. Harmonic attention (optional)
        5. Octave-band pooling (optional)
        6. Output projection to model dimension

    Attributes:
        input_dim: Raw input dimension.
        output_dim: Output dimension (model hidden dim).
        use_dft: Whether to use learnable DFT.
        use_wavelet: Whether to use wavelet decomposition.
        use_harmonic: Whether to use harmonic attention.
        use_octave_pooling: Whether to use octave pooling.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 1024,
        use_dft: bool = True,
        dft_dim: int = 512,
        use_wavelet: bool = True,
        wavelet_levels: int = 6,
        use_harmonic: bool = True,
        max_harmonics: int = 16,
        use_octave_pooling: bool = True,
        num_octave_bands: int = 10,
        dropout: float = 0.1,
    ):
        """Initialize spectral input encoder.

        Args:
            input_dim: Input signal dimension.
            output_dim: Output feature dimension.
            use_dft: Whether to include learnable DFT.
            dft_dim: DFT output dimension.
            use_wavelet: Whether to include wavelet decomposition.
            wavelet_levels: Number of wavelet levels.
            use_harmonic: Whether to include harmonic attention.
            max_harmonics: Maximum harmonics.
            use_octave_pooling: Whether to include octave pooling.
            num_octave_bands: Number of octave bands.
            dropout: Dropout rate.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_dft = use_dft
        self.use_wavelet = use_wavelet
        self.use_harmonic = use_harmonic
        self.use_octave_pooling = use_octave_pooling
        self.dropout = dropout

        # Input projection
        self.input_proj = np.random.randn(input_dim, output_dim) * 0.02
        self.input_bias = np.zeros(output_dim)

        # Component layers
        feature_dims = [output_dim]

        if use_dft:
            self.dft_layer = LearnableDFTLayer(
                input_dim=input_dim,
                dft_dim=dft_dim,
            )
            # DFT produces magnitude + phase
            self.dft_proj = np.random.randn(dft_dim * 2, output_dim) * 0.02
            feature_dims.append(output_dim)

        if use_wavelet:
            self.wavelet_layer = WaveletDecompositionLayer(
                input_dim=input_dim,
                num_levels=wavelet_levels,
                output_dim=output_dim // wavelet_levels,
            )
            self.wavelet_proj = np.random.randn(
                wavelet_levels * (output_dim // wavelet_levels), output_dim
            ) * 0.02
            feature_dims.append(output_dim)

        if use_harmonic:
            self.harmonic_layer = HarmonicAttentionLayer(
                input_dim=input_dim,
                max_harmonics=max_harmonics,
            )
            self.harmonic_proj = np.random.randn(input_dim, output_dim) * 0.02
            feature_dims.append(output_dim)

        if use_octave_pooling:
            self.octave_layer = OctaveBandPooling(
                input_dim=input_dim,
                num_octaves=num_octave_bands,
                output_dim=output_dim // num_octave_bands,
            )
            self.octave_proj = np.random.randn(
                num_octave_bands * (output_dim // num_octave_bands), output_dim
            ) * 0.02
            feature_dims.append(output_dim)

        # Feature fusion
        num_features = len(feature_dims)
        self.fusion_weights = np.ones(num_features) / num_features
        self.fusion_proj = np.random.randn(output_dim, output_dim) * 0.02

        # Layer normalization
        from mesie.foundation.models.transformer_blocks import RMSNorm
        self.output_norm = RMSNorm(output_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode spectral input.

        Args:
            x: Raw spectral input [..., input_dim].

        Returns:
            Tuple of (encoded_features, encoder_info).
            Features shape: [..., output_dim].
        """
        encoder_info: Dict[str, Any] = {}
        features = []

        # 1. Direct input projection
        direct = np.einsum("...d,do->...o", x, self.input_proj) + self.input_bias
        features.append(direct)
        encoder_info["direct_energy"] = float(np.mean(direct ** 2))

        # 2. Learnable DFT
        if self.use_dft:
            magnitude, phase = self.dft_layer.forward(x)
            dft_features = np.concatenate([magnitude, phase], axis=-1)
            dft_projected = np.einsum("...d,do->...o", dft_features, self.dft_proj)
            features.append(dft_projected)
            encoder_info["dft_magnitude_mean"] = float(np.mean(magnitude))

        # 3. Wavelet decomposition
        if self.use_wavelet:
            wavelet_features, wavelet_info = self.wavelet_layer.forward(x)
            wavelet_projected = np.einsum(
                "...d,do->...o", wavelet_features, self.wavelet_proj
            )
            features.append(wavelet_projected)
            encoder_info["wavelet_info"] = wavelet_info

        # 4. Harmonic attention
        if self.use_harmonic:
            harmonic_features, harmonic_mask = self.harmonic_layer.forward(x)
            harmonic_projected = np.einsum(
                "...d,do->...o", harmonic_features, self.harmonic_proj
            )
            features.append(harmonic_projected)
            encoder_info["harmonic_mask_mean"] = float(np.mean(harmonic_mask))

        # 5. Octave-band pooling
        if self.use_octave_pooling:
            octave_features, octave_info = self.octave_layer.forward(x)
            octave_projected = np.einsum(
                "...d,do->...o", octave_features, self.octave_proj
            )
            features.append(octave_projected)
            encoder_info["octave_info"] = octave_info

        # 6. Feature fusion (weighted sum)
        fused = np.zeros_like(features[0])
        for i, feat in enumerate(features):
            fused = fused + self.fusion_weights[i] * feat

        # 7. Final projection and normalization
        output = np.einsum("...d,do->...o", fused, self.fusion_proj)
        output = self.output_norm.forward(output)

        encoder_info["num_features_fused"] = len(features)
        encoder_info["output_norm"] = float(np.mean(np.sqrt(np.sum(output ** 2, axis=-1))))

        return output, encoder_info
