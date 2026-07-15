"""Positional encoding strategies for spectral transformers.

This module implements various positional encoding schemes specifically
designed for spectral data, including frequency-aware encodings that
capture the structure of frequency-domain signals.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE) for spectral transformers.

    Implements the rotary position embedding from the RoFormer paper,
    adapted for spectral data where position correlates with frequency.

    Attributes:
        dim: Embedding dimension.
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.
        precision: Numerical precision for computation.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        precision: str = "float32",
    ):
        """Initialize rotary embeddings.

        Args:
            dim: Embedding dimension (must be even).
            max_seq_len: Maximum sequence length to precompute.
            base: Base frequency for position encoding.
            precision: Numerical precision ('float32' or 'float64').
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.precision = precision

        # Precompute frequency bands
        self._freqs = self._compute_freqs()
        self._cos_cache: Optional[np.ndarray] = None
        self._sin_cache: Optional[np.ndarray] = None
        self._build_cache(max_seq_len)

    def _compute_freqs(self) -> np.ndarray:
        """Compute frequency bands for rotary embedding."""
        dtype = np.float64 if self.precision == "float64" else np.float32
        freqs = 1.0 / (
            self.base ** (np.arange(0, self.dim, 2, dtype=dtype) / self.dim)
        )
        return freqs

    def _build_cache(self, seq_len: int) -> None:
        """Build cosine and sine caches for given sequence length."""
        dtype = np.float64 if self.precision == "float64" else np.float32
        t = np.arange(seq_len, dtype=dtype)
        freqs = np.outer(t, self._freqs)
        # Create [seq_len, dim] by repeating each frequency pair
        emb = np.concatenate([freqs, freqs], axis=-1)
        self._cos_cache = np.cos(emb)
        self._sin_cache = np.sin(emb)

    def get_cos_sin(
        self, seq_len: int, offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get cosine and sine values for given sequence length.

        Args:
            seq_len: Length of the sequence.
            offset: Position offset.

        Returns:
            Tuple of (cos, sin) arrays with shape [seq_len, dim].
        """
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset)
            self.max_seq_len = seq_len + offset

        cos = self._cos_cache[offset:offset + seq_len]
        sin = self._sin_cache[offset:offset + seq_len]
        return cos, sin

    def rotate_half(self, x: np.ndarray) -> np.ndarray:
        """Rotate half of the dimensions.

        Args:
            x: Input array with shape [..., dim].

        Returns:
            Rotated array with shape [..., dim].
        """
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return np.concatenate([-x2, x1], axis=-1)

    def apply_rotary(
        self,
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
    ) -> np.ndarray:
        """Apply rotary embedding to input.

        Args:
            x: Input tensor with shape [batch, heads, seq_len, head_dim].
            cos: Cosine values with shape [seq_len, head_dim].
            sin: Sine values with shape [seq_len, head_dim].

        Returns:
            Position-encoded tensor.
        """
        # Broadcast cos/sin to match x dimensions
        while cos.ndim < x.ndim:
            cos = np.expand_dims(cos, 0)
            sin = np.expand_dims(sin, 0)

        return x * cos + self.rotate_half(x) * sin


class FrequencyAwarePositionalEncoding:
    """Positional encoding that explicitly encodes frequency information.

    This encoding combines standard positional information with
    frequency-domain specific features like octave position,
    harmonic relationships, and spectral resolution.

    Attributes:
        dim: Encoding dimension.
        max_seq_len: Maximum sequence length.
        min_freq: Minimum frequency in the spectrum (Hz).
        max_freq: Maximum frequency in the spectrum (Hz).
        freq_scale: Frequency axis scale ('linear', 'log', 'mel').
        num_octaves: Number of octave bands.
        include_harmonic: Whether to include harmonic encoding.
        max_harmonics: Maximum harmonics to encode.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        min_freq: float = 0.01,
        max_freq: float = 100000.0,
        freq_scale: str = "log",
        num_octaves: int = 10,
        include_harmonic: bool = True,
        max_harmonics: int = 16,
    ):
        """Initialize frequency-aware positional encoding.

        Args:
            dim: Total encoding dimension.
            max_seq_len: Maximum sequence length.
            min_freq: Minimum frequency (Hz).
            max_freq: Maximum frequency (Hz).
            freq_scale: Frequency axis scaling.
            num_octaves: Number of octave bands.
            include_harmonic: Whether to encode harmonics.
            max_harmonics: Maximum harmonics to track.
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_scale = freq_scale
        self.num_octaves = num_octaves
        self.include_harmonic = include_harmonic
        self.max_harmonics = max_harmonics

        # Allocate dimensions to different encoding components
        self._position_dim = dim // 4
        self._frequency_dim = dim // 4
        self._octave_dim = dim // 4
        self._harmonic_dim = dim - 3 * (dim // 4)

        # Precompute encoding matrices
        self._position_enc = self._build_position_encoding()
        self._frequency_enc = self._build_frequency_encoding()
        self._octave_enc = self._build_octave_encoding()
        self._harmonic_enc = self._build_harmonic_encoding()

    def _build_position_encoding(self) -> np.ndarray:
        """Build standard sinusoidal position encoding."""
        pe = np.zeros((self.max_seq_len, self._position_dim))
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, self._position_dim, 2) * (-math.log(10000.0) / self._position_dim)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _build_frequency_encoding(self) -> np.ndarray:
        """Build frequency-dependent encoding.

        Maps each position to its corresponding frequency value and
        encodes it using sinusoidal functions at different scales.
        """
        fe = np.zeros((self.max_seq_len, self._frequency_dim))

        if self.freq_scale == "log":
            freqs = np.logspace(
                np.log10(self.min_freq),
                np.log10(self.max_freq),
                self.max_seq_len,
            )
        elif self.freq_scale == "mel":
            mel_min = 2595.0 * np.log10(1.0 + self.min_freq / 700.0)
            mel_max = 2595.0 * np.log10(1.0 + self.max_freq / 700.0)
            mels = np.linspace(mel_min, mel_max, self.max_seq_len)
            freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        else:
            freqs = np.linspace(self.min_freq, self.max_freq, self.max_seq_len)

        # Normalize frequencies to [0, 1]
        log_freqs = np.log10(freqs + 1e-10)
        norm_freqs = (log_freqs - log_freqs.min()) / (log_freqs.max() - log_freqs.min() + 1e-10)

        # Encode at multiple scales
        scales = np.exp(
            np.arange(0, self._frequency_dim, 2) * (-math.log(1000.0) / self._frequency_dim)
        )
        for i, scale in enumerate(scales):
            if 2 * i < self._frequency_dim:
                fe[:, 2 * i] = np.sin(2 * np.pi * norm_freqs * (i + 1))
            if 2 * i + 1 < self._frequency_dim:
                fe[:, 2 * i + 1] = np.cos(2 * np.pi * norm_freqs * (i + 1))

        return fe

    def _build_octave_encoding(self) -> np.ndarray:
        """Build octave-band encoding.

        Encodes the octave position of each frequency bin, capturing
        the logarithmic structure of frequency perception.
        """
        oe = np.zeros((self.max_seq_len, self._octave_dim))

        # Compute octave positions
        freqs = np.logspace(
            np.log10(max(self.min_freq, 1e-6)),
            np.log10(self.max_freq),
            self.max_seq_len,
        )
        octaves = np.log2(freqs / max(self.min_freq, 1e-6))
        norm_octaves = octaves / (np.log2(self.max_freq / max(self.min_freq, 1e-6)) + 1e-10)

        # One-hot-like octave band encoding
        for i in range(min(self.num_octaves, self._octave_dim)):
            band_center = (i + 0.5) / self.num_octaves
            band_width = 1.0 / self.num_octaves
            oe[:, i] = np.exp(-0.5 * ((norm_octaves - band_center) / band_width) ** 2)

        # Continuous octave encoding for remaining dimensions
        remaining = self._octave_dim - self.num_octaves
        if remaining > 0:
            for i in range(remaining):
                freq_scale = (i + 1) * np.pi
                oe[:, self.num_octaves + i] = np.sin(norm_octaves * freq_scale)

        return oe

    def _build_harmonic_encoding(self) -> np.ndarray:
        """Build harmonic relationship encoding.

        Encodes potential harmonic relationships between frequency positions,
        allowing the model to recognize overtone series and harmonic structures.
        """
        he = np.zeros((self.max_seq_len, self._harmonic_dim))

        if not self.include_harmonic or self._harmonic_dim == 0:
            return he

        # For each position, encode its relationship to potential fundamentals
        positions = np.arange(self.max_seq_len, dtype=np.float64)
        for h in range(min(self.max_harmonics, self._harmonic_dim)):
            harmonic_ratio = (h + 2)  # 2nd, 3rd, 4th harmonics, etc.
            # Encode distance to nearest harmonic position
            harmonic_positions = positions / harmonic_ratio
            fractional_part = harmonic_positions - np.floor(harmonic_positions)
            # Use sine/cosine of fractional part
            if h < self._harmonic_dim:
                he[:, h] = np.cos(2 * np.pi * fractional_part)

        return he

    def encode(self, seq_len: int, offset: int = 0) -> np.ndarray:
        """Get positional encoding for given sequence length.

        Args:
            seq_len: Sequence length.
            offset: Position offset.

        Returns:
            Positional encoding with shape [seq_len, dim].
        """
        start = offset
        end = offset + seq_len

        if end > self.max_seq_len:
            # Extend encodings if needed
            self.max_seq_len = end
            self._position_enc = self._build_position_encoding()
            self._frequency_enc = self._build_frequency_encoding()
            self._octave_enc = self._build_octave_encoding()
            self._harmonic_enc = self._build_harmonic_encoding()

        # Concatenate all encoding components
        encoding = np.concatenate([
            self._position_enc[start:end],
            self._frequency_enc[start:end],
            self._octave_enc[start:end],
            self._harmonic_enc[start:end],
        ], axis=-1)

        return encoding


class SpectralHarmonicEncoding:
    """Harmonic series-aware positional encoding for spectral data.

    This encoding explicitly represents harmonic relationships in
    frequency-domain data, enabling the model to recognize and
    leverage overtone structures commonly found in physical signals.

    Attributes:
        dim: Encoding dimension.
        max_seq_len: Maximum sequence length.
        num_harmonics: Number of harmonic overtones to encode.
        fundamental_range: Range of fundamental frequencies.
        decay_rate: Amplitude decay rate per harmonic.
        use_subharmonics: Whether to include subharmonic encoding.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        num_harmonics: int = 32,
        fundamental_range: Tuple[float, float] = (20.0, 2000.0),
        decay_rate: float = 0.7,
        use_subharmonics: bool = True,
    ):
        """Initialize harmonic encoding.

        Args:
            dim: Encoding dimension.
            max_seq_len: Maximum sequence length.
            num_harmonics: Number of harmonics to encode.
            fundamental_range: Range of fundamental frequencies (Hz).
            decay_rate: Amplitude decay per harmonic.
            use_subharmonics: Whether to encode subharmonics.
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_harmonics = num_harmonics
        self.fundamental_range = fundamental_range
        self.decay_rate = decay_rate
        self.use_subharmonics = use_subharmonics

        # Allocate dimensions
        self._harmonic_dim = dim // 2
        self._subharmonic_dim = dim // 4 if use_subharmonics else 0
        self._structural_dim = dim - self._harmonic_dim - self._subharmonic_dim

        self._encoding = self._build_encoding()

    def _build_encoding(self) -> np.ndarray:
        """Build the complete harmonic encoding matrix."""
        encoding = np.zeros((self.max_seq_len, self.dim))

        # Frequency positions (log-spaced)
        freq_positions = np.logspace(
            np.log10(self.fundamental_range[0]),
            np.log10(self.fundamental_range[1]),
            self.max_seq_len,
        )

        # Harmonic encoding: for each position, encode its harmonic signature
        for h in range(min(self.num_harmonics, self._harmonic_dim)):
            harmonic_number = h + 1
            amplitude = self.decay_rate ** h

            # Phase encoding for this harmonic
            phase = 2 * np.pi * harmonic_number * np.log2(
                freq_positions / self.fundamental_range[0]
            )
            if h < self._harmonic_dim:
                encoding[:, h] = amplitude * np.sin(phase)
            if h + self.num_harmonics < self._harmonic_dim:
                encoding[:, h + self.num_harmonics] = amplitude * np.cos(phase)

        # Subharmonic encoding
        if self.use_subharmonics and self._subharmonic_dim > 0:
            offset = self._harmonic_dim
            for s in range(min(8, self._subharmonic_dim)):
                subharmonic = 1.0 / (s + 2)
                phase = 2 * np.pi * subharmonic * np.log2(
                    freq_positions / self.fundamental_range[0]
                )
                encoding[:, offset + s] = np.sin(phase)

        # Structural encoding: beat frequencies and combination tones
        struct_offset = self._harmonic_dim + self._subharmonic_dim
        if self._structural_dim > 0:
            for i in range(min(self._structural_dim, 16)):
                # Encode difference tones (f2 - f1 patterns)
                beat_freq = (i + 1) * 0.1
                encoding[:, struct_offset + i] = np.sin(
                    beat_freq * np.arange(self.max_seq_len) * 2 * np.pi / self.max_seq_len
                )

        return encoding

    def encode(self, seq_len: int, offset: int = 0) -> np.ndarray:
        """Get harmonic encoding for given sequence length.

        Args:
            seq_len: Sequence length.
            offset: Position offset.

        Returns:
            Encoding with shape [seq_len, dim].
        """
        if offset + seq_len > self.max_seq_len:
            self.max_seq_len = offset + seq_len
            self._encoding = self._build_encoding()
        return self._encoding[offset:offset + seq_len]


class ALiBiEncoding:
    """Attention with Linear Biases (ALiBi) for spectral transformers.

    Implements ALiBi-style position encoding that adds linear biases
    to attention scores, allowing length extrapolation without
    explicit position embeddings.

    Attributes:
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        slope_strategy: Strategy for computing head slopes.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 8192,
        slope_strategy: str = "geometric",
    ):
        """Initialize ALiBi encoding.

        Args:
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length.
            slope_strategy: How to compute slopes ('geometric', 'linear', 'spectral').
        """
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.slope_strategy = slope_strategy

        self._slopes = self._compute_slopes()
        self._bias_cache = self._build_bias_cache()

    def _compute_slopes(self) -> np.ndarray:
        """Compute per-head slopes for ALiBi."""
        if self.slope_strategy == "geometric":
            # Standard geometric sequence as in original paper
            ratio = 2.0 ** (-8.0 / self.num_heads)
            slopes = np.array([ratio ** (i + 1) for i in range(self.num_heads)])
        elif self.slope_strategy == "linear":
            slopes = np.linspace(0.01, 0.5, self.num_heads)
        elif self.slope_strategy == "spectral":
            # Spectral-aware slopes: lower frequencies get larger slopes
            # (broader attention), higher frequencies get smaller slopes
            # (more local attention)
            base_slopes = 2.0 ** (-8.0 / self.num_heads)
            slopes = np.array([base_slopes ** (i + 1) for i in range(self.num_heads)])
            # Modulate by frequency band importance
            freq_modulation = np.logspace(-1, 0, self.num_heads)
            slopes = slopes * freq_modulation
        else:
            slopes = np.array([2.0 ** (-(i + 1)) for i in range(self.num_heads)])

        return slopes

    def _build_bias_cache(self) -> np.ndarray:
        """Build the attention bias cache."""
        # Create distance matrix
        positions = np.arange(self.max_seq_len)
        distances = positions[np.newaxis, :] - positions[:, np.newaxis]

        # Apply slopes: [num_heads, seq_len, seq_len]
        biases = np.zeros((self.num_heads, self.max_seq_len, self.max_seq_len))
        for h in range(self.num_heads):
            biases[h] = -self._slopes[h] * np.abs(distances)

        return biases

    def get_bias(self, seq_len: int) -> np.ndarray:
        """Get attention bias for given sequence length.

        Args:
            seq_len: Sequence length.

        Returns:
            Bias tensor with shape [num_heads, seq_len, seq_len].
        """
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._bias_cache = self._build_bias_cache()

        return self._bias_cache[:, :seq_len, :seq_len]


class SpectralPositionalEncoding:
    """Unified positional encoding factory for spectral transformers.

    Provides a unified interface to create and apply different positional
    encoding strategies based on configuration.

    Attributes:
        encoding_type: Type of positional encoding.
        dim: Encoding dimension.
        max_seq_len: Maximum sequence length.
        config: Additional configuration parameters.
    """

    def __init__(
        self,
        encoding_type: str = "rotary",
        dim: int = 1024,
        max_seq_len: int = 8192,
        num_heads: int = 16,
        min_freq: float = 0.01,
        max_freq: float = 100000.0,
        freq_scale: str = "log",
        **kwargs,
    ):
        """Initialize positional encoding.

        Args:
            encoding_type: Type of encoding to use.
            dim: Encoding dimension.
            max_seq_len: Maximum sequence length.
            num_heads: Number of attention heads (for ALiBi).
            min_freq: Minimum frequency for frequency-aware encoding.
            max_freq: Maximum frequency for frequency-aware encoding.
            freq_scale: Frequency axis scale.
            **kwargs: Additional configuration parameters.
        """
        self.encoding_type = encoding_type
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        # Initialize the appropriate encoder
        if encoding_type == "rotary":
            self._encoder = RotaryEmbedding(
                dim=dim,
                max_seq_len=max_seq_len,
                base=kwargs.get("base", 10000.0),
            )
        elif encoding_type == "frequency_aware":
            self._encoder = FrequencyAwarePositionalEncoding(
                dim=dim,
                max_seq_len=max_seq_len,
                min_freq=min_freq,
                max_freq=max_freq,
                freq_scale=freq_scale,
            )
        elif encoding_type == "spectral_harmonic":
            self._encoder = SpectralHarmonicEncoding(
                dim=dim,
                max_seq_len=max_seq_len,
                num_harmonics=kwargs.get("num_harmonics", 32),
            )
        elif encoding_type == "alibi":
            self._encoder = ALiBiEncoding(
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                slope_strategy=kwargs.get("slope_strategy", "spectral"),
            )
        elif encoding_type == "sinusoidal":
            self._encoder = self._build_sinusoidal(dim, max_seq_len)
        elif encoding_type == "learned":
            # Learned embeddings are initialized randomly
            self._encoder = np.random.randn(max_seq_len, dim) * 0.02
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _build_sinusoidal(self, dim: int, max_seq_len: int) -> np.ndarray:
        """Build standard sinusoidal positional encoding."""
        pe = np.zeros((max_seq_len, dim))
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:dim // 2]) if dim % 2 == 0 else np.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        return pe

    def encode(self, seq_len: int, offset: int = 0) -> np.ndarray:
        """Get positional encoding for given sequence length.

        Args:
            seq_len: Sequence length.
            offset: Position offset.

        Returns:
            Encoding array. Shape depends on encoding type:
            - Most types: [seq_len, dim]
            - ALiBi: [num_heads, seq_len, seq_len]
        """
        if self.encoding_type == "rotary":
            cos, sin = self._encoder.get_cos_sin(seq_len, offset)
            return cos  # Return cos; sin available via get_rotary_components
        elif self.encoding_type in ("frequency_aware", "spectral_harmonic"):
            return self._encoder.encode(seq_len, offset)
        elif self.encoding_type == "alibi":
            return self._encoder.get_bias(seq_len)
        elif self.encoding_type == "sinusoidal":
            return self._encoder[offset:offset + seq_len]
        elif self.encoding_type == "learned":
            return self._encoder[offset:offset + seq_len]
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def get_rotary_components(
        self, seq_len: int, offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get rotary embedding components (cos, sin).

        Only valid for rotary encoding type.

        Args:
            seq_len: Sequence length.
            offset: Position offset.

        Returns:
            Tuple of (cos, sin) arrays.

        Raises:
            ValueError: If encoding type is not rotary.
        """
        if self.encoding_type != "rotary":
            raise ValueError("get_rotary_components only valid for rotary encoding")
        return self._encoder.get_cos_sin(seq_len, offset)
