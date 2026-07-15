"""Transformer pipeline for spectral intelligence.

Provides a high-level pipeline that integrates transformer-based
models with the MESIE spectral processing system. Supports
spectral tokenization, positional encoding, and multi-scale
attention for frequency-domain analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TransformerConfig:
    """Configuration for the spectral transformer pipeline.

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of encoder layers.
        n_decoder_layers: Number of decoder layers.
        d_feedforward: Feed-forward network dimension.
        max_seq_len: Maximum input sequence length.
        dropout: Dropout rate.
        tokenization: Spectral tokenization strategy.
        pooling: Output pooling strategy.
    """

    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    d_feedforward: int = 512
    max_seq_len: int = 512
    dropout: float = 0.1
    tokenization: str = "frequency_bins"  # 'frequency_bins', 'wavelets', 'patches'
    pooling: str = "mean"  # 'mean', 'cls', 'max'


@dataclass
class TransformerOutput:
    """Output from the spectral transformer pipeline.

    Args:
        embeddings: Sequence embeddings of shape (seq_len, d_model).
        pooled_output: Pooled representation of shape (d_model,).
        attention_maps: Attention weight matrices per layer.
        token_predictions: Optional per-token predictions.
        metadata: Pipeline execution metadata.
    """

    embeddings: np.ndarray
    pooled_output: np.ndarray
    attention_maps: list[np.ndarray] = field(default_factory=list)
    token_predictions: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SpectralTokenizer:
    """Tokenizes spectral data into sequences for transformer input.

    Converts continuous spectral data into discrete token sequences
    using configurable strategies.

    Args:
        strategy: Tokenization strategy.
        n_tokens: Target number of tokens.
        d_model: Token embedding dimension.
    """

    def __init__(
        self,
        strategy: str = "frequency_bins",
        n_tokens: int = 64,
        d_model: int = 128,
    ) -> None:
        self.strategy = strategy
        self.n_tokens = n_tokens
        self.d_model = d_model
        self._projection = np.random.randn(1, d_model) * 0.02
        self._cls_token = np.random.randn(1, d_model) * 0.02

    def tokenize(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert spectrum to token embeddings.

        Args:
            spectrum: Input spectrum of arbitrary length.

        Returns:
            Token embeddings of shape (n_tokens, d_model).
        """
        spectrum = np.atleast_1d(spectrum).flatten()

        if self.strategy == "frequency_bins":
            tokens = self._bin_tokenize(spectrum)
        elif self.strategy == "wavelets":
            tokens = self._wavelet_tokenize(spectrum)
        elif self.strategy == "patches":
            tokens = self._patch_tokenize(spectrum)
        else:
            tokens = self._bin_tokenize(spectrum)

        return tokens

    def _bin_tokenize(self, spectrum: np.ndarray) -> np.ndarray:
        """Tokenize by dividing spectrum into equal frequency bins."""
        # Resample to n_tokens points
        resampled = np.interp(
            np.linspace(0, 1, self.n_tokens),
            np.linspace(0, 1, len(spectrum)),
            spectrum,
        )
        # Project each bin value to d_model dimension
        tokens = resampled[:, np.newaxis] * self._projection
        return tokens

    def _wavelet_tokenize(self, spectrum: np.ndarray) -> np.ndarray:
        """Tokenize using multi-scale wavelet-like decomposition."""
        tokens = np.zeros((self.n_tokens, self.d_model))
        scales = [2**i for i in range(min(8, self.n_tokens // 8 + 1))]

        token_idx = 0
        for scale in scales:
            if token_idx >= self.n_tokens:
                break
            # Downsample at this scale
            step = max(1, len(spectrum) // scale)
            coeffs = spectrum[::step][:min(scale, self.n_tokens - token_idx)]
            for c in coeffs:
                if token_idx >= self.n_tokens:
                    break
                tokens[token_idx] = c * self._projection.flatten()
                token_idx += 1

        return tokens

    def _patch_tokenize(self, spectrum: np.ndarray) -> np.ndarray:
        """Tokenize by extracting overlapping patches."""
        patch_size = max(1, len(spectrum) // self.n_tokens)
        stride = max(1, (len(spectrum) - patch_size) // max(1, self.n_tokens - 1))

        tokens = np.zeros((self.n_tokens, self.d_model))
        for i in range(self.n_tokens):
            start = i * stride
            end = min(start + patch_size, len(spectrum))
            if start >= len(spectrum):
                break
            patch = spectrum[start:end]
            # Use patch statistics as features
            patch_features = np.array([
                np.mean(patch),
                np.std(patch),
                np.max(patch),
                np.min(patch),
            ])
            # Project to d_model
            tokens[i, :4] = patch_features
            tokens[i] *= np.linalg.norm(self._projection) * 0.1

        return tokens

    def prepend_cls(self, tokens: np.ndarray) -> np.ndarray:
        """Prepend a [CLS] token to the sequence.

        Args:
            tokens: Token embeddings of shape (seq_len, d_model).

        Returns:
            Tokens with CLS prepended, shape (seq_len+1, d_model).
        """
        return np.vstack([self._cls_token, tokens])


class PositionalEncoder:
    """Sinusoidal and learnable positional encodings.

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        encoding_type: 'sinusoidal' or 'learnable'.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_len: int = 512,
        encoding_type: str = "sinusoidal",
    ) -> None:
        self.d_model = d_model
        self.max_len = max_len
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            self._encoding = self._build_sinusoidal(max_len, d_model)
        else:
            self._encoding = np.random.randn(max_len, d_model) * 0.02

    def _build_sinusoidal(self, max_len: int, d_model: int) -> np.ndarray:
        """Build sinusoidal positional encoding table."""
        encoding = np.zeros((max_len, d_model))
        positions = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        encoding[:, 0::2] = np.sin(positions * div_term)
        encoding[:, 1::2] = np.cos(positions * div_term[:d_model // 2])
        return encoding

    def encode(self, tokens: np.ndarray) -> np.ndarray:
        """Add positional encoding to token embeddings.

        Args:
            tokens: Token embeddings of shape (seq_len, d_model).

        Returns:
            Position-encoded tokens.
        """
        seq_len = tokens.shape[0]
        return tokens + self._encoding[:seq_len]


class MultiHeadSpectralAttention:
    """Multi-head attention optimized for spectral sequences.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Weight matrices
        scale = 0.02
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute multi-head self-attention.

        Args:
            x: Input of shape (seq_len, d_model).

        Returns:
            Tuple of (output, attention_weights).
        """
        seq_len = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        scores = np.einsum("qhd,khd->hqk", Q, K) / np.sqrt(self.head_dim)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-10)

        # Apply attention to values
        output = np.einsum("hqk,khd->qhd", attn_weights, V)
        output = output.reshape(seq_len, self.d_model)
        output = output @ self.W_o

        # Average attention across heads for visualization
        avg_attn = np.mean(attn_weights, axis=0)
        return output, avg_attn


class TransformerEncoderLayer:
    """Single transformer encoder layer with spectral optimizations.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_feedforward: Feed-forward dimension.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_feedforward: int = 512,
    ) -> None:
        self.attention = MultiHeadSpectralAttention(d_model, n_heads)
        self.ff_1 = np.random.randn(d_model, d_feedforward) * 0.02
        self.ff_2 = np.random.randn(d_feedforward, d_model) * 0.02
        self.d_model = d_model

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return (x - mean) / std

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through encoder layer.

        Args:
            x: Input of shape (seq_len, d_model).

        Returns:
            Tuple of (output, attention_weights).
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention.forward(x)
        x = self._layer_norm(x + attn_out)

        # Feed-forward with residual
        ff_out = self._gelu(x @ self.ff_1) @ self.ff_2
        x = self._layer_norm(x + ff_out)

        return x, attn_weights


class SpectralTransformerPipeline:
    """End-to-end transformer pipeline for spectral intelligence.

    Combines tokenization, positional encoding, multi-layer
    transformer encoding, and output pooling into a unified
    pipeline for spectral data analysis.

    Args:
        config: Transformer pipeline configuration.
    """

    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        self.config = config or TransformerConfig()

        # Initialize components
        self.tokenizer = SpectralTokenizer(
            strategy=self.config.tokenization,
            n_tokens=self.config.max_seq_len,
            d_model=self.config.d_model,
        )
        self.positional_encoder = PositionalEncoder(
            d_model=self.config.d_model,
            max_len=self.config.max_seq_len + 1,  # +1 for CLS
        )
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_feedforward=self.config.d_feedforward,
            )
            for _ in range(self.config.n_encoder_layers)
        ]
        self._is_fitted = False
        self._processing_count = 0

    def process(self, spectrum: np.ndarray) -> TransformerOutput:
        """Process a spectrum through the full transformer pipeline.

        Args:
            spectrum: Input spectral data of arbitrary length.

        Returns:
            TransformerOutput with embeddings, pooled output, and attention maps.
        """
        self._processing_count += 1

        # Tokenize
        tokens = self.tokenizer.tokenize(spectrum)

        # Add CLS token for pooling
        if self.config.pooling == "cls":
            tokens = self.tokenizer.prepend_cls(tokens)

        # Add positional encoding
        encoded = self.positional_encoder.encode(tokens)

        # Pass through encoder layers
        attention_maps = []
        x = encoded
        for layer in self.encoder_layers:
            x, attn_weights = layer.forward(x)
            attention_maps.append(attn_weights)

        # Pooling
        if self.config.pooling == "cls":
            pooled = x[0]  # CLS token
        elif self.config.pooling == "max":
            pooled = np.max(x, axis=0)
        else:  # mean
            pooled = np.mean(x, axis=0)

        return TransformerOutput(
            embeddings=x,
            pooled_output=pooled,
            attention_maps=attention_maps,
            metadata={
                "n_tokens": tokens.shape[0],
                "d_model": self.config.d_model,
                "n_layers": self.config.n_encoder_layers,
                "pooling": self.config.pooling,
                "processing_id": self._processing_count,
            },
        )

    def batch_process(self, spectra: list[np.ndarray]) -> list[TransformerOutput]:
        """Process multiple spectra through the pipeline.

        Args:
            spectra: List of spectral arrays.

        Returns:
            List of TransformerOutput results.
        """
        return [self.process(s) for s in spectra]

    def extract_embeddings(self, spectrum: np.ndarray) -> np.ndarray:
        """Extract a fixed-size embedding vector from spectrum.

        Args:
            spectrum: Input spectral data.

        Returns:
            Embedding vector of shape (d_model,).
        """
        output = self.process(spectrum)
        return output.pooled_output

    def get_attention_analysis(self, spectrum: np.ndarray) -> dict[str, Any]:
        """Analyze attention patterns for a spectrum.

        Args:
            spectrum: Input spectral data.

        Returns:
            Dictionary with attention analysis results.
        """
        output = self.process(spectrum)

        analysis = {
            "n_layers": len(output.attention_maps),
            "layer_analyses": [],
        }

        for i, attn_map in enumerate(output.attention_maps):
            layer_info = {
                "layer": i,
                "attention_entropy": float(-np.sum(
                    attn_map * np.log(attn_map + 1e-10)
                ) / attn_map.size),
                "max_attention": float(np.max(attn_map)),
                "attention_sparsity": float(np.mean(attn_map < 0.01)),
            }
            analysis["layer_analyses"].append(layer_info)

        return analysis

    @property
    def processing_count(self) -> int:
        """Total number of spectra processed."""
        return self._processing_count
