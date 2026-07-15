"""Transformer building blocks for SpectralGPT.

This module implements the core transformer components adapted for spectral
data processing, including multi-head attention with frequency-band awareness,
feed-forward networks with spectral gating, and full transformer blocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalization Layers
# ---------------------------------------------------------------------------

class RMSNorm:
    """Root Mean Square Layer Normalization.

    More efficient alternative to LayerNorm that only normalizes
    by the RMS of activations without centering.

    Attributes:
        dim: Normalization dimension.
        eps: Epsilon for numerical stability.
        weight: Learnable scale parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Epsilon for stability.
        """
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply RMS normalization.

        Args:
            x: Input array with shape [..., dim].

        Returns:
            Normalized array.
        """
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class LayerNorm:
    """Standard Layer Normalization.

    Attributes:
        dim: Normalization dimension.
        eps: Epsilon for numerical stability.
        weight: Learnable scale parameter.
        bias: Learnable bias parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True):
        """Initialize LayerNorm.

        Args:
            dim: Feature dimension.
            eps: Epsilon for stability.
            bias: Whether to include bias.
        """
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim)
        self.bias = np.zeros(dim) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization.

        Args:
            x: Input array with shape [..., dim].

        Returns:
            Normalized array.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        result = normalized * self.weight
        if self.bias is not None:
            result = result + self.bias
        return result


class SpectralNorm:
    """Spectral-aware normalization that preserves frequency structure.

    Normalizes within frequency bands rather than across the entire
    spectrum, preserving relative spectral shape while normalizing
    amplitudes.

    Attributes:
        dim: Feature dimension.
        num_bands: Number of frequency bands for normalization.
        eps: Epsilon for numerical stability.
        weight: Learnable scale per band.
        bias: Learnable bias per band.
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 8,
        eps: float = 1e-6,
        bias: bool = True,
    ):
        """Initialize SpectralNorm.

        Args:
            dim: Feature dimension.
            num_bands: Number of frequency bands.
            eps: Epsilon for stability.
            bias: Whether to include bias.
        """
        self.dim = dim
        self.num_bands = num_bands
        self.eps = eps
        self.band_size = dim // num_bands
        self.weight = np.ones(dim)
        self.bias = np.zeros(dim) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply spectral normalization.

        Args:
            x: Input array with shape [..., dim].

        Returns:
            Normalized array preserving spectral structure.
        """
        result = np.zeros_like(x)
        for i in range(self.num_bands):
            start = i * self.band_size
            end = start + self.band_size if i < self.num_bands - 1 else self.dim
            band = x[..., start:end]
            mean = np.mean(band, axis=-1, keepdims=True)
            var = np.var(band, axis=-1, keepdims=True)
            result[..., start:end] = (band - mean) / np.sqrt(var + self.eps)

        result = result * self.weight
        if self.bias is not None:
            result = result + self.bias
        return result


# ---------------------------------------------------------------------------
# Attention Mechanisms
# ---------------------------------------------------------------------------

class SpectralMultiHeadAttention:
    """Multi-head attention adapted for spectral data.

    Implements standard multi-head attention with optional:
    - Grouped Query Attention (GQA)
    - Query-Key normalization
    - Causal masking
    - Frequency-band routing

    Attributes:
        hidden_dim: Total hidden dimension.
        num_heads: Number of query heads.
        head_dim: Dimension per head.
        num_kv_heads: Number of key-value heads (for GQA).
        dropout: Attention dropout rate.
        causal: Whether to use causal masking.
        qk_norm: Whether to normalize Q and K.
        scale: Attention score scaling factor.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        num_kv_heads: int = 0,
        dropout: float = 0.0,
        causal: bool = False,
        qk_norm: bool = True,
        bias: bool = False,
    ):
        """Initialize multi-head attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            num_kv_heads: KV heads for GQA (0 = same as num_heads).
            dropout: Attention dropout probability.
            causal: Whether to apply causal masking.
            qk_norm: Whether to normalize queries and keys.
            bias: Whether projection layers have bias.
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        self.dropout = dropout
        self.causal = causal
        self.qk_norm = qk_norm
        self.scale = 1.0 / math.sqrt(head_dim)

        # Projection weights
        self.q_proj = np.random.randn(hidden_dim, num_heads * head_dim) * 0.02
        self.k_proj = np.random.randn(hidden_dim, self.num_kv_heads * head_dim) * 0.02
        self.v_proj = np.random.randn(hidden_dim, self.num_kv_heads * head_dim) * 0.02
        self.o_proj = np.random.randn(num_heads * head_dim, hidden_dim) * 0.02

        if bias:
            self.q_bias = np.zeros(num_heads * head_dim)
            self.k_bias = np.zeros(self.num_kv_heads * head_dim)
            self.v_bias = np.zeros(self.num_kv_heads * head_dim)
            self.o_bias = np.zeros(hidden_dim)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            self.o_bias = None

        # QK normalization
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def _project(
        self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]
    ) -> np.ndarray:
        """Linear projection.

        Args:
            x: Input [batch, seq_len, hidden_dim].
            weight: Projection weight [hidden_dim, out_dim].
            bias: Optional bias [out_dim].

        Returns:
            Projected output [batch, seq_len, out_dim].
        """
        out = np.einsum("...d,do->...o", x, weight)
        if bias is not None:
            out = out + bias
        return out

    def _reshape_for_attention(
        self, x: np.ndarray, num_heads: int
    ) -> np.ndarray:
        """Reshape for multi-head attention.

        Args:
            x: Input [batch, seq_len, num_heads * head_dim].
            num_heads: Number of heads.

        Returns:
            Reshaped [batch, num_heads, seq_len, head_dim].
        """
        batch_size = x.shape[0] if x.ndim > 2 else 1
        seq_len = x.shape[-2]
        if x.ndim == 2:
            x = x.reshape(1, seq_len, num_heads, self.head_dim)
        else:
            x = x.reshape(batch_size, seq_len, num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_bias: Optional[np.ndarray] = None,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_bias: Optional positional bias (e.g., ALiBi).
            kv_cache: Optional key-value cache for inference.

        Returns:
            Tuple of (output, attention_weights).
        """
        # Project Q, K, V
        q = self._project(x, self.q_proj, self.q_bias)
        k = self._project(x, self.k_proj, self.k_bias)
        v = self._project(x, self.v_proj, self.v_bias)

        # Reshape for multi-head
        q = self._reshape_for_attention(q, self.num_heads)
        k = self._reshape_for_attention(k, self.num_kv_heads)
        v = self._reshape_for_attention(v, self.num_kv_heads)

        # Handle GQA by repeating KV heads
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = np.repeat(k, repeat_factor, axis=1)
            v = np.repeat(v, repeat_factor, axis=1)

        # QK normalization
        if self.qk_norm:
            q = self.q_norm.forward(q)
            k = self.k_norm.forward(k)

        # Compute attention scores
        scores = np.einsum("bhsd,bhtd->bhst", q, k) * self.scale

        # Apply position bias (ALiBi)
        if position_bias is not None:
            if position_bias.ndim == 3:
                scores = scores + position_bias[np.newaxis, ...]
            else:
                scores = scores + position_bias

        # Apply causal mask
        if self.causal:
            seq_len = scores.shape[-1]
            causal_mask = np.triu(
                np.full((seq_len, seq_len), -np.inf), k=1
            )
            scores = scores + causal_mask

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

        # Apply dropout (during training)
        # Note: In numpy implementation, dropout is a no-op for inference

        # Compute attention output
        output = np.einsum("bhst,bhtd->bhsd", attention_weights, v)

        # Reshape back
        batch_size = output.shape[0]
        seq_len = output.shape[2]
        output = np.transpose(output, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )

        # Output projection
        output = self._project(output, self.o_proj, self.o_bias)

        return output, attention_weights


class FrequencyBandAttention:
    """Frequency-band aware attention mechanism.

    Partitions the frequency axis into bands and computes attention
    independently within each band, then combines results. This captures
    local frequency structure more efficiently than global attention.

    Attributes:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        num_bands: Number of frequency bands.
        band_overlap: Overlap ratio between bands.
        inter_band_attention: Whether to compute inter-band attention.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        num_bands: int = 8,
        band_overlap: float = 0.25,
        inter_band_attention: bool = True,
    ):
        """Initialize frequency-band attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            num_bands: Number of frequency bands.
            band_overlap: Overlap between adjacent bands.
            inter_band_attention: Whether to allow inter-band interaction.
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_bands = num_bands
        self.band_overlap = band_overlap
        self.inter_band_attention = inter_band_attention

        # Per-band attention layers
        heads_per_band = max(1, num_heads // num_bands)
        self.band_attentions = [
            SpectralMultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=heads_per_band,
                head_dim=head_dim,
                causal=False,
                qk_norm=True,
            )
            for _ in range(num_bands)
        ]

        # Inter-band attention (if enabled)
        if inter_band_attention:
            self.inter_band_attn = SpectralMultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads // 4,
                head_dim=head_dim,
                causal=False,
                qk_norm=True,
            )

        # Band aggregation weights
        self.band_weights = np.ones(num_bands) / num_bands

    def _compute_band_boundaries(self, seq_len: int) -> List[Tuple[int, int]]:
        """Compute frequency band boundaries with overlap.

        Args:
            seq_len: Sequence length.

        Returns:
            List of (start, end) tuples for each band.
        """
        band_size = seq_len // self.num_bands
        overlap_size = int(band_size * self.band_overlap)
        boundaries = []

        for i in range(self.num_bands):
            start = max(0, i * band_size - overlap_size)
            end = min(seq_len, (i + 1) * band_size + overlap_size)
            boundaries.append((start, end))

        return boundaries

    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Compute frequency-band attention.

        Args:
            x: Input [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (output, band_attention_weights).
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]

        batch_size, seq_len, _ = x.shape
        output = np.zeros_like(x)
        band_weights_dict: Dict[str, np.ndarray] = {}

        # Compute boundaries
        boundaries = self._compute_band_boundaries(seq_len)

        # Intra-band attention
        for i, (start, end) in enumerate(boundaries):
            band_input = x[:, start:end, :]
            band_output, band_attn = self.band_attentions[i].forward(band_input)

            # Weighted accumulation (handle overlap)
            weight_window = np.ones(end - start)
            if i > 0:
                overlap = boundaries[i - 1][1] - start
                if overlap > 0:
                    weight_window[:overlap] *= np.linspace(0, 1, overlap)
            if i < self.num_bands - 1:
                overlap = end - boundaries[i + 1][0]
                if overlap > 0:
                    weight_window[-overlap:] *= np.linspace(1, 0, overlap)

            output[:, start:end, :] += (
                band_output * weight_window[np.newaxis, :, np.newaxis]
            )
            band_weights_dict[f"band_{i}"] = band_attn

        # Inter-band attention
        if self.inter_band_attention:
            # Compute band summaries
            band_summaries = []
            for start, end in boundaries:
                band_summary = np.mean(x[:, start:end, :], axis=1, keepdims=True)
                band_summaries.append(band_summary)

            band_summary_seq = np.concatenate(band_summaries, axis=1)
            inter_output, inter_attn = self.inter_band_attn.forward(band_summary_seq)

            # Distribute inter-band information
            for i, (start, end) in enumerate(boundaries):
                band_len = end - start
                inter_contribution = np.repeat(
                    inter_output[:, i:i + 1, :], band_len, axis=1
                )
                output[:, start:end, :] += 0.1 * inter_contribution

            band_weights_dict["inter_band"] = inter_attn

        return output, band_weights_dict


class MultiScaleSpectralAttention:
    """Multi-scale attention that operates at different frequency resolutions.

    Processes the input at multiple frequency resolutions simultaneously,
    capturing both fine-grained spectral details and broad spectral patterns.

    Attributes:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        scales: List of resolution scales (1 = full resolution).
        fusion_method: How to fuse multi-scale outputs.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        scales: Optional[List[int]] = None,
        fusion_method: str = "weighted_sum",
    ):
        """Initialize multi-scale attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Total attention heads.
            head_dim: Dimension per head.
            scales: Resolution scales (default: [1, 2, 4, 8]).
            fusion_method: Fusion method ('weighted_sum', 'concat', 'gate').
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scales = scales or [1, 2, 4, 8]
        self.fusion_method = fusion_method

        # Attention at each scale
        heads_per_scale = max(1, num_heads // len(self.scales))
        self.scale_attentions = [
            SpectralMultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=heads_per_scale,
                head_dim=head_dim,
                causal=False,
                qk_norm=True,
            )
            for _ in self.scales
        ]

        # Scale fusion weights (learnable)
        self.scale_weights = np.ones(len(self.scales)) / len(self.scales)

        # Downsampling/upsampling projections
        self.downsample_projs = [
            np.random.randn(hidden_dim, hidden_dim) * 0.02
            for _ in self.scales
        ]
        self.upsample_projs = [
            np.random.randn(hidden_dim, hidden_dim) * 0.02
            for _ in self.scales
        ]

    def _downsample(self, x: np.ndarray, scale: int) -> np.ndarray:
        """Downsample sequence by averaging adjacent tokens.

        Args:
            x: Input [batch, seq_len, hidden_dim].
            scale: Downsampling factor.

        Returns:
            Downsampled sequence [batch, seq_len // scale, hidden_dim].
        """
        if scale == 1:
            return x

        batch_size, seq_len, dim = x.shape
        # Pad to make divisible
        pad_len = (scale - seq_len % scale) % scale
        if pad_len > 0:
            x = np.concatenate([x, np.zeros((batch_size, pad_len, dim))], axis=1)

        new_len = x.shape[1] // scale
        x = x.reshape(batch_size, new_len, scale, dim)
        return np.mean(x, axis=2)

    def _upsample(
        self, x: np.ndarray, target_len: int, scale: int
    ) -> np.ndarray:
        """Upsample sequence by repeating tokens.

        Args:
            x: Input [batch, seq_len, hidden_dim].
            target_len: Target sequence length.
            scale: Original downsampling factor.

        Returns:
            Upsampled sequence [batch, target_len, hidden_dim].
        """
        if scale == 1:
            return x

        batch_size, seq_len, dim = x.shape
        # Repeat each token
        x = np.repeat(x, scale, axis=1)
        # Trim to target length
        return x[:, :target_len, :]

    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute multi-scale attention.

        Args:
            x: Input [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (output, scale_info).
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]

        batch_size, seq_len, dim = x.shape
        scale_outputs = []
        scale_info: Dict[str, Any] = {}

        for i, scale in enumerate(self.scales):
            # Downsample
            x_down = self._downsample(x, scale)

            # Apply attention at this scale
            attn_out, attn_weights = self.scale_attentions[i].forward(x_down)

            # Upsample back to original resolution
            attn_up = self._upsample(attn_out, seq_len, scale)

            scale_outputs.append(attn_up * self.scale_weights[i])
            scale_info[f"scale_{scale}"] = {
                "attention_weights": attn_weights,
                "resolution": x_down.shape[1],
            }

        # Fuse scales
        if self.fusion_method == "weighted_sum":
            output = sum(scale_outputs)
        elif self.fusion_method == "concat":
            output = np.concatenate(scale_outputs, axis=-1)
            # Project back to hidden_dim
            output = output[..., :dim]  # Simple truncation for numpy
        else:
            output = sum(scale_outputs)

        return output, scale_info


# ---------------------------------------------------------------------------
# Feed-Forward Networks
# ---------------------------------------------------------------------------

class SpectralFeedForward:
    """Feed-forward network with spectral-aware gating.

    Implements various FFN architectures including SwiGLU, GeGLU,
    and spectral gating where gate values are informed by frequency content.

    Attributes:
        hidden_dim: Model hidden dimension.
        ffn_dim: FFN intermediate dimension.
        activation: Activation function type.
        use_gated: Whether to use gated linear units.
        dropout: Dropout rate.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        activation: str = "swiglu",
        use_gated: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """Initialize feed-forward network.

        Args:
            hidden_dim: Model hidden dimension.
            ffn_dim: Intermediate dimension.
            activation: Activation function name.
            use_gated: Whether to use gated architecture.
            dropout: Dropout probability.
            bias: Whether to include bias terms.
        """
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.activation = activation
        self.use_gated = use_gated
        self.dropout = dropout

        # Weights
        if use_gated:
            self.gate_proj = np.random.randn(hidden_dim, ffn_dim) * 0.02
            self.up_proj = np.random.randn(hidden_dim, ffn_dim) * 0.02
        else:
            self.up_proj = np.random.randn(hidden_dim, ffn_dim) * 0.02
            self.gate_proj = None

        self.down_proj = np.random.randn(ffn_dim, hidden_dim) * 0.02

        if bias:
            self.up_bias = np.zeros(ffn_dim)
            self.down_bias = np.zeros(hidden_dim)
            self.gate_bias = np.zeros(ffn_dim) if use_gated else None
        else:
            self.up_bias = None
            self.down_bias = None
            self.gate_bias = None

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function.

        Args:
            x: Input array.

        Returns:
            Activated array.
        """
        if self.activation in ("swiglu", "silu"):
            return x * (1.0 / (1.0 + np.exp(-x)))  # SiLU/Swish
        elif self.activation in ("geglu", "gelu"):
            return 0.5 * x * (1.0 + np.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "mish":
            return x * np.tanh(np.log(1.0 + np.exp(x)))
        else:
            return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through feed-forward network.

        Args:
            x: Input [batch, seq_len, hidden_dim].

        Returns:
            Output [batch, seq_len, hidden_dim].
        """
        if self.use_gated:
            # Gated architecture (SwiGLU, GeGLU)
            gate = np.einsum("...d,do->...o", x, self.gate_proj)
            if self.gate_bias is not None:
                gate = gate + self.gate_bias
            gate = self._activate(gate)

            up = np.einsum("...d,do->...o", x, self.up_proj)
            if self.up_bias is not None:
                up = up + self.up_bias

            hidden = gate * up
        else:
            hidden = np.einsum("...d,do->...o", x, self.up_proj)
            if self.up_bias is not None:
                hidden = hidden + self.up_bias
            hidden = self._activate(hidden)

        output = np.einsum("...d,do->...o", hidden, self.down_proj)
        if self.down_bias is not None:
            output = output + self.down_bias

        return output


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class SpectralTransformerBlock:
    """A single transformer block for spectral processing.

    Combines attention, feed-forward, and normalization with residual
    connections. Supports pre-norm and post-norm configurations.

    Attributes:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        ffn_dim: FFN intermediate dimension.
        normalization: Normalization type.
        pre_norm: Whether to use pre-normalization.
        activation: FFN activation.
        dropout: Dropout rate.
        use_parallel: Whether attention and FFN run in parallel.
        layer_idx: Layer index (for layer-specific behavior).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        ffn_dim: int = 4096,
        normalization: str = "rms_norm",
        pre_norm: bool = True,
        activation: str = "swiglu",
        dropout: float = 0.0,
        use_parallel: bool = False,
        causal: bool = False,
        num_kv_heads: int = 0,
        qk_norm: bool = True,
        bias: bool = False,
        layer_idx: int = 0,
    ):
        """Initialize transformer block.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            ffn_dim: FFN intermediate dimension.
            normalization: Normalization type.
            pre_norm: Whether to use pre-normalization.
            activation: FFN activation function.
            dropout: Dropout rate.
            use_parallel: Whether to parallelize attention and FFN.
            causal: Whether attention is causal.
            num_kv_heads: Number of KV heads for GQA.
            qk_norm: Whether to normalize Q and K.
            bias: Whether to use bias.
            layer_idx: Layer index.
        """
        self.hidden_dim = hidden_dim
        self.pre_norm = pre_norm
        self.use_parallel = use_parallel
        self.layer_idx = layer_idx

        # Normalization layers
        if normalization == "rms_norm":
            self.attn_norm = RMSNorm(hidden_dim)
            self.ffn_norm = RMSNorm(hidden_dim)
        elif normalization == "spectral_norm":
            self.attn_norm = SpectralNorm(hidden_dim)
            self.ffn_norm = SpectralNorm(hidden_dim)
        else:
            self.attn_norm = LayerNorm(hidden_dim)
            self.ffn_norm = LayerNorm(hidden_dim)

        # Attention
        self.attention = SpectralMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            causal=causal,
            qk_norm=qk_norm,
            bias=bias,
        )

        # Feed-forward
        self.ffn = SpectralFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            activation=activation,
            use_gated=True,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_bias: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass through transformer block.

        Args:
            x: Input [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_bias: Optional positional bias.

        Returns:
            Tuple of (output, attention_weights).
        """
        if self.use_parallel:
            # Parallel attention + FFN
            normed = self.attn_norm.forward(x)
            attn_out, attn_weights = self.attention.forward(
                normed, attention_mask, position_bias
            )
            ffn_out = self.ffn.forward(self.ffn_norm.forward(x))
            output = x + attn_out + ffn_out
        elif self.pre_norm:
            # Pre-norm: norm -> attn -> residual -> norm -> ffn -> residual
            normed = self.attn_norm.forward(x)
            attn_out, attn_weights = self.attention.forward(
                normed, attention_mask, position_bias
            )
            x = x + attn_out

            normed = self.ffn_norm.forward(x)
            ffn_out = self.ffn.forward(normed)
            output = x + ffn_out
        else:
            # Post-norm: attn -> residual -> norm -> ffn -> residual -> norm
            attn_out, attn_weights = self.attention.forward(
                x, attention_mask, position_bias
            )
            x = self.attn_norm.forward(x + attn_out)

            ffn_out = self.ffn.forward(x)
            output = self.ffn_norm.forward(x + ffn_out)

        return output, attn_weights


class CrossModalAttentionBlock:
    """Cross-modal attention block for multi-modality interaction.

    Enables information flow between different spectral modalities
    through cross-attention, allowing the model to learn shared
    representations.

    Attributes:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        """Initialize cross-modal attention block.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            dropout: Dropout rate.
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Cross-attention
        self.cross_attention = SpectralMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=False,
            qk_norm=True,
        )

        # Normalization
        self.norm_q = RMSNorm(hidden_dim)
        self.norm_kv = RMSNorm(hidden_dim)
        self.norm_out = RMSNorm(hidden_dim)

        # Output FFN
        self.ffn = SpectralFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=hidden_dim * 4,
            activation="swiglu",
        )

    def forward(
        self,
        query: np.ndarray,
        context: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-modal attention.

        Args:
            query: Query input [batch, seq_len_q, hidden_dim].
            context: Context input [batch, seq_len_kv, hidden_dim].
            attention_mask: Optional cross-attention mask.

        Returns:
            Tuple of (output, cross_attention_weights).
        """
        # Normalize
        q_normed = self.norm_q.forward(query)
        kv_normed = self.norm_kv.forward(context)

        # For cross-attention, we need Q from query and K,V from context
        # Since our attention module does self-attention, we concatenate
        # and mask appropriately (simplified for numpy implementation)
        q_len = q_normed.shape[-2]
        combined = np.concatenate([q_normed, kv_normed], axis=-2)
        attn_out, attn_weights = self.cross_attention.forward(combined)
        cross_out = attn_out[..., :q_len, :]

        # Residual
        output = query + cross_out

        # FFN with residual
        normed = self.norm_out.forward(output)
        ffn_out = self.ffn.forward(normed)
        output = output + ffn_out

        return output, attn_weights
