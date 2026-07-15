"""Spectral Foundation Model Architecture.

Provides a scalable, pre-training-ready architecture for spectral intelligence.
Designed as a foundation model that can be pre-trained on diverse spectral data
and fine-tuned for specific downstream tasks.

Architecture Components:
    - SpectralPatchEmbedding: Converts spectral data to patch embeddings
    - RotaryPositionalEncoding: Position-aware encoding for frequency ordering
    - GatedAttention: Gated multi-head attention with spectral bias
    - MixtureOfExperts: Sparse MoE layer for efficient capacity scaling
    - FoundationEncoder: Full foundation model encoder stack
    - SpectralFoundationModel: Complete model with task heads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FoundationModelConfig:
    """Configuration for the Spectral Foundation Model.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_feedforward: Feed-forward network dimension.
        n_experts: Number of experts in MoE layers.
        top_k_experts: Number of experts to route to per token.
        patch_size: Size of spectral patches.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        use_rotary: Whether to use rotary positional encoding.
        use_gated_attention: Whether to use gated attention.
        use_moe: Whether to use Mixture of Experts.
        vocab_size: Vocabulary size for discrete tokenization.
        n_registers: Number of register tokens.
        layer_scale_init: Initial value for layer scale.
    """

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 12
    d_feedforward: int = 1024
    n_experts: int = 8
    top_k_experts: int = 2
    patch_size: int = 16
    max_seq_len: int = 512
    dropout: float = 0.1
    use_rotary: bool = True
    use_gated_attention: bool = True
    use_moe: bool = True
    vocab_size: int = 8192
    n_registers: int = 4
    layer_scale_init: float = 0.1


@dataclass
class FoundationModelOutput:
    """Output from the Spectral Foundation Model.

    Args:
        sequence_output: Full sequence hidden states (seq_len, d_model).
        pooled_output: Pooled representation (d_model,).
        attention_maps: Per-layer attention maps.
        expert_assignments: Per-layer expert routing decisions.
        register_states: Register token final states.
        metadata: Processing metadata.
    """

    sequence_output: np.ndarray
    pooled_output: np.ndarray
    attention_maps: List[np.ndarray] = field(default_factory=list)
    expert_assignments: List[np.ndarray] = field(default_factory=list)
    register_states: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Patch Embedding
# =============================================================================


class SpectralPatchEmbedding:
    """Convert spectral data into patch embeddings.

    Divides the input spectrum into non-overlapping patches and
    projects each patch into the model dimension space.

    Args:
        patch_size: Number of spectral points per patch.
        d_model: Output embedding dimension.
        overlap: Patch overlap (0 = no overlap).
    """

    def __init__(
        self,
        patch_size: int = 16,
        d_model: int = 256,
        overlap: int = 0,
    ) -> None:
        self.patch_size = patch_size
        self.d_model = d_model
        self.overlap = overlap
        self.stride = patch_size - overlap

        # Projection matrix
        self._projection = np.random.randn(patch_size, d_model) * (2.0 / (patch_size + d_model)) ** 0.5
        # Bias
        self._bias = np.zeros(d_model)
        # [CLS] token
        self._cls_token = np.random.randn(1, d_model) * 0.02

    def embed(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert spectrum to patch embeddings.

        Args:
            spectrum: Input spectrum of arbitrary length.

        Returns:
            Patch embeddings of shape (n_patches + 1, d_model) including CLS.
        """
        spectrum = np.atleast_1d(spectrum).flatten()

        # Pad to multiple of stride
        n_patches = max(1, (len(spectrum) - self.patch_size) // self.stride + 1)
        padded_len = (n_patches - 1) * self.stride + self.patch_size
        if padded_len > len(spectrum):
            spectrum = np.pad(spectrum, (0, padded_len - len(spectrum)))

        # Extract patches
        patches = np.zeros((n_patches, self.patch_size))
        for i in range(n_patches):
            start = i * self.stride
            patches[i] = spectrum[start:start + self.patch_size]

        # Project to d_model
        embeddings = patches @ self._projection + self._bias

        # Prepend CLS token
        embeddings = np.vstack([self._cls_token, embeddings])

        return embeddings

    @property
    def n_parameters(self) -> int:
        """Number of learnable parameters."""
        return self.patch_size * self.d_model + self.d_model + self.d_model


# =============================================================================
# Rotary Positional Encoding
# =============================================================================


class RotaryPositionalEncoding:
    """Rotary Position Embedding (RoPE) for spectral sequences.

    Implements rotation-based positional encoding that naturally
    captures relative position information without explicit position IDs.

    Args:
        d_model: Model dimension.
        max_seq_len: Maximum sequence length.
        base: Base frequency for rotation.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_seq_len: int = 512,
        base: float = 10000.0,
    ) -> None:
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        dim = d_model // 2
        freqs = 1.0 / (base ** (np.arange(0, dim, dtype=float) / dim))
        positions = np.arange(max_seq_len, dtype=float)
        self._angles = np.outer(positions, freqs)
        self._cos = np.cos(self._angles)
        self._sin = np.sin(self._angles)

    def apply(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """Apply rotary positional encoding.

        Args:
            x: Input tensor of shape (seq_len, d_model).
            offset: Position offset for continuation.

        Returns:
            Position-encoded tensor of same shape.
        """
        seq_len, d = x.shape
        half_d = d // 2

        # Split into two halves
        x1 = x[:, :half_d]
        x2 = x[:, half_d:2 * half_d]

        # Get relevant rotations
        cos = self._cos[offset:offset + seq_len, :half_d]
        sin = self._sin[offset:offset + seq_len, :half_d]

        # Apply rotation
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Reconstruct
        if d > 2 * half_d:
            return np.concatenate([out1, out2, x[:, 2 * half_d:]], axis=1)
        return np.concatenate([out1, out2], axis=1)


# =============================================================================
# Gated Multi-Head Attention
# =============================================================================


class GatedMultiHeadAttention:
    """Gated multi-head attention with spectral bias.

    Extends standard multi-head attention with:
    - Gate mechanism for selective information flow
    - Spectral bias that naturally attends to harmonic relationships
    - Relative position awareness via rotary encoding

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        use_gate: Whether to use the gating mechanism.
        spectral_bias_strength: Strength of the spectral harmonic bias.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        use_gate: bool = True,
        spectral_bias_strength: float = 0.1,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_gate = use_gate
        self.spectral_bias_strength = spectral_bias_strength

        scale = 0.02
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        if use_gate:
            self.W_gate = np.random.randn(d_model, d_model) * scale
            self.gate_bias = np.zeros(d_model)

    def _compute_spectral_bias(self, seq_len: int) -> np.ndarray:
        """Compute harmonic-aware attention bias matrix."""
        bias = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if j > 0 and i > 0:
                    ratio = max(i, j) / min(i, j)
                    # Boost attention for harmonic ratios (2x, 3x, etc.)
                    if abs(ratio - round(ratio)) < 0.1:
                        bias[i, j] = self.spectral_bias_strength
        return bias

    def forward(
        self,
        x: np.ndarray,
        rotary_encoder: Optional[RotaryPositionalEncoding] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gated multi-head attention.

        Args:
            x: Input of shape (seq_len, d_model).
            rotary_encoder: Optional rotary position encoder.

        Returns:
            Tuple of (output, attention_weights).
        """
        seq_len = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Apply rotary encoding to Q and K
        if rotary_encoder is not None:
            Q = rotary_encoder.apply(Q)
            K = rotary_encoder.apply(K)

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_heads, self.head_dim)

        # Scaled dot-product attention
        scores = np.einsum("qhd,khd->hqk", Q, K) / np.sqrt(self.head_dim)

        # Add spectral bias
        if self.spectral_bias_strength > 0:
            bias = self._compute_spectral_bias(seq_len)
            scores += bias[np.newaxis, :, :]

        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-10)

        # Apply attention to values
        output = np.einsum("hqk,khd->qhd", attn_weights, V)
        output = output.reshape(seq_len, self.d_model)
        output = output @ self.W_o

        # Gating mechanism
        if self.use_gate:
            gate = self._sigmoid(x @ self.W_gate + self.gate_bias)
            output = output * gate

        # Average attention across heads
        avg_attn = np.mean(attn_weights, axis=0)
        return output, avg_attn

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# =============================================================================
# Mixture of Experts
# =============================================================================


class ExpertNetwork:
    """A single expert feed-forward network.

    Args:
        d_model: Input/output dimension.
        d_expert: Hidden dimension of this expert.
    """

    def __init__(self, d_model: int = 256, d_expert: int = 512) -> None:
        self.d_model = d_model
        self.d_expert = d_expert
        scale = (2.0 / (d_model + d_expert)) ** 0.5
        self.W1 = np.random.randn(d_model, d_expert) * scale
        self.W2 = np.random.randn(d_expert, d_model) * scale
        self.b1 = np.zeros(d_expert)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through expert.

        Args:
            x: Input of shape (seq_len, d_model).

        Returns:
            Output of shape (seq_len, d_model).
        """
        # SwiGLU-inspired activation
        h = x @ self.W1 + self.b1
        h = h * self._silu(h)
        return h @ self.W2 + self.b2

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) activation."""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))


class MixtureOfExperts:
    """Sparse Mixture of Experts layer for efficient capacity scaling.

    Routes each token to a subset of experts, allowing the model
    to have large total capacity while keeping per-token computation
    manageable.

    Args:
        d_model: Model dimension.
        n_experts: Total number of experts.
        top_k: Number of experts to route each token to.
        d_expert: Hidden dimension per expert.
        load_balancing_weight: Weight for load balancing auxiliary loss.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_experts: int = 8,
        top_k: int = 2,
        d_expert: int = 512,
        load_balancing_weight: float = 0.01,
    ) -> None:
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight

        # Router
        self.router = np.random.randn(d_model, n_experts) * 0.02

        # Experts
        self.experts = [
            ExpertNetwork(d_model, d_expert) for _ in range(n_experts)
        ]

        # Statistics
        self._routing_counts = np.zeros(n_experts)
        self._total_tokens = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Route tokens to experts and combine outputs.

        Args:
            x: Input of shape (seq_len, d_model).

        Returns:
            Tuple of (output, routing_assignments).
        """
        seq_len = x.shape[0]
        self._total_tokens += seq_len

        # Compute routing logits
        logits = x @ self.router  # (seq_len, n_experts)

        # Top-k routing
        top_k_indices = np.argsort(logits, axis=1)[:, -self.top_k:]
        top_k_logits = np.take_along_axis(logits, top_k_indices, axis=1)

        # Softmax over selected experts
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=1, keepdims=True))
        routing_weights = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)

        # Process through experts
        output = np.zeros_like(x)
        for token_idx in range(seq_len):
            for k in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k]
                weight = routing_weights[token_idx, k]
                expert_output = self.experts[expert_idx].forward(
                    x[token_idx:token_idx + 1]
                )
                output[token_idx] += weight * expert_output[0]
                self._routing_counts[expert_idx] += 1

        return output, top_k_indices

    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get expert load balancing statistics.

        Returns:
            Dictionary with routing distribution info.
        """
        total = np.sum(self._routing_counts) + 1e-12
        distribution = self._routing_counts / total
        entropy = float(-np.sum(distribution * np.log(distribution + 1e-12)))
        max_load = float(np.max(distribution))
        min_load = float(np.min(distribution))

        return {
            "routing_distribution": distribution.tolist(),
            "entropy": entropy,
            "max_load": max_load,
            "min_load": min_load,
            "load_imbalance": max_load - min_load,
            "total_tokens_routed": int(self._total_tokens),
        }


# =============================================================================
# Foundation Encoder Layer
# =============================================================================


class FoundationEncoderLayer:
    """A single layer of the Foundation Model encoder.

    Combines gated attention, optional MoE feed-forward, layer normalization,
    and residual connections with layer scale.

    Args:
        config: Foundation model configuration.
        layer_idx: Index of this layer in the stack.
    """

    def __init__(self, config: FoundationModelConfig, layer_idx: int = 0) -> None:
        self.config = config
        self.layer_idx = layer_idx

        # Attention
        self.attention = GatedMultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            use_gate=config.use_gated_attention,
        )

        # Feed-forward (MoE or standard)
        if config.use_moe and layer_idx % 2 == 0:  # MoE every other layer
            self.feed_forward: Any = MixtureOfExperts(
                d_model=config.d_model,
                n_experts=config.n_experts,
                top_k=config.top_k_experts,
                d_expert=config.d_feedforward,
            )
            self.is_moe = True
        else:
            self.ff_w1 = np.random.randn(config.d_model, config.d_feedforward) * 0.02
            self.ff_w2 = np.random.randn(config.d_feedforward, config.d_model) * 0.02
            self.is_moe = False

        # Layer scale parameters
        self.attn_scale = np.ones(config.d_model) * config.layer_scale_init
        self.ff_scale = np.ones(config.d_model) * config.layer_scale_init

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """RMS Layer Normalization."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(
        self,
        x: np.ndarray,
        rotary_encoder: Optional[RotaryPositionalEncoding] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Forward pass through encoder layer.

        Args:
            x: Input of shape (seq_len, d_model).
            rotary_encoder: Optional rotary position encoder.

        Returns:
            Tuple of (output, attention_weights, expert_assignments).
        """
        # Pre-norm attention
        normed = self._layer_norm(x)
        attn_out, attn_weights = self.attention.forward(normed, rotary_encoder)
        x = x + self.attn_scale * attn_out

        # Pre-norm feed-forward
        normed = self._layer_norm(x)
        expert_assignments = None

        if self.is_moe:
            ff_out, expert_assignments = self.feed_forward.forward(normed)
        else:
            ff_out = self._gelu(normed @ self.ff_w1) @ self.ff_w2

        x = x + self.ff_scale * ff_out

        return x, attn_weights, expert_assignments


# =============================================================================
# Spectral Foundation Model
# =============================================================================


class SpectralFoundationModel:
    """Complete Spectral Foundation Model for pre-training and fine-tuning.

    Provides a scalable transformer architecture optimized for spectral data,
    with support for:
    - Patch-based spectral embedding
    - Rotary positional encoding for frequency ordering
    - Gated attention with harmonic bias
    - Mixture of Experts for efficient scaling
    - Register tokens for global information aggregation
    - Multiple output heads for different downstream tasks

    Args:
        config: Foundation model configuration.
    """

    def __init__(self, config: Optional[FoundationModelConfig] = None) -> None:
        self.config = config or FoundationModelConfig()

        # Patch embedding
        self.patch_embedding = SpectralPatchEmbedding(
            patch_size=self.config.patch_size,
            d_model=self.config.d_model,
        )

        # Rotary positional encoding
        self.rotary_encoder = RotaryPositionalEncoding(
            d_model=self.config.d_model,
            max_seq_len=self.config.max_seq_len,
        ) if self.config.use_rotary else None

        # Register tokens (for global information aggregation)
        self.register_tokens = np.random.randn(
            self.config.n_registers, self.config.d_model
        ) * 0.02

        # Encoder layers
        self.encoder_layers = [
            FoundationEncoderLayer(self.config, layer_idx=i)
            for i in range(self.config.n_layers)
        ]

        # Output heads
        self._classification_head = np.random.randn(self.config.d_model, 10) * 0.02
        self._reconstruction_head = np.random.randn(
            self.config.d_model, self.config.patch_size
        ) * 0.02
        self._contrastive_head = np.random.randn(
            self.config.d_model, self.config.d_model // 2
        ) * 0.02

        # State
        self._processing_count = 0
        self._total_tokens_processed = 0

    def encode(self, spectrum: np.ndarray) -> FoundationModelOutput:
        """Encode a spectrum through the full foundation model.

        Args:
            spectrum: Input spectral data of arbitrary length.

        Returns:
            FoundationModelOutput with all outputs and intermediate states.
        """
        self._processing_count += 1
        spectrum = np.atleast_1d(spectrum).flatten()

        # Patch embedding (includes CLS)
        x = self.patch_embedding.embed(spectrum)

        # Prepend register tokens
        x = np.vstack([self.register_tokens, x])
        n_registers = self.config.n_registers

        self._total_tokens_processed += x.shape[0]

        # Pass through encoder layers
        attention_maps = []
        expert_assignments_all = []

        for layer in self.encoder_layers:
            x, attn_weights, expert_assignments = layer.forward(x, self.rotary_encoder)
            attention_maps.append(attn_weights)
            if expert_assignments is not None:
                expert_assignments_all.append(expert_assignments)

        # Extract outputs
        register_states = x[:n_registers]
        cls_token = x[n_registers]  # First token after registers
        sequence_output = x[n_registers + 1:]  # Patch tokens

        # Pooled output: combine CLS with register mean
        register_mean = np.mean(register_states, axis=0) if n_registers > 0 else np.zeros(self.config.d_model)
        pooled_output = (cls_token + register_mean) / 2.0

        return FoundationModelOutput(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            attention_maps=attention_maps,
            expert_assignments=expert_assignments_all,
            register_states=register_states,
            metadata={
                "n_patches": sequence_output.shape[0],
                "n_registers": n_registers,
                "n_layers": self.config.n_layers,
                "d_model": self.config.d_model,
                "processing_id": self._processing_count,
                "total_tokens": self._total_tokens_processed,
            },
        )

    def get_embedding(self, spectrum: np.ndarray) -> np.ndarray:
        """Get a fixed-size embedding for a spectrum.

        Args:
            spectrum: Input spectral data.

        Returns:
            Embedding vector of shape (d_model,).
        """
        output = self.encode(spectrum)
        return output.pooled_output

    def get_contrastive_embedding(self, spectrum: np.ndarray) -> np.ndarray:
        """Get a contrastive learning embedding.

        Args:
            spectrum: Input spectral data.

        Returns:
            Projected embedding for contrastive learning.
        """
        pooled = self.get_embedding(spectrum)
        projected = pooled @ self._contrastive_head
        # L2 normalize
        norm = np.linalg.norm(projected) + 1e-12
        return projected / norm

    def reconstruct_patches(self, spectrum: np.ndarray) -> np.ndarray:
        """Predict/reconstruct spectral patches (for masked pre-training).

        Args:
            spectrum: Input spectral data.

        Returns:
            Reconstructed patches.
        """
        output = self.encode(spectrum)
        return output.sequence_output @ self._reconstruction_head

    def classify(self, spectrum: np.ndarray) -> np.ndarray:
        """Classify a spectrum into categories.

        Args:
            spectrum: Input spectral data.

        Returns:
            Class logits.
        """
        pooled = self.get_embedding(spectrum)
        logits = pooled @ self._classification_head
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def get_attention_analysis(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """Analyze attention patterns across all layers.

        Args:
            spectrum: Input spectral data.

        Returns:
            Dictionary with per-layer and aggregate attention metrics.
        """
        output = self.encode(spectrum)
        layer_analyses = []

        for i, attn_map in enumerate(output.attention_maps):
            # Compute metrics
            entropy = float(-np.sum(attn_map * np.log(attn_map + 1e-10)) / attn_map.size)
            max_attn = float(np.max(attn_map))
            sparsity = float(np.mean(attn_map < 0.01))

            layer_analyses.append({
                "layer": i,
                "attention_entropy": entropy,
                "maximum_attention": max_attn,
                "attention_sparsity": sparsity,
                "mean_attention": float(np.mean(attn_map)),
            })

        # MoE statistics
        moe_stats = {}
        for layer in self.encoder_layers:
            if layer.is_moe:
                stats = layer.feed_forward.get_load_balance_stats()
                moe_stats[f"layer_{layer.layer_idx}"] = stats

        return {
            "n_layers": len(output.attention_maps),
            "layer_analyses": layer_analyses,
            "mean_entropy": float(np.mean([a["attention_entropy"] for a in layer_analyses])),
            "mean_sparsity": float(np.mean([a["attention_sparsity"] for a in layer_analyses])),
            "moe_stats": moe_stats,
            "n_patches": output.metadata["n_patches"],
            "total_parameters_estimate": self.n_parameters,
        }

    @property
    def n_parameters(self) -> int:
        """Estimated total number of parameters."""
        d = self.config.d_model
        ff = self.config.d_feedforward
        n_layers = self.config.n_layers
        n_experts = self.config.n_experts

        # Attention: 4 * d^2 per layer
        attn_params = 4 * d * d * n_layers
        # FF: 2 * d * ff per non-MoE layer, n_experts * 2 * d * ff per MoE layer
        n_moe_layers = n_layers // 2 if self.config.use_moe else 0
        n_ff_layers = n_layers - n_moe_layers
        ff_params = n_ff_layers * 2 * d * ff + n_moe_layers * n_experts * 2 * d * ff
        # Embedding
        embed_params = self.config.patch_size * d
        # Heads
        head_params = d * 10 + d * self.config.patch_size + d * d // 2

        return attn_params + ff_params + embed_params + head_params

    @property
    def processing_count(self) -> int:
        """Total number of forward passes."""
        return self._processing_count


# =============================================================================
# Pre-training Objectives
# =============================================================================


class MaskedSpectralModeling:
    """Masked Spectral Modeling (MSM) pre-training objective.

    Similar to BERT's masked language modeling but for spectral patches.
    Randomly masks patches and trains the model to reconstruct them.

    Args:
        mask_ratio: Fraction of patches to mask.
        mask_value: Value to replace masked patches with.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_value: float = 0.0,
    ) -> None:
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    def create_masked_input(
        self,
        spectrum: np.ndarray,
        patch_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a masked version of the input spectrum.

        Args:
            spectrum: Input spectrum.
            patch_size: Size of each patch.

        Returns:
            Tuple of (masked_spectrum, mask_indices, original_patches).
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        n_patches = len(spectrum) // patch_size

        # Select patches to mask
        n_masked = max(1, int(n_patches * self.mask_ratio))
        mask_indices = np.random.choice(n_patches, n_masked, replace=False)

        # Create masked version
        masked = spectrum.copy()
        original_patches = np.zeros((n_masked, patch_size))

        for i, idx in enumerate(mask_indices):
            start = idx * patch_size
            end = start + patch_size
            original_patches[i] = spectrum[start:end]
            masked[start:end] = self.mask_value

        return masked, mask_indices, original_patches

    def compute_loss(
        self,
        predicted_patches: np.ndarray,
        original_patches: np.ndarray,
    ) -> float:
        """Compute MSM reconstruction loss.

        Args:
            predicted_patches: Model's patch predictions.
            original_patches: Original patch values.

        Returns:
            Mean squared error loss.
        """
        min_len = min(len(predicted_patches), len(original_patches))
        return float(np.mean(
            (predicted_patches[:min_len] - original_patches[:min_len]) ** 2
        ))


class ContrastiveSpectralLearning:
    """Contrastive learning objective for spectral representations.

    Learns embeddings by contrasting positive pairs (augmented views
    of the same spectrum) against negative pairs (different spectra).

    Args:
        temperature: Contrastive loss temperature.
        projection_dim: Dimension of projected embeddings.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        projection_dim: int = 128,
    ) -> None:
        self.temperature = temperature
        self.projection_dim = projection_dim

    def augment_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Create an augmented view of a spectrum.

        Applies random augmentations suitable for spectral data:
        - Random scaling
        - Gaussian noise addition
        - Random frequency masking
        - Time stretching

        Args:
            spectrum: Input spectrum.

        Returns:
            Augmented spectrum.
        """
        augmented = spectrum.copy()

        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        augmented *= scale

        # Gaussian noise
        noise_level = np.random.uniform(0.0, 0.05) * np.std(augmented)
        augmented += np.random.randn(len(augmented)) * noise_level

        # Random frequency masking (mask a random band)
        mask_width = int(len(augmented) * np.random.uniform(0.0, 0.1))
        if mask_width > 0:
            mask_start = np.random.randint(0, max(1, len(augmented) - mask_width))
            augmented[mask_start:mask_start + mask_width] = 0.0

        return augmented

    def compute_contrastive_loss(
        self,
        embeddings: np.ndarray,
        positive_pairs: List[Tuple[int, int]],
    ) -> float:
        """Compute NT-Xent contrastive loss.

        Args:
            embeddings: Matrix of embeddings (n_samples, dim).
            positive_pairs: List of (i, j) positive pair indices.

        Returns:
            Contrastive loss value.
        """
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        normalized = embeddings / norms

        # Similarity matrix
        similarity = normalized @ normalized.T / self.temperature

        # NT-Xent loss
        total_loss = 0.0
        for i, j in positive_pairs:
            # Numerator: similarity of positive pair
            pos_sim = similarity[i, j]
            # Denominator: sum of similarities with all negatives
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            neg_sims = similarity[i, mask]
            log_sum_exp = np.log(np.sum(np.exp(neg_sims)) + 1e-12)
            total_loss += -pos_sim + log_sum_exp

        return float(total_loss / max(1, len(positive_pairs)))


# =============================================================================
# Transfer Learning Adapter
# =============================================================================


class SpectralTransferHead:
    """Task-specific head for transfer learning from the foundation model.

    Provides lightweight adaptation layers that can be trained
    on top of frozen foundation model representations.

    Args:
        input_dim: Dimension of foundation model output.
        output_dim: Task-specific output dimension.
        n_hidden_layers: Number of hidden layers in the head.
        hidden_dim: Hidden layer dimension.
        task_type: Type of downstream task.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 10,
        n_hidden_layers: int = 2,
        hidden_dim: int = 128,
        task_type: str = "classification",
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type

        # Build MLP head
        self.layers: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        prev_dim = input_dim
        for _ in range(n_hidden_layers):
            self.layers.append(
                np.random.randn(prev_dim, hidden_dim) * (2.0 / (prev_dim + hidden_dim)) ** 0.5
            )
            self.biases.append(np.zeros(hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.layers.append(
            np.random.randn(prev_dim, output_dim) * (2.0 / (prev_dim + output_dim)) ** 0.5
        )
        self.biases.append(np.zeros(output_dim))

    def forward(self, embedding: np.ndarray) -> np.ndarray:
        """Forward pass through the transfer head.

        Args:
            embedding: Foundation model embedding.

        Returns:
            Task-specific output.
        """
        x = embedding
        for i, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            x = x @ layer + bias
            if i < len(self.layers) - 1:
                # GELU activation for hidden layers
                x = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

        # Final activation based on task type
        if self.task_type == "classification":
            exp_x = np.exp(x - np.max(x))
            x = exp_x / (np.sum(exp_x) + 1e-12)
        elif self.task_type == "regression":
            pass  # Linear output
        elif self.task_type == "detection":
            x = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # Sigmoid

        return x
