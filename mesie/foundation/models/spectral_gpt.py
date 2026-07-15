"""SpectralGPT — Foundation model for universal spectral intelligence.

This module implements the complete SpectralGPT architecture, a transformer-based
foundation model designed to learn the structure of frequency-domain reality
across multiple spectral modalities (seismic, vibration, EEG/ECG, audio, RF,
and synthetic physics simulations).

Architecture:
    Input → SpectralInputEncoder → Positional Encoding → 
    N × (TransformerBlock + optional MoE + optional CrossModal) →
    OutputHeads → Loss
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesie.foundation.models.transformer_blocks import (
    SpectralTransformerBlock,
    CrossModalAttentionBlock,
    RMSNorm,
)
from mesie.foundation.models.positional_encoding import SpectralPositionalEncoding
from mesie.foundation.models.spectral_encoder import SpectralInputEncoder
from mesie.foundation.models.mixture_of_experts import MixtureOfExperts
from mesie.foundation.models.output_heads import (
    SpectralReconstructionHead,
    NextWindowPredictionHead,
    ContrastiveProjectionHead,
    ClassificationHead,
    MultiTaskHead,
)


class SpectralEmbedding:
    """Embedding layer for spectral tokens.

    Converts discrete token IDs or continuous spectral patches into
    dense embeddings suitable for transformer processing.

    Attributes:
        vocab_size: Number of discrete tokens.
        embedding_dim: Embedding dimension.
        max_seq_len: Maximum sequence length.
        padding_idx: Index for padding token.
        use_continuous: Whether to support continuous inputs.
        continuous_dim: Dimension for continuous input projection.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        embedding_dim: int = 1024,
        max_seq_len: int = 8192,
        padding_idx: int = 0,
        use_continuous: bool = True,
        continuous_dim: int = 256,
        num_modalities: int = 7,
    ):
        """Initialize spectral embedding.

        Args:
            vocab_size: Vocabulary size for discrete tokens.
            embedding_dim: Output embedding dimension.
            max_seq_len: Maximum sequence length.
            padding_idx: Padding token index.
            use_continuous: Support continuous input.
            continuous_dim: Continuous input dimension.
            num_modalities: Number of modalities.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.use_continuous = use_continuous
        self.continuous_dim = continuous_dim

        # Token embedding table
        self.token_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.token_embeddings[padding_idx] = 0.0  # Zero padding

        # Continuous projection
        if use_continuous:
            self.continuous_proj = np.random.randn(continuous_dim, embedding_dim) * 0.02
            self.continuous_bias = np.zeros(embedding_dim)

        # Modality embeddings
        self.modality_embeddings = np.random.randn(num_modalities, embedding_dim) * 0.02

        # Segment embeddings (for multi-segment inputs)
        self.segment_embeddings = np.random.randn(8, embedding_dim) * 0.02

        # Embedding normalization
        self.norm = RMSNorm(embedding_dim)

        # Scale factor
        self.scale = math.sqrt(embedding_dim)

    def embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """Embed discrete token IDs.

        Args:
            token_ids: Token indices [...] (integer array).

        Returns:
            Embeddings [..., embedding_dim].
        """
        flat_ids = token_ids.flatten().astype(int)
        embeddings = self.token_embeddings[flat_ids]
        output_shape = list(token_ids.shape) + [self.embedding_dim]
        return embeddings.reshape(output_shape) * self.scale

    def embed_continuous(self, x: np.ndarray) -> np.ndarray:
        """Embed continuous spectral values.

        Args:
            x: Continuous input [..., continuous_dim].

        Returns:
            Embeddings [..., embedding_dim].
        """
        if not self.use_continuous:
            raise RuntimeError("Continuous embedding not enabled")

        return np.einsum("...d,do->...o", x, self.continuous_proj) + self.continuous_bias

    def add_modality_embedding(
        self, embeddings: np.ndarray, modality_id: int
    ) -> np.ndarray:
        """Add modality-specific embedding.

        Args:
            embeddings: Base embeddings [..., embedding_dim].
            modality_id: Modality index.

        Returns:
            Embeddings with modality information [..., embedding_dim].
        """
        return embeddings + self.modality_embeddings[modality_id]

    def forward(
        self,
        token_ids: Optional[np.ndarray] = None,
        continuous_input: Optional[np.ndarray] = None,
        modality_id: Optional[int] = None,
        segment_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute embeddings from tokens or continuous input.

        Args:
            token_ids: Optional discrete token IDs.
            continuous_input: Optional continuous spectral input.
            modality_id: Optional modality identifier.
            segment_ids: Optional segment identifiers.

        Returns:
            Combined embeddings [..., embedding_dim].
        """
        if token_ids is not None:
            embeddings = self.embed_tokens(token_ids)
        elif continuous_input is not None:
            embeddings = self.embed_continuous(continuous_input)
        else:
            raise ValueError("Either token_ids or continuous_input must be provided")

        # Add modality embedding
        if modality_id is not None:
            embeddings = self.add_modality_embedding(embeddings, modality_id)

        # Add segment embedding
        if segment_ids is not None:
            seg_flat = segment_ids.flatten().astype(int)
            seg_emb = self.segment_embeddings[seg_flat]
            seg_emb = seg_emb.reshape(list(segment_ids.shape) + [self.embedding_dim])
            embeddings = embeddings + seg_emb

        # Normalize
        embeddings = self.norm.forward(embeddings)

        return embeddings


class SpectralGPT:
    """SpectralGPT — Foundation model for frequency-domain intelligence.

    A transformer-based foundation model that learns universal spectral
    representations through self-supervised pretraining on diverse
    frequency-domain data. The model architecture combines:

    1. Spectral-aware input encoding with learnable DFT and wavelet layers
    2. Multi-head attention with frequency-band and multi-scale variants
    3. Mixture of Experts for modality specialization
    4. Cross-modal attention for multi-domain learning
    5. Multi-task output heads for diverse pretraining objectives

    Just as GPT learned the structure of language, SpectralGPT learns
    the structure of frequency-domain reality.

    Attributes:
        config: Model configuration dictionary.
        hidden_dim: Model hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        max_seq_len: Maximum sequence length.
        vocab_size: Vocabulary size.
        num_params: Total number of parameters.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        head_dim: int = 64,
        vocab_size: int = 32768,
        max_seq_len: int = 8192,
        ffn_dim: int = 4096,
        num_experts: int = 8,
        top_k_experts: int = 2,
        use_moe: bool = True,
        moe_layers: Optional[List[int]] = None,
        use_cross_modal: bool = True,
        cross_modal_layers: Optional[List[int]] = None,
        positional_encoding: str = "rotary",
        normalization: str = "rms_norm",
        activation: str = "swiglu",
        dropout: float = 0.0,
        num_modalities: int = 7,
        use_spectral_encoder: bool = True,
        spectral_input_dim: int = 1024,
        continuous_dim: int = 256,
        causal: bool = False,
        num_kv_heads: int = 0,
        qk_norm: bool = True,
        tie_embeddings: bool = True,
        init_std: float = 0.02,
    ):
        """Initialize SpectralGPT.

        Args:
            hidden_dim: Model hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            vocab_size: Vocabulary size for discrete tokens.
            max_seq_len: Maximum sequence length.
            ffn_dim: Feed-forward dimension.
            num_experts: Number of MoE experts.
            top_k_experts: Active experts per token.
            use_moe: Whether to use Mixture of Experts.
            moe_layers: Which layers use MoE (default: every 2nd layer).
            use_cross_modal: Whether to use cross-modal attention.
            cross_modal_layers: Which layers have cross-modal attention.
            positional_encoding: Positional encoding type.
            normalization: Normalization type.
            activation: Activation function.
            dropout: Dropout rate.
            num_modalities: Number of spectral modalities.
            use_spectral_encoder: Whether to use spectral input encoder.
            spectral_input_dim: Raw spectral input dimension.
            continuous_dim: Continuous token dimension.
            causal: Whether to use causal attention.
            num_kv_heads: Number of KV heads for GQA.
            qk_norm: Whether to normalize Q and K.
            tie_embeddings: Whether to tie input/output embeddings.
            init_std: Weight initialization standard deviation.
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.ffn_dim = ffn_dim
        self.num_modalities = num_modalities
        self.use_moe = use_moe
        self.use_cross_modal = use_cross_modal
        self.causal = causal
        self.tie_embeddings = tie_embeddings

        # Determine MoE and cross-modal layers
        self.moe_layers = moe_layers or list(range(1, num_layers, 2))
        self.cross_modal_layers = cross_modal_layers or [
            num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1
        ]

        # Store config
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "ffn_dim": ffn_dim,
            "num_experts": num_experts,
            "use_moe": use_moe,
            "positional_encoding": positional_encoding,
            "activation": activation,
            "num_modalities": num_modalities,
        }

        # ---- Build model components ----

        # 1. Embedding layer
        self.embedding = SpectralEmbedding(
            vocab_size=vocab_size,
            embedding_dim=hidden_dim,
            max_seq_len=max_seq_len,
            use_continuous=True,
            continuous_dim=continuous_dim,
            num_modalities=num_modalities,
        )

        # 2. Spectral input encoder (optional)
        self.use_spectral_encoder = use_spectral_encoder
        if use_spectral_encoder:
            self.spectral_encoder = SpectralInputEncoder(
                input_dim=spectral_input_dim,
                output_dim=hidden_dim,
                use_dft=True,
                use_wavelet=True,
                use_harmonic=True,
                use_octave_pooling=True,
            )

        # 3. Positional encoding
        self.pos_encoding = SpectralPositionalEncoding(
            encoding_type=positional_encoding,
            dim=head_dim if positional_encoding == "rotary" else hidden_dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
        )

        # 4. Transformer layers
        self.layers: List[Dict[str, Any]] = []
        for i in range(num_layers):
            layer: Dict[str, Any] = {
                "idx": i,
                "transformer": SpectralTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_dim=ffn_dim,
                    normalization=normalization,
                    pre_norm=True,
                    activation=activation,
                    dropout=dropout,
                    causal=causal,
                    num_kv_heads=num_kv_heads,
                    qk_norm=qk_norm,
                    layer_idx=i,
                ),
            }

            # Add MoE layer
            if use_moe and i in self.moe_layers:
                layer["moe"] = MixtureOfExperts(
                    hidden_dim=hidden_dim,
                    num_experts=num_experts,
                    top_k=top_k_experts,
                    expert_dim=ffn_dim,
                    modality_aware=True,
                    num_modalities=num_modalities,
                    activation=activation,
                )

            # Add cross-modal attention
            if use_cross_modal and i in self.cross_modal_layers:
                layer["cross_modal"] = CrossModalAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads // 2,
                    head_dim=head_dim,
                    dropout=dropout,
                )

            self.layers.append(layer)

        # 5. Final normalization
        self.final_norm = RMSNorm(hidden_dim)

        # 6. Output heads
        self.output_heads = MultiTaskHead(
            input_dim=hidden_dim,
            task_configs={
                "reconstruction": {"type": "reconstruction", "output_dim": hidden_dim},
                "next_window": {"type": "next_window", "prediction_steps": 4},
                "contrastive": {"type": "contrastive", "projection_dim": 256},
            },
        )

        # 7. Language model head (for discrete tokens)
        if tie_embeddings:
            self.lm_head_weight = self.embedding.token_embeddings.T  # [hidden, vocab]
        else:
            self.lm_head_weight = np.random.randn(hidden_dim, vocab_size) * init_std

        # Compute parameter count
        self.num_params = self._count_parameters()

    def _count_parameters(self) -> int:
        """Estimate total number of parameters."""
        params = 0

        # Embeddings
        params += self.vocab_size * self.hidden_dim  # token embeddings
        params += self.num_modalities * self.hidden_dim  # modality embeddings
        params += 8 * self.hidden_dim  # segment embeddings

        # Transformer layers
        per_layer = (
            # Attention: Q, K, V, O projections
            4 * self.hidden_dim * self.hidden_dim +
            # FFN (gated): gate, up, down
            3 * self.hidden_dim * self.ffn_dim +
            # Norms
            2 * self.hidden_dim
        )
        params += per_layer * self.num_layers

        # MoE layers (additional expert params)
        if self.use_moe:
            experts_per_layer = (
                3 * self.hidden_dim * self.ffn_dim * self.config["num_experts"]
            )
            params += experts_per_layer * len(self.moe_layers)

        # LM head (if not tied)
        if not self.tie_embeddings:
            params += self.hidden_dim * self.vocab_size

        return params

    def forward(
        self,
        token_ids: Optional[np.ndarray] = None,
        continuous_input: Optional[np.ndarray] = None,
        spectral_input: Optional[np.ndarray] = None,
        modality_id: Optional[int] = None,
        attention_mask: Optional[np.ndarray] = None,
        segment_ids: Optional[np.ndarray] = None,
        return_hidden_states: bool = False,
        return_attention: bool = False,
        active_tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Forward pass through SpectralGPT.

        Args:
            token_ids: Optional discrete token IDs [batch, seq_len].
            continuous_input: Optional continuous input [batch, seq_len, continuous_dim].
            spectral_input: Optional raw spectral input [batch, seq_len, spectral_dim].
            modality_id: Optional modality identifier.
            attention_mask: Optional attention mask.
            segment_ids: Optional segment identifiers.
            return_hidden_states: Whether to return all hidden states.
            return_attention: Whether to return attention weights.
            active_tasks: Optional list of active output tasks.

        Returns:
            Dictionary with model outputs including:
            - 'last_hidden_state': Final hidden states [batch, seq_len, hidden_dim]
            - 'logits': Token prediction logits (if applicable)
            - 'task_outputs': Output head predictions
            - 'hidden_states': All layer hidden states (if requested)
            - 'attentions': Attention weights (if requested)
            - 'moe_info': MoE routing information
        """
        outputs: Dict[str, Any] = {}
        all_hidden_states: List[np.ndarray] = []
        all_attentions: List[Optional[np.ndarray]] = []
        moe_losses: List[float] = []

        # ---- Input Processing ----

        # Option 1: Raw spectral input through spectral encoder
        if spectral_input is not None and self.use_spectral_encoder:
            hidden_states, encoder_info = self.spectral_encoder.forward(spectral_input)
            outputs["encoder_info"] = encoder_info
            # Ensure 3D: [batch, seq_len, hidden_dim]
            if hidden_states.ndim == 2:
                hidden_states = hidden_states[np.newaxis, ...]
            elif hidden_states.ndim == 1:
                hidden_states = hidden_states[np.newaxis, np.newaxis, :]

        # Option 2: Embedding from tokens or continuous input
        else:
            hidden_states = self.embedding.forward(
                token_ids=token_ids,
                continuous_input=continuous_input,
                modality_id=modality_id,
                segment_ids=segment_ids,
            )
            # Ensure 3D
            if hidden_states.ndim == 2:
                hidden_states = hidden_states[np.newaxis, ...]

        # ---- Positional Encoding ----
        seq_len = hidden_states.shape[-2]
        pos_bias = None

        if self.pos_encoding.encoding_type == "alibi":
            pos_bias = self.pos_encoding.encode(seq_len)
        elif self.pos_encoding.encoding_type in ("sinusoidal", "frequency_aware", "spectral_harmonic", "learned"):
            pos_enc = self.pos_encoding.encode(seq_len)
            hidden_states = hidden_states + pos_enc[:seq_len, :self.hidden_dim]
        # For rotary: applied inside attention (handled by attention layer)

        if return_hidden_states:
            all_hidden_states.append(hidden_states.copy())

        # ---- Transformer Layers ----
        for layer_dict in self.layers:
            layer_idx = layer_dict["idx"]

            # Self-attention + FFN
            transformer = layer_dict["transformer"]
            hidden_states, attn_weights = transformer.forward(
                hidden_states, attention_mask, pos_bias
            )

            if return_attention:
                all_attentions.append(attn_weights)

            # MoE (if present in this layer)
            if "moe" in layer_dict:
                moe_output, moe_info = layer_dict["moe"].forward(
                    hidden_states, modality_id=modality_id
                )
                hidden_states = hidden_states + moe_output
                moe_losses.append(moe_info.get("load_balance_loss", 0.0))

            # Cross-modal attention (if present in this layer)
            if "cross_modal" in layer_dict:
                # Use hidden states as both query and context
                # In practice, context comes from other modalities
                cross_output, cross_attn = layer_dict["cross_modal"].forward(
                    hidden_states, hidden_states
                )
                hidden_states = cross_output

            if return_hidden_states:
                all_hidden_states.append(hidden_states.copy())

        # ---- Final Normalization ----
        hidden_states = self.final_norm.forward(hidden_states)

        # ---- Output Processing ----
        outputs["last_hidden_state"] = hidden_states

        # Token prediction logits
        logits = np.einsum("...d,dv->...v", hidden_states, self.lm_head_weight)
        outputs["logits"] = logits

        # Task-specific outputs
        if active_tasks is not None:
            # Pool for task heads (use mean pooling)
            pooled = np.mean(hidden_states, axis=-2)
            outputs["task_outputs"] = self.output_heads.forward(pooled, active_tasks)

        # Additional outputs
        if return_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        if return_attention:
            outputs["attentions"] = all_attentions
        if moe_losses:
            outputs["moe_loss"] = float(np.mean(moe_losses))
        else:
            outputs["moe_loss"] = 0.0

        outputs["num_params"] = self.num_params

        return outputs

    def get_embeddings(
        self,
        token_ids: Optional[np.ndarray] = None,
        continuous_input: Optional[np.ndarray] = None,
        spectral_input: Optional[np.ndarray] = None,
        modality_id: Optional[int] = None,
        pooling: str = "mean",
        layer: int = -1,
    ) -> np.ndarray:
        """Extract embeddings from the model.

        Args:
            token_ids: Optional token IDs.
            continuous_input: Optional continuous input.
            spectral_input: Optional raw spectral input.
            modality_id: Optional modality ID.
            pooling: Pooling method ('mean', 'max', 'cls', 'last').
            layer: Which layer to extract from (-1 = last).

        Returns:
            Embeddings [batch, hidden_dim].
        """
        outputs = self.forward(
            token_ids=token_ids,
            continuous_input=continuous_input,
            spectral_input=spectral_input,
            modality_id=modality_id,
            return_hidden_states=(layer != -1),
        )

        if layer == -1:
            hidden = outputs["last_hidden_state"]
        else:
            hidden = outputs["hidden_states"][layer]

        # Pool
        if pooling == "mean":
            return np.mean(hidden, axis=-2)
        elif pooling == "max":
            return np.max(hidden, axis=-2)
        elif pooling == "cls":
            return hidden[..., 0, :]
        elif pooling == "last":
            return hidden[..., -1, :]
        else:
            return np.mean(hidden, axis=-2)

    def generate(
        self,
        prompt: np.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        modality_id: Optional[int] = None,
    ) -> np.ndarray:
        """Generate spectral tokens autoregressively.

        Args:
            prompt: Initial token sequence [seq_len] or [1, seq_len].
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Top-P (nucleus) sampling parameter.
            modality_id: Optional modality for generation.

        Returns:
            Generated token sequence [seq_len + max_new_tokens].
        """
        if prompt.ndim == 1:
            prompt = prompt[np.newaxis, :]

        generated = prompt.copy()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = generated[:, -self.max_seq_len:]

            # Forward pass
            outputs = self.forward(
                token_ids=context,
                modality_id=modality_id,
            )

            # Get logits for last position
            next_logits = outputs["logits"][:, -1, :]  # [1, vocab_size]

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_indices = np.argsort(next_logits[0])[::-1][:top_k]
                mask = np.full(next_logits.shape[-1], -np.inf)
                mask[top_k_indices] = 0.0
                next_logits = next_logits + mask

            # Softmax
            logits_max = np.max(next_logits, axis=-1, keepdims=True)
            probs = np.exp(next_logits - logits_max)
            probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-10)

            # Top-P filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(probs[0])[::-1]
                sorted_probs = probs[0][sorted_indices]
                cumsum = np.cumsum(sorted_probs)
                cutoff_idx = np.searchsorted(cumsum, top_p) + 1
                mask_indices = sorted_indices[cutoff_idx:]
                probs[0][mask_indices] = 0.0
                probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-10)

            # Sample
            next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])
            generated = np.concatenate([generated, next_token], axis=-1)

            # Stop if EOS token (index 2)
            if next_token[0, 0] == 2:
                break

        return generated[0]  # Remove batch dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.

        Returns:
            Dictionary with model architecture details.
        """
        return {
            "model_type": "SpectralGPT",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "ffn_dim": self.ffn_dim,
            "num_params": self.num_params,
            "num_params_readable": self._format_params(self.num_params),
            "use_moe": self.use_moe,
            "moe_layers": self.moe_layers,
            "use_cross_modal": self.use_cross_modal,
            "cross_modal_layers": self.cross_modal_layers,
            "num_modalities": self.num_modalities,
            "causal": self.causal,
        }

    @staticmethod
    def _format_params(num_params: int) -> str:
        """Format parameter count in human-readable form."""
        if num_params >= 1e9:
            return f"{num_params / 1e9:.1f}B"
        elif num_params >= 1e6:
            return f"{num_params / 1e6:.1f}M"
        elif num_params >= 1e3:
            return f"{num_params / 1e3:.1f}K"
        return str(num_params)


# ---------------------------------------------------------------------------
# Model Factory Functions
# ---------------------------------------------------------------------------

def create_spectral_gpt_tiny() -> SpectralGPT:
    """Create a tiny SpectralGPT for testing."""
    return SpectralGPT(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        head_dim=64,
        vocab_size=4096,
        max_seq_len=512,
        ffn_dim=1024,
        num_experts=4,
        top_k_experts=2,
        use_moe=False,
        use_cross_modal=False,
        use_spectral_encoder=False,
        continuous_dim=64,
    )


def create_spectral_gpt_small() -> SpectralGPT:
    """Create a small SpectralGPT for single-GPU training."""
    return SpectralGPT(
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        head_dim=64,
        vocab_size=16384,
        max_seq_len=2048,
        ffn_dim=2048,
        num_experts=8,
        top_k_experts=2,
        use_moe=True,
        use_cross_modal=True,
    )


def create_spectral_gpt_base() -> SpectralGPT:
    """Create the base SpectralGPT model."""
    return SpectralGPT(
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        head_dim=64,
        vocab_size=32768,
        max_seq_len=8192,
        ffn_dim=4096,
        num_experts=8,
        top_k_experts=2,
        use_moe=True,
        use_cross_modal=True,
    )


def create_spectral_gpt_large() -> SpectralGPT:
    """Create a large SpectralGPT model."""
    return SpectralGPT(
        hidden_dim=2048,
        num_layers=36,
        num_heads=32,
        head_dim=64,
        vocab_size=65536,
        max_seq_len=16384,
        ffn_dim=8192,
        num_experts=16,
        top_k_experts=4,
        use_moe=True,
        use_cross_modal=True,
    )
