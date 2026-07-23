"""Native Auro-4B architecture contract and parameter accounting.

This module defines the actual 4B geometry. The full configuration is never
silently materialized in CI or on a laptop: callers must explicitly request the
full model. A ratio-preserving proxy is provided for executable tests and
pipeline validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal

Auro4BScale = Literal["proxy", "full"]


@dataclass(frozen=True)
class Auro4BArchitecture:
    model_id: str = "Auro-4B"
    architecture_version: str = "auro4b.native.v1"
    parameter_target: int = 4_000_000_000
    hidden_dim: int = 3072
    num_layers: int = 32
    num_attention_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn_dim: int = 10240
    vocab_size: int = 65536
    max_seq_len: int = 16384
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    qk_norm: bool = True
    bias: bool = False
    activation: str = "swiglu"
    normalization: str = "rms_norm"
    positional_encoding: str = "rotary"
    attention: str = "grouped_query_attention"
    structured_residual_every: int = 4
    structured_basis: str = "walsh_hadamard"
    checkpoint_constitution: str = "auro.substrate.checkpoint.v1"
    claims: Dict[str, bool] = field(default_factory=lambda: {
        "trained_general_knowledge": False,
        "benchmark_superiority": False,
        "hallucination_reduction": False,
        "long_context_extrapolation_verified": False,
    })

    def validate(self) -> None:
        if self.hidden_dim != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_dim must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.ffn_dim % 256:
            raise ValueError("ffn_dim must be hardware-aligned to 256")
        if self.max_seq_len <= 0 or self.vocab_size <= 0 or self.num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if self.structured_residual_every <= 0:
            raise ValueError("structured_residual_every must be positive")

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def parameter_estimate(self) -> Dict[str, int]:
        """Estimate trainable parameters for the decoder-only model.

        The count models tied token embeddings, GQA projections, SwiGLU MLPs,
        two RMSNorm vectors per block, and a final RMSNorm. It deliberately
        excludes optimizer state and KV-cache memory.
        """
        self.validate()
        d = self.hidden_dim
        kv = self.kv_dim
        attention_per_layer = d * d + d * kv + d * kv + d * d
        swiglu_per_layer = 3 * d * self.ffn_dim
        norms_per_layer = 2 * d
        block_total = attention_per_layer + swiglu_per_layer + norms_per_layer
        embeddings = self.vocab_size * d
        lm_head = 0 if self.tie_embeddings else d * self.vocab_size
        final_norm = d
        structured_diagonals = (
            self.num_layers // self.structured_residual_every
        ) * 2 * d
        total = (
            embeddings
            + lm_head
            + self.num_layers * block_total
            + final_norm
            + structured_diagonals
        )
        return {
            "embeddings": embeddings,
            "attention": self.num_layers * attention_per_layer,
            "swiglu": self.num_layers * swiglu_per_layer,
            "norms": self.num_layers * norms_per_layer + final_norm,
            "structured_residuals": structured_diagonals,
            "lm_head": lm_head,
            "total": total,
            "target": self.parameter_target,
            "delta_from_target": total - self.parameter_target,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["kv_dim"] = self.kv_dim
        payload["parameter_estimate"] = self.parameter_estimate()
        payload["gqa_group_size"] = self.num_attention_heads // self.num_kv_heads
        return payload


PROXY_ARCHITECTURE = Auro4BArchitecture(
    architecture_version="auro4b.proxy.v1",
    parameter_target=4_000_000_000,
    hidden_dim=384,
    num_layers=6,
    num_attention_heads=6,
    num_kv_heads=2,
    head_dim=64,
    ffn_dim=1280,
    vocab_size=8192,
    max_seq_len=1024,
    rope_theta=100_000.0,
    structured_residual_every=2,
)

FULL_ARCHITECTURE = Auro4BArchitecture()


def auro4b_architecture(scale: Auro4BScale = "proxy") -> Auro4BArchitecture:
    if scale == "proxy":
        return PROXY_ARCHITECTURE
    if scale == "full":
        return FULL_ARCHITECTURE
    raise ValueError(f"unknown Auro-4B scale: {scale}")
