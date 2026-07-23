"""Native Auro-4B MoE architecture contract and parameter accounting.

The Auro-4B name denotes active parameters per token. Sparse top-2 routing
increases stored expert capacity without silently changing the active-compute
class. The full configuration is never materialized in CI or on a laptop unless
callers explicitly request it. A ratio-preserving proxy validates the pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal

Auro4BScale = Literal["proxy", "full"]


@dataclass(frozen=True)
class Auro4BArchitecture:
    model_id: str = "Auro-4B"
    architecture_version: str = "auro4b.moe-long-context.v2"
    parameter_target: int = 4_000_000_000
    hidden_dim: int = 3072
    num_layers: int = 32
    num_attention_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn_dim: int = 8192
    vocab_size: int = 65536
    max_seq_len: int = 65536
    rope_theta: float = 4_000_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    qk_norm: bool = True
    bias: bool = False
    activation: str = "swiglu"
    normalization: str = "rms_norm"
    positional_encoding: str = "rotary"
    attention: str = "grouped_query_attention"
    use_moe: bool = True
    num_experts: int = 8
    top_k_experts: int = 2
    moe_every: int = 4
    structured_residual_every: int = 4
    structured_basis: str = "walsh_hadamard"
    checkpoint_constitution: str = "auro.substrate.checkpoint.v1"
    claims: Dict[str, bool] = field(default_factory=lambda: {
        "trained_general_knowledge": False,
        "benchmark_superiority": False,
        "hallucination_reduction": False,
        "long_context_extrapolation_verified": False,
        "moe_routing_quality_verified": False,
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
        if not self.use_moe or self.num_experts < 2:
            raise ValueError("Auro-4B requires a Mixture-of-Experts path")
        if not 1 <= self.top_k_experts <= self.num_experts:
            raise ValueError("top_k_experts must be within the expert count")
        if self.moe_every <= 0:
            raise ValueError("moe_every must be positive")
        if self.structured_residual_every <= 0:
            raise ValueError("structured_residual_every must be positive")

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def moe_layer_count(self) -> int:
        return len(range(self.moe_every - 1, self.num_layers, self.moe_every))

    @property
    def dense_layer_count(self) -> int:
        return self.num_layers - self.moe_layer_count

    def parameter_estimate(self) -> Dict[str, int]:
        """Estimate active and stored trainable parameter capacity.

        ``total`` remains the active-parameter count used by the Auro-4B family
        name and compatibility gates. ``stored_total`` includes all dormant MoE
        experts. Optimizer state and KV-cache memory are excluded.
        """
        self.validate()
        d = self.hidden_dim
        kv = self.kv_dim
        attention_per_layer = d * d + d * kv + d * kv + d * d
        expert_per_layer = 3 * d * self.ffn_dim
        norms_per_layer = 2 * d
        embeddings = self.vocab_size * d
        lm_head = 0 if self.tie_embeddings else d * self.vocab_size
        final_norm = d
        structured_diagonals = (
            self.num_layers // self.structured_residual_every
        ) * 2 * d
        router_parameters = self.moe_layer_count * d * self.num_experts

        attention = self.num_layers * attention_per_layer
        dense_ffn = self.dense_layer_count * expert_per_layer
        active_moe_ffn = self.moe_layer_count * self.top_k_experts * expert_per_layer
        stored_moe_ffn = self.moe_layer_count * self.num_experts * expert_per_layer
        norms = self.num_layers * norms_per_layer + final_norm

        active_total = (
            embeddings + lm_head + attention + dense_ffn + active_moe_ffn
            + norms + structured_diagonals + router_parameters
        )
        stored_total = (
            embeddings + lm_head + attention + dense_ffn + stored_moe_ffn
            + norms + structured_diagonals + router_parameters
        )
        return {
            "embeddings": embeddings,
            "attention": attention,
            "dense_swiglu": dense_ffn,
            "active_moe_swiglu": active_moe_ffn,
            "stored_moe_swiglu": stored_moe_ffn,
            "routers": router_parameters,
            "norms": norms,
            "structured_residuals": structured_diagonals,
            "lm_head": lm_head,
            "total": active_total,
            "active_total": active_total,
            "stored_total": stored_total,
            "target": self.parameter_target,
            "delta_from_target": active_total - self.parameter_target,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["kv_dim"] = self.kv_dim
        payload["parameter_estimate"] = self.parameter_estimate()
        payload["gqa_group_size"] = self.num_attention_heads // self.num_kv_heads
        payload["moe_layer_count"] = self.moe_layer_count
        payload["dense_layer_count"] = self.dense_layer_count
        payload["context_multiplier_from_v1"] = 4
        return payload


PROXY_ARCHITECTURE = Auro4BArchitecture(
    architecture_version="auro4b.proxy.moe-long-context.v2",
    parameter_target=4_000_000_000,
    hidden_dim=384,
    num_layers=6,
    num_attention_heads=6,
    num_kv_heads=2,
    head_dim=64,
    ffn_dim=1024,
    vocab_size=8192,
    max_seq_len=4096,
    rope_theta=400_000.0,
    num_experts=8,
    top_k_experts=2,
    moe_every=2,
    structured_residual_every=2,
)

FULL_ARCHITECTURE = Auro4BArchitecture()


def auro4b_architecture(scale: Auro4BScale = "proxy") -> Auro4BArchitecture:
    if scale == "proxy":
        return PROXY_ARCHITECTURE
    if scale == "full":
        return FULL_ARCHITECTURE
    raise ValueError(f"unknown Auro-4B scale: {scale}")
