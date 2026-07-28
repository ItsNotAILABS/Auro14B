"""AURO ST-14B dense efficiency profile.

This module turns the ST-14B research geometry into an executable AURO
configuration contract without misrepresenting unrun hardware benchmarks as
measured results. The existing AURO/HIM/MESIE organism remains available; this
profile is a dense, serving-optimized lane for production inference studies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from auro_native_llm.model.config import AuroLMConfig


@dataclass(frozen=True)
class ST14BArchitecture:
    model_id: str = "AURO-ST-14B"
    total_parameter_target: int = 14_200_000_000
    vocab_size: int = 128_000
    hidden_dim: int = 5_120
    num_layers: int = 40
    num_heads: int = 40
    num_kv_heads: int = 5
    head_dim: int = 128
    ffn_dim: int = 18_432
    max_seq_len: int = 8_192
    rope_theta: float = 10_000.0
    normalization: str = "rms_norm"
    activation: str = "swiglu"
    positional_encoding: str = "rotary"
    tie_embeddings: bool = True
    bias: bool = False
    status: str = "PROTOTYPE"

    def validate(self) -> None:
        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.num_layers < 1 or self.max_seq_len < 8:
            raise ValueError("layers and context must be positive")
        if self.activation != "swiglu":
            raise ValueError("ST-14B requires SwiGLU")
        if self.normalization != "rms_norm":
            raise ValueError("ST-14B requires RMSNorm")
        if self.positional_encoding != "rotary":
            raise ValueError("ST-14B requires RoPE")

    @property
    def gqa_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def dense_parameter_estimate(self) -> int:
        """Estimate parameters for the implemented tied-embedding dense decoder.

        Attention: Q + K + V + O.
        SwiGLU: gate + up + down.
        Norms: two per layer plus final norm.
        """
        self.validate()
        d = self.hidden_dim
        kv = self.kv_dim
        embeddings = self.vocab_size * d
        output = 0 if self.tie_embeddings else embeddings
        attention = (d * d) + (d * kv) + (d * kv) + (d * d)
        swiglu = 3 * d * self.ffn_dim
        norms = 2 * d
        return embeddings + output + self.num_layers * (attention + swiglu + norms) + d

    def kv_cache_bytes(
        self,
        *,
        batch_size: int = 1,
        sequence_length: int | None = None,
        bytes_per_element: int = 2,
    ) -> int:
        """K+V cache size when KV heads are stored before query-head expansion."""
        self.validate()
        seq = self.max_seq_len if sequence_length is None else sequence_length
        if batch_size < 1 or seq < 1 or bytes_per_element < 1:
            raise ValueError("cache dimensions and precision must be positive")
        return (
            2
            * batch_size
            * self.num_layers
            * self.num_kv_heads
            * seq
            * self.head_dim
            * bytes_per_element
        )

    def mha_equivalent_kv_cache_bytes(
        self,
        *,
        batch_size: int = 1,
        sequence_length: int | None = None,
        bytes_per_element: int = 2,
    ) -> int:
        seq = self.max_seq_len if sequence_length is None else sequence_length
        return (
            2
            * batch_size
            * self.num_layers
            * self.num_heads
            * seq
            * self.head_dim
            * bytes_per_element
        )

    def report(self) -> dict[str, Any]:
        gqa = self.kv_cache_bytes()
        mha = self.mha_equivalent_kv_cache_bytes()
        estimate = self.dense_parameter_estimate()
        return {
            "schema": "auro.st14b.architecture-report.v1",
            "status": self.status,
            "architecture": asdict(self),
            "derived": {
                "gqa_ratio": self.gqa_ratio,
                "dense_parameter_estimate": estimate,
                "parameter_target_delta": estimate - self.total_parameter_target,
                "kv_cache_bytes_b1_bf16_8192": gqa,
                "kv_cache_mib_b1_bf16_8192": gqa / (1024 * 1024),
                "mha_equivalent_kv_cache_bytes_b1_bf16_8192": mha,
                "theoretical_kv_cache_reduction": 1.0 - (gqa / mha),
            },
            "claim_boundary": {
                "architecture_contract": True,
                "trained_checkpoint": False,
                "h100_fp8_benchmark": False,
                "ttft_under_12ms": False,
                "throughput_142_tps_per_stream": False,
                "mfu_64_2_percent": False,
                "throughput_speedup_1_8x": False,
                "memory_bandwidth_reduction_42_percent": False,
            },
        }


def build_st14b_config(*, mode: str = "full", **overrides: Any) -> AuroLMConfig:
    """Build the dense AURO ST-14B serving lane as an AuroLMConfig.

    The profile intentionally disables core MoE and cross-modal/spectral blocks
    so the stated 14B-class dense parameter geometry remains meaningful. NOVA,
    Chimeria, MedinaMemorySystems, and AURO's outer cognition/runtime organs can
    still compose around this serving core through canonical interfaces.
    """
    spec = ST14BArchitecture()
    spec.validate()
    config = AuroLMConfig(
        model_id=spec.model_id,
        tier="orchestrator-serving",
        parameter_target=spec.total_parameter_target,
        mode=mode,  # type: ignore[arg-type]
        mesie_preset="st14b_dense_efficiency",
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        ffn_dim=spec.ffn_dim,
        vocab_size=spec.vocab_size,
        max_seq_len=spec.max_seq_len,
        use_moe=False,
        num_experts=1,
        top_k_experts=1,
        moe_layers=[],
        use_cross_modal=False,
        cross_modal_layers=[],
        use_spectral_encoder=False,
        spectral_input_dim=spec.hidden_dim,
        continuous_dim=spec.hidden_dim,
        num_modalities=1,
        positional_encoding=spec.positional_encoding,
        normalization=spec.normalization,
        activation=spec.activation,
        dropout=0.0,
        num_kv_heads=spec.num_kv_heads,
        qk_norm=True,
        tie_embeddings=spec.tie_embeddings,
        causal=True,
        use_meaning=False,
        use_spectral_fusion=False,
        use_helix=False,
        use_token_governor=False,
        multi_task=False,
        use_delta_attention=False,
    )
    config.extra.update(
        {
            "architecture_profile": "auro.st14b.efficiency.v1",
            "status": spec.status,
            "gqa_ratio": spec.gqa_ratio,
            "dense_parameter_estimate": spec.dense_parameter_estimate(),
            "kv_cache_policy": "store-unexpanded-kv-heads",
            "quantization_targets": ["bf16", "fp16", "fp8", "int8", "int4"],
            "serving_targets": ["vllm", "tensorrt-llm", "transformers", "llama.cpp"],
            "benchmark_required": True,
            "research_claims_are_targets_not_results": True,
        }
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.extra[key] = value
    return config


def st14b_quantization_matrix() -> list[dict[str, Any]]:
    return [
        {
            "precision": "bf16",
            "status": "REFERENCE",
            "purpose": "quality and numerical baseline",
            "promotion_gate": "exact-checkpoint perplexity and task evaluation",
        },
        {
            "precision": "fp8",
            "status": "PLANNED",
            "purpose": "H100-class throughput lane",
            "promotion_gate": "TensorRT-LLM or vLLM benchmark with accuracy delta",
        },
        {
            "precision": "int8",
            "status": "PLANNED",
            "purpose": "enterprise GPU and CPU compatibility",
            "promotion_gate": "calibration receipt plus exact evaluation",
        },
        {
            "precision": "int4",
            "status": "RESEARCH",
            "purpose": "single-workstation and edge deployment",
            "promotion_gate": "AWQ/GPTQ/GGUF comparison with quality floor",
        },
    ]
