"""Auro-4B native construction lane.

The lane has two modes:
- proxy: executable ratio-preserving model for CI, tokenizer, checkpoint, API,
  and training-pipeline validation;
- full: the actual ~4B decoder geometry, materialized only by explicit request.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional
import hashlib
import json

from auro_native_llm.model.auro4b_architecture import (
    Auro4BArchitecture,
    auro4b_architecture,
)

if TYPE_CHECKING:
    from auro_native_llm.model.auro_lm import AuroLanguageModel
    from auro_native_llm.model.prewiring import PrewiringConfig, PrewiringReceipt
else:
    AuroLanguageModel = Any
    PrewiringConfig = Any
    PrewiringReceipt = Any


def architecture_to_overrides(architecture: Auro4BArchitecture) -> Dict[str, Any]:
    architecture.validate()
    return {
        "hidden_dim": architecture.hidden_dim,
        "num_layers": architecture.num_layers,
        "num_heads": architecture.num_attention_heads,
        "num_kv_heads": architecture.num_kv_heads,
        "head_dim": architecture.head_dim,
        "ffn_dim": architecture.ffn_dim,
        "vocab_size": architecture.vocab_size,
        "max_seq_len": architecture.max_seq_len,
        "use_moe": False,
        "num_experts": 1,
        "top_k_experts": 1,
        "positional_encoding": architecture.positional_encoding,
        "normalization": architecture.normalization,
        "activation": architecture.activation,
        "qk_norm": architecture.qk_norm,
        "tie_embeddings": architecture.tie_embeddings,
        "extra": {
            "architecture_version": architecture.architecture_version,
            "rope_theta": architecture.rope_theta,
            "rms_norm_eps": architecture.rms_norm_eps,
            "structured_residual_every": architecture.structured_residual_every,
            "structured_basis": architecture.structured_basis,
            "checkpoint_constitution": architecture.checkpoint_constitution,
            "claims": dict(architecture.claims),
            "parameter_estimate": architecture.parameter_estimate(),
        },
    }


def build_auro4b(
    *,
    scale: str = "proxy",
    structured: bool = True,
    materialize_full: bool = False,
    prewiring: Optional[PrewiringConfig] = None,
    **overrides: Any,
) -> tuple[AuroLanguageModel, Optional[PrewiringReceipt]]:
    """Build Auro-4B without silently pretending a proxy is four billion weights."""
    architecture = auro4b_architecture(scale)  # type: ignore[arg-type]
    if scale == "full" and not materialize_full:
        raise RuntimeError(
            "full Auro-4B materialization requires materialize_full=True; "
            "use build_auro4b_config(scale='full') for planning"
        )

    from auro_native_llm.model.auro_lm import AuroLanguageModel as _AuroLanguageModel
    from auro_native_llm.model.prewiring import apply_structured_prewiring

    model_overrides = architecture_to_overrides(architecture)
    model_overrides.update(overrides)
    model = _AuroLanguageModel.build("Auro-4B", mode="dev", **model_overrides)
    model.auro4b_architecture = architecture.to_dict()
    receipt = apply_structured_prewiring(model, prewiring) if structured else None
    return model, receipt


def build_auro4b_config(scale: str = "full") -> Dict[str, Any]:
    architecture = auro4b_architecture(scale)  # type: ignore[arg-type]
    return architecture.to_dict()


def write_birth_certificate(
    model: AuroLanguageModel,
    output: str | Path,
    receipt: Optional[PrewiringReceipt],
    *,
    checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    architecture = getattr(model, "auro4b_architecture", build_auro4b_config("proxy"))
    payload: Dict[str, Any] = {
        "schema": "auro.model.birth.v2",
        "model_id": model.model_id,
        "architecture": architecture,
        "parameter_target": int(model.config.parameter_target),
        "live_parameter_count": int(model.num_params),
        "compute_plane": "MESIE",
        "native": True,
        "external_model_fallback": False,
        "structured_prewiring": receipt.to_dict() if receipt else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_constitution": "auro.substrate.checkpoint.v1",
        "claim_boundary": {
            "structured_inductive_bias": receipt is not None,
            "trained_general_knowledge": False,
            "dpo_alignment_completed": False,
            "long_context_extrapolation_verified": False,
            "benchmark_superiority_claimed": False,
            "hallucination_reduction_claimed": False,
        },
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["birth_sha256"] = hashlib.sha256(unsigned).hexdigest()
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
