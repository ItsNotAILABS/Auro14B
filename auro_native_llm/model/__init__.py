"""AURO native model family on the MESIE compute plane.

Canonical construction covers 156K, 250M, 500M, 2B, 4B, 8B, 14B and 100B.
Every family config is MoE-enabled. Architecture targets remain separate from
trained checkpoint evidence.
"""
from __future__ import annotations

from typing import Any

from auro_native_llm.model import config as _config
from auro_native_llm.model.config import (
    AuroLMConfig,
    family_config as _base_family_config,
    family_config_from_mesie as _base_family_config_from_mesie,
    family_scale_table as _base_family_scale_table,
    list_mesie_presets,
    mesie_preset_dims,
)
from auro_native_llm.model.family_upgrade import (
    CONTEXT_MULTIPLIER,
    POLICY_VERSION,
    apply_family_upgrade,
    build_auro156k_config,
    upgraded_family_config,
)
from auro_native_llm.model.atomic_family import (
    ATOMIC_LADDER,
    AURO_500M_TRIAD,
    AtomicArchitecture,
    AtomicVariant,
    atomic_config_overrides,
    architecture_for,
    sub2b_manifest,
)


def _requested_model_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("model_id") or (args[0] if args else "Auro-2B"))


def _atomic_config(model_id: str, mode: str = "dev", **overrides: Any) -> AuroLMConfig:
    base_id = architecture_for(model_id).model_id
    arch = architecture_for(model_id)
    if mode == "full":
        values = atomic_config_overrides(model_id)
    elif base_id == "Auro-250M":
        values = {
            "model_id": model_id, "tier": "atomic", "parameter_target": arch.parameter_target,
            "mode": "dev", "mesie_preset": "spectral_gpt_tiny", "hidden_dim": 256,
            "num_layers": 6, "num_heads": 4, "num_kv_heads": 2, "head_dim": 64,
            "ffn_dim": 768, "vocab_size": 4096, "max_seq_len": 2048,
            "use_moe": True, "num_experts": 8, "top_k_experts": 2, "moe_every": 2,
        }
    elif base_id == "Auro-500M":
        values = {
            "model_id": model_id, "tier": "atomic", "parameter_target": arch.parameter_target,
            "mode": "dev", "mesie_preset": "spectral_gpt_tiny", "hidden_dim": 384,
            "num_layers": 8, "num_heads": 6, "num_kv_heads": 2, "head_dim": 64,
            "ffn_dim": 1536, "vocab_size": 8192, "max_seq_len": 4096,
            "use_moe": True, "num_experts": 8, "top_k_experts": 2, "moe_every": 2,
        }
    else:
        return build_auro156k_config(AuroLMConfig, **overrides)
    values.update({key: value for key, value in overrides.items() if key in AuroLMConfig.__dataclass_fields__})
    config = AuroLMConfig(**{key: value for key, value in values.items() if key in AuroLMConfig.__dataclass_fields__})
    for key, value in overrides.items():
        if key not in AuroLMConfig.__dataclass_fields__:
            config.extra[key] = value
    config.extra.update({
        "family_upgrade_policy": POLICY_VERSION,
        "legacy_max_seq_len": max(1, config.max_seq_len // CONTEXT_MULTIPLIER),
        "context_multiplier": CONTEXT_MULTIPLIER,
        "declared_max_seq_len": config.max_seq_len,
        "all_family_members_moe": True,
        "long_context_quality_verified": False,
        "long_context_training_required": True,
        "architecture_contract": arch.to_dict(),
        "specialization_variant": model_id if model_id != base_id else None,
    })
    return config


def family_config(*args: Any, **kwargs: Any) -> AuroLMConfig:
    model_id = _requested_model_id(args, kwargs)
    mode = str(kwargs.get("mode", "dev"))
    if model_id == "Auro-156K" or model_id in ATOMIC_LADDER or any(item.variant_id == model_id for item in AURO_500M_TRIAD):
        atomic_overrides = dict(kwargs)
        atomic_overrides.pop("model_id", None)
        atomic_overrides.pop("mode", None)
        atomic_overrides.pop("sync_mesie", None)
        return _atomic_config(model_id, mode, **atomic_overrides)
    return upgraded_family_config(_base_family_config, *args, **kwargs)


def family_config_from_mesie(*args: Any, **kwargs: Any) -> AuroLMConfig:
    model_id = _requested_model_id(args, kwargs)
    if model_id in ATOMIC_LADDER or any(item.variant_id == model_id for item in AURO_500M_TRIAD):
        return family_config(*args, **kwargs)
    return upgraded_family_config(_base_family_config_from_mesie, *args, **kwargs)


def family_scale_table() -> dict[str, dict[str, Any]]:
    table = _base_family_scale_table()
    for model_id in ("Auro-156K", "Auro-250M", "Auro-500M"):
        arch = architecture_for(model_id)
        table[model_id] = {
            "parameter_target": arch.parameter_target,
            "tier": "atomic",
            "dev": family_config(model_id, mode="dev").to_dict(),
            "full": family_config(model_id, mode="full").to_dict(),
            "parameter_accounting": arch.parameter_accounting(),
        }
    return table


_config.family_config = family_config
_config.family_config_from_mesie = family_config_from_mesie
_config.family_scale_table = family_scale_table

from auro_native_llm.model.auro_lm import AuroGenerateResult, AuroLanguageModel
from auro_native_llm.model.long_context import AuroLongContextModel, LongContextForward
from auro_native_llm.model.auro4b import architecture_to_overrides, build_auro4b, build_auro4b_config, write_birth_certificate
from auro_native_llm.model.auro4b_architecture import Auro4BArchitecture, FULL_ARCHITECTURE, PROXY_ARCHITECTURE, auro4b_architecture
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.train import TrainConfig, train_language_model
from auro_native_llm.model.checkpoint import save_checkpoint, load_checkpoint
from auro_native_llm.model.jobs import submit_pretrain_job, build_pretrain_command
from auro_native_llm.model.taxonomy import MODEL_CLASSES, RELEASE_LADDER, ModelClass, ModelClassSpec, classify_parameter_count, release_ladder
from auro_native_llm.model.fluidizer import FluidizedResult, fluidize_report
from auro_native_llm.model.triad_swarm import Auro2BTriadSwarm, ModelExecutor, ModelIdentity, TopicSwarmPlanner

__all__ = [
    "ATOMIC_LADDER", "AURO_500M_TRIAD", "AtomicArchitecture", "AtomicVariant",
    "Auro2BTriadSwarm", "Auro4BArchitecture", "AuroGenerateResult", "AuroLMConfig",
    "AuroLanguageModel", "AuroLongContextModel", "AuroTokenizer", "CONTEXT_MULTIPLIER",
    "FULL_ARCHITECTURE", "FluidizedResult", "LongContextForward", "MODEL_CLASSES",
    "ModelClass", "ModelClassSpec", "ModelExecutor", "ModelIdentity", "POLICY_VERSION",
    "PROXY_ARCHITECTURE", "RELEASE_LADDER", "TopicSwarmPlanner", "TrainConfig",
    "apply_family_upgrade", "architecture_for", "architecture_to_overrides", "atomic_config_overrides",
    "auro4b_architecture", "build_auro156k_config", "build_auro4b", "build_auro4b_config",
    "build_pretrain_command", "classify_parameter_count", "family_config", "family_config_from_mesie",
    "family_scale_table", "fluidize_report", "list_mesie_presets", "load_checkpoint",
    "mesie_preset_dims", "release_ladder", "save_checkpoint", "sub2b_manifest",
    "submit_pretrain_job", "train_language_model", "write_birth_certificate",
]
