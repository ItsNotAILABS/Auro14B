"""Mandatory AURO family upgrade and atomic-lane construction policy.

All standard AURO model construction passes through this policy. It enforces a
Mixture-of-Experts path, raises the declared context window exactly fourfold,
and installs conservative neuromorphic-residual configuration metadata. These
are architecture/runtime policies; they do not claim trained capability gains.
"""
from __future__ import annotations

from typing import Any, Callable

CONTEXT_MULTIPLIER = 4
MIN_EXPERTS = 8
MIN_TOP_K = 2
POLICY_VERSION = "auro.family.moe-context.v1"
NEUROMORPHIC_POLICY_VERSION = "auro.family.neuromorphic-residual.v1"
ATOMIC_POLICY_VERSION = "auro.atomic-family.v2.1"


def _install_neuromorphic_policy(config: Any) -> None:
    extra = config.extra
    extra.setdefault("neuromorphic_residual_policy", NEUROMORPHIC_POLICY_VERSION)
    extra.setdefault("use_neuromorphic_residual", True)
    extra.setdefault("neuromorphic_target_activity", 0.18)
    extra.setdefault("neuromorphic_threshold_quantile", 0.82)
    extra.setdefault("neuromorphic_minimum_gate", 0.35)
    extra.setdefault("neuromorphic_energy_penalty_weight", 0.01)
    extra.setdefault("neuromorphic_activity_penalty_weight", 0.02)
    extra.setdefault("neuromorphic_inhibitory_gain", 0.35)
    extra.setdefault("neuromorphic_checkpoint_quality_verified", False)
    extra.setdefault("neuromorphic_physical_energy_verified", False)


def apply_family_upgrade(config: Any) -> Any:
    """Apply mandatory family policy to an AuroLMConfig-compatible object."""
    _install_neuromorphic_policy(config)
    if config.extra.get("family_upgrade_policy") == POLICY_VERSION:
        return config

    original_context = int(config.max_seq_len)
    config.max_seq_len = original_context * CONTEXT_MULTIPLIER
    config.use_moe = True
    config.num_experts = max(MIN_EXPERTS, int(config.num_experts or 0))
    config.top_k_experts = max(MIN_TOP_K, int(config.top_k_experts or 0))
    config.top_k_experts = min(config.top_k_experts, config.num_experts)
    config.moe_every = max(1, int(getattr(config, "moe_every", 2) or 2))

    config.extra["family_upgrade_policy"] = POLICY_VERSION
    config.extra["legacy_max_seq_len"] = original_context
    config.extra["context_multiplier"] = CONTEXT_MULTIPLIER
    config.extra["declared_max_seq_len"] = config.max_seq_len
    config.extra["all_family_members_moe"] = True
    config.extra["long_context_quality_verified"] = False
    config.extra["long_context_training_required"] = True
    config.extra.setdefault("architecture_configuration_is_not_checkpoint", True)
    config.extra.setdefault("checkpoint_evidence_required", True)
    return config


def _atomic_config(
    config_cls: Any,
    *,
    model_id: str,
    parameter_target: int,
    mode: str,
    full: dict[str, Any],
    dev: dict[str, Any],
    roles: list[str],
    deploy_profiles: list[str],
    **overrides: Any,
) -> Any:
    if mode not in {"dev", "full"}:
        raise ValueError("mode must be dev or full")
    dims = dict(full if mode == "full" else dev)
    config = config_cls(
        model_id=model_id,
        tier="atomic",
        parameter_target=parameter_target,
        mode=mode,
        mesie_preset=str(dims.pop("mesie_preset")),
        use_moe=True,
        num_experts=8,
        top_k_experts=2,
        moe_every=2,
        use_cross_modal=bool(dims.pop("use_cross_modal", False)),
        use_spectral_encoder=bool(dims.pop("use_spectral_encoder", True)),
        num_modalities=int(dims.pop("num_modalities", 4)),
        use_meaning=True,
        use_spectral_fusion=True,
        use_helix=True,
        use_token_governor=True,
        multi_task=True,
        **dims,
    )
    config.extra.update(
        {
            "atomic_family_policy": ATOMIC_POLICY_VERSION,
            "model_class": "atomic",
            "subagent_roles": list(roles),
            "deploy_profiles": list(deploy_profiles),
            "checkpoint_quality_verified": False,
            "checkpoint_promotion_verified": False,
            "architecture_target_not_trained_checkpoint": True,
        }
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.extra[key] = value
    return apply_family_upgrade(config)


def build_auro156k_config(config_cls: Any, mode: str = "dev", **overrides: Any) -> Any:
    """Create the smallest executable AURO MoE rung."""
    del mode  # The reference seed geometry is intentionally identical in both modes.
    config = config_cls(
        model_id="Auro-156K",
        tier="atomic",
        parameter_target=156_000,
        mode="dev",
        mesie_preset="auro_seed_moe",
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_dim=64,
        vocab_size=1024,
        max_seq_len=256,
        use_moe=True,
        num_experts=8,
        top_k_experts=2,
        moe_every=2,
        use_cross_modal=False,
        use_spectral_encoder=False,
        continuous_dim=32,
        spectral_input_dim=64,
        num_modalities=1,
        use_meaning=True,
        use_spectral_fusion=True,
        use_helix=True,
        use_token_governor=True,
        multi_task=True,
    )
    config.extra.update(
        {
            "atomic_family_policy": ATOMIC_POLICY_VERSION,
            "model_class": "atomic",
            "subagent_roles": ["routing_seed", "classifier", "json_repair", "tool_selection"],
            "deploy_profiles": ["wasm", "embedded", "high-multiplicity-swarm"],
            "checkpoint_quality_verified": False,
            "checkpoint_promotion_verified": False,
            "architecture_target_not_trained_checkpoint": True,
        }
    )
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.extra[key] = value
    return apply_family_upgrade(config)


def build_auro250m_config(config_cls: Any, mode: str = "dev", **overrides: Any) -> Any:
    """Create the phone/browser atomic expert lane.

    Dev mode is a ratio-preserving executable proxy. Full mode carries the
    canonical 250M architecture target. Neither mode implies trained weights.
    """
    return _atomic_config(
        config_cls,
        model_id="Auro-250M",
        parameter_target=250_000_000,
        mode=mode,
        dev={
            "mesie_preset": "spectral_gpt_tiny",
            "hidden_dim": 192,
            "num_layers": 6,
            "num_heads": 3,
            "num_kv_heads": 1,
            "head_dim": 64,
            "ffn_dim": 512,
            "vocab_size": 4096,
            "max_seq_len": 1024,
            "continuous_dim": 64,
            "spectral_input_dim": 192,
        },
        full={
            "mesie_preset": "spectral_gpt_small",
            "hidden_dim": 768,
            "num_layers": 16,
            "num_heads": 12,
            "num_kv_heads": 4,
            "head_dim": 64,
            "ffn_dim": 2048,
            "vocab_size": 64000,
            "max_seq_len": 1024,
            "continuous_dim": 192,
            "spectral_input_dim": 768,
        },
        roles=["intent_extract", "retrieval_filter", "structured_transform", "code_triage", "memory_consolidation", "semantic_outline"],
        deploy_profiles=["phone", "browser-wasm", "cpu", "embedded-expert"],
        **overrides,
    )


def build_auro500m_config(config_cls: Any, mode: str = "dev", **overrides: Any) -> Any:
    """Create the edge-worker and embedded specialist lane."""
    return _atomic_config(
        config_cls,
        model_id="Auro-500M",
        parameter_target=500_000_000,
        mode=mode,
        dev={
            "mesie_preset": "spectral_gpt_tiny",
            "hidden_dim": 256,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 1,
            "head_dim": 64,
            "ffn_dim": 1024,
            "vocab_size": 8192,
            "max_seq_len": 2048,
            "continuous_dim": 96,
            "spectral_input_dim": 256,
        },
        full={
            "mesie_preset": "spectral_gpt_base",
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 4,
            "head_dim": 64,
            "ffn_dim": 4096,
            "vocab_size": 64000,
            "max_seq_len": 2048,
            "continuous_dim": 256,
            "spectral_input_dim": 1024,
        },
        roles=["tool_execution_plan", "code_patch", "evidence_review", "local_worker", "expert_consensus", "text_expansion"],
        deploy_profiles=["phone-high-memory", "laptop", "edge-gpu", "embedded-expert"],
        **overrides,
    )


def upgraded_family_config(base_factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return apply_family_upgrade(base_factory(*args, **kwargs))
