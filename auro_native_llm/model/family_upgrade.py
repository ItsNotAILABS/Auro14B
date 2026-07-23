"""Mandatory AURO family upgrade policy.

All standard AURO model construction passes through this policy. It enforces a
Mixture-of-Experts path and raises the declared context window exactly fourfold
relative to each repository family configuration. This changes architecture
capacity; it does not claim trained or validated long-context performance.
"""
from __future__ import annotations

from typing import Any, Callable

CONTEXT_MULTIPLIER = 4
MIN_EXPERTS = 8
MIN_TOP_K = 2
POLICY_VERSION = "auro.family.moe-context.v1"


def apply_family_upgrade(config: Any) -> Any:
    """Apply the mandatory family policy to an AuroLMConfig-compatible object."""
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
    return config


def build_auro156k_config(config_cls: Any, **overrides: Any) -> Any:
    """Create the smallest executable AURO MoE rung.

    The 156K name remains a target-class label; the exact live count depends on
    the MESIE implementation and auxiliary heads. Its legacy 256-token context
    is raised to 1,024 by the same family policy used everywhere else.
    """
    config = config_cls(
        model_id="Auro-156K",
        tier="seed",
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
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.extra[key] = value
    return apply_family_upgrade(config)


def upgraded_family_config(base_factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call the repository factory and apply the mandatory family policy."""
    return apply_family_upgrade(base_factory(*args, **kwargs))
