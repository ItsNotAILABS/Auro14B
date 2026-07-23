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


def upgraded_family_config(base_factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call the repository factory and apply the mandatory family policy."""
    return apply_family_upgrade(base_factory(*args, **kwargs))
