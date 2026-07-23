"""Auro text LLM — first-class language model family on the MESIE compute engine.

This package installs the mandatory AURO family policy before importing the
runtime model: every standard family member is MoE-enabled and receives a 4x
declared context target. The context expansion remains an architecture target
until exact checkpoints pass long-context training and evaluation.

Family: Auro-156K · Auro-2B · Auro-4B · Auro-8B · Auro-14B · Auro-100B
"""

from auro_native_llm.model import config as _config
from auro_native_llm.model.config import (
    AuroLMConfig,
    family_config as _base_family_config,
    family_config_from_mesie as _base_family_config_from_mesie,
    family_scale_table,
    list_mesie_presets,
    mesie_preset_dims,
)
from auro_native_llm.model.family_upgrade import (
    CONTEXT_MULTIPLIER,
    POLICY_VERSION,
    apply_family_upgrade,
    upgraded_family_config,
)


def family_config(*args, **kwargs):
    return upgraded_family_config(_base_family_config, *args, **kwargs)


def family_config_from_mesie(*args, **kwargs):
    return upgraded_family_config(_base_family_config_from_mesie, *args, **kwargs)


# AuroLanguageModel imports family_config directly from config.py. Install the
# policy before importing auro_lm so direct and package-level construction agree.
_config.family_config = family_config
_config.family_config_from_mesie = family_config_from_mesie

from auro_native_llm.model.auro_lm import AuroLanguageModel, AuroGenerateResult
from auro_native_llm.model.auro4b import (
    architecture_to_overrides,
    build_auro4b,
    build_auro4b_config,
    write_birth_certificate,
)
from auro_native_llm.model.auro4b_architecture import (
    Auro4BArchitecture,
    FULL_ARCHITECTURE,
    PROXY_ARCHITECTURE,
    auro4b_architecture,
)
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.train import TrainConfig, train_language_model
from auro_native_llm.model.checkpoint import save_checkpoint, load_checkpoint
from auro_native_llm.model.jobs import submit_pretrain_job, build_pretrain_command

__all__ = [
    "Auro4BArchitecture",
    "AuroGenerateResult",
    "AuroLMConfig",
    "AuroLanguageModel",
    "AuroTokenizer",
    "CONTEXT_MULTIPLIER",
    "FULL_ARCHITECTURE",
    "POLICY_VERSION",
    "PROXY_ARCHITECTURE",
    "TrainConfig",
    "apply_family_upgrade",
    "architecture_to_overrides",
    "auro4b_architecture",
    "build_auro4b",
    "build_auro4b_config",
    "build_pretrain_command",
    "family_config",
    "family_config_from_mesie",
    "family_scale_table",
    "list_mesie_presets",
    "load_checkpoint",
    "mesie_preset_dims",
    "save_checkpoint",
    "submit_pretrain_job",
    "train_language_model",
    "write_birth_certificate",
]
