"""Canonical AURO native model family on the MESIE compute plane.

Every standard AURO family configuration is MoE-enabled and receives the
family context policy. Architecture targets, trained checkpoints, personas,
and verified capability claims remain distinct.
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
    build_auro156k_config,
    upgraded_family_config,
)


def _requested_model_id(args, kwargs):
    if "model_id" in kwargs:
        return kwargs["model_id"]
    return args[0] if args else "Auro-2B"


def family_config(*args, **kwargs):
    if _requested_model_id(args, kwargs) == "Auro-156K":
        overrides = dict(kwargs)
        overrides.pop("model_id", None)
        overrides.pop("mode", None)
        overrides.pop("sync_mesie", None)
        return build_auro156k_config(AuroLMConfig, **overrides)
    return upgraded_family_config(_base_family_config, *args, **kwargs)


def family_config_from_mesie(*args, **kwargs):
    return upgraded_family_config(_base_family_config_from_mesie, *args, **kwargs)


# AuroLanguageModel imports factories from config.py; install policy first.
_config.family_config = family_config
_config.family_config_from_mesie = family_config_from_mesie

from auro_native_llm.model.auro_lm import AuroGenerateResult, AuroLanguageModel
from auro_native_llm.model.long_context import AuroLongContextModel, LongContextForward
from auro_native_llm.model.auro4b import architecture_to_overrides, build_auro4b, build_auro4b_config, write_birth_certificate
from auro_native_llm.model.auro4b_architecture import Auro4BArchitecture, FULL_ARCHITECTURE, PROXY_ARCHITECTURE, auro4b_architecture
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.train import TrainConfig, train_language_model
from auro_native_llm.model.checkpoint import load_checkpoint, save_checkpoint
from auro_native_llm.model.jobs import build_pretrain_command, submit_pretrain_job
from auro_native_llm.model.taxonomy import (
    MODEL_CLASSES,
    RELEASE_LADDER,
    ModelClass,
    ModelClassSpec,
    classify_parameter_count,
    release_ladder,
)
from auro_native_llm.model.registry import MODELS, MODEL_BY_ID, ModelProfile, get_model_profile, model_manifest

__all__ = [
    "Auro4BArchitecture", "AuroGenerateResult", "AuroLMConfig", "AuroLanguageModel",
    "AuroLongContextModel", "LongContextForward", "AuroTokenizer", "CONTEXT_MULTIPLIER",
    "FULL_ARCHITECTURE", "POLICY_VERSION", "PROXY_ARCHITECTURE", "TrainConfig",
    "MODEL_CLASSES", "RELEASE_LADDER", "ModelClass", "ModelClassSpec",
    "MODELS", "MODEL_BY_ID", "ModelProfile", "apply_family_upgrade",
    "architecture_to_overrides", "auro4b_architecture", "build_auro156k_config",
    "build_auro4b", "build_auro4b_config", "build_pretrain_command",
    "classify_parameter_count", "family_config", "family_config_from_mesie",
    "family_scale_table", "get_model_profile", "list_mesie_presets", "load_checkpoint",
    "mesie_preset_dims", "model_manifest", "release_ladder", "save_checkpoint",
    "submit_pretrain_job", "train_language_model", "write_birth_certificate",
]
