"""Canonical AURO native model family on the MESIE compute plane.

Every standard AURO family configuration is MoE-enabled and receives the
family context policy. Atomic lanes are first-class constructors. Architecture
targets, trained checkpoints, specialist identities, and verified capability
claims remain distinct evidence states.
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
    ATOMIC_POLICY_VERSION,
    CONTEXT_MULTIPLIER,
    POLICY_VERSION,
    apply_family_upgrade,
    build_auro156k_config,
    build_auro250m_config,
    build_auro500m_config,
    upgraded_family_config,
)

_ATOMIC_BUILDERS = {
    "Auro-156K": build_auro156k_config,
    "Auro-250M": build_auro250m_config,
    "Auro-500M": build_auro500m_config,
}


def _requested_model_id(args, kwargs):
    if "model_id" in kwargs:
        return kwargs["model_id"]
    return args[0] if args else "Auro-2B"


def _atomic_config(model_id, kwargs):
    overrides = dict(kwargs)
    overrides.pop("model_id", None)
    mode = str(overrides.pop("mode", "dev"))
    overrides.pop("sync_mesie", None)
    return _ATOMIC_BUILDERS[model_id](AuroLMConfig, mode=mode, **overrides)


def family_config(*args, **kwargs):
    model_id = _requested_model_id(args, kwargs)
    if model_id in _ATOMIC_BUILDERS:
        return _atomic_config(model_id, kwargs)
    return upgraded_family_config(_base_family_config, *args, **kwargs)


def family_config_from_mesie(*args, **kwargs):
    model_id = _requested_model_id(args, kwargs)
    if model_id in _ATOMIC_BUILDERS:
        return _atomic_config(model_id, kwargs)
    return upgraded_family_config(_base_family_config_from_mesie, *args, **kwargs)


# AuroLanguageModel imports factories from config.py; install policy first.
_config.family_config = family_config
_config.family_config_from_mesie = family_config_from_mesie

from auro_native_llm.model.atomic_family import (
    ATOMIC_LADDER,
    SUB2B_CONTRACT_VERSION,
    AtomicArchitecture,
    CouncilResult,
    ExpertObservation,
    HierarchicalAtomicCouncil,
    TaskCapsule,
    sub2b_manifest,
)
from auro_native_llm.model.auro_lm import AuroGenerateResult, AuroLanguageModel
from auro_native_llm.model.long_context import AuroLongContextModel, LongContextForward
from auro_native_llm.model.auro4b import architecture_to_overrides, build_auro4b, build_auro4b_config, write_birth_certificate
from auro_native_llm.model.auro4b_architecture import Auro4BArchitecture, FULL_ARCHITECTURE, PROXY_ARCHITECTURE, auro4b_architecture
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.train import TrainConfig, train_language_model
from auro_native_llm.model.checkpoint import load_checkpoint, save_checkpoint
from auro_native_llm.model.jobs import build_pretrain_command, submit_pretrain_job
from auro_native_llm.model.taxonomy import (
    CANONICAL_RELEASE_ORDER,
    MODEL_CLASSES,
    RELEASE_LADDER,
    ModelClass,
    ModelClassSpec,
    classify_parameter_count,
    release_ladder,
)
from auro_native_llm.model.registry import MODELS, MODEL_BY_ID, ModelProfile, get_model_profile, model_manifest

__all__ = [
    "ATOMIC_LADDER", "ATOMIC_POLICY_VERSION", "AtomicArchitecture",
    "Auro4BArchitecture", "AuroGenerateResult", "AuroLMConfig", "AuroLanguageModel",
    "AuroLongContextModel", "CANONICAL_RELEASE_ORDER", "CONTEXT_MULTIPLIER",
    "CouncilResult", "ExpertObservation", "FULL_ARCHITECTURE", "HierarchicalAtomicCouncil",
    "LongContextForward", "AuroTokenizer", "POLICY_VERSION", "PROXY_ARCHITECTURE",
    "SUB2B_CONTRACT_VERSION", "TaskCapsule", "TrainConfig", "MODEL_CLASSES",
    "RELEASE_LADDER", "ModelClass", "ModelClassSpec", "MODELS", "MODEL_BY_ID",
    "ModelProfile", "apply_family_upgrade", "architecture_to_overrides", "auro4b_architecture",
    "build_auro156k_config", "build_auro250m_config", "build_auro500m_config",
    "build_auro4b", "build_auro4b_config", "build_pretrain_command", "classify_parameter_count",
    "family_config", "family_config_from_mesie", "family_scale_table", "get_model_profile",
    "list_mesie_presets", "load_checkpoint", "mesie_preset_dims", "model_manifest",
    "release_ladder", "save_checkpoint", "sub2b_manifest", "submit_pretrain_job",
    "train_language_model", "write_birth_certificate",
]
