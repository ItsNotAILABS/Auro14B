"""Auro text LLM — first-class language model family on the MESIE compute engine.

This is not a wrapper around OpenAI/Ollama/HF remote models. The backbone is
MESIE SpectralGPT plus native AURO checkpoint, structured-weight, meaning, and
spectral subsystems.

Family: Auro-156K · Auro-2B · Auro-4B · Auro-8B · Auro-14B · Auro-100B
"""

from auro_native_llm.model.config import (
    AuroLMConfig,
    family_config,
    family_config_from_mesie,
    family_scale_table,
    list_mesie_presets,
    mesie_preset_dims,
)
from auro_native_llm.model.auro_lm import AuroLanguageModel, AuroGenerateResult
from auro_native_llm.model.long_context import AuroLongContextModel, LongContextForward
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
    "AuroLongContextModel",
    "LongContextForward",
    "AuroTokenizer",
    "FULL_ARCHITECTURE",
    "PROXY_ARCHITECTURE",
    "TrainConfig",
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
