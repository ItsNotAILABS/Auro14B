"""Auro text LLM — first-class language model family on the MESIE compute engine.

This is not a wrapper around OpenAI/Ollama/HF remote models. The backbone is
MESIE SpectralGPT (causal MoE transformer) plus:

- φ / golden-ratio mathematics (SOLUS)
- Meaning engines (Latin · Sanskrit · Nahuatl cosmic layers)
- First-class MESIE spectral vectors (SpectralVectorizer + Helix)
- Multi-task heads (LM + spectral + meaning)
- Training fabric jobs for each family lane

Family: Auro-2B · Auro-4B · Auro-8B · Auro-14B · Auro-100B
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
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.train import TrainConfig, train_language_model
from auro_native_llm.model.checkpoint import save_checkpoint, load_checkpoint
from auro_native_llm.model.jobs import submit_pretrain_job, build_pretrain_command

__all__ = [
    "AuroGenerateResult",
    "AuroLMConfig",
    "AuroLanguageModel",
    "AuroTokenizer",
    "TrainConfig",
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
]
