"""MESIE Foundation Model — Spectral GPT Pretraining Framework.

This package implements a foundation-model-level pretraining system for spectral
intelligence. It provides a universal spectral latent space learned from diverse
frequency-domain data including seismic windows, vibration frames, EEG/ECG corpora,
audio spectrograms, RF sweeps, and synthetic physics simulations.

Architecture Overview:
    - SpectralGPT: Transformer-based foundation model for spectral sequences
    - SpectralTokenizer: Converts raw spectral data into discrete/continuous tokens
    - UniversalSpectralLatentSpace: Shared representation across all modalities
    - PretrainingEngine: Orchestrates large-scale pretraining with multiple objectives

Key Concepts:
    Just as GPT learned the structure of language, MESIE learns the structure
    of frequency-domain reality through self-supervised pretraining on massive
    spectral corpora spanning multiple physical domains.
"""

from mesie.foundation.config.pretraining_config import (
    PretrainingConfig,
    ModelConfig,
    TokenizerConfig,
    DataConfig,
    TrainingConfig,
    LatentSpaceConfig,
    ObjectiveConfig,
    EvaluationConfig,
)
from mesie.foundation.models.spectral_gpt import SpectralGPT
from mesie.foundation.tokenizers.spectral_tokenizer import SpectralTokenizer
from mesie.foundation.latent.universal_latent_space import UniversalSpectralLatentSpace
from mesie.foundation.training.pretraining_engine import PretrainingEngine

__all__ = [
    "PretrainingConfig",
    "ModelConfig",
    "TokenizerConfig",
    "DataConfig",
    "TrainingConfig",
    "LatentSpaceConfig",
    "ObjectiveConfig",
    "EvaluationConfig",
    "SpectralGPT",
    "SpectralTokenizer",
    "UniversalSpectralLatentSpace",
    "PretrainingEngine",
]
