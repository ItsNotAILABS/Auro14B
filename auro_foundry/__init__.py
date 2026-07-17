"""Auro Foundry repository-native training and inference runtime."""

from .config import ModelConfig, TrainConfig
from .generation import TextGenerator, load_model
from .model import AuroForCausalLM
from .tokenizer import AuroBPETokenizer
from .training import TrainingResult, train

__all__ = [
    "AuroBPETokenizer",
    "AuroForCausalLM",
    "ModelConfig",
    "TextGenerator",
    "TrainConfig",
    "TrainingResult",
    "load_model",
    "train",
]
__version__ = "1.0.0-alpha"
