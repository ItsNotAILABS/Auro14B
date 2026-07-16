"""Auro Foundry: repository-native corpus, tokenizer, training, generation, and serving.

The same code path runs small local models immediately and scales through
PyTorch distributed launchers when additional Medina-owned nodes are registered
in MESIE's training fabric.
"""

from .config import ModelConfig, TrainConfig
from .tokenizer import AuroBPETokenizer

__all__ = ["ModelConfig", "TrainConfig", "AuroBPETokenizer"]
__version__ = "1.0.0-alpha"
