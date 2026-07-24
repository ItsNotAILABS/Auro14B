"""Auro Foundry repository-native training and inference runtime."""

__version__ = "1.0.0-alpha"

# Lazy exports — avoid hard torch import on platforms without wheels (Win ARM64)
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


def __getattr__(name: str):
    if name in ("ModelConfig", "TrainConfig"):
        from .config import ModelConfig, TrainConfig

        return {"ModelConfig": ModelConfig, "TrainConfig": TrainConfig}[name]
    if name in ("TextGenerator", "load_model"):
        from .generation import TextGenerator, load_model

        return {"TextGenerator": TextGenerator, "load_model": load_model}[name]
    if name == "AuroForCausalLM":
        from .model import AuroForCausalLM

        return AuroForCausalLM
    if name == "AuroBPETokenizer":
        from .tokenizer import AuroBPETokenizer

        return AuroBPETokenizer
    if name in ("TrainingResult", "train"):
        from .training import TrainingResult, train

        return {"TrainingResult": TrainingResult, "train": train}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
