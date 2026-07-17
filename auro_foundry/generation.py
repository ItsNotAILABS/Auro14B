from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import AuroForCausalLM
from .tokenizer import AuroBPETokenizer


def load_checkpoint(checkpoint: str | Path, *, device: str = "auto") -> tuple[AuroForCausalLM, AuroBPETokenizer, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])
    tokenizer_path = payload.get("tokenizer_path") or checkpoint_path.parent / "tokenizer.json"
    tokenizer = AuroBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("checkpoint vocabulary does not match tokenizer")
    model = AuroForCausal