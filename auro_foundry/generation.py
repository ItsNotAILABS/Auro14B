from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, TrainConfig
from .model import AuroForCausalLM
from .tokenizer import AuroBPETokenizer


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str | Path, device: str = "auto") -> tuple[AuroForCausalLM, AuroBPETokenizer, dict[str, Any], torch.device]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_config = payload.get("config") or payload.get("train_config")
    if raw_config:
        train_config = TrainConfig.from_dict(raw_config)
        model_config = train_config.model
        tokenizer_path = Path(train_config.tokenizer_path)
    else:
        model_config = ModelConfig.from_dict(payload["model_config"])
        tokenizer_path = Path(payload.get("tokenizer_path", path.parent / "tokenizer.json"))
    if not tokenizer_path.is_absolute():
        tokenizer_path = (Path.cwd() / tokenizer_path).resolve()
    tokenizer = AuroBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint vocabulary does not match tokenizer")
    model = AuroForCausalLM(model_config)
    model.load_state_dict(payload["model"])
    target = resolve_device(device)
    model.to(target).eval()
    metadata = {
        "id": model_config.model_id,
        "model_id": model_config.model_id,
        "checkpoint": str(path),
        "step": int(payload.get("step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "parameters": model.parameter_report()["parameters"],
        "device": str(target),
    }
    return model, tokenizer, metadata, target


class TextGenerator:
    def __init__(self, checkpoint: str | Path, *, device: str = "auto") -> None:
        self.model, self.tokenizer, self.metadata, self.device = load_model(checkpoint, device)

    @torch.no_grad()
    def generate(self, prompt: str, *, max_new_tokens: int = 128, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.05) -> str:
        token_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_id,
        )
        generated = output[0, len(token_ids):].tolist()
        return self.tokenizer.decode(generated).strip()
