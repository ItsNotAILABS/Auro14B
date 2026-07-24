"""Small, auditable, fully local autoregressive neural language model.

This reference lane exists to prove the complete open-weight path without any
downloaded base model. It is intentionally compact enough to train on CPU.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CONTROL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>",
    "<tool>", "<receipt>", "<spectral>", "<memory>", "<repository>",
    "<code>", "<test>", "<execution>", "<nova>", "<mesie>",
)
BYTE_OFFSET = len(CONTROL_TOKENS)
VOCAB_SIZE = BYTE_OFFSET + 256


class ByteTokenizer:
    """Immutable UTF-8 byte fallback: zero unknown tokens, exact round trips."""
    pad_id, bos_id, eos_id = 0, 1, 2

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decode(self, ids: Iterable[int]) -> str:
        data = bytes(i - BYTE_OFFSET for i in ids if BYTE_OFFSET <= int(i) < VOCAB_SIZE)
        return data.decode("utf-8", errors="replace")

    def manifest(self) -> dict:
        return {"schema":"auro.byte_tokenizer.v1","vocab_size":VOCAB_SIZE,"byte_offset":BYTE_OFFSET,
                "control_tokens":list(CONTROL_TOKENS),"unknown_token":None,"byte_round_trip":True}


@dataclass(frozen=True)
class OpenHIMConfig:
    context_length: int = 16
    embedding_dim: int = 48
    hidden_dim: int = 128
    seed: int = 20260718


class OpenHIM:
    """Context MLP causal LM with trainable embeddings and explicit weights."""
    def __init__(self, config: OpenHIMConfig = OpenHIMConfig()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = .02
        self.weights = {
            "embedding": rng.normal(0, scale, (VOCAB_SIZE, config.embedding_dim)).astype(np.float32),
            "w1": rng.normal(0, scale, (config.context_length * config.embedding_dim, config.hidden_dim)).astype(np.float32),
            "b1": np.zeros(config.hidden_dim, np.float32),
            "w2": rng.normal(0, scale, (config.hidden_dim, VOCAB_SIZE)).astype(np.float32),
            "b2": np.zeros(VOCAB_SIZE, np.float32),
        }
        self.tokenizer = ByteTokenizer()

    @property
    def num_parameters(self) -> int:
        return sum(int(value.size) for value in self.weights.values())

    def logits(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = self.weights["embedding"][contexts]
        flat = embedded.reshape(len(contexts), -1)
        hidden = np.tanh(flat @ self.weights["w1"] + self.weights["b1"])
        return hidden @ self.weights["w2"] + self.weights["b2"], hidden, flat

    def generate(self, prompt: str, *, max_new_tokens: int = 120, temperature: float = .7,
                 top_k: int = 24, seed: int = 7) -> str:
        rng = np.random.default_rng(seed)
        ids = self.tokenizer.encode(prompt, bos=True)
        for _ in range(max_new_tokens):
            context = ([self.tokenizer.pad_id] * self.config.context_length + ids)[-self.config.context_length:]
            logits, _, _ = self.logits(np.asarray([context], dtype=np.int64))
            scores = logits[0].astype(np.float64) / max(float(temperature), .05)
            scores[:BYTE_OFFSET] = -1e9
            if 0 < top_k < len(scores):
                keep = np.argpartition(scores, -top_k)[-top_k:]
                mask = np.ones(len(scores), dtype=bool); mask[keep] = False; scores[mask] = -1e9
            scores -= scores.max(); probabilities = np.exp(scores); probabilities /= probabilities.sum()
            token = int(rng.choice(VOCAB_SIZE, p=probabilities)); ids.append(token)
            if token == self.tokenizer.eos_id: break
        return self.tokenizer.decode(ids)

    def save(self, directory: str | Path, report: dict) -> dict:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        raw = io.BytesIO(); np.savez_compressed(raw, **self.weights)
        encoded = base64.b64encode(raw.getvalue()).decode("ascii")
        (directory / "weights.npz.b64").write_text(encoded, encoding="ascii")
        (directory / "config.json").write_text(json.dumps(self.config.__dict__, indent=2), encoding="utf-8")
        (directory / "tokenizer.json").write_text(json.dumps(self.tokenizer.manifest(), indent=2), encoding="utf-8")
        (directory / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"weights_bytes":len(raw.getvalue()),"encoded_bytes":len(encoded)}

    @classmethod
    def load(cls, directory: str | Path) -> "OpenHIM":
        directory = Path(directory)
        config = OpenHIMConfig(**json.loads((directory / "config.json").read_text()))
        model = cls(config)
        raw = base64.b64decode((directory / "weights.npz.b64").read_text(encoding="ascii"))
        with np.load(io.BytesIO(raw)) as values:
            model.weights = {name: values[name].astype(np.float32) for name in values.files}
        return model
