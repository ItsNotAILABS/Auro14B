from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Auro model configuration.

    The same implementation is used by the local micro lane and the dense
    206.7B target lane. Scale is a configuration decision, not a separate code
    path.
    """

    model_id: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seq_len: int = 4096
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    bias: bool = False
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.vocab_size < 264:
            raise ValueError("vocab_size must leave room for byte and special tokens")
        if self.num_layers < 1 or self.max_seq_len < 8:
            raise ValueError("model depth and context must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.num_kv_heads

    def estimated_parameters(self) -> int:
        """Return a dense parameter estimate matching the implemented model."""
        self.validate()
        d = self.hidden_size
        kv = self.kv_dim
        embeddings = self.vocab_size * d
        output = 0 if self.tie_word_embeddings else embeddings
        attention = (d * d) + (d * kv * 2) + (d * d)
        mlp = 3 * d * self.intermediate_size
        norms = 2 * d
        per_layer = attention + mlp + norms
        final_norm = d
        return embeddings + output + self.num_layers * per_layer + final_norm

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["estimated_parameters"] = self.estimated_parameters()
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        clean = {k: v for k, v in value.items() if k != "estimated_parameters"}
        config = cls(**clean)
        config.validate()
        return config


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    run_name: str = "auro-local-run"
    output_dir: str = "artifacts/auro-foundry"
    dataset_dir: str = "artifacts/auro-foundry/dataset"
    tokenizer_path: str = "artifacts/auro-foundry/tokenizer.json"
    sequence_length: int = 512
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_steps: int = 200
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    checkpoint_interval: int = 100
    log_interval: int = 5
    seed: int = 1337
    precision: str = "auto"
    strategy: str = "auto"
    compile_model: bool = False
    device: str = "auto"
    resume_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.model.validate()
        if self.sequence_length > self.model.max_seq_len:
            raise ValueError("sequence_length exceeds model max_seq_len")
        for name in ("micro_batch_size", "gradient_accumulation_steps", "max_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.strategy not in {"auto", "single", "ddp", "fsdp"}:
            raise ValueError("strategy must be auto, single, ddp, or fsdp")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model"] = self.model.to_dict()
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainConfig":
        data = dict(value)
        data["model"] = ModelConfig.from_dict(data["model"])
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target


def preset(name: str) -> ModelConfig:
    presets = {
        "micro": ModelConfig("Auro-Micro", 4096, 256, 6, 8, 2, 768, 1024, dropout=0.1),
        "local": ModelConfig("Auro-Local-110M", 16384, 768, 12, 12, 4, 3072, 4096, dropout=0.1),
        "14b": ModelConfig("Auro-14B", 110000, 5120, 48, 40, 8, 13824, 32768, gradient_checkpointing=True),
        "206.7b": ModelConfig(
            "Auro-206.7B-Dense",
            110000,
            12288,
            96,
            96,
            8,
            49152,
            131072,
            gradient_checkpointing=True,
        ),
    }
    try:
        config = presets[name.lower()]
    except KeyError as exc:
        raise KeyError(f"unknown model preset: {name}; choose {sorted(presets)}") from exc
    config.validate()
    return config
