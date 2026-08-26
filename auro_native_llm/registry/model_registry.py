from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelRecord:
    model_id: str
    architecture_hash: str
    tokenizer_hash: str
    corpus_hash: str
    weights_hash: str
    parent_checkpoint: str | None = None
    optimizer_state_hash: str | None = None
    quantization: str = "bf16"
    serving_compatibility: list[str] = field(default_factory=list)
    evaluation_receipts: list[str] = field(default_factory=list)
    promotion_level: str = "PROTOTYPE"
    rollback_target: str | None = None

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": "auro.model_registry.v1", "models": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def put(self, record: ModelRecord) -> dict[str, Any]:
        data = self._load()
        item = asdict(record)
        item["record_sha256"] = record.digest()
        data["models"][record.model_id] = item
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        return item

    def get(self, model_id: str) -> dict[str, Any] | None:
        return self._load()["models"].get(model_id)

    def validate(self, model_id: str) -> bool:
        item = self.get(model_id)
        if item is None:
            return False
        expected = item.pop("record_sha256", None)
        actual = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return bool(expected) and expected == actual
