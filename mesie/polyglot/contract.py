"""AIS Vector Polyglot contract — canonical cross-language message schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "1.0.0"
SUITE_NAME = "AISVectorPolyglot"


class PolyglotAction(str, Enum):
    VALIDATE = "validate"
    MATCH = "match"
    EMBED = "embed"
    RANK = "rank"
    FINGERPRINT = "fingerprint"
    HEALTH = "health"


class RuntimeId(str, Enum):
    PYTHON = "python"
    RUST = "rust"
    JULIA = "julia"
    MOTOKO = "motoko"
    TYPESCRIPT = "typescript"
    HTTP = "http"
    THIRD_PARTY = "third_party"


@dataclass
class AISVectorMessage:
    """Wire-format message for any AIS polyglot runtime."""

    action: PolyglotAction
    runtime: RuntimeId
    contract_version: str = CONTRACT_VERSION
    record: Optional[Dict[str, Any]] = None
    record_a: Optional[Dict[str, Any]] = None
    record_b: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    top_k: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, raw: str) -> "AISVectorMessage":
        data = json.loads(raw)
        data["action"] = PolyglotAction(data["action"])
        data["runtime"] = RuntimeId(data["runtime"])
        return cls(**data)


@dataclass
class AISVectorResponse:
    ok: bool
    runtime: RuntimeId
    action: PolyglotAction
    data: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "runtime": self.runtime.value,
            "action": self.action.value,
            "data": self.data,
            "vector": self.vector,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "contract_version": self.contract_version,
        }


def record_to_dict(record: Any) -> Dict[str, Any]:
    """Serialize a MultiElementRecord or loader input to contract JSON."""
    from mesie.io.loaders import load_record

    rec = load_record(record)
    return {
        "record_id": rec.record_id,
        "components": [
            {
                "name": c.name,
                "frequency": c.frequency.tolist() if hasattr(c.frequency, "tolist") else list(c.frequency),
                "amplitude": c.amplitude.tolist() if hasattr(c.amplitude, "tolist") else list(c.amplitude),
            }
            for c in rec.components
        ],
    }