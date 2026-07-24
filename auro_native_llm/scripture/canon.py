"""Canon — loadable, hash-bound scriptural doctrine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def default_canon_path() -> Path:
    return Path(__file__).resolve().parents[2] / "native_llm" / "scripture" / "AURO_CANON.v1.json"


@dataclass
class Article:
    id: str
    title: str
    text: str
    binds: List[str] = field(default_factory=list)
    severity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "binds": list(self.binds),
            "severity": self.severity,
        }


@dataclass
class Canon:
    """Executable scripture for Auro / MESIE systems."""

    canon_id: str
    title: str
    version: str
    principle: str
    articles: List[Article]
    gates: List[str]
    denied_intents: List[str]
    allowed_ops: List[str]
    role_host_matrix: Dict[str, List[str]]
    family: List[str]
    claim_boundary: str
    memory: Dict[str, Any]
    governance: Dict[str, Any]
    compute_plane: str = "MESIE"
    principles: List[Dict[str, Any]] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    decision_rules: List[Dict[str, Any]] = field(default_factory=list)
    memory_rules: List[Dict[str, Any]] = field(default_factory=list)
    process_model: Dict[str, Any] = field(default_factory=dict)
    integration_level: str = "hybrid_neuro_symbolic"
    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    content_sha256: str = ""

    def articles_for(self, op: str) -> List[Article]:
        return [a for a in self.articles if op in a.binds or not a.binds]

    def severity_articles_for(self, op: str) -> List[Article]:
        return [a for a in self.articles_for(op) if a.severity]

    def may_host(self, parent: str, child: str) -> bool:
        allowed = self.role_host_matrix.get(parent)
        if allowed is None:
            return False
        return child in allowed

    def preamble(self, max_articles: int = 4) -> str:
        lines = [
            f"[CANON {self.canon_id} v{self.version}]",
            self.principle,
            f"compute_plane={self.compute_plane}",
            f"claim_boundary={self.claim_boundary}",
        ]
        for a in self.articles[:max_articles]:
            lines.append(f"- {a.id}: {a.title}: {a.text}")
        lines.append("[/CANON]")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canon_id": self.canon_id,
            "title": self.title,
            "version": self.version,
            "principle": self.principle,
            "compute_plane": self.compute_plane,
            "content_sha256": self.content_sha256,
            "source_path": self.source_path,
            "family": list(self.family),
            "gates": list(self.gates),
            "articles": [a.to_dict() for a in self.articles],
            "denied_intents": list(self.denied_intents),
            "allowed_ops": list(self.allowed_ops),
            "claim_boundary": self.claim_boundary,
            "memory": dict(self.memory),
            "governance": dict(self.governance),
            "principles": list(self.principles),
            "invariants": list(self.invariants),
            "decision_rules": list(self.decision_rules),
            "memory_rules": list(self.memory_rules),
            "process_model": dict(self.process_model),
            "integration_level": self.integration_level,
        }


def _sha256_obj(obj: Dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_canon(path: Optional[str | Path] = None) -> Canon:
    p = Path(path) if path else default_canon_path()
    if not p.exists():
        raise FileNotFoundError(f"canon not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    articles = [
        Article(
            id=str(a["id"]),
            title=str(a["title"]),
            text=str(a["text"]),
            binds=list(a.get("binds", [])),
            severity=bool(a.get("severity", False)),
        )
        for a in raw.get("articles", [])
    ]
    return Canon(
        canon_id=str(raw.get("canon_id", "UNKNOWN")),
        title=str(raw.get("title", "Canon")),
        version=str(raw.get("version", "0")),
        principle=str(raw.get("principle", "")),
        articles=articles,
        gates=list(raw.get("gates", [])),
        denied_intents=list(raw.get("denied_intents", [])),
        allowed_ops=list(raw.get("allowed_ops", [])),
        role_host_matrix={k: list(v) for k, v in raw.get("role_host_matrix", {}).items()},
        family=list(raw.get("family", [])),
        claim_boundary=str(raw.get("claim_boundary", "")),
        memory=dict(raw.get("memory", {})),
        governance=dict(raw.get("governance", {"fail_closed": True})),
        compute_plane=str(raw.get("compute_plane", "MESIE")),
        principles=list(raw.get("principles", [])),
        invariants=list(raw.get("invariants", [])),
        decision_rules=list(raw.get("decision_rules", [])),
        memory_rules=list(raw.get("memory_rules", [])),
        process_model=dict(raw.get("process_model", {})),
        integration_level=str(raw.get("integration_level", "hybrid_neuro_symbolic")),
        raw=raw,
        source_path=str(p),
        content_sha256=_sha256_obj(raw),
    )
