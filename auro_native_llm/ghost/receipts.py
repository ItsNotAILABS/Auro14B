"""Traceable receipt chain of custody (GHOST T pillar)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class GhostReceipt:
    """One linked receipt in the chain of custody."""

    receipt_id: str
    kind: str  # intent | policy | mesie | llm | tool | validate | result
    parent_hash: Optional[str]
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)
    actor: str = "auro.ghost"
    model_id: Optional[str] = None
    ok: bool = True

    def body(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "parent_hash": self.parent_hash,
            "payload": self.payload,
            "ts": self.ts,
            "actor": self.actor,
            "model_id": self.model_id,
            "ok": self.ok,
        }

    def content_hash(self) -> str:
        return _sha(self.body())

    def to_dict(self) -> Dict[str, Any]:
        b = self.body()
        b["content_hash"] = self.content_hash()
        return b


class GhostReceiptChain:
    """Hash-linked receipts answering who/what/policy/tools/change/validation/replay."""

    def __init__(self, chain_id: Optional[str] = None) -> None:
        self.chain_id = chain_id or f"ghost-{uuid.uuid4().hex[:12]}"
        self.receipts: List[GhostReceipt] = []
        self._tip_hash: Optional[str] = None

    def append(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        actor: str = "auro.ghost",
        model_id: Optional[str] = None,
        ok: bool = True,
    ) -> GhostReceipt:
        rec = GhostReceipt(
            receipt_id=f"r-{uuid.uuid4().hex[:10]}",
            kind=kind,
            parent_hash=self._tip_hash,
            payload=payload,
            actor=actor,
            model_id=model_id,
            ok=ok,
        )
        self._tip_hash = rec.content_hash()
        self.receipts.append(rec)
        return rec

    @property
    def tip_hash(self) -> Optional[str]:
        return self._tip_hash

    def verify(self) -> Dict[str, Any]:
        """Replay-check parent links and hashes."""
        prev = None
        for i, r in enumerate(self.receipts):
            if r.parent_hash != prev:
                return {"ok": False, "broken_at": i, "receipt_id": r.receipt_id}
            prev = r.content_hash()
        return {
            "ok": True,
            "n": len(self.receipts),
            "tip": self._tip_hash,
            "chain_id": self.chain_id,
        }

    def custody_answers(self) -> Dict[str, Any]:
        """Who/what/policy/tools/change/validation/replay."""
        kinds = [r.kind for r in self.receipts]
        actors = sorted({r.actor for r in self.receipts})
        models = sorted({r.model_id for r in self.receipts if r.model_id})
        policy = next((r.payload for r in self.receipts if r.kind == "policy"), {})
        tools = [
            r.payload.get("tool") or r.payload.get("engine")
            for r in self.receipts
            if r.kind in ("mesie", "tool", "llm")
        ]
        return {
            "who_initiated": next(
                (r.payload.get("initiator") for r in self.receipts if r.kind == "intent"),
                actors[0] if actors else None,
            ),
            "actors": actors,
            "policy": policy,
            "tools_models": {"tools": [t for t in tools if t], "models": models},
            "kinds_sequence": kinds,
            "what_changed": [
                r.payload.get("change")
                for r in self.receipts
                if r.payload.get("change")
            ],
            "validated": any(r.kind == "validate" and r.ok for r in self.receipts),
            "replayable": self.verify().get("ok"),
            "tip_hash": self._tip_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.ghost.receipt_chain.v1",
            "chain_id": self.chain_id,
            "tip_hash": self._tip_hash,
            "n_receipts": len(self.receipts),
            "verify": self.verify(),
            "custody": self.custody_answers(),
            "receipts": [r.to_dict() for r in self.receipts],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
