"""Receipt and append-only event primitives for AURO continuous learning."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class AgentReceipt:
    run_id: str
    agent: str
    status: str
    started_at_unix: int
    completed_at_unix: int
    inputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()
    schema: str = "auro.continuous.agent-receipt.v1"
    receipt_sha256: str = ""

    def seal(self) -> "AgentReceipt":
        payload = asdict(self)
        payload["receipt_sha256"] = ""
        digest = hashlib.sha256(canonical_json(payload)).hexdigest()
        return AgentReceipt(**{**payload, "receipt_sha256": digest})

    def verify(self) -> bool:
        return self.seal().receipt_sha256 == self.receipt_sha256


class ReceiptStore:
    """Append-only JSONL event log plus immutable per-run receipt files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.receipts = self.root / "receipts"
        self.events = self.root / "events.jsonl"
        self.receipts.mkdir(parents=True, exist_ok=True)

    def write(self, receipt: AgentReceipt) -> Path:
        sealed = receipt if receipt.receipt_sha256 else receipt.seal()
        if not sealed.verify():
            raise ValueError("agent receipt integrity verification failed")
        path = self.receipts / f"{sealed.run_id}.{sealed.agent}.json"
        if path.exists():
            raise FileExistsError(f"immutable receipt already exists: {path}")
        document = asdict(sealed)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        event = {
            "event": "agent_run_completed",
            "run_id": sealed.run_id,
            "agent": sealed.agent,
            "status": sealed.status,
            "receipt_sha256": sealed.receipt_sha256,
            "recorded_at_unix": int(time.time()),
        }
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return path
