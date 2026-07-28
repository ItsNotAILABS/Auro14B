"""Tamper-evident receipt primitives for AURO continuous learning."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
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
    schema: str = "auro.continuous.agent-receipt.v2"
    previous_receipt_sha256: str = ""
    signer_id: str = ""
    receipt_sha256: str = ""
    signature_hmac_sha256: str = ""

    def seal(
        self,
        *,
        previous_receipt_sha256: str | None = None,
        signing_key: str | bytes | None = None,
        signer_id: str | None = None,
    ) -> "AgentReceipt":
        payload = asdict(self)
        if previous_receipt_sha256 is not None:
            payload["previous_receipt_sha256"] = previous_receipt_sha256
        if signer_id is not None:
            payload["signer_id"] = signer_id
        payload["receipt_sha256"] = ""
        payload["signature_hmac_sha256"] = ""
        digest = hashlib.sha256(canonical_json(payload)).hexdigest()
        signature = ""
        if signing_key:
            key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
            signature = hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()
        return AgentReceipt(**{**payload, "receipt_sha256": digest, "signature_hmac_sha256": signature})

    def verify(self, signing_key: str | bytes | None = None) -> bool:
        resealed = self.seal(
            previous_receipt_sha256=self.previous_receipt_sha256,
            signing_key=signing_key if self.signature_hmac_sha256 else None,
            signer_id=self.signer_id,
        )
        if resealed.receipt_sha256 != self.receipt_sha256:
            return False
        if self.signature_hmac_sha256:
            return hmac.compare_digest(resealed.signature_hmac_sha256, self.signature_hmac_sha256)
        return True

    @property
    def signed(self) -> bool:
        return bool(self.signer_id and self.signature_hmac_sha256)


class ReceiptStore:
    """Hash-chained JSONL event log plus immutable signed receipt files.

    Set ``AURO_RECEIPT_SIGNING_KEY`` and ``AURO_RECEIPT_SIGNER_ID`` in a trusted
    worker. Unsigned receipts remain usable for development but cannot satisfy a
    constitutional promotion gate.
    """

    def __init__(self, root: str | Path, signing_key: str | None = None, signer_id: str | None = None):
        self.root = Path(root)
        self.receipts = self.root / "receipts"
        self.events = self.root / "events.jsonl"
        self.chain_head = self.root / "chain-head.json"
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.signing_key = signing_key if signing_key is not None else os.environ.get("AURO_RECEIPT_SIGNING_KEY", "")
        self.signer_id = signer_id if signer_id is not None else os.environ.get("AURO_RECEIPT_SIGNER_ID", "")

    def _head(self) -> str:
        if not self.chain_head.is_file():
            return ""
        try:
            return str(json.loads(self.chain_head.read_text(encoding="utf-8")).get("receipt_sha256") or "")
        except Exception:
            raise ValueError("receipt chain head is unreadable")

    def write(self, receipt: AgentReceipt) -> Path:
        previous = self._head()
        sealed = receipt.seal(
            previous_receipt_sha256=previous,
            signing_key=self.signing_key or None,
            signer_id=self.signer_id if self.signing_key else "",
        )
        if not sealed.verify(self.signing_key or None):
            raise ValueError("agent receipt integrity or signature verification failed")
        path = self.receipts / f"{sealed.run_id}.{sealed.agent}.json"
        if path.exists():
            raise FileExistsError(f"immutable receipt already exists: {path}")
        document = asdict(sealed)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        event = {
            "event": "agent_run_completed",
            "run_id": sealed.run_id,
            "agent": sealed.agent,
            "status": sealed.status,
            "receipt_sha256": sealed.receipt_sha256,
            "previous_receipt_sha256": sealed.previous_receipt_sha256,
            "signed": sealed.signed,
            "signer_id": sealed.signer_id,
            "recorded_at_unix": int(time.time()),
        }
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        head = {
            "schema": "auro.continuous.receipt-chain-head.v1",
            "receipt_sha256": sealed.receipt_sha256,
            "previous_receipt_sha256": sealed.previous_receipt_sha256,
            "signed": sealed.signed,
            "signer_id": sealed.signer_id,
        }
        temporary_head = self.chain_head.with_suffix(".tmp")
        temporary_head.write_text(json.dumps(head, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_head.replace(self.chain_head)
        return path
