"""Tamper-evident durable receipt chain for model, capability, and organ activity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any


@dataclass(frozen=True)
class Receipt:
    sequence: int
    timestamp_ns: int
    kind: str
    subject: str
    ok: bool
    previous_hash: str
    payload_hash: str
    receipt_hash: str
    metadata: dict[str, Any]


class ReceiptLedger:
    def __init__(self, path: str | Path | None = None):
        configured = path if path is not None else os.getenv("AURO_RECEIPT_LEDGER")
        self.path = Path(configured) if configured else None
        self._lock = threading.Lock()
        self._receipts: list[Receipt] = []
        if self.path and self.path.exists():
            self._load()

    def record(self, kind: str, subject: str, ok: bool, payload: Any, metadata: dict[str, Any] | None = None) -> Receipt:
        with self._lock:
            sequence = len(self._receipts) + 1
            previous = self._receipts[-1].receipt_hash if self._receipts else "GENESIS"
            payload_hash = _hash(payload)
            timestamp = time.time_ns()
            meta = dict(metadata or {})
            material = {
                "sequence": sequence,
                "timestamp_ns": timestamp,
                "kind": kind,
                "subject": subject,
                "ok": bool(ok),
                "previous_hash": previous,
                "payload_hash": payload_hash,
                "metadata": meta,
            }
            receipt = Receipt(**material, receipt_hash=_hash(material))
            if self.path:
                self._durable_append(receipt)
            self._receipts.append(receipt)
            return receipt

    def verify(self) -> dict[str, Any]:
        previous = "GENESIS"
        for index, receipt in enumerate(self._receipts, 1):
            material = {key: value for key, value in asdict(receipt).items() if key != "receipt_hash"}
            if receipt.sequence != index or receipt.previous_hash != previous or _hash(material) != receipt.receipt_hash:
                return {"valid": False, "failed_sequence": index, "head": previous}
            previous = receipt.receipt_hash
        return {"valid": True, "count": len(self._receipts), "head": previous}

    def tail(self, limit: int = 20):
        return [asdict(item) for item in self._receipts[-max(0, int(limit)):]]

    def _durable_append(self, receipt: Receipt) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(asdict(receipt), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.path.parent)

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise ValueError(f"unable to read receipt ledger: {exc}") from exc
        if not data:
            return
        if not data.endswith(b"\n"):
            raise ValueError("receipt ledger has an incomplete trailing record")
        for index, raw_line in enumerate(data.splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
                self._receipts.append(Receipt(**value))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid receipt ledger record at sequence {index}") from exc
        status = self.verify()
        if not status["valid"]:
            raise ValueError(f"invalid receipt ledger at sequence {status['failed_sequence']}")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
