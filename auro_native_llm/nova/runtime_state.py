"""Durable NOVA execution state backed by SQLite WAL.

This replaces process-local approval and replay bookkeeping with one-time,
action-bound authorization and persistent session/receipt custody.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def action_sha256(action: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(action)).hexdigest()


class NovaRuntimeState:
    def __init__(self, path: str | Path = "state/nova-runtime.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              principal TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              status TEXT NOT NULL,
              state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approvals (
              approval_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              principal TEXT NOT NULL,
              action_sha256 TEXT NOT NULL,
              issued_at INTEGER NOT NULL,
              expires_at INTEGER NOT NULL,
              consumed_at INTEGER,
              revoked_at INTEGER,
              FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS replay_keys (
              replay_key TEXT PRIMARY KEY,
              recorded_at INTEGER NOT NULL,
              receipt_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS receipts (
              receipt_sha256 TEXT PRIMARY KEY,
              previous_receipt_sha256 TEXT NOT NULL,
              session_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              recorded_at INTEGER NOT NULL
            );
            """
        )
        self.db.commit()

    def create_session(self, principal: str, initial_state: Mapping[str, Any] | None = None) -> str:
        now = int(time.time())
        session_id = f"nova-{now}-{secrets.token_hex(12)}"
        self.db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, 'active', ?)",
            (session_id, principal, now, now, json.dumps(dict(initial_state or {}), sort_keys=True)),
        )
        self.db.commit()
        return session_id

    def session(self, session_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "state": json.loads(row["state_json"])}

    def issue_approval(
        self,
        session_id: str,
        principal: str,
        action: Mapping[str, Any],
        ttl_seconds: int = 300,
    ) -> dict[str, Any]:
        session = self.session(session_id)
        if session is None or session["status"] != "active" or session["principal"] != principal:
            raise PermissionError("active principal-bound NOVA session required")
        now = int(time.time())
        approval_id = f"approval-{secrets.token_urlsafe(24)}"
        digest = action_sha256(action)
        self.db.execute(
            "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
            (approval_id, session_id, principal, digest, now, now + max(1, min(ttl_seconds, 3600))),
        )
        self.db.commit()
        return {"approval_id": approval_id, "action_sha256": digest, "expires_at": now + ttl_seconds}

    def consume_approval(
        self,
        approval_id: str,
        session_id: str,
        principal: str,
        action: Mapping[str, Any],
        now: int | None = None,
    ) -> dict[str, Any]:
        current = int(time.time()) if now is None else int(now)
        digest = action_sha256(action)
        self.db.execute("BEGIN IMMEDIATE")
        row = self.db.execute("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
        if row is None:
            self.db.rollback()
            raise PermissionError("unknown approval_id")
        reasons = []
        if row["session_id"] != session_id: reasons.append("session mismatch")
        if row["principal"] != principal: reasons.append("principal mismatch")
        if row["action_sha256"] != digest: reasons.append("action mismatch")
        if row["consumed_at"] is not None: reasons.append("approval already consumed")
        if row["revoked_at"] is not None: reasons.append("approval revoked")
        if current > row["expires_at"]: reasons.append("approval expired")
        if reasons:
            self.db.rollback()
            raise PermissionError("; ".join(reasons))
        self.db.execute("UPDATE approvals SET consumed_at = ? WHERE approval_id = ?", (current, approval_id))
        self.db.commit()
        return {"approval_id": approval_id, "authorized": True, "action_sha256": digest, "consumed_at": current}

    def record_receipt(self, session_id: str, event_type: str, payload: Mapping[str, Any], replay_key: str = "") -> dict[str, Any]:
        if self.session(session_id) is None:
            raise ValueError("unknown session")
        now = int(time.time())
        previous = self.db.execute(
            "SELECT receipt_sha256 FROM receipts ORDER BY recorded_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        previous_sha = str(previous[0]) if previous else ""
        document = {
            "schema": "nova.runtime.receipt.v1",
            "session_id": session_id,
            "event_type": event_type,
            "payload": dict(payload),
            "previous_receipt_sha256": previous_sha,
            "recorded_at": now,
        }
        receipt_sha = hashlib.sha256(canonical_json(document)).hexdigest()
        self.db.execute("BEGIN IMMEDIATE")
        if replay_key:
            existing = self.db.execute("SELECT 1 FROM replay_keys WHERE replay_key = ?", (replay_key,)).fetchone()
            if existing:
                self.db.rollback()
                raise ValueError("replayed NOVA event")
            self.db.execute("INSERT INTO replay_keys VALUES (?, ?, ?)", (replay_key, now, receipt_sha))
        self.db.execute(
            "INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?)",
            (receipt_sha, previous_sha, session_id, event_type, json.dumps(dict(payload), sort_keys=True), now),
        )
        self.db.commit()
        return {**document, "receipt_sha256": receipt_sha}

    def close(self) -> None:
        self.db.close()
