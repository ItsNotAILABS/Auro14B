"""Read-only Browser-Brain service operations for API and UI surfaces."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .conversation_archive import ConversationRecord, build_archive_index, read_conversation


class BrowserBrainService:
    """Indexes, searches, verifies, and replays preserved conversation artifacts."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()

    def index(self) -> dict[str, Any]:
        return build_archive_index(self.repository_root)

    def list(self, query: str = "", limit: int = 100) -> dict[str, Any]:
        index = self.index()
        rows = list(index["conversations"])
        needle = query.casefold().strip()
        if needle:
            rows = [
                row for row in rows
                if needle in " ".join(str(value) for value in row.values() if value is not None).casefold()
            ]
        rows = rows[: max(1, min(int(limit), 500))]
        return {
            "schema": "auro.browser-brain.conversation-list.v1",
            "query": query,
            "count": len(rows),
            "conversations": rows,
            "index_sha256": index["index_sha256"],
        }

    def get(self, archive_id: str) -> dict[str, Any] | None:
        for row in self.index()["conversations"]:
            if row["archive_id"] == archive_id:
                allowed = {field.name for field in fields(ConversationRecord)}
                record = ConversationRecord(**{key: value for key, value in row.items() if key in allowed})
                return read_conversation(record)
        return None

    def timeline(self, archive_id: str) -> dict[str, Any] | None:
        payload = self.get(archive_id)
        if payload is None:
            return None
        events = []
        for position, turn in enumerate(payload["turns"], 1):
            events.append({
                "position": position,
                "sequence": turn.get("sequence", position),
                "kind": turn.get("kind", "conversation"),
                "phase": turn.get("phase") or turn.get("case_id") or "turn",
                "prompt": turn.get("prompt") or turn.get("instruction") or "",
                "answer": turn.get("answer") or turn.get("output") or "",
                "ok": bool(turn.get("ok", True)),
                "receipt": turn.get("hash") or turn.get("receipt_sha256"),
                "previous_receipt": turn.get("previous_hash"),
                "latency_ms": turn.get("latency_ms"),
            })
        return {
            "schema": "auro.browser-brain.timeline.v1",
            "archive_id": archive_id,
            "event_count": len(events),
            "events": events,
            "record": payload["record"],
        }

    def continuation_context(self, archive_id: str, max_characters: int = 24000) -> dict[str, Any] | None:
        payload = self.get(archive_id)
        if payload is None:
            return None
        transcript = payload["transcript"].strip()
        if not transcript:
            transcript = "\n\n".join(
                f"USER: {turn.get('prompt') or turn.get('instruction') or ''}\nASSISTANT: {turn.get('answer') or turn.get('output') or ''}"
                for turn in payload["turns"]
            )
        clipped = transcript[-max(1000, min(int(max_characters), 200000)):]
        return {
            "schema": "auro.browser-brain.continuation-context.v1",
            "archive_id": archive_id,
            "context": clipped,
            "characters": len(clipped),
            "source_sha256": payload["record"]["sha256"],
            "instruction": "Continue from this preserved conversation without rewriting its history or inventing missing turns.",
        }
