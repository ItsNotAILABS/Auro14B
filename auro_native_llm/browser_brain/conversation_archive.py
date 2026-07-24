"""Durable, discoverable Browser-Brain conversation archive.

This module indexes human-readable and JSONL conversation artifacts produced by
HIM/AURO runs without changing the model, MESIE, scripture, memory, or agent
architecture. It provides one stable place for browser and CLI surfaces to list,
inspect, verify, and export prior conversations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ConversationRecord:
    archive_id: str
    source_kind: str
    source_path: str
    title: str
    turn_count: int
    session_id: str | None
    receipt_head: str | None
    sha256: str
    transcript_path: str | None = None
    jsonl_path: str | None = None
    summary_path: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _record_for_directory(directory: Path) -> ConversationRecord | None:
    transcript = directory / "TRANSCRIPT.md"
    conversation = directory / "conversation.jsonl"
    summary = directory / "summary.json"
    if not transcript.is_file() and not conversation.is_file():
        return None

    metadata = _read_json(summary) if summary.is_file() else {}
    identity = metadata.get("identity") if isinstance(metadata.get("identity"), dict) else {}
    session_id = identity.get("session_id") or metadata.get("session_id")
    turn_count = _count_jsonl(conversation) if conversation.is_file() else int(metadata.get("turns") or metadata.get("dialogue_turns") or 0)
    material = transcript if transcript.is_file() else conversation
    digest = sha256_file(material)
    archive_id = f"conversation-{digest[:16]}"
    title = directory.name.replace("-", " ").replace("_", " ").strip().title()
    return ConversationRecord(
        archive_id=archive_id,
        source_kind="artifact-directory",
        source_path=str(directory),
        title=title,
        turn_count=turn_count,
        session_id=str(session_id) if session_id else None,
        receipt_head=str(metadata.get("receipt_head")) if metadata.get("receipt_head") else None,
        sha256=digest,
        transcript_path=str(transcript) if transcript.is_file() else None,
        jsonl_path=str(conversation) if conversation.is_file() else None,
        summary_path=str(summary) if summary.is_file() else None,
    )


def discover_conversations(root: str | Path) -> list[ConversationRecord]:
    root = Path(root)
    records: list[ConversationRecord] = []
    if not root.exists():
        return records
    candidates = {path.parent for path in root.rglob("conversation.jsonl")}
    candidates.update(path.parent for path in root.rglob("TRANSCRIPT.md"))
    for directory in sorted(candidates):
        record = _record_for_directory(directory)
        if record:
            records.append(record)
    return records


def build_archive_index(
    repository_root: str | Path,
    output_path: str | Path | None = None,
    additional_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    repository_root = Path(repository_root)
    roots = [repository_root / "artifacts", repository_root / "deliverables", *[Path(x) for x in additional_roots]]
    records: dict[str, ConversationRecord] = {}
    for root in roots:
        for record in discover_conversations(root):
            records.setdefault(record.archive_id, record)
    payload = {
        "schema": "auro.browser-brain.conversation-archive.v1",
        "repository_root": str(repository_root),
        "roots_scanned": [str(root) for root in roots],
        "conversation_count": len(records),
        "conversations": [record.public() for record in sorted(records.values(), key=lambda item: item.archive_id)],
        "known_historical_runs": {
            "pr_57": "HIM birth observation conversation",
            "pr_58": "HIM language maturation conversation and autonomous work",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["index_sha256"] = hashlib.sha256(canonical).hexdigest()
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def read_conversation(record: ConversationRecord) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    if record.jsonl_path:
        path = Path(record.jsonl_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            turns.append(value if isinstance(value, dict) else {"value": value})
    transcript = Path(record.transcript_path).read_text(encoding="utf-8") if record.transcript_path else ""
    return {"record": record.public(), "turns": turns, "transcript": transcript}
