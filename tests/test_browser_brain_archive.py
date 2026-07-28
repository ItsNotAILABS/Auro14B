import json
from pathlib import Path

from auro_native_llm.browser_brain import build_archive_index, discover_conversations
from auro_native_llm.browser_brain.service import BrowserBrainService


def make_conversation(root: Path) -> Path:
    run = root / "artifacts" / "him-birth-observation"
    run.mkdir(parents=True)
    rows = [
        {
            "sequence": 1,
            "case_id": "identity",
            "prompt": "Who are you?",
            "answer": "I am AURO running through the preserved HIM runtime.",
            "ok": True,
            "previous_hash": "0" * 64,
            "hash": "1" * 64,
        },
        {
            "sequence": 2,
            "kind": "dialogue",
            "phase": "continuity",
            "instruction": "Preserve continuity.",
            "output": "Continuity is preserved through state, memory, and receipts.",
            "ok": True,
            "previous_hash": "1" * 64,
            "hash": "2" * 64,
        },
    ]
    (run / "conversation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run / "TRANSCRIPT.md").write_text(
        "# HIM Birth Observation\n\nUSER: Who are you?\n\nHIM: I am AURO.\n",
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"identity": {"session_id": "session-1"}, "receipt_head": "2" * 64}),
        encoding="utf-8",
    )
    return run


def test_archive_discovers_and_hashes_conversations(tmp_path):
    make_conversation(tmp_path)
    records = discover_conversations(tmp_path / "artifacts")
    assert len(records) == 1
    assert records[0].turn_count == 2
    assert records[0].session_id == "session-1"
    assert len(records[0].sha256) == 64

    index = build_archive_index(tmp_path)
    assert index["conversation_count"] == 1
    assert len(index["index_sha256"]) == 64
    assert index["known_historical_runs"]["pr_57"]


def test_service_search_timeline_and_continuation(tmp_path):
    make_conversation(tmp_path)
    service = BrowserBrainService(tmp_path)
    listing = service.list("birth")
    assert listing["count"] == 1
    archive_id = listing["conversations"][0]["archive_id"]

    payload = service.get(archive_id)
    assert payload and len(payload["turns"]) == 2

    timeline = service.timeline(archive_id)
    assert timeline and timeline["event_count"] == 2
    assert timeline["events"][1]["phase"] == "continuity"
    assert timeline["events"][1]["receipt"] == "2" * 64

    continuation = service.continuation_context(archive_id)
    assert continuation
    assert "HIM Birth Observation" in continuation["context"]
    assert continuation["source_sha256"] == payload["record"]["sha256"]


def test_unknown_archive_is_not_invented(tmp_path):
    service = BrowserBrainService(tmp_path)
    assert service.get("missing") is None
    assert service.timeline("missing") is None
    assert service.continuation_context("missing") is None
