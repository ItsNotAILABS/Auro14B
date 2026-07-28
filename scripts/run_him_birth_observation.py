"""Run a fail-closed HIM dialogue and autonomous-development observation.

This is an observation instrument, not proof of durable cross-session memory,
checkpoint promotion, or external receipt custody unless promotion mode is used
with a trusted signing key.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

from auro_native_llm.him.mature import awaken_mature_him
from auro_native_llm.model.usable import is_usable_text
from auro_native_llm.organism.checkpoint import load_mind
from auro_native_llm.organism.family import build_mind

MISSION = """NEXUS Relay / SignalLens watches LangChain SSRF hardening, LiteLLM proxy controls, MCP SDK authorization patterns, and Qdrant/Milvus retrieval features. Use primary sources, separate verified facts from inference, preserve uncertainty, and produce actionable signals rather than marketing summaries."""
SNAPSHOT = """Observer-supplied source snapshot, 2026-07-23: LangChain advisories and releases include SSRF hardening themes. LiteLLM documents centralized proxy controls. MCP authorization varies by transport. Qdrant and Milvus are monitored for advanced retrieval and operational changes. Exact current claims require source verification before alerts."""
DIALOGUE = [
    ("mission", f"Read this mission and state the work you understand.\n\n{MISSION}"),
    ("evidence", "What evidence rules will prevent security, authorization, proxy, and retrieval claims from being overstated?"),
    ("language", "Explain how your word, character, phrase, spatial, grabbing, creative, and revision algorithms will support this work."),
    ("handoff", "After this answer, stop treating later stages as conversation. State the autonomous generate, read, challenge, revise, and report sequence."),
]


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load(checkpoint: str | None, allow_lightweight_fallback: bool):
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"requested checkpoint does not exist: {path}")
        try:
            return load_mind(path, chrome_mock=True), str(path), "checkpoint"
        except Exception as exc:
            raise RuntimeError(f"requested checkpoint failed to load: {path}: {type(exc).__name__}: {exc}") from exc

    candidates = (
        Path("checkpoints/auro_minds/Auro-2B_continual"),
        Path("checkpoints/auro_minds/Auro-2B_physics"),
        Path("checkpoints/open/HIM-native-v0"),
    )
    failures = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            return load_mind(path, chrome_mock=True), str(path), "checkpoint"
        except Exception as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("checkpoint candidates existed but failed to load; fallback denied: " + " | ".join(failures))
    if not allow_lightweight_fallback:
        raise FileNotFoundError("no checkpoint was found and lightweight fallback was not authorized")
    return build_mind("Auro-2B", lite=True, chrome_mock=True), "built:Auro-2B-lite", "lightweight_fixture"


def record(him, kind: str, phase: str, instruction: str, sequence: int, previous_hash: str) -> dict[str, Any]:
    started = time.time()
    error = None
    try:
        report = him.run(instruction, max_actions=5)
    except Exception as exc:
        report = {"ok": False, "answer": "", "method": "exception", "steps": [], "language_receipt": None}
        error = f"{type(exc).__name__}: {exc}"
    output = str(report.get("answer") or report.get("text") or "")
    row = {
        "sequence": sequence,
        "kind": kind,
        "phase": phase,
        "instruction": instruction,
        "output": output,
        "ok": bool(report.get("ok")),
        "usable": is_usable_text(output, min_len=40),
        "method": report.get("method"),
        "plan": report.get("plan"),
        "steps": report.get("steps"),
        "latency_ms": report.get("latency_ms") or (time.time() - started) * 1000,
        "language_receipt": report.get("language_receipt"),
        "previous_hash": previous_hash,
        "error": error,
    }
    row["hash"] = digest(row)
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-lightweight-fallback", action="store_true")
    parser.add_argument("--promotion-observation", action="store_true")
    parser.add_argument("--output", default="artifacts/him-birth-observation")
    args = parser.parse_args(argv)

    signing_key = os.environ.get("HIM_OBSERVATION_SIGNING_KEY", "")
    signer_id = os.environ.get("HIM_OBSERVATION_SIGNER_ID", "")
    if args.promotion_observation and (not signing_key or not signer_id):
        raise SystemExit("promotion observation requires HIM_OBSERVATION_SIGNING_KEY and HIM_OBSERVATION_SIGNER_ID")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    mind, checkpoint_source, checkpoint_class = load(args.checkpoint, args.allow_lightweight_fallback)
    him = awaken_mature_him(mind, n_germs=20, context_tokens=500_000)
    him.colony.context.ingest(MISSION, kind="system", meta={"program": "NEXUS Relay / SignalLens"})
    him.colony.context.ingest(SNAPSHOT, kind="evidence", meta={"as_of": "2026-07-23"})
    him.language.observe(MISSION, source="mission")
    him.language.observe(SNAPSHOT, source="source-snapshot")

    rows: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    for phase, prompt in DIALOGUE:
        row = record(him, "dialogue", phase, prompt, len(rows) + 1, previous_hash)
        previous_hash = row["hash"]
        rows.append(row)
        print(json.dumps({"kind": "dialogue", "phase": phase, "ok": row["ok"], "usable": row["usable"]}), flush=True)

    task = f"{MISSION}\n\n{SNAPSHOT}\n\nProduce an evidence-aware research report, language-engine analysis, operational watch design, and falsifiable competitive evaluation contract."
    development = him.develop(task, cycles=4)
    for stage in development["stages"]:
        report = stage["report"]
        output = str(report.get("answer") or report.get("text") or "")
        row = {
            "sequence": len(rows) + 1,
            "kind": "autonomous_work",
            "phase": f"development_{stage['stage']}",
            "instruction": stage["instruction"],
            "output": output,
            "ok": bool(report.get("ok")),
            "usable": is_usable_text(output, min_len=40),
            "method": report.get("method"),
            "plan": report.get("plan"),
            "steps": report.get("steps"),
            "latency_ms": report.get("latency_ms"),
            "language_receipt": report.get("language_receipt"),
            "previous_hash": previous_hash,
            "error": None,
        }
        row["hash"] = digest(row)
        previous_hash = row["hash"]
        rows.append(row)
        (root / f"{row['sequence']:02d}-{row['phase']}.md").write_text(output, encoding="utf-8")

    final = development["final"]
    final_row = {
        "sequence": len(rows) + 1,
        "kind": "autonomous_work",
        "phase": "final_report",
        "instruction": "Seal the final report after generation, readback, red-team, rewrite, and language-engine revision.",
        "output": final,
        "ok": bool(final),
        "usable": is_usable_text(final, min_len=40),
        "method": "mature_him_development_final",
        "plan": None,
        "steps": [],
        "latency_ms": None,
        "language_receipt": development["revision_receipt"],
        "previous_hash": previous_hash,
        "error": None,
    }
    final_row["hash"] = digest(final_row)
    previous_hash = final_row["hash"]
    rows.append(final_row)

    all_ok = bool(rows) and all(row["ok"] for row in rows)
    all_usable = bool(rows) and all(row["usable"] for row in rows)
    all_language_receipts = bool(rows) and all(bool(row["language_receipt"]) for row in rows)
    chain_valid = all(row["previous_hash"] == ("0" * 64 if index == 0 else rows[index - 1]["hash"]) for index, row in enumerate(rows))
    signature = hmac.new(signing_key.encode(), previous_hash.encode(), hashlib.sha256).hexdigest() if signing_key else ""

    (root / "FINAL_HIM_LANGUAGE_REPORT.md").write_text(final, encoding="utf-8")
    write_jsonl(root / "cycle.jsonl", rows)
    write_jsonl(root / "conversation.jsonl", [row for row in rows if row["kind"] == "dialogue"])
    write_jsonl(root / "autonomous_work.jsonl", [row for row in rows if row["kind"] == "autonomous_work"])
    write_jsonl(root / "language_receipts.jsonl", [{"sequence": row["sequence"], "phase": row["phase"], "language_receipt": row["language_receipt"], "hash": row["hash"]} for row in rows])
    lexical_receipt = him.language.save(root / "LEXICAL_SPATIAL_LIBRARY.json")
    (root / "WEIGHT_UPDATE_STATUS.md").write_text("# Weight Update Status\n\nNo optimizer step or checkpoint weight update occurred in this run. The trajectory is potential training data, not a trained checkpoint.\n", encoding="utf-8")

    promotion_eligible = args.promotion_observation and checkpoint_class == "checkpoint" and all_ok and all_usable and all_language_receipts and chain_valid and bool(signature)
    summary = {
        "schema": "auro.him.language-maturation.v3",
        "checkpoint_source": checkpoint_source,
        "checkpoint_class": checkpoint_class,
        "identity": him.whoami(),
        "dialogue_turns": 4,
        "autonomous_work_records": len(rows) - 4,
        "successful_records": sum(row["ok"] for row in rows),
        "usable_records": sum(row["usable"] for row in rows),
        "records_with_language_receipts": sum(bool(row["language_receipt"]) for row in rows),
        "total_records": len(rows),
        "all_records_successful": all_ok,
        "all_records_usable": all_usable,
        "chain_valid": chain_valid,
        "receipt_head": previous_hash,
        "receipt_signature_hmac_sha256": signature,
        "signer_id": signer_id if signature else "",
        "external_custody_claim": bool(signature),
        "lexicon": him.lexicon.manifest(),
        "lexical_snapshot": lexical_receipt,
        "weight_update_performed": False,
        "durable_cross_session_memory_tested": False,
        "promotion_observation": args.promotion_observation,
        "promotion_eligible": promotion_eligible,
        "claim_boundary": "This run measures one bounded trajectory. It does not prove durable cross-session memory, trained weight improvement, or checkpoint promotion.",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown = ["# HIM Language Maturation Transcript", "", f"Checkpoint: `{checkpoint_source}`", f"Checkpoint class: `{checkpoint_class}`", f"Session: `{summary['identity'].get('session_id')}`", ""]
    for row in rows:
        markdown += [f"## {row['sequence']}. {row['kind']}: {row['phase']}", "", f"**Instruction:** {row['instruction']}", "", f"**HIM:** {row['output'] or '[no output]'}", "", f"**Method:** `{row['method']}`", f"**Receipt:** `{row['hash']}`", ""]
    (root / "TRANSCRIPT.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if all_ok and all_usable and all_language_receipts and chain_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
