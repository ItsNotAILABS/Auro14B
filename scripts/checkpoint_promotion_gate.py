#!/usr/bin/env python3
"""Exact-checkpoint production promotion gate for AURO/HIM.

Checkpoint integrity is necessary but not sufficient. This gate joins immutable
checkpoint verification with release evidence and the human-deliverable
readiness score. It never promotes a checkpoint from naming alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from auro_native_llm.open_weights import CheckpointIntegrityError, verify_checkpoint
from readiness_score import score_readiness

REQUIRED_EVIDENCE = {
    "tokenizer_audit",
    "corpus_manifest",
    "training_report",
    "official_benchmarks",
    "coding_execution",
    "governed_execution",
    "api_chat_smoke",
    "browser_chat_smoke",
    "clean_install",
    "model_card",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release evidence must be a JSON object")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("release evidence requires an artifacts object")
    return payload


def _verify_evidence_files(evidence_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    base = evidence_path.parent
    artifacts = payload["artifacts"]
    verified: dict[str, Any] = {}
    blockers: list[str] = []
    for name in sorted(REQUIRED_EVIDENCE):
        item = artifacts.get(name)
        if not isinstance(item, dict):
            blockers.append(f"missing evidence artifact: {name}")
            continue
        rel = item.get("path")
        expected = str(item.get("sha256") or "")
        if not isinstance(rel, str) or not rel:
            blockers.append(f"invalid evidence path: {name}")
            continue
        path = (base / rel).resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError:
            blockers.append(f"evidence path escapes manifest directory: {name}")
            continue
        if not path.is_file():
            blockers.append(f"evidence file not found: {name}")
            continue
        actual = _sha256_file(path)
        if len(expected) != 64 or expected != actual:
            blockers.append(f"evidence hash mismatch: {name}")
            continue
        verified[name] = {"path": str(path), "sha256": actual}
    return verified, blockers


def evaluate(checkpoint: Path, evidence_path: Path, threshold: float, runner_key: str | None) -> dict[str, Any]:
    blockers: list[str] = []
    checkpoint_result: dict[str, Any]
    try:
        checkpoint_result = verify_checkpoint(checkpoint, runner_signing_key=runner_key)
    except (CheckpointIntegrityError, OSError, ValueError) as exc:
        checkpoint_result = {"verified": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        blockers.append("checkpoint integrity/provenance verification failed")

    evidence = _load_evidence(evidence_path)
    verified_artifacts, artifact_blockers = _verify_evidence_files(evidence_path, evidence)
    blockers.extend(artifact_blockers)

    readiness_input = evidence.get("readiness")
    if not isinstance(readiness_input, dict):
        readiness_input = {"gates": {}, "unresolved_blockers": ["readiness input missing"]}
    readiness_blockers = list(readiness_input.get("unresolved_blockers") or [])
    readiness_input = dict(readiness_input)
    readiness_input["unresolved_blockers"] = readiness_blockers + blockers
    readiness = score_readiness(readiness_input, threshold)

    expected_checkpoint_manifest = str(evidence.get("checkpoint_manifest_sha256") or "")
    actual_checkpoint_manifest = str(checkpoint_result.get("manifest_sha256") or "")
    if checkpoint_result.get("verified") and expected_checkpoint_manifest:
        if expected_checkpoint_manifest != actual_checkpoint_manifest:
            blockers.append("release evidence references a different checkpoint manifest")
    elif checkpoint_result.get("verified") and not expected_checkpoint_manifest:
        blockers.append("release evidence does not pin checkpoint_manifest_sha256")

    claimed_source_commit = str(evidence.get("source_commit") or "")
    actual_source_commit = str(checkpoint_result.get("source_commit") or "")
    if checkpoint_result.get("verified") and claimed_source_commit and claimed_source_commit != actual_source_commit:
        blockers.append("release evidence source_commit differs from training receipt")

    final_ready = bool(checkpoint_result.get("verified") and not blockers and readiness["ready"])
    result = {
        "schema": "auro.checkpoint-promotion.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_verification": checkpoint_result,
        "evidence_manifest": str(evidence_path),
        "verified_evidence_artifacts": verified_artifacts,
        "threshold": threshold,
        "readiness": readiness,
        "blockers": blockers,
        "promote": final_ready,
        "claim_boundary": {
            "integrity_equals_quality": False,
            "readiness_equals_benchmark_accuracy": False,
            "promotion_requires_exact_checkpoint_evidence": True,
        },
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate production promotion of one exact AURO checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runner-signing-key-env", default="AURO_TRAINING_RECEIPT_HMAC_KEY")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    runner_key = os.getenv(args.runner_signing_key_env) or None
    result = evaluate(args.checkpoint, args.evidence, args.threshold, runner_key)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["promote"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
