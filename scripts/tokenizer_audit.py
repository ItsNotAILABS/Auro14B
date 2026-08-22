#!/usr/bin/env python3
"""Audit an AURO tokenizer manifest for release-critical invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_CONTROL_TOKENS = (
    "<system>", "<user>", "<assistant>", "<tool>", "<receipt>",
    "<spectral>", "<memory>", "<repository>", "<code>", "<test>",
    "<execution>", "<nova>", "<mesie>", "<mathesis>", "<cain>", "<oro>",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    controls = manifest.get("control_tokens") or []
    if not isinstance(controls, list) or not all(isinstance(item, str) for item in controls):
        controls = []
    missing = [token for token in REQUIRED_CONTROL_TOKENS if token not in controls]
    duplicate_controls = sorted({token for token in controls if controls.count(token) > 1})
    unknown_token_absent = manifest.get("unknown_token") is None
    byte_round_trip = manifest.get("byte_round_trip") is True
    ids_stable = len(controls) == len(set(controls))
    ready = byte_round_trip and unknown_token_absent and ids_stable and not missing and not duplicate_controls
    result = {
        "schema": "auro.tokenizer-audit.v1",
        "tokenizer_schema": manifest.get("schema"),
        "vocab_size": manifest.get("vocab_size"),
        "byte_round_trip": byte_round_trip,
        "unknown_token_absent": unknown_token_absent,
        "required_control_tokens": list(REQUIRED_CONTROL_TOKENS),
        "present_control_tokens": controls,
        "missing_control_tokens": missing,
        "duplicate_control_tokens": duplicate_controls,
        "stable_control_ids": ids_stable,
        "ready": ready,
        "claim_boundary": {
            "manifest_audit_proves_corpus_quality": False,
            "manifest_audit_proves_model_quality": False,
        },
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AURO tokenizer manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        parser.error("tokenizer manifest must be a JSON object")
    result = audit_manifest(manifest)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["ready"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
