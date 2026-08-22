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

IMMUTABLE_CONTROL_IDS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<system>": 3,
    "<user>": 4,
    "<assistant>": 5,
    "<tool>": 6,
    "<receipt>": 7,
    "<spectral>": 8,
    "<memory>": 9,
    "<repository>": 10,
    "<code>": 11,
    "<test>": 12,
    "<execution>": 13,
    "<nova>": 14,
    "<mesie>": 15,
    "<mathesis>": 272,
    "<cain>": 273,
    "<oro>": 274,
}


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

    raw_ids = manifest.get("control_token_ids")
    control_ids = raw_ids if isinstance(raw_ids, dict) else {}
    id_mismatches = {
        token: {"expected": expected, "actual": control_ids.get(token)}
        for token, expected in IMMUTABLE_CONTROL_IDS.items()
        if control_ids.get(token) != expected
    }
    byte_offset = manifest.get("byte_offset")
    byte_end = manifest.get("byte_vocab_end_exclusive")
    immutable_byte_range = byte_offset == 16 and byte_end == 272
    unique_ids = len(control_ids.values()) == len(set(control_ids.values())) if control_ids else False
    ids_stable = bool(control_ids) and unique_ids and not id_mismatches and immutable_byte_range
    ready = byte_round_trip and unknown_token_absent and ids_stable and not missing and not duplicate_controls

    result = {
        "schema": "auro.tokenizer-audit.v2",
        "tokenizer_schema": manifest.get("schema"),
        "tokenizer_version": manifest.get("version"),
        "vocab_size": manifest.get("vocab_size"),
        "byte_round_trip": byte_round_trip,
        "unknown_token_absent": unknown_token_absent,
        "byte_offset": byte_offset,
        "byte_vocab_end_exclusive": byte_end,
        "immutable_byte_range": immutable_byte_range,
        "required_control_tokens": list(REQUIRED_CONTROL_TOKENS),
        "present_control_tokens": controls,
        "missing_control_tokens": missing,
        "duplicate_control_tokens": duplicate_controls,
        "control_token_ids": control_ids,
        "immutable_control_ids": IMMUTABLE_CONTROL_IDS,
        "control_id_mismatches": id_mismatches,
        "stable_control_ids": ids_stable,
        "ready": ready,
        "claim_boundary": {
            "manifest_audit_proves_corpus_quality": False,
            "manifest_audit_proves_model_quality": False,
            "tokenizer_v2_changes_legacy_byte_ids": False,
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
