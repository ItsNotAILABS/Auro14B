#!/usr/bin/env python3
"""Build the evidence-driven AURO release train through Auro-2B."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auro_native_llm.release_train import build_release_train
from scripts.inventory_auro_checkpoints import inventory


def _validate_json_artifact(path: Path | None, label: str) -> list[str]:
    if path is None:
        return [f"{label} not supplied"]
    if not path.is_file():
        return [f"{label} not found: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{label} is not valid JSON: {type(exc).__name__}: {str(exc)[:200]}"]
    if not isinstance(payload, dict):
        return [f"{label} root must be an object"]
    blockers = []
    if not payload.get("schema"):
        blockers.append(f"{label} has no schema")
    if not payload.get("sha256") and not payload.get("manifest_sha256") and not payload.get("content_sha256"):
        blockers.append(f"{label} has no content or manifest hash")
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an AURO 156K/250M/500M/triad/2B release-train plan"
    )
    parser.add_argument("--checkpoint-root", default="checkpoints/auro_minds")
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--candidate-output-root", default="checkpoints/auro_release_candidates")
    parser.add_argument("--promotion-signing-key-env", default="AURO_CHECKPOINT_SIGNING_KEY")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sub2b-release/release-train.json"))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    signing_key = os.getenv(args.promotion_signing_key_env) or None
    checkpoint_inventory = inventory(
        Path(args.checkpoint_root),
        promotion_signing_key=signing_key,
    )
    plan = build_release_train(
        checkpoint_inventory,
        corpus_manifest=args.corpus_manifest,
        tokenizer_manifest=args.tokenizer_manifest,
        output_root=args.candidate_output_root,
    )
    input_blockers = [
        *_validate_json_artifact(args.corpus_manifest, "corpus manifest"),
        *_validate_json_artifact(args.tokenizer_manifest, "tokenizer manifest"),
    ]
    plan["input_validation_blockers"] = input_blockers
    plan["operator_ready"] = bool(not input_blockers and plan["release_ready"])

    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.require_ready and not plan["operator_ready"]:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
