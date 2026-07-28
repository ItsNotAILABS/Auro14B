#!/usr/bin/env python3
"""Audit local AURO checkpoint custody without trusting directory names.

Presence, integrity, evaluation, and promotion are separate states. A directory
containing a JSON file and arbitrary bytes is never considered complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

MANIFEST_NAMES = ("manifest.json", "checkpoint_manifest.json", "constitutional_manifest.json")
WEIGHT_PATTERNS = ("*.pt", "*.pth", "*.safetensors", "*.npz", "*.npz.b64", "*.bin")
TOKENIZER_PATTERNS = ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "tokenizer_config.json")
EVALUATION_NAMES = ("evaluation.json", "benchmark_results.json", "PRO_EVALUATION_REPORT.json", "HIM_SFT_REPORT.json")
GEOMETRY_KEYS = ("hidden_dim", "num_layers", "num_heads", "vocab_size", "max_seq_len")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"parse_error": "root must be an object"}
    except Exception as exc:
        return {"parse_error": str(exc)}


def _first_mapping(payloads: list[Mapping[str, Any]], keys: tuple[str, ...]) -> Mapping[str, Any]:
    for payload in payloads:
        current: Any = payload
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, Mapping):
            return current
    return {}


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    manifests = []
    payloads: list[Mapping[str, Any]] = []
    for name in MANIFEST_NAMES:
        candidate = path / name
        if candidate.is_file():
            payload = _read_json(candidate)
            payloads.append(payload)
            manifests.append({"path": str(candidate), "sha256": sha256_file(candidate), "payload": payload})

    weight_files = []
    for pattern in WEIGHT_PATTERNS:
        for candidate in sorted(path.glob(pattern)):
            weight_files.append({"path": str(candidate), "name": candidate.name, "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)})

    tokenizer_files = []
    for name in TOKENIZER_PATTERNS:
        candidate = path / name
        if candidate.is_file():
            tokenizer_files.append({"path": str(candidate), "name": candidate.name, "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)})

    evaluations = []
    for name in EVALUATION_NAMES:
        candidate = path / name
        if candidate.is_file():
            payload = _read_json(candidate)
            evaluations.append({"path": str(candidate), "sha256": sha256_file(candidate), "payload": payload})

    declared_files = _first_mapping(payloads, ("files",)) or _first_mapping(payloads, ("artifacts",))
    hash_checks = []
    for item in weight_files + tokenizer_files:
        declared = declared_files.get(item["name"]) if isinstance(declared_files, Mapping) else None
        if isinstance(declared, Mapping):
            declared = declared.get("sha256")
        hash_checks.append({
            "name": item["name"],
            "declared_sha256": str(declared or ""),
            "actual_sha256": item["sha256"],
            "matches": bool(declared) and str(declared) == item["sha256"],
        })

    geometry: Mapping[str, Any] = {}
    for payload in payloads:
        for key in ("geometry", "model_config", "config"):
            candidate = payload.get(key) if isinstance(payload, Mapping) else None
            if isinstance(candidate, Mapping):
                geometry = candidate
                break
        if geometry:
            break
    geometry_missing = [key for key in GEOMETRY_KEYS if not isinstance(geometry.get(key), int) or int(geometry.get(key, 0)) <= 0]
    parameter_count = next((payload.get("parameter_count") for payload in payloads if isinstance(payload.get("parameter_count"), int)), None)
    model_id = next((payload.get("model_id") for payload in payloads if payload.get("model_id")), None)

    promotion = next((payload for payload in payloads if payload.get("promotion_status") or payload.get("authorized_by")), {})
    promotion_status = str(promotion.get("promotion_status") or "unverified")
    authorized_by = str(promotion.get("authorized_by") or "")
    signature = str(promotion.get("signature") or promotion.get("manifest_signature") or "")
    signed_promotion = promotion_status == "promoted" and bool(authorized_by) and len(signature) >= 64

    artifact_present = bool(weight_files)
    manifest_present = bool(manifests)
    tokenizer_custody = bool(tokenizer_files) and all(check["matches"] for check in hash_checks if check["name"] in {x["name"] for x in tokenizer_files})
    weight_hash_agreement = bool(weight_files) and all(check["matches"] for check in hash_checks if check["name"] in {x["name"] for x in weight_files})
    geometry_verified = not geometry_missing and bool(model_id) and isinstance(parameter_count, int) and parameter_count > 0
    evaluation_verified = bool(evaluations) and any(
        bool(item["payload"].get("all_passed") or item["payload"].get("ok") or item["payload"].get("promotion_ready"))
        for item in evaluations
    )
    integrity_verified = manifest_present and artifact_present and tokenizer_custody and weight_hash_agreement and geometry_verified
    promotion_ready = integrity_verified and evaluation_verified and signed_promotion

    blockers = []
    if not artifact_present: blockers.append("no weight artifact")
    if not manifest_present: blockers.append("no canonical checkpoint manifest")
    if not tokenizer_files: blockers.append("no tokenizer custody artifact")
    if not weight_hash_agreement: blockers.append("manifest-to-weight hash agreement missing or failed")
    if tokenizer_files and not tokenizer_custody: blockers.append("manifest-to-tokenizer hash agreement missing or failed")
    if geometry_missing: blockers.append(f"model geometry incomplete: {', '.join(geometry_missing)}")
    if not model_id: blockers.append("model_id missing from manifest")
    if not isinstance(parameter_count, int) or parameter_count <= 0: blockers.append("verified parameter_count missing")
    if not evaluation_verified: blockers.append("passing exact-checkpoint evaluation receipt missing")
    if not signed_promotion: blockers.append("signed constitutional promotion authorization missing")

    return {
        "schema": "auro.checkpoint.audit.v2",
        "path": str(path),
        "name": path.name,
        "model_id": model_id,
        "parameter_count": parameter_count,
        "geometry": dict(geometry),
        "geometry_missing": geometry_missing,
        "manifests": manifests,
        "weight_files": weight_files,
        "tokenizer_files": tokenizer_files,
        "evaluations": evaluations,
        "hash_checks": hash_checks,
        "artifact_present": artifact_present,
        "manifest_present": manifest_present,
        "integrity_verified": integrity_verified,
        "evaluation_verified": evaluation_verified,
        "signed_promotion": signed_promotion,
        "promotion_ready": promotion_ready,
        "evidence_complete": promotion_ready,
        "blockers": blockers,
    }


def inventory(root: Path) -> dict[str, Any]:
    candidates = []
    if root.exists():
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            if "auro" in path.name.lower() or "him" in path.name.lower():
                candidates.append(inspect_checkpoint(path))
    auro_2b = [item for item in candidates if "2b" in item["name"].lower() or str(item.get("model_id", "")).lower() == "auro-2b"]
    return {
        "schema": "auro.checkpoint.inventory.v2",
        "root": str(root),
        "checkpoints": candidates,
        "auro_2b_candidates": auro_2b,
        "auro_2b_artifact_present": any(item["artifact_present"] for item in auro_2b),
        "auro_2b_integrity_verified": any(item["integrity_verified"] for item in auro_2b),
        "auro_2b_promotion_ready": any(item["promotion_ready"] for item in auro_2b),
        "auro_2b_evidence_complete": any(item["evidence_complete"] for item in auro_2b),
        "claim_boundary": "directory names and architecture configs are not checkpoint evidence; completeness requires hash-bound weights and tokenizer, verified geometry and parameter count, exact-checkpoint evaluation, and signed constitutional promotion",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/auro_minds")
    parser.add_argument("--output")
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    report = inventory(Path(args.root))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not args.require_promotion or report["auro_2b_promotion_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
