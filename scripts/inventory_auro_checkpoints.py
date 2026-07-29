#!/usr/bin/env python3
"""Audit local AURO checkpoint custody without trusting directory names.

Presence, integrity, evaluation, specialization, and promotion remain separate.
The report now provides a complete ship-through-2B matrix for 156K, 250M,
500M, its three specialist identities, and 2B.
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
EVALUATION_NAMES = ("evaluation.json", "benchmark_results.json", "PRO_EVALUATION_REPORT.json", "HIM_SFT_REPORT.json", "triad_evaluation.json")
GEOMETRY_KEYS = ("hidden_dim", "num_layers", "num_heads", "vocab_size", "max_seq_len")
REQUIRED_THROUGH_2B = ("Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B")
TRIAD_VARIANTS = ("Auro-500M-SENSUS", "Auro-500M-PRAXIS", "Auro-500M-VERBUM")


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


def _valid_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


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
        hash_checks.append({"name": item["name"], "declared_sha256": str(declared or ""), "actual_sha256": item["sha256"], "matches": bool(declared) and str(declared) == item["sha256"]})

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
    signed_promotion = promotion_status == "promoted" and bool(authorized_by) and _valid_sha(signature)

    adapters = []
    for payload in payloads:
        for item in payload.get("adapters", []) if isinstance(payload.get("adapters"), list) else []:
            if isinstance(item, Mapping):
                adapters.append({"adapter_id": item.get("adapter_id"), "model_id": item.get("model_id"), "sha256": item.get("sha256"), "hash_valid": _valid_sha(item.get("sha256")), "evaluation_passed": bool(item.get("evaluation_passed")), "promotion_status": item.get("promotion_status")})

    artifact_present = bool(weight_files)
    manifest_present = bool(manifests)
    tokenizer_names = {item["name"] for item in tokenizer_files}
    weight_names = {item["name"] for item in weight_files}
    tokenizer_custody = bool(tokenizer_files) and all(check["matches"] for check in hash_checks if check["name"] in tokenizer_names)
    weight_hash_agreement = bool(weight_files) and all(check["matches"] for check in hash_checks if check["name"] in weight_names)
    geometry_verified = not geometry_missing and bool(model_id) and isinstance(parameter_count, int) and parameter_count > 0
    evaluation_verified = bool(evaluations) and any(bool(item["payload"].get("all_passed") or item["payload"].get("ok") or item["payload"].get("promotion_ready")) for item in evaluations)
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
        "schema": "auro.checkpoint.audit.v3",
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
        "adapters": adapters,
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


def _status_for(model_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    matching = [item for item in candidates if str(item.get("model_id") or "").lower() == model_id.lower() or model_id.lower().replace("-", "") in item["name"].lower().replace("-", "_").replace("_", "")]
    return {
        "model_id": model_id,
        "candidate_count": len(matching),
        "artifact_present": any(item["artifact_present"] for item in matching),
        "integrity_verified": any(item["integrity_verified"] for item in matching),
        "promotion_ready": any(item["promotion_ready"] for item in matching),
        "candidate_paths": [item["path"] for item in matching],
        "blockers": sorted({blocker for item in matching for blocker in item["blockers"]}) if matching else ["no checkpoint candidate found"],
    }


def inventory(root: Path) -> dict[str, Any]:
    candidates = []
    if root.exists():
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            if "auro" in path.name.lower() or "him" in path.name.lower():
                candidates.append(inspect_checkpoint(path))

    matrix = {model_id: _status_for(model_id, candidates) for model_id in REQUIRED_THROUGH_2B}
    triad_direct = {model_id: _status_for(model_id, candidates) for model_id in TRIAD_VARIANTS}
    base_adapters = [adapter for checkpoint in candidates if str(checkpoint.get("model_id")) == "Auro-500M" for adapter in checkpoint.get("adapters", [])]
    triad_adapter_status = {
        model_id: any(adapter.get("model_id") == model_id and adapter.get("hash_valid") and adapter.get("evaluation_passed") and adapter.get("promotion_status") == "promoted" for adapter in base_adapters)
        for model_id in TRIAD_VARIANTS
    }
    triad_ready = all(triad_direct[model_id]["promotion_ready"] or triad_adapter_status[model_id] for model_id in TRIAD_VARIANTS)
    ship_through_2b_ready = all(item["promotion_ready"] for item in matrix.values())
    auro_2b = [item for item in candidates if "2b" in item["name"].lower() or str(item.get("model_id", "")).lower() == "auro-2b"]
    return {
        "schema": "auro.checkpoint.inventory.v3",
        "root": str(root),
        "checkpoints": candidates,
        "through_2b_release_matrix": matrix,
        "ship_through_2b_ready": ship_through_2b_ready,
        "triad_direct_checkpoints": triad_direct,
        "triad_adapter_status": triad_adapter_status,
        "triad_specialization_ready": triad_ready,
        "auro_2b_candidates": auro_2b,
        "auro_2b_artifact_present": any(item["artifact_present"] for item in auro_2b),
        "auro_2b_integrity_verified": any(item["integrity_verified"] for item in auro_2b),
        "auro_2b_promotion_ready": any(item["promotion_ready"] for item in auro_2b),
        "auro_2b_evidence_complete": any(item["evidence_complete"] for item in auro_2b),
        "claim_boundary": "directory names and architecture configs are not checkpoint evidence; shipping through 2B requires independent promoted bundles for 156K, 250M, 500M and 2B, while triad specialization additionally requires three promoted checkpoints or adapters",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="checkpoints/auro_minds")
    parser.add_argument("--output")
    parser.add_argument("--require-promotion", action="store_true")
    parser.add_argument("--require-through-2b", action="store_true")
    parser.add_argument("--require-triad", action="store_true")
    args = parser.parse_args()
    report = inventory(Path(args.root))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.require_promotion and not report["auro_2b_promotion_ready"]:
        return 3
    if args.require_through_2b and not report["ship_through_2b_ready"]:
        return 4
    if args.require_triad and not report["triad_specialization_ready"]:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
