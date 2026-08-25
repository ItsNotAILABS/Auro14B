#!/usr/bin/env python3
"""Audit AURO checkpoint custody without trusting directory names.

The inventory separates artifact presence, cryptographic integrity, training
provenance, exact-checkpoint evaluation, signed promotion, and release-group
readiness. Architecture files and plausible directory names are never accepted
as checkpoint evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

CONSTITUTIONAL_MANIFEST = "constitutional_manifest.json"
LEGACY_MANIFEST_NAMES = ("manifest.json", "checkpoint_manifest.json")
WEIGHT_PATTERNS = ("*.pt", "*.pth", "*.safetensors", "*.npz", "*.npz.b64", "*.bin")
TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
)
EVALUATION_NAMES = (
    "evaluation.json",
    "benchmark_results.json",
    "PRO_EVALUATION_REPORT.json",
    "conversation-gate.json",
    "checkpoint-evaluation.json",
)
TRAINING_RECEIPT_NAMES = (
    "TRAINING_EXECUTION_RECEIPT.json",
    "training_receipt.json",
    "train_report.json",
    "HIM_SFT_REPORT.json",
)
NORMALIZED_GEOMETRY_KEYS = (
    "hidden_dim",
    "num_layers",
    "num_heads",
    "num_kv_heads",
    "ffn_dim",
    "vocab_size",
    "max_seq_len",
)
THROUGH_2B_LANES = ("Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B")
TRIAD_VARIANTS = (
    "Auro-500M-SENSUS",
    "Auro-500M-PRAXIS",
    "Auro-500M-VERBUM",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"parse_error": "root must be an object"}
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def _first_value(payloads: Iterable[Mapping[str, Any]], *keys: str) -> Any:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return value
    return None


def _file_inventory(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("files", "artifacts"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _declared_hash(inventory: Mapping[str, Any], relative: str, basename: str) -> str:
    value = inventory.get(relative)
    if value is None:
        value = inventory.get(basename)
    if isinstance(value, Mapping):
        value = value.get("sha256")
    return str(value or "")


def _normalise_geometry(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    source: Mapping[str, Any] = config
    for key in ("geometry", "model_config", "config", "architecture"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            source = {**value, **config}
            break
    aliases = {
        "hidden_dim": ("hidden_dim", "hidden_size"),
        "num_layers": ("num_layers", "layers"),
        "num_heads": ("num_heads", "attention_heads"),
        "num_kv_heads": ("num_kv_heads", "kv_heads"),
        "ffn_dim": ("ffn_dim", "intermediate_size"),
        "vocab_size": ("vocab_size", "vocab_size_target"),
        "max_seq_len": ("max_seq_len", "context_window_tokens_target"),
    }
    geometry: dict[str, Any] = {}
    for normalized, candidates in aliases.items():
        value = next((source.get(item) for item in candidates if source.get(item) is not None), None)
        geometry[normalized] = value
    return geometry


def _verify_constitutional_manifest(
    root: Path,
    payload: Mapping[str, Any],
    signing_key: str | None,
) -> dict[str, Any]:
    supplied_hash = str(payload.get("manifest_sha256") or "")
    supplied_signature = str(payload.get("authorization_hmac_sha256") or "")
    unsigned = dict(payload)
    unsigned["manifest_sha256"] = None
    unsigned["authorization_hmac_sha256"] = None
    encoded = canonical(unsigned)
    seal_matches = _is_sha256(supplied_hash) and hmac.compare_digest(
        supplied_hash,
        hashlib.sha256(encoded).hexdigest(),
    )

    file_results = []
    for relative, expected in sorted(_file_inventory(payload).items()):
        path = root / str(relative)
        declared = str(expected.get("sha256") if isinstance(expected, Mapping) else expected or "")
        actual = sha256_file(path) if path.is_file() else ""
        file_results.append(
            {
                "path": str(relative),
                "present": path.is_file(),
                "declared_sha256": declared,
                "actual_sha256": actual,
                "matches": bool(path.is_file() and _is_sha256(declared) and hmac.compare_digest(declared, actual)),
            }
        )
    inventory_verified = bool(file_results) and all(item["matches"] for item in file_results)
    protocols = list(payload.get("protocols") or [])
    protocols_verified = bool(protocols) and all(bool(item.get("passed")) for item in protocols if isinstance(item, Mapping))
    promotion_status = str(payload.get("promotion_status") or "quarantined")
    authorization_present = bool(
        promotion_status == "promoted"
        and payload.get("authorized_by")
        and _is_sha256(supplied_signature)
    )
    authorization_verified = False
    if authorization_present and signing_key:
        expected_signature = hmac.new(signing_key.encode("utf-8"), encoded, hashlib.sha256).hexdigest()
        authorization_verified = hmac.compare_digest(supplied_signature, expected_signature)

    return {
        "kind": "constitutional",
        "seal_matches": seal_matches,
        "inventory_verified": inventory_verified,
        "protocols_verified": protocols_verified,
        "promotion_status": promotion_status,
        "authorization_present": authorization_present,
        "authorization_verified": authorization_verified,
        "file_results": file_results,
        "integrity_verified": bool(seal_matches and inventory_verified),
        "signed_promotion": bool(
            promotion_status == "promoted"
            and protocols_verified
            and authorization_verified
        ),
    }


def _verify_legacy_manifest(
    root: Path,
    payload: Mapping[str, Any],
    signing_key: str | None,
) -> dict[str, Any]:
    inventory = _file_inventory(payload)
    file_results = []
    for relative, expected in sorted(inventory.items()):
        path = root / str(relative)
        declared = str(expected.get("sha256") if isinstance(expected, Mapping) else expected or "")
        actual = sha256_file(path) if path.is_file() else ""
        file_results.append(
            {
                "path": str(relative),
                "present": path.is_file(),
                "declared_sha256": declared,
                "actual_sha256": actual,
                "matches": bool(path.is_file() and _is_sha256(declared) and hmac.compare_digest(declared, actual)),
            }
        )
    inventory_verified = bool(file_results) and all(item["matches"] for item in file_results)
    signature = str(payload.get("signature") or payload.get("manifest_signature") or "")
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    unsigned.pop("manifest_signature", None)
    authorization_present = bool(
        str(payload.get("promotion_status") or "") == "promoted"
        and payload.get("authorized_by")
        and _is_sha256(signature)
    )
    authorization_verified = False
    if authorization_present and signing_key:
        expected = hmac.new(signing_key.encode("utf-8"), canonical(unsigned), hashlib.sha256).hexdigest()
        authorization_verified = hmac.compare_digest(signature, expected)
    return {
        "kind": "legacy",
        "inventory_verified": inventory_verified,
        "promotion_status": str(payload.get("promotion_status") or "unverified"),
        "authorization_present": authorization_present,
        "authorization_verified": authorization_verified,
        "file_results": file_results,
        "integrity_verified": inventory_verified,
        "signed_promotion": bool(authorization_verified),
    }


def _evaluation_passed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("all_passed")
        or payload.get("promotion_ready")
        or payload.get("passed") is True
        or payload.get("gate_passed") is True
    )


def _pins_checkpoint(
    payload: Mapping[str, Any],
    checkpoint_id: str,
    manifest_sha256: str,
) -> bool:
    return bool(
        (checkpoint_id and str(payload.get("checkpoint_id") or "") == checkpoint_id)
        or (
            manifest_sha256
            and str(payload.get("checkpoint_manifest_sha256") or "") == manifest_sha256
        )
    )


def _verify_training_receipt(payload: Mapping[str, Any], checkpoint_id: str) -> bool:
    supplied = str(payload.get("receipt_sha256") or "")
    if supplied:
        unsigned = dict(payload)
        unsigned.pop("receipt_sha256", None)
        if not _is_sha256(supplied) or not hmac.compare_digest(
            supplied,
            hashlib.sha256(canonical(unsigned)).hexdigest(),
        ):
            return False
    elif not payload.get("ok"):
        return False
    pinned = str(payload.get("checkpoint_id") or "")
    output = str(payload.get("output_checkpoint") or payload.get("checkpoint") or "")
    return bool((checkpoint_id and pinned == checkpoint_id) or (checkpoint_id and Path(output).name == checkpoint_id))


def _content_hash(items: Iterable[Mapping[str, Any]]) -> str:
    material = [
        {
            "path": item.get("relative") or item.get("name"),
            "sha256": item.get("sha256"),
        }
        for item in items
    ]
    return hashlib.sha256(canonical(material)).hexdigest()


def inspect_checkpoint(path: Path, promotion_signing_key: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    config_path = path / "config.json"
    meta_path = path / "meta.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    meta = _read_json(meta_path) if meta_path.is_file() else {}

    manifests: list[dict[str, Any]] = []
    constitutional_payload: Mapping[str, Any] = {}
    constitutional_path = path / CONSTITUTIONAL_MANIFEST
    if constitutional_path.is_file():
        constitutional_payload = _read_json(constitutional_path)
        manifests.append(
            {
                "path": str(constitutional_path),
                "sha256": sha256_file(constitutional_path),
                "payload": constitutional_payload,
            }
        )
    legacy_payloads: list[Mapping[str, Any]] = []
    for name in LEGACY_MANIFEST_NAMES:
        candidate = path / name
        if candidate.is_file():
            payload = _read_json(candidate)
            legacy_payloads.append(payload)
            manifests.append({"path": str(candidate), "sha256": sha256_file(candidate), "payload": payload})

    manifest_payload: Mapping[str, Any] = constitutional_payload or (legacy_payloads[0] if legacy_payloads else {})
    payloads = [meta, config, constitutional_payload, *legacy_payloads]

    artifacts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for pattern in (*WEIGHT_PATTERNS, *TOKENIZER_PATTERNS):
        for candidate in sorted(path.rglob(pattern)):
            if candidate.is_file() and candidate not in seen_paths:
                seen_paths.add(candidate)
                artifacts.append(
                    {
                        "path": str(candidate),
                        "relative": str(candidate.relative_to(path)).replace("\\", "/"),
                        "name": candidate.name,
                        "bytes": candidate.stat().st_size,
                        "sha256": sha256_file(candidate),
                    }
                )
    weight_files = [item for item in artifacts if any(Path(item["name"]).match(pattern) for pattern in WEIGHT_PATTERNS)]
    tokenizer_files = [item for item in artifacts if item["name"] in TOKENIZER_PATTERNS]

    evaluations = []
    for name in EVALUATION_NAMES:
        candidate = path / name
        if candidate.is_file():
            payload = _read_json(candidate)
            evaluations.append({"path": str(candidate), "sha256": sha256_file(candidate), "payload": payload})

    training_receipts = []
    for name in TRAINING_RECEIPT_NAMES:
        candidate = path / name
        if candidate.is_file():
            payload = _read_json(candidate)
            training_receipts.append({"path": str(candidate), "sha256": sha256_file(candidate), "payload": payload})

    model_id = str(_first_value(payloads, "model_id", "variant_id") or "")
    checkpoint_id = str(_first_value(payloads, "checkpoint_id") or "")
    parameter_count = _first_value(payloads, "num_params", "parameter_count", "measured_parameters")
    parameter_target = _first_value(payloads, "parameter_target")
    geometry = _normalise_geometry(config, manifest_payload)
    geometry_missing = [
        key
        for key in NORMALIZED_GEOMETRY_KEYS
        if not isinstance(geometry.get(key), int) or int(geometry.get(key, 0)) <= 0
    ]

    verification: dict[str, Any]
    if constitutional_payload:
        verification = _verify_constitutional_manifest(path, constitutional_payload, promotion_signing_key)
        manifest_sha256 = str(constitutional_payload.get("manifest_sha256") or "")
    elif legacy_payloads:
        verification = _verify_legacy_manifest(path, legacy_payloads[0], promotion_signing_key)
        manifest_sha256 = manifests[0]["sha256"]
    else:
        verification = {
            "kind": "missing",
            "integrity_verified": False,
            "signed_promotion": False,
            "authorization_present": False,
            "authorization_verified": False,
            "file_results": [],
        }
        manifest_sha256 = ""

    declared_files = _file_inventory(manifest_payload)
    hash_checks = []
    for item in artifacts:
        declared = _declared_hash(declared_files, str(item["relative"]), str(item["name"]))
        hash_checks.append(
            {
                "name": item["name"],
                "relative": item["relative"],
                "declared_sha256": declared,
                "actual_sha256": item["sha256"],
                "matches": bool(declared and _is_sha256(declared) and hmac.compare_digest(declared, item["sha256"])),
            }
        )

    artifact_present = bool(weight_files)
    manifest_present = bool(manifests)
    tokenizer_custody = bool(tokenizer_files) and all(
        item["matches"] for item in hash_checks if item["relative"] in {x["relative"] for x in tokenizer_files}
    )
    weight_hash_agreement = bool(weight_files) and all(
        item["matches"] for item in hash_checks if item["relative"] in {x["relative"] for x in weight_files}
    )
    geometry_verified = bool(
        not geometry_missing
        and model_id
        and isinstance(parameter_count, int)
        and parameter_count > 0
        and isinstance(parameter_target, int)
        and parameter_target > 0
    )
    evaluation_verified = any(
        _evaluation_passed(item["payload"])
        and _pins_checkpoint(item["payload"], checkpoint_id, manifest_sha256)
        for item in evaluations
    )
    training_provenance_verified = any(
        _verify_training_receipt(item["payload"], checkpoint_id)
        for item in training_receipts
    )
    integrity_verified = bool(
        manifest_present
        and artifact_present
        and tokenizer_custody
        and weight_hash_agreement
        and geometry_verified
        and verification.get("integrity_verified")
    )
    signed_promotion = bool(verification.get("signed_promotion"))
    promotion_ready = bool(
        integrity_verified
        and training_provenance_verified
        and evaluation_verified
        and signed_promotion
    )

    blockers: list[str] = []
    if not artifact_present:
        blockers.append("no weight artifact")
    if not manifest_present:
        blockers.append("no canonical checkpoint manifest")
    if not tokenizer_files:
        blockers.append("no tokenizer custody artifact")
    if not weight_hash_agreement:
        blockers.append("manifest-to-weight hash agreement missing or failed")
    if tokenizer_files and not tokenizer_custody:
        blockers.append("manifest-to-tokenizer hash agreement missing or failed")
    if not verification.get("integrity_verified"):
        blockers.append("checkpoint manifest seal or file inventory verification failed")
    if geometry_missing:
        blockers.append(f"model geometry incomplete: {', '.join(geometry_missing)}")
    if not model_id:
        blockers.append("model_id missing from checkpoint metadata")
    if not checkpoint_id:
        blockers.append("checkpoint_id missing from checkpoint metadata")
    if not isinstance(parameter_count, int) or parameter_count <= 0:
        blockers.append("verified measured parameter count missing")
    if not isinstance(parameter_target, int) or parameter_target <= 0:
        blockers.append("parameter_target missing")
    if not training_provenance_verified:
        blockers.append("hash-valid training provenance pinned to this checkpoint is missing")
    if not evaluation_verified:
        blockers.append("passing exact-checkpoint evaluation pinned to this checkpoint is missing")
    if verification.get("authorization_present") and not verification.get("authorization_verified"):
        blockers.append("promotion signature is present but no matching verification key was supplied")
    if not signed_promotion:
        blockers.append("cryptographically verified constitutional promotion authorization missing")

    checkpoint_content_sha256 = _content_hash([*weight_files, *tokenizer_files]) if artifacts else ""
    return {
        "schema": "auro.checkpoint.audit.v3",
        "path": str(path),
        "name": path.name,
        "model_id": model_id or None,
        "checkpoint_id": checkpoint_id or None,
        "parameter_count": parameter_count,
        "parameter_target": parameter_target,
        "geometry": geometry,
        "geometry_missing": geometry_missing,
        "manifests": manifests,
        "manifest_verification": verification,
        "manifest_sha256": manifest_sha256 or None,
        "checkpoint_content_sha256": checkpoint_content_sha256 or None,
        "weight_files": weight_files,
        "tokenizer_files": tokenizer_files,
        "evaluations": evaluations,
        "training_receipts": training_receipts,
        "hash_checks": hash_checks,
        "artifact_present": artifact_present,
        "manifest_present": manifest_present,
        "tokenizer_custody": tokenizer_custody,
        "weight_hash_agreement": weight_hash_agreement,
        "geometry_verified": geometry_verified,
        "integrity_verified": integrity_verified,
        "training_provenance_verified": training_provenance_verified,
        "evaluation_verified": evaluation_verified,
        "promotion_signature_present": bool(verification.get("authorization_present")),
        "promotion_signature_verified": bool(verification.get("authorization_verified")),
        "signed_promotion": signed_promotion,
        "promotion_ready": promotion_ready,
        "evidence_complete": promotion_ready,
        "blockers": blockers,
    }


def _candidate_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates: set[Path] = set()
    for name in (CONSTITUTIONAL_MANIFEST, *LEGACY_MANIFEST_NAMES):
        for manifest in root.rglob(name):
            candidates.add(manifest.parent)
    for directory in root.iterdir():
        if directory.is_dir() and any(directory.glob(pattern) for pattern in WEIGHT_PATTERNS):
            candidates.add(directory)
    return sorted(candidates)


def _lane_summary(checkpoints: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    matches = [item for item in checkpoints if str(item.get("model_id") or "").lower() == model_id.lower()]
    best = next((item for item in matches if item["promotion_ready"]), None)
    if best is None:
        best = next((item for item in matches if item["integrity_verified"]), None)
    if best is None and matches:
        best = matches[0]
    return {
        "model_id": model_id,
        "candidate_count": len(matches),
        "artifact_present": any(item["artifact_present"] for item in matches),
        "integrity_verified": any(item["integrity_verified"] for item in matches),
        "training_provenance_verified": any(item["training_provenance_verified"] for item in matches),
        "evaluation_verified": any(item["evaluation_verified"] for item in matches),
        "promotion_ready": any(item["promotion_ready"] for item in matches),
        "best_candidate": best["path"] if best else None,
        "blockers": list(best["blockers"] if best else ["no checkpoint candidate"]),
    }


def inventory(root: Path, promotion_signing_key: str | None = None) -> dict[str, Any]:
    checkpoints = [
        inspect_checkpoint(path, promotion_signing_key=promotion_signing_key)
        for path in _candidate_directories(root)
    ]
    through_2b = {model_id: _lane_summary(checkpoints, model_id) for model_id in THROUGH_2B_LANES}
    triad = {model_id: _lane_summary(checkpoints, model_id) for model_id in TRIAD_VARIANTS}
    auro_2b = through_2b["Auro-2B"]
    return {
        "schema": "auro.checkpoint.inventory.v3",
        "root": str(root),
        "checkpoints": checkpoints,
        "through_2b": through_2b,
        "triad": triad,
        "through_2b_promotion_ready": all(item["promotion_ready"] for item in through_2b.values()),
        "triad_promotion_ready": all(item["promotion_ready"] for item in triad.values()),
        "ship_through_2b_ready": bool(
            all(item["promotion_ready"] for item in through_2b.values())
            and all(item["promotion_ready"] for item in triad.values())
        ),
        "auro_2b_candidates": [item for item in checkpoints if str(item.get("model_id") or "").lower() == "auro-2b"],
        "auro_2b_artifact_present": auro_2b["artifact_present"],
        "auro_2b_integrity_verified": auro_2b["integrity_verified"],
        "auro_2b_promotion_ready": auro_2b["promotion_ready"],
        "auro_2b_evidence_complete": auro_2b["promotion_ready"],
        "promotion_signature_verification_key_supplied": bool(promotion_signing_key),
        "claim_boundary": (
            "directory names and architecture configs are not checkpoint evidence; "
            "shipping through 2B requires hash-bound weights and tokenizer custody, "
            "verified geometry and measured parameter accounting, checkpoint-pinned "
            "training and evaluation evidence, and cryptographically verified promotion "
            "for 156K, 250M, 500M, the three 500M specialist variants, and 2B"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AURO checkpoint custody and release readiness")
    parser.add_argument("--root", default="checkpoints/auro_minds")
    parser.add_argument("--output")
    parser.add_argument("--require-promotion", action="store_true", help="Legacy alias: require Auro-2B promotion")
    parser.add_argument("--require-through-2b", action="store_true")
    parser.add_argument("--require-triad", action="store_true")
    parser.add_argument("--require-ship-through-2b", action="store_true")
    parser.add_argument("--promotion-signing-key-env", default="AURO_CHECKPOINT_SIGNING_KEY")
    args = parser.parse_args()
    signing_key = os.getenv(args.promotion_signing_key_env) or None
    report = inventory(Path(args.root), promotion_signing_key=signing_key)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.require_promotion and not report["auro_2b_promotion_ready"]:
        return 3
    if args.require_through_2b and not report["through_2b_promotion_ready"]:
        return 4
    if args.require_triad and not report["triad_promotion_ready"]:
        return 5
    if args.require_ship_through_2b and not report["ship_through_2b_ready"]:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
