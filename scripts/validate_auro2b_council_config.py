#!/usr/bin/env python3
"""Validate Auro-2B council configuration without contacting model endpoints.

The validator distinguishes a structurally valid wiring template from an
identity-complete release configuration. It never interprets endpoint names or
placeholder hashes as checkpoint evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auro_native_llm.production_fleet.council_service import (  # noqa: E402
    CONFIG_SCHEMA,
    EndpointConfig,
)

EXPECTED_SPECIALISTS = (
    "Auro-500M-SENSUS",
    "Auro-500M-PRAXIS",
    "Auro-500M-VERBUM",
)
EXPECTED_ATOMICS = ("Auro-156K", "Auro-250M")
EXPECTED_TARGETS = {
    "Auro-156K": 156_000,
    "Auro-250M": 250_000_000,
    "Auro-500M-SENSUS": 500_000_000,
    "Auro-500M-PRAXIS": 500_000_000,
    "Auro-500M-VERBUM": 500_000_000,
    "Auro-2B": 2_000_000_000,
}
PLACEHOLDER_MARKERS = (
    "replace-with",
    "placeholder",
    "todo",
    "example",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("council configuration must contain a JSON object")
    return value


def _canonical_sha(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: str | None) -> bool:
    if not value or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_placeholder(value: str | None) -> bool:
    lower = str(value or "").strip().lower()
    return not lower or any(marker in lower for marker in PLACEHOLDER_MARKERS)


def validate(value: Mapping[str, Any], *, require_evidence: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    endpoints: list[EndpointConfig] = []

    if value.get("schema") != CONFIG_SCHEMA:
        errors.append(f"schema must be {CONFIG_SCHEMA}")

    main_raw = value.get("main")
    specialist_raw = value.get("specialists")
    atomic_raw = value.get("atomic")
    if not isinstance(main_raw, Mapping):
        errors.append("main must be an object")
    if not isinstance(specialist_raw, list):
        errors.append("specialists must be an array")
        specialist_raw = []
    if not isinstance(atomic_raw, list):
        errors.append("atomic must be an array")
        atomic_raw = []

    sections: list[Mapping[str, Any]] = []
    if isinstance(main_raw, Mapping):
        sections.append(main_raw)
    sections.extend(item for item in specialist_raw if isinstance(item, Mapping))
    sections.extend(item for item in atomic_raw if isinstance(item, Mapping))

    for index, item in enumerate(sections):
        try:
            endpoints.append(EndpointConfig.from_mapping(item))
        except (TypeError, ValueError) as exc:
            errors.append(f"endpoint[{index}] invalid: {exc}")

    identities = [item.model_id for item in endpoints]
    if identities.count("Auro-2B") != 1:
        errors.append("configuration must contain exactly one Auro-2B main endpoint")
    specialists = tuple(
        str(item.get("model_id") or "")
        for item in specialist_raw
        if isinstance(item, Mapping)
    )
    if specialists != EXPECTED_SPECIALISTS:
        errors.append(
            f"specialists must be ordered exactly as {list(EXPECTED_SPECIALISTS)}; got {list(specialists)}"
        )
    atomics = tuple(
        str(item.get("model_id") or "")
        for item in atomic_raw
        if isinstance(item, Mapping)
    )
    if atomics != EXPECTED_ATOMICS:
        errors.append(
            f"atomic endpoints must be ordered exactly as {list(EXPECTED_ATOMICS)}; got {list(atomics)}"
        )
    if len(identities) != len(set(identities)):
        errors.append("model identities must be unique")

    for endpoint in endpoints:
        expected_target = EXPECTED_TARGETS.get(endpoint.model_id)
        if expected_target is None:
            errors.append(f"unsupported council identity: {endpoint.model_id}")
        elif endpoint.parameter_target != expected_target:
            errors.append(
                f"{endpoint.model_id} parameter_target {endpoint.parameter_target} != {expected_target}"
            )

        checkpoint_placeholder = _is_placeholder(endpoint.checkpoint_id)
        hash_placeholder = _is_placeholder(endpoint.checkpoint_sha256)
        checkpoint_hash_valid = _is_sha256(endpoint.checkpoint_sha256)
        if checkpoint_placeholder or hash_placeholder or not checkpoint_hash_valid:
            message = f"{endpoint.model_id} lacks a non-placeholder 64-character checkpoint SHA-256"
            (errors if require_evidence else warnings).append(message)

        if endpoint.model_id in EXPECTED_SPECIALISTS:
            adapter_missing = (
                _is_placeholder(endpoint.adapter_id)
                or _is_placeholder(endpoint.adapter_sha256)
                or not _is_sha256(endpoint.adapter_sha256)
            )
            if adapter_missing:
                message = f"{endpoint.model_id} lacks distinct adapter or checkpoint specialization evidence"
                (errors if require_evidence else warnings).append(message)

    specialist_tokens = []
    for endpoint in endpoints:
        if endpoint.model_id not in EXPECTED_SPECIALISTS:
            continue
        token = (
            f"adapter:{endpoint.adapter_sha256}"
            if _is_sha256(endpoint.adapter_sha256)
            and not _is_placeholder(endpoint.adapter_sha256)
            else f"checkpoint:{endpoint.checkpoint_sha256}"
            if _is_sha256(endpoint.checkpoint_sha256)
            and not _is_placeholder(endpoint.checkpoint_sha256)
            else None
        )
        specialist_tokens.append(token)
    if require_evidence and (
        any(token is None for token in specialist_tokens)
        or len(set(specialist_tokens)) != len(EXPECTED_SPECIALISTS)
    ):
        errors.append("three distinct specialist checkpoint or adapter proofs are required")

    max_workers = int(value.get("max_workers", 12))
    if not 3 <= max_workers <= 32:
        errors.append("max_workers must be between 3 and 32")

    identity_complete = not any(
        "lacks" in item or "distinct specialist" in item for item in errors + warnings
    )
    result = {
        "schema": "auro.2b-council.config-validation.v1",
        "config_schema": value.get("schema"),
        "config_sha256": _canonical_sha(value),
        "endpoint_count": len(endpoints),
        "identities": identities,
        "topology_valid": not errors,
        "identity_evidence_complete": identity_complete and not errors,
        "require_evidence": require_evidence,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "status": "pass" if not errors else "fail",
        "truth_boundary": (
            "A passing topology check proves configuration shape only. Endpoint health, exact checkpoint "
            "custody, tokenizer integrity, model quality, latency, and promotion require separate evidence."
        ),
    }
    result["receipt_sha256"] = _canonical_sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default=str(ROOT / "native_llm" / "configs" / "auro_2b_council.example.json"),
    )
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="Fail when exact checkpoint or specialist adapter evidence remains missing or placeholder.",
    )
    parser.add_argument("--output", help="Optional path for the JSON validation receipt.")
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    try:
        result = validate(_load(path), require_evidence=args.require_evidence)
    except Exception as exc:
        result = {
            "schema": "auro.2b-council.config-validation.v1",
            "status": "fail",
            "errors": [str(exc)],
            "warnings": [],
            "config": str(path),
        }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
