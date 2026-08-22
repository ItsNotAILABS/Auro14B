#!/usr/bin/env python3
"""Canonical AURO human-deliverable readiness scorer.

This score is a release-readiness measure, not benchmark accuracy or an
intelligence percentage. Critical gates must individually meet the threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = {
    "checkpoint_integrity": 0.14,
    "training_provenance": 0.10,
    "tokenizer_integrity": 0.10,
    "corpus_provenance": 0.08,
    "official_benchmarks": 0.13,
    "coding_execution": 0.08,
    "governed_execution": 0.10,
    "api_chat_smoke": 0.07,
    "browser_chat_smoke": 0.05,
    "portability": 0.06,
    "clean_install": 0.06,
    "model_card_claims": 0.03,
}

CRITICAL_GATES = {
    "checkpoint_integrity",
    "training_provenance",
    "tokenizer_integrity",
    "corpus_provenance",
    "official_benchmarks",
    "governed_execution",
    "api_chat_smoke",
    "clean_install",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def score_readiness(payload: dict[str, Any], threshold: float = 0.85) -> dict[str, Any]:
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("readiness input requires a gates object")
    weights = dict(DEFAULT_WEIGHTS)
    supplied_weights = payload.get("weights")
    if supplied_weights is not None:
        if not isinstance(supplied_weights, dict):
            raise ValueError("weights must be an object")
        for name, value in supplied_weights.items():
            if name not in weights:
                raise ValueError(f"unknown readiness weight: {name}")
            numeric = float(value)
            if numeric < 0:
                raise ValueError(f"negative readiness weight: {name}")
            weights[name] = numeric
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("readiness weights must sum to a positive value")

    normalized: dict[str, float] = {}
    details: dict[str, Any] = {}
    for name in weights:
        item = gates.get(name, 0.0)
        if isinstance(item, dict):
            value = _clamp(item.get("score", 0.0))
            evidence = item.get("evidence")
            note = item.get("note")
        else:
            value = _clamp(item)
            evidence = None
            note = None
        normalized[name] = value
        details[name] = {
            "score": value,
            "weight": weights[name],
            "critical": name in CRITICAL_GATES,
            "evidence": evidence,
            "note": note,
            "passes_threshold": value >= threshold,
        }

    weighted_score = sum(normalized[name] * weights[name] for name in weights) / total_weight
    blockers = payload.get("unresolved_blockers") or []
    if not isinstance(blockers, list):
        raise ValueError("unresolved_blockers must be an array")
    blockers = [str(item).strip() for item in blockers if str(item).strip()]
    failed_critical = sorted(name for name in CRITICAL_GATES if normalized.get(name, 0.0) < threshold)
    ready = weighted_score >= threshold and not failed_critical and not blockers

    result = {
        "schema": "auro.release-readiness.v1",
        "threshold": threshold,
        "weighted_human_deliverable_readiness": round(weighted_score, 6),
        "ready": ready,
        "critical_gates": sorted(CRITICAL_GATES),
        "failed_critical_gates": failed_critical,
        "unresolved_blockers": blockers,
        "gates": details,
        "claim_boundary": {
            "readiness_is_benchmark_accuracy": False,
            "readiness_is_intelligence_percentage": False,
            "ready_requires_no_unresolved_blockers": True,
        },
    }
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Score AURO human-deliverable release readiness")
    parser.add_argument("input", type=Path)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = score_readiness(payload, args.threshold)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
