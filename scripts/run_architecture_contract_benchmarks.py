#!/usr/bin/env python3
"""Benchmark AURO's MoE family policy and governed 294K context envelope.

These are executable architecture-contract benchmarks, not language-quality or
routing-quality claims. The report keeps those claim classes separate.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main() -> int:
    from auro_native_llm.context.envelope import ACCEPTED_CONTEXT_TOKENS, DEFAULT_DENSE_WINDOW, ContextEnvelope
    from auro_native_llm.model import family_config

    family = []
    for mode, model_ids in (
        ("dev", ("Auro-156K", "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B")),
        ("full", ("Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B")),
    ):
        for model_id in model_ids:
            config = family_config(model_id, mode=mode)
            row = {
                "model_id": model_id,
                "mode": mode,
                "use_moe": bool(config.use_moe),
                "num_experts": int(config.num_experts),
                "top_k_experts": int(config.top_k_experts),
                "max_seq_len": int(config.max_seq_len),
                "policy": config.extra.get("family_upgrade_policy"),
                "long_context_quality_verified": bool(config.extra.get("long_context_quality_verified")),
            }
            row["passed"] = (
                row["use_moe"]
                and row["num_experts"] >= 8
                and 2 <= row["top_k_experts"] <= row["num_experts"]
                and row["policy"] == "auro.family.moe-context.v1"
                and row["long_context_quality_verified"] is False
            )
            family.append(row)

    envelope = ContextEnvelope()
    ids = np.arange(ACCEPTED_CONTEXT_TOKENS + 4096, dtype=np.int64) % 50021
    dense_a, receipt_a, chunks_a = envelope.ingest(ids)
    dense_b, receipt_b, chunks_b = envelope.ingest(ids.copy())
    context_checks = {
        "accepted_limit": receipt_a.accepted_tokens == ACCEPTED_CONTEXT_TOKENS,
        "dense_window_bounded": receipt_a.dense_tokens <= DEFAULT_DENSE_WINDOW,
        "input_truncation_accounted": receipt_a.truncated_input_tokens == 4096,
        "receipt_deterministic": receipt_a.envelope_sha256 == receipt_b.envelope_sha256,
        "dense_view_deterministic": hashlib.sha256(dense_a.tobytes()).hexdigest() == hashlib.sha256(dense_b.tobytes()).hexdigest(),
        "chunk_inventory_deterministic": [c.sha256 for c in chunks_a] == [c.sha256 for c in chunks_b],
        "retrieval_selection_deterministic": receipt_a.selected_chunk_indexes == receipt_b.selected_chunk_indexes,
    }

    payload = {
        "schema": "auro.architecture-contract-benchmark.v1",
        "claim_class": "architecture-contract-only",
        "family": {
            "cases": len(family),
            "passed": sum(bool(row["passed"]) for row in family),
            "all_passed": all(bool(row["passed"]) for row in family),
            "results": family,
        },
        "context_294k": {
            "accepted_context_tokens": ACCEPTED_CONTEXT_TOKENS,
            "dense_window": DEFAULT_DENSE_WINDOW,
            "chunk_count": receipt_a.chunk_count,
            "retrieved_tokens": receipt_a.retrieved_tokens,
            "selected_chunk_indexes": receipt_a.selected_chunk_indexes,
            "receipt_sha256": receipt_a.envelope_sha256,
            "checks": context_checks,
            "all_passed": all(context_checks.values()),
        },
        "unverified": [
            "MoE routing quality on trained checkpoints",
            "language quality at enlarged context lengths",
            "needle retrieval accuracy across 294,912 accepted tokens",
            "throughput and memory efficiency on deployment hardware",
        ],
    }
    payload["all_passed"] = payload["family"]["all_passed"] and payload["context_294k"]["all_passed"]
    payload["report_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()

    output = ROOT / "artifacts/architecture-contract-benchmark/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
