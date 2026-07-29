#!/usr/bin/env python3
"""Run AURO 2B-triad tests without confusing fixtures for model quality.

`fixture` validates routing, MESIE receipts, consensus, fluidization and
parameter-accounting boundaries. `exact` requires AURO_TRIAD_FLEET_JSON and
calls the configured checkpoints/endpoints. Neither mode silently downloads a
base model or substitutes hosted intelligence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auro_native_llm.model.atomic_family import AURO_500M_TRIAD
from auro_native_llm.model.triad_swarm import Auro2BTriadSwarm, ModelExecutor, ModelIdentity

CASES = (
    "Research the evidence for a design claim, identify uncertainty, and explain the safest conclusion.",
    "Inspect a Python service failure, propose a bounded patch, and state how to test it without claiming execution.",
    "Create a concise but creative launch explanation that remains technically accurate and conversational.",
    "Use prior context carefully, separate remembered facts from inference, and explain the next operational step.",
)


class FixtureMesie:
    def analyze(self, text, model_id):
        payload = {
            "model_id": model_id,
            "backend": "mesie.deterministic-fixture",
            "spectral_metrics": {"spectral_entropy": 0.5, "spectral_centroid": 0.25},
            "embedding_sha256": hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest(),
        }
        payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload


def fixture_generator(label):
    def generate(messages, options):
        system = messages[0]["content"]
        if "one member of a three-model" in system:
            payload = {"consensus": "Preserve evidence boundaries and perform the next verifiable action", "confidence": 0.75, "disagreements": [], "evidence": [f"fixture:{label}"]}
        elif "Auro-2B, parent" in system:
            payload = {"answer": "Use the bounded specialist result and keep unverified checkpoints quarantined", "reasoning_summary": ["Three specialists returned contracts", "MESIE receipts were attached"], "key_points": ["Use bounded task capsules", "Keep checkpoint identities separate"], "caveats": ["Fixture outputs are not quality evidence"], "next_steps": ["Run exact promoted checkpoints"], "confidence": 0.7}
        else:
            payload = {"analysis": f"{label} fixture analysis", "draft": f"{label} fixture draft", "recommendations": ["run exact evaluation"], "evidence": [f"fixture:{label}"], "confidence": 0.7}
        return {"text": json.dumps(payload), "usage": {"completion_tokens": 24}}

    return generate


def fixture_runtime():
    main = ModelExecutor(ModelIdentity("Auro-2B", 2_000_000_000, provider="fixture-no-checkpoint"), fixture_generator("main"))
    specialists = [ModelExecutor(ModelIdentity(item.variant_id, 500_000_000, provider="fixture-no-checkpoint"), fixture_generator(item.variant_id)) for item in AURO_500M_TRIAD]
    return Auro2BTriadSwarm(main_2b=main, specialists=specialists, atomic_executors={}, mesie=FixtureMesie())


def run_fixture():
    runtime = fixture_runtime()
    records = []
    for index, case in enumerate(CASES):
        started = time.perf_counter()
        result = runtime.run_turn(case, full_parent_context=(case + " retained context ") * 100).to_dict()
        records.append({
            "case_id": f"fixture-{index + 1}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "atomic_agent_count": result["atomic_agent_count"],
            "model_backed_atomic_count": result["model_backed_atomic_count"],
            "mesie_receipt_count": len(result["mesie_receipts"]),
            "specialist_contract_passes": sum(item["contract_valid"] for item in result["specialist_reports"]),
            "consensus_contract_passes": sum(item["contract_valid"] for item in result["consensus_votes"]),
            "estimated_text_reduction": result["estimated_text_reduction"],
            "promotion_ready": result["promotion_ready"],
            "blockers": result["blockers"],
            "output_excerpt": result["text"][:300],
            "runtime_receipt_sha256": result["runtime_receipt_sha256"],
        })
    mechanics_pass = all(row["specialist_contract_passes"] == 3 and row["consensus_contract_passes"] == 3 and row["mesie_receipt_count"] >= 14 and row["promotion_ready"] is False for row in records)
    return {
        "schema": "auro.2b_triad.benchmark.v1",
        "mode": "fixture",
        "mechanics_pass": mechanics_pass,
        "quality_claim": False,
        "checkpoint_claim": False,
        "promotion_status": "permanently_quarantined_fixture",
        "cases": records,
        "summary": {
            "case_count": len(records),
            "median_latency_ms": statistics.median(row["latency_ms"] for row in records),
            "mean_estimated_text_reduction": statistics.fmean(row["estimated_text_reduction"] for row in records),
            "total_atomic_agents": sum(row["atomic_agent_count"] for row in records),
            "total_mesie_receipts": sum(row["mesie_receipt_count"] for row in records),
        },
        "boundaries": [
            "Fixture generators are deterministic control doubles, not trained AURO checkpoints.",
            "Text reduction is a transport estimate, not exact tokenizer or throughput evidence.",
            "Official quality claims require exact checkpoint hashes, tokenizer custody, benchmark versions and failure samples.",
        ],
    }


def run_exact():
    if not os.getenv("AURO_TRIAD_FLEET_JSON", "").strip():
        raise SystemExit("exact mode requires AURO_TRIAD_FLEET_JSON")
    from auro_native_llm.production_fleet.runtime import NovaRuntime

    runtime = NovaRuntime()
    if runtime.triad is None:
        raise SystemExit(runtime.triad_error or "triad initialization failed")
    records = []
    for index, case in enumerate(CASES):
        started = time.perf_counter()
        result = runtime.triad_respond(case)
        records.append({
            "case_id": f"exact-{index + 1}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "promotion_ready": result["promotion_ready"],
            "blockers": result["blockers"],
            "atomic_agent_count": result["atomic_agent_count"],
            "model_backed_atomic_count": result["model_backed_atomic_count"],
            "estimated_text_reduction": result["estimated_text_reduction"],
            "output": result["text"],
            "structured_answer": result["structured_answer"],
            "runtime_receipt_sha256": result["runtime_receipt_sha256"],
        })
    return {
        "schema": "auro.2b_triad.benchmark.v1",
        "mode": "exact",
        "checkpoint_claim": all(row["promotion_ready"] for row in records),
        "quality_claim": False,
        "cases": records,
        "boundaries": [
            "This runner records conversation behavior and evidence custody; official benchmark accuracy remains separate.",
            "Promotion still requires the constitutional gate and the repository's official benchmark harnesses.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "exact"), default="fixture")
    parser.add_argument("--output", default="evidence/auro-2b-triad-benchmark.json")
    args = parser.parse_args()
    report = run_fixture() if args.mode == "fixture" else run_exact()
    report["generated_at_unix"] = int(time.time())
    unsigned = dict(report)
    report["report_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "mode": report["mode"], "report_sha256": report["report_sha256"], "mechanics_pass": report.get("mechanics_pass")}))


if __name__ == "__main__":
    main()
