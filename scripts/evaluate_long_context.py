#!/usr/bin/env python3
"""Build AURO long-context and MoE evidence from exact runner observations.

Input JSON must contain checkpoint identity plus raw observations emitted by the
actual model runtime. This command does not fabricate inference. A deterministic
--smoke mode exists only to validate the evaluation pipeline and is permanently
marked non-promotable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from auro_native_llm.evaluation.long_context import (
    build_evidence_receipt,
    geometric_curriculum,
    perplexity_by_position,
    regression_report,
    retrieval_report,
    routing_balance,
    write_receipt,
)


def smoke_payload() -> Dict[str, Any]:
    assignments = [[i % 8, (i + 1) % 8] for i in range(128)]
    retrieval_cases = []
    for index, position in enumerate((0.05, 0.25, 0.50, 0.75, 0.95) * 4):
        expected = f"needle-{index}"
        retrieval_cases.append({
            "case_id": f"smoke-{index}",
            "context_length": 4096,
            "needle_position": position,
            "expected": expected,
            "observed": expected,
        })
    losses = [2.0] * 4096
    return {
        "model_id": "Auro-smoke",
        "checkpoint_sha256": "smoke-not-a-checkpoint",
        "exact_checkpoint": False,
        "base_context": 1024,
        "target_context": 4096,
        "curriculum_tokens": 4_000_000,
        "retrieval_cases": retrieval_cases,
        "token_losses": losses,
        "routing_assignments": assignments,
        "num_experts": 8,
        "baseline": {
            "retrieval": {"accuracy": 1.0},
            "perplexity": {"first_to_last_ratio": 1.0},
            "routing": {"coefficient_of_variation": 0.0},
            "protected_metrics": {"short_context_accuracy": 0.90},
        },
        "protected_metrics": {"short_context_accuracy": 0.90},
        "runner": {"kind": "deterministic_pipeline_smoke", "quality_evidence": False},
    }


def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    curriculum = geometric_curriculum(
        payload["model_id"],
        int(payload["base_context"]),
        int(payload["target_context"]),
        int(payload["curriculum_tokens"]),
        int(payload.get("curriculum_stages", 4)),
    )
    retrieval = retrieval_report(payload["retrieval_cases"])
    perplexity = perplexity_by_position(payload["token_losses"], int(payload.get("perplexity_buckets", 8)))
    routing = routing_balance(payload["routing_assignments"], int(payload["num_experts"]))
    candidate = {
        "retrieval": retrieval,
        "perplexity": perplexity,
        "routing": routing,
        "protected_metrics": payload.get("protected_metrics", {}),
    }
    regression = regression_report(payload["baseline"], candidate)
    return build_evidence_receipt(
        model_id=payload["model_id"],
        checkpoint_sha256=payload["checkpoint_sha256"],
        curriculum=curriculum,
        retrieval=retrieval,
        perplexity=perplexity,
        routing=routing,
        regression=regression,
        runner=payload.get("runner", {}),
        exact_checkpoint=bool(payload.get("exact_checkpoint", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke == bool(args.input):
        parser.error("choose exactly one of --input or --smoke")
    payload = smoke_payload() if args.smoke else json.loads(args.input.read_text(encoding="utf-8"))
    receipt = evaluate(payload)
    write_receipt(args.output, receipt)
    print(json.dumps({
        "output": str(args.output),
        "decision": receipt["promotion"]["decision"],
        "evidence_sha256": receipt["evidence_sha256"],
        "exact_checkpoint": receipt["exact_checkpoint"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
