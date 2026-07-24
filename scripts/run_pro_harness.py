#!/usr/bin/env python3
"""Run AURO's professional multitask/creativity harness on an exact checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/pro-evaluation/report.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    from auro_native_llm.evaluation.pro_harness import ProEvaluationHarness, default_cases
    from auro_native_llm.model.usable import generate_usable
    from auro_native_llm.organism.checkpoint import load_mind

    checkpoint = Path(args.checkpoint)
    mind = load_mind(checkpoint, chrome_mock=True, full_runtime=False)

    def generate(prompt: str, metadata: dict) -> str:
        result = generate_usable(
            mind,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            prefer_lm=True,
        )
        return str(result.get("text") or result.get("answer") or "")

    report = ProEvaluationHarness(default_cases()).run(
        generate,
        {
            "path": str(checkpoint),
            "model_id": getattr(mind, "model_id", checkpoint.name),
            "train_steps": getattr(getattr(mind, "language", None), "train_steps", None),
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_passed": report["all_passed"],
        "mean_score": report["mean_score"],
        "receipt_sha256": report["receipt_sha256"],
        "output": str(output),
    }, indent=2))
    return 0 if report["all_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
