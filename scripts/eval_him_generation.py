"""Evaluate HIM generation quality: usable text + keyword grounding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

# id → required substrings (case-insensitive) for grounded answers
EVAL_CASES: List[Dict[str, Any]] = [
    {"q": "What is MESIE?", "must": ["spectral", "mesie"], "id": "mesie"},
    {"q": "What does MESIE stand for?", "must": ["multi-element", "spectral"], "id": "stand"},
    {"q": "What is GHOST?", "must": ["grounded", "receipt"], "id": "ghost"},
    {"q": "Who are you?", "must": ["him"], "id": "who"},
    {"q": "What is Auro's compute plane?", "must": ["mesie"], "id": "plane"},
    {"q": "What is the golden ratio phi approximately?", "must": ["1.618"], "id": "phi"},
    {"q": "How do I train Auro?", "must": ["train", "checkpoint"], "id": "train"},
    {"q": "Should every step call a large language model?", "must": ["no", "mesie"], "id": "hybrid"},
    {"q": "Where should RPC API keys live?", "must": ["server", ".env"], "id": "keys"},
    {"q": "How does the 500k context window work?", "must": ["hierarchical", "500"], "id": "ctx"},
]


def main(argv: list[str] | None = None) -> int:
    import sys

    sys.path.insert(0, str(ROOT))

    p = argparse.ArgumentParser()
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_him_sft")
    p.add_argument("--fallback", default="checkpoints/auro_minds/Auro-2B_physics")
    p.add_argument("--prefer-lm", action="store_true")
    args = p.parse_args(argv)

    from auro_native_llm.model.usable import is_usable_text
    from auro_native_llm.organism.checkpoint import load_mind

    resume = Path(args.resume)
    if not resume.exists():
        resume = Path(args.fallback)
    print(f"[eval] load {resume}", flush=True)
    mind = load_mind(resume, chrome_mock=True, full_runtime=False)

    results = []
    t0 = time.time()
    for case in EVAL_CASES:
        r = mind.chat(case["q"], prefer_lm=args.prefer_lm)
        ans = (r.get("answer") or r.get("text") or "").strip()
        low = ans.lower()
        usable = is_usable_text(ans, min_len=8)
        hits = [m for m in case["must"] if m.lower() in low]
        grounded = len(hits) == len(case["must"])
        passed = usable and grounded
        results.append(
            {
                "id": case["id"],
                "q": case["q"],
                "passed": passed,
                "usable": usable,
                "grounded": grounded,
                "hits": hits,
                "method": r.get("method"),
                "answer_preview": ans[:180],
            }
        )
        mark = "OK" if passed else "FAIL"
        print(f"  [{mark}] {case['id']} method={r.get('method')} :: {ans[:100]}", flush=True)

    n = len(results)
    passed_n = sum(1 for x in results if x["passed"])
    usable_n = sum(1 for x in results if x["usable"])
    report = {
        "schema": "auro.him.eval.v1",
        "ok": passed_n == n,
        "checkpoint": str(resume),
        "n": n,
        "passed": passed_n,
        "usable": usable_n,
        "pass_rate": passed_n / max(n, 1),
        "usable_rate": usable_n / max(n, 1),
        "results": results,
        "elapsed_s": time.time() - t0,
        "num_params_live": mind.language.num_params,
        "train_steps": mind.language.train_steps,
    }
    out = ROOT / "artifacts" / "him"
    out.mkdir(parents=True, exist_ok=True)
    (out / "EVAL_GENERATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "pass_rate": report["pass_rate"],
                "usable_rate": report["usable_rate"],
                "passed": f"{passed_n}/{n}",
                "saved": str(out / "EVAL_GENERATION.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
