"""Mature smoke: LLM answers must be usable and fast."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auro_native_llm.model.usable import hybrid_answer, is_usable_text
from auro_native_llm.organism.checkpoint import load_mind


def main() -> int:
    ckpt = ROOT / "checkpoints" / "auro_minds" / "Auro-2B_physics"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "auro_minds" / "Auro-2B_continual"
    print(f"loading {ckpt} (fast path)…", flush=True)
    t0 = time.time()
    mind = load_mind(ckpt, chrome_mock=True, full_runtime=False)
    print(f"loaded in {time.time()-t0:.2f}s params={mind.language.num_params:,}", flush=True)

    qs = [
        "What is MESIE SpectralGPT?",
        "What is GHOST hybrid architecture?",
        "What is the golden ratio phi approx?",
        "How do I train Auro?",
        "What is Auro compute plane?",
    ]
    ok_n = 0
    for q in qs:
        t1 = time.time()
        r = mind.chat(q, prefer_lm=False)
        ans = (r.get("answer") or r.get("text") or "").strip()
        usable = is_usable_text(ans, min_len=8)
        ok_n += int(usable)
        print(
            f"[{'OK' if usable else 'FAIL'}] {time.time()-t1:.2f}s method={r.get('method')} "
            f":: {q}\n  → {ans[:160].replace(chr(10),' ')}\n",
            flush=True,
        )

    # coding
    from auro_foundry.coding_harness import CodingTask
    from auro_native_llm.intelligence.coding import CodingOrchestrator

    att = CodingOrchestrator(mind).solve_task(
        CodingTask("add", "sum two numbers", "assert solution(2,3)==5\n")
    )
    print(f"CODE passed={att.passed} method={att.method}", flush=True)
    print(att.source[:100] if att.source else "no source", flush=True)

    # pure hybrid without mind load path
    a, m = hybrid_answer("What does MESIE stand for?", mind)
    print(f"hybrid {m}: {a[:100]}", flush=True)

    print(
        f"\nSUMMARY chat_usable={ok_n}/{len(qs)} coding={att.passed} load_s={time.time()-t0:.1f}",
        flush=True,
    )
    return 0 if ok_n == len(qs) and att.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
