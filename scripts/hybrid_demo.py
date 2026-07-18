"""MESIE hybrid killer-use-case demo."""
from __future__ import annotations

import json

from auro_native_llm.vproc.hybrid import HybridRuntime
from auro_native_llm.vproc import run_work_call


def main() -> int:
    rt = HybridRuntime(None)
    demo = rt.batch_demo()
    print(
        f"n={demo['n']} mesie_only={demo['n_mesie_only']} "
        f"llm={demo['n_llm_escalations']} frac={demo['llm_fraction']:.0%}"
    )
    for r in demo["rows"]:
        print(
            f"  [{r['routing']}] llm={r['llm']} :: {r['prompt'][:55]} "
            f"| {r['reason']}"
        )
    w = run_work_call(
        "Filter smooth stream and compute spectral buckets",
        force_mesie_only=True,
    )
    c = w["call"]
    print(
        "CALL",
        c["metrics"]["routing"],
        "bytes",
        c["metrics"]["bytes_processed"],
        "nova",
        round(c["metrics"]["nova_cycles"], 2),
        "R",
        round(c["metrics"]["resonance"], 3),
        "tip",
        (c.get("receipt_tip") or "")[:16],
    )
    print(json.dumps({"demo": demo, "sample_call_metrics": c["metrics"]}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
