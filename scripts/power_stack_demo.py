"""Run coupled physics + economy + algorithms + transformers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auro_native_llm.engines.orchestra import PowerStack


def main() -> int:
    mind = None
    ckpt = ROOT / "checkpoints" / "auro_minds" / "Auro-2B_physics"
    if ckpt.exists():
        try:
            from auro_native_llm.organism.checkpoint import load_mind

            print(f"loading {ckpt}", flush=True)
            mind = load_mind(ckpt, chrome_mock=True)
        except Exception as exc:
            print(f"mind optional: {exc}", flush=True)

    stack = PowerStack(mind)
    rep = stack.run(
        "Spectral market under Hamiltonian field stress — route agents by resonance",
        rounds=6,
        physics_steps=3,
    )
    hist = rep["history"]
    print(f"rounds={rep['rounds']} engines={len(rep['engines'])} saved={rep.get('saved')}", flush=True)
    for i, h in enumerate(hist):
        p = h["physics"]
        e = h["economy"]
        r = h["route"]
        print(
            f"  r{i+1}: E={p['energy']:.4f} g={p['coupling']:.3f} "
            f"M={p['metrics'].get('magnetization', 0):.3f} "
            f"W={e['wealth']:.4f} U={e['utility']:.4f} F={e['free_energy']:.4f} "
            f"z={e['excess_demand_l2']:.4f} "
            f"route={r.get('route')} score={r.get('score', 0):.3f} "
            f"ot={h['algorithms'].get('ot_cost', 0):.4f} "
            f"xf={h['transformer'].get('backend')}",
            flush=True,
        )
    last = hist[-1]
    print(json.dumps({"last_route": last["route"], "engines": rep["engines"][:8], "...": "..."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
