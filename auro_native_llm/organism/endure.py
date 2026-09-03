"""Endure mode — experiment variants, keep looping, never die after one shot.

The best Auro mode for long work: try N experimental angles, then endure on
the winner with extra cycles. Every cycle writes a receipt. Failures are
fuel, not a stop.

  python -m auro_native_llm.organism.endure --goal "…"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "auro.endure.v1"


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _one(goal: str, *, tag: str, i: int) -> Dict[str, Any]:
    t0 = time.time()
    rec: Dict[str, Any] = {"i": i, "tag": tag, "goal": goal[:400], "ok": False, "reward": 0.0}
    try:
        from auro_native_llm.organism.autocycle import AutocycleConfig, run_autocycle

        cfg = AutocycleConfig(
            cycles=1,
            train_steps_per_cycle=1,
            lite=True,
            show=False,
            sleep_heartbeat=False,
            goals=[goal],
        )
        out = run_autocycle(cfg)
        rec["ok"] = bool(out.get("ok", True))
        rec["reward"] = 0.8 if rec["ok"] else 0.25
        rec["via"] = "autocycle"
        rec["detail"] = {k: out.get(k) for k in ("cycles", "ok", "model_id") if k in out}
    except Exception as e:
        rec["ok"] = False
        rec["via"] = "error"
        rec["error"] = str(e)[:240]
        rec["reward"] = 0.1
        try:
            from auro_native_llm.organism.self_train import Experience

            rec["experience"] = Experience(text=goal[:500], kind="error", model_id="Auro-2B", reward=0.1).to_dict()
        except Exception:
            pass
    rec["ms"] = int((time.time() - t0) * 1000)
    return rec


def run_endure(
    goal: str,
    *,
    experiments: int = 4,
    cycles: int = 6,
    model_id: str = "Auro-2B",
) -> Dict[str, Any]:
    g = (goal or "").strip() or "Stay useful: try, fail, keep going, write receipts."
    t0 = time.time()
    receipts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    n_exp = max(1, int(experiments))
    n_end = max(0, int(cycles))
    for i in range(n_exp):
        rec = _one(f"{g} [experiment {i + 1}/{n_exp}]", tag="experiment", i=i)
        receipts.append(rec)
        if best is None or rec["reward"] >= best["reward"]:
            best = rec
    seed = (best or {}).get("goal") or g
    for j in range(n_end):
        rec = _one(f"{seed} [endure {j + 1}/{n_end}]", tag="endure", i=n_exp + j)
        receipts.append(rec)
        if best is None or rec["reward"] >= best["reward"]:
            best = rec
    out_dir = _root() / "artifacts" / "auro-endure"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "schema": SCHEMA,
        "native": True,
        "via": "auro_native_llm.organism.endure",
        "model_id": model_id,
        "goal": g[:400],
        "experiments": n_exp,
        "endure_cycles": n_end,
        "best": best,
        "receipts": receipts,
        "ms": int((time.time() - t0) * 1000),
        "doctrine": "Failures endure. The mind does not stop after one shot.",
    }
    path = out_dir / "LAST.json"
    path.write_text(json.dumps(payload, indent=2, default=str)[:120_000], encoding="utf-8")
    payload["path"] = str(path)
    payload["summary"] = (
        f"Auro Endure: {n_exp} experiments + {n_end} endure cycles. "
        f"Best { (best or {}).get('tag') } reward={(best or {}).get('reward')} via={(best or {}).get('via')}."
    )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="auro-endure")
    p.add_argument("--goal", default="Keep a useful experiment alive")
    p.add_argument("--experiments", type=int, default=3)
    p.add_argument("--cycles", type=int, default=4)
    args = p.parse_args(argv)
    r = run_endure(args.goal, experiments=args.experiments, cycles=args.cycles)
    print(r.get("summary") or json.dumps(r, default=str)[:2000])
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
