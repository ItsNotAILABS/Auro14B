"""Hybrid runtime facade — killer GHOST/MESIE use case."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.vproc.processor import MesieVirtualProcessor, WorkCall


class HybridRuntime:
    """MESIE first. LLM rare. Full work-call inspection."""

    def __init__(self, mind: Any = None) -> None:
        self.mind = mind
        self.vproc = MesieVirtualProcessor(mind)
        self.history: List[Dict[str, Any]] = []

    def execute(
        self,
        prompt: str,
        *,
        force_mesie_only: bool = False,
        save: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        call = self.vproc.work(prompt, force_mesie_only=force_mesie_only)
        # optional ghost supervisor receipt merge
        ghost_sup: Dict[str, Any] = {}
        try:
            from auro_native_llm.ghost.supervisor import GhostSupervisor

            # spectral-only intents: still run policy but mesie-first
            if not call.result.get("escalate"):
                ghost_sup = {
                    "note": "LLM not invoked — Ghost/MESIE path completed",
                    "routing": call.metrics.routing,
                }
            else:
                # already escalated inside vproc; supervisor policy stamp
                from auro_native_llm.ghost.policy import PolicyGate

                dec = PolicyGate().decide(prompt)
                ghost_sup = {"policy": dec.to_dict(), "escalated": True}
        except Exception as exc:
            ghost_sup = {"error": str(exc)[:120]}

        out = {
            "schema": "auro.hybrid.execution.v1",
            "ok": call.ok,
            "work_call": call.to_dict(),
            "ghost": ghost_sup,
            "processor": self.vproc.health(),
            "elapsed_s": time.time() - t0,
            "killer_use_case": {
                "mesie_ghost": "deterministic filter/features/shadow/stream",
                "llm": "only if escalate justified by cleaned spectral features",
                "llm_used": bool(call.result.get("escalate")),
                "escalate_reason": call.result.get("escalate_reason"),
            },
            "counterpoint_to_monolithic_llm": True,
        }
        self.history.append(out)
        if save:
            p = Path("artifacts/auro-vproc")
            p.mkdir(parents=True, exist_ok=True)
            (p / "LAST_WORK_CALL.json").write_text(
                json.dumps(out, indent=2, default=str), encoding="utf-8"
            )
            # chain
            if self.vproc.chain.receipts:
                self.vproc.chain.save(p / "LAST_RECEIPT_CHAIN.json")
            out["saved"] = str(p / "LAST_WORK_CALL.json")
        return out

    def batch_demo(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Show most steps skip LLM."""
        prompts = prompts or [
            "Filter and smooth sensor stream; report spectral coherence",
            "Compute PSD-like band energy for vibration signature",
            "Match two spectral envelopes for predictive maintenance",
            "Explain strategy for multi-site agent fleet given coherence metrics",
            "Stream frequency-domain features from edge device (no prose)",
        ]
        rows = []
        for p in prompts:
            r = self.execute(p, save=False)
            wc = r["work_call"]
            rows.append(
                {
                    "prompt": p[:60],
                    "routing": wc["metrics"]["routing"],
                    "llm": wc["result"]["escalate"],
                    "reason": wc["result"]["escalate_reason"],
                    "nova_cycles": wc["metrics"]["nova_cycles"],
                    "coherence": wc["metrics"]["coherence"],
                    "resonance": wc["metrics"]["resonance"],
                    "bytes": wc["metrics"]["bytes_processed"],
                }
            )
        n_llm = sum(1 for x in rows if x["llm"])
        summary = {
            "schema": "auro.hybrid.batch_demo.v1",
            "n": len(rows),
            "n_llm_escalations": n_llm,
            "n_mesie_only": len(rows) - n_llm,
            "llm_fraction": n_llm / max(len(rows), 1),
            "rows": rows,
            "lesson": "Most steps skip LLM — faster, cheaper, auditable.",
        }
        p = Path("artifacts/auro-vproc")
        p.mkdir(parents=True, exist_ok=True)
        (p / "BATCH_DEMO.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def hybrid_execute(prompt: str, mind: Any = None, **kw: Any) -> Dict[str, Any]:
    return HybridRuntime(mind).execute(prompt, **kw)
