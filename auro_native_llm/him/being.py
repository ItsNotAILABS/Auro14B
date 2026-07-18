"""HIM — agentic multi-model organism.

Identity
--------
Name: HIM
Nature: host of mini Python models (germs) + tools + GHOST + MESIE
Loop:  SENSE → PLAN → ACT → OBSERVE → REFLECT → (repeat | done)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.colony import build_colony
from auro_native_llm.model.usable import hybrid_answer, is_usable_text


HIM_NAME = "HIM"
HIM_DOCTRINE = (
    "I am HIM. I am not a single opaque LLM. I am a colony of specialist "
    "mini-models (germs) hosted in Python: skills, spectral, code, reason, "
    "planner, critic, writer. MESIE is my deterministic math. GHOST is my "
    "audit spine. I plan, act with tools, observe results, and only escalate "
    "to free language when spectral lock is insufficient."
)


@dataclass
class HimStep:
    phase: str
    ok: bool
    detail: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "ok": self.ok, "detail": self.detail, "ts": self.ts}


class HIM:
    """The agentic being — multi-mini-model + tools + autonomous loop."""

    def __init__(
        self,
        mind: Any = None,
        *,
        n_germs: int = 40,
        context_tokens: int = 500_000,
        name: str = HIM_NAME,
    ) -> None:
        self.name = name
        self.mind = mind
        self.session_id = f"him-{uuid.uuid4().hex[:10]}"
        self.born_at = time.time()
        self.colony = build_colony(
            mind, n_extra_germs=n_germs, context_tokens=context_tokens
        )
        # seed identity into 500k context
        self.colony.context.ingest(HIM_DOCTRINE, kind="system", meta={"who": self.name})
        self.colony.context.ingest(
            f"{self.name} session {self.session_id} awakened.",
            kind="system",
        )
        self.memory: List[Dict[str, Any]] = []
        self.steps: List[HimStep] = []
        self.goals_done: List[str] = []

    # ---------------------------------------------------------------- identity
    def whoami(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "session_id": self.session_id,
            "doctrine": HIM_DOCTRINE,
            "n_germs": self.colony.n_germs,
            "num_params_live": self.colony.num_params_live
            + int(getattr(getattr(self.mind, "language", None), "num_params", 0) or 0),
            "colony_params": self.colony.num_params_live,
            "host_params": int(
                getattr(getattr(self.mind, "language", None), "num_params", 0) or 0
            ),
            "context_window_tokens": self.colony.context.token_budget,
            "context_used": self.colony.context.tokens_used,
            "agentic": True,
            "loops": ["SENSE", "PLAN", "ACT", "OBSERVE", "REFLECT"],
            "tools": [
                "colony_generate",
                "hybrid_answer",
                "coding",
                "reasoning",
                "ghost",
                "hybrid_vproc",
                "power_stack",
                "google_envelope",
                "github",
                "web3_install",
                "web3_api",
                "vault",
            ],
            "web3": {
                "applet": "him-web3",
                "install": "npm run install:applet -- ethers viem",
                "api": "http://127.0.0.1:8787/api/*",
                "security": "RPC keys server-side only; sealed refs in vault ledgers",
            },
            "vault": {
                "ledgers": ["keys", "rpc", "high_value", "agent", "github"],
                "module": "auro_native_llm.vault",
                "reveal": "metadata by default; reveal only in server process",
            },
        }

    # ---------------------------------------------------------------- agent loop
    def sense(self, goal: str) -> Dict[str, Any]:
        ctx = self.colony.context.retrieve(goal, top_k=8, token_cap=4000)
        ans, method = hybrid_answer(goal, self.mind)
        out = {
            "goal": goal,
            "context_tokens": self.colony.context.tokens_used,
            "context_preview": ctx[:600],
            "hybrid_method": method,
            "hybrid_preview": (ans or "")[:400],
        }
        self.steps.append(HimStep("SENSE", True, out))
        return out

    def plan(self, goal: str, sense: Dict[str, Any]) -> Dict[str, Any]:
        low = goal.lower()
        actions: List[Dict[str, str]] = []
        # always plan
        actions.append({"tool": "colony_generate", "why": "multi-germ prose + skills"})
        if any(k in low for k in ("code", "function", "implement", "assert", "python")):
            actions.append({"tool": "coding", "why": "assert-backed code"})
        if any(k in low for k in ("spectral", "mesie", "match", "embed", "psd")):
            actions.append({"tool": "hybrid_vproc", "why": "MESIE deterministic path"})
        if any(k in low for k in ("ghost", "policy", "receipt", "audit")):
            actions.append({"tool": "ghost", "why": "GHOST policy + receipts"})
        if any(k in low for k in ("github", "repo", "push", "pr")):
            actions.append({"tool": "github", "why": "GitHub CLI access"})
        if any(k in low for k in ("train", "power stack", "economy", "physics engine")):
            actions.append({"tool": "power_stack", "why": "coupled physics+econ engines"})
        if any(k in low for k in ("google", "mail", "collab", "browser")):
            actions.append({"tool": "google", "why": "sandbox workspace"})
        if any(
            k in low
            for k in (
                "web3",
                "ethereum",
                "ethers",
                "viem",
                "blockchain",
                "rpc",
                "alchemy",
                "infura",
                "smart contract",
                "on-chain",
                "block number",
            )
        ):
            actions.append({"tool": "web3", "why": "secure him-web3 API / packages"})
        if any(
            k in low
            for k in (
                "vault",
                "secret",
                "secrets",
                "api key",
                "api keys",
                "ledger",
                "password",
                "mnemonic",
                "pat ",
                "sealed",
            )
        ):
            actions.append({"tool": "vault", "why": "multi-ledger sealed secrets (metadata only)"})
        # ensure at least hybrid answer
        if not any(a["tool"] == "colony_generate" for a in actions):
            actions.insert(0, {"tool": "colony_generate", "why": "default voice"})
        plan = {
            "goal": goal,
            "actions": actions[:6],
            "max_steps": len(actions[:6]) + 1,
            "from_sense": sense.get("hybrid_method"),
        }
        self.steps.append(HimStep("PLAN", True, plan))
        return plan

    def act(self, tool: str, goal: str) -> Dict[str, Any]:
        mind = self.mind
        try:
            if tool == "colony_generate":
                out = self.colony.generate(goal)
                return {"ok": out.get("ok"), "tool": tool, "text": out.get("text"), "meta": {
                    "n_germs_active": out.get("n_germs_active"),
                    "params": out.get("num_params_live"),
                }}
            if tool == "coding" and mind is not None:
                from auro_foundry.coding_harness import CodingTask
                from auro_native_llm.intelligence.coding import CodingOrchestrator

                att = CodingOrchestrator(mind).solve_task(
                    CodingTask("him_code", goal, "assert solution is not None\n")
                )
                return {
                    "ok": bool(att.passed or att.source),
                    "tool": tool,
                    "text": att.source or "",
                    "meta": {"passed": att.passed, "method": att.method},
                }
            if tool == "hybrid_vproc":
                from auro_native_llm.vproc.hybrid import HybridRuntime

                r = HybridRuntime(mind).execute(goal, force_mesie_only=False, save=False)
                wc = r.get("work_call") or {}
                return {
                    "ok": r.get("ok"),
                    "tool": tool,
                    "text": json.dumps({
                        "routing": (wc.get("metrics") or {}).get("routing"),
                        "resonance": (wc.get("metrics") or {}).get("resonance"),
                        "escalate": (wc.get("result") or {}).get("escalate"),
                    }),
                    "meta": r.get("killer_use_case"),
                }
            if tool == "ghost" and mind is not None:
                if hasattr(mind, "ghost_run"):
                    g = mind.ghost_run(goal)
                else:
                    from auro_native_llm.ghost.supervisor import GhostSupervisor

                    g = GhostSupervisor(mind).run(goal).to_dict()
                return {
                    "ok": g.get("ok"),
                    "tool": tool,
                    "text": json.dumps({
                        "risk": g.get("risk_class"),
                        "haunt": g.get("haunt_flags"),
                        "custody_tip": ((g.get("receipt_chain") or {}).get("tip_hash") or "")[:16],
                    }),
                    "meta": {"claims": len(g.get("claims") or [])},
                }
            if tool == "github":
                from auro_native_llm.github_access import GitHubAccess

                st = GitHubAccess().status()
                return {
                    "ok": st.get("signed_in"),
                    "tool": tool,
                    "text": f"GitHub signed_in={st.get('signed_in')} user={((st.get('user') or {}).get('login'))}",
                    "meta": st,
                }
            if tool == "power_stack" and mind is not None:
                from auro_native_llm.engines.orchestra import PowerStack

                rep = PowerStack(mind).run(goal, rounds=2, physics_steps=1)
                last = (rep.get("history") or [{}])[-1]
                return {
                    "ok": rep.get("ok"),
                    "tool": tool,
                    "text": json.dumps({
                        "route": (last.get("route") or {}).get("route"),
                        "wealth": (last.get("economy") or {}).get("wealth"),
                        "E": (last.get("physics") or {}).get("energy"),
                    }),
                    "meta": {"engines": len(rep.get("engines") or [])},
                }
            if tool == "google" and mind is not None:
                if hasattr(mind, "google"):
                    g = mind.google("status")
                    return {"ok": True, "tool": tool, "text": str(g)[:500], "meta": {}}
            if tool == "web3":
                from auro_native_llm.him.web3_tools import HimWeb3Tools

                w3 = HimWeb3Tools()
                st = w3.status()
                health = w3.api_health()
                block = w3.block_number() if health.get("ok") and health.get("rpc_configured") else {}
                text = json.dumps(
                    {
                        "web3_status": st,
                        "api_health": health,
                        "block_number": block,
                        "architecture": (
                            "React → /api/* → ethers/viem on server; "
                            "keys only in him-web3/.env"
                        ),
                    },
                    default=str,
                )[:2000]
                return {
                    "ok": bool(st.get("ok")),
                    "tool": tool,
                    "text": text,
                    "meta": {"rpc": health.get("rpc_configured")},
                }
            if tool == "vault":
                from auro_native_llm.vault import get_vault

                v = get_vault()
                health = v.health()
                listed = v.list()
                hint = v.export_rpc_env_hint()
                # Never include plaintext secrets in agent text
                text = json.dumps(
                    {
                        "health": health,
                        "list": listed,
                        "rpc_env_hint": hint,
                        "doctrine": (
                            "Vault stores sealed secrets in ledgers keys/rpc/high_value/"
                            "agent/github. list() is metadata only; reveal only in server process."
                        ),
                    },
                    default=str,
                )[:2000]
                return {
                    "ok": bool(health.get("ledgers")),
                    "tool": tool,
                    "text": text,
                    "meta": {"total": health.get("total"), "root": health.get("root")},
                }
            # default hybrid
            ans, meth = hybrid_answer(goal, mind)
            return {"ok": is_usable_text(ans), "tool": "hybrid", "text": ans, "meta": {"method": meth}}
        except Exception as exc:
            return {"ok": False, "tool": tool, "text": "", "error": str(exc)[:300]}

    def observe(self, action_result: Dict[str, Any]) -> Dict[str, Any]:
        obs = {
            "ok": bool(action_result.get("ok")),
            "tool": action_result.get("tool"),
            "has_text": bool(action_result.get("text")),
            "error": action_result.get("error"),
            "chars": len(str(action_result.get("text") or "")),
        }
        self.steps.append(HimStep("OBSERVE", obs["ok"], obs))
        # write into long context
        snippet = str(action_result.get("text") or action_result.get("error") or "")[:1500]
        self.colony.context.ingest(
            f"OBSERVE {obs['tool']}: {snippet}",
            kind="chat",
            meta={"phase": "observe"},
        )
        return obs

    def reflect(self, goal: str, observations: List[Dict[str, Any]], artifacts: List[str]) -> Dict[str, Any]:
        success = any(o.get("ok") for o in observations) and any(artifacts)
        summary = "\n\n".join(a[:800] for a in artifacts if a)[:3500]
        if not summary:
            summary, meth = hybrid_answer(goal, self.mind)
            method = meth
        else:
            method = "him_agent_compose"
        # final colony polish
        try:
            polish = self.colony.generate(
                f"HIM final answer for: {goal}\nEvidence:\n{summary[:1200]}"
            )
            if polish.get("ok") and is_usable_text(polish.get("text") or ""):
                summary = polish["text"]
                method = "him_colony_final"
        except Exception:
            pass
        out = {
            "ok": success or is_usable_text(summary),
            "goal": goal,
            "method": method,
            "answer": summary,
            "text": summary,
            "n_observations": len(observations),
            "name": self.name,
        }
        self.steps.append(HimStep("REFLECT", out["ok"], {"method": method, "chars": len(summary)}))
        if out["ok"]:
            self.goals_done.append(goal[:200])
        self.memory.append({"goal": goal, "answer": summary[:500], "ts": time.time()})
        return out

    def run(self, goal: str, *, max_actions: int = 5) -> Dict[str, Any]:
        """Full agentic loop for HIM."""
        t0 = time.time()
        self.steps = []
        sense = self.sense(goal)
        self.steps.append(HimStep("SENSE", True, {"keys": list(sense.keys())}))
        plan = self.plan(goal, sense)
        artifacts: List[str] = []
        observations: List[Dict[str, Any]] = []

        for action in plan.get("actions", [])[:max_actions]:
            tool = action.get("tool") or "colony_generate"
            self.steps.append(HimStep("ACT", True, {"tool": tool, "why": action.get("why")}))
            result = self.act(tool, goal)
            obs = self.observe(result)
            observations.append(obs)
            if result.get("text"):
                artifacts.append(str(result["text"]))
            # early stop if coding passed or strong answer
            if tool == "coding" and (result.get("meta") or {}).get("passed"):
                break
            if tool == "colony_generate" and is_usable_text(str(result.get("text") or ""), min_len=80):
                # still run other tools if planned, but keep going
                pass

        final = self.reflect(goal, observations, artifacts)
        report = {
            "schema": "auro.him.run.v1",
            "name": self.name,
            "session_id": self.session_id,
            "ok": final.get("ok"),
            "goal": goal,
            "plan": plan,
            "steps": [s.to_dict() for s in self.steps],
            "answer": final.get("answer"),
            "text": final.get("text"),
            "method": final.get("method"),
            "whoami": self.whoami(),
            "latency_ms": (time.time() - t0) * 1000.0,
            "agentic": True,
        }
        # persist last run
        try:
            outp = Path("artifacts/him")
            outp.mkdir(parents=True, exist_ok=True)
            (outp / "LAST_HIM_RUN.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            report["saved"] = str(outp / "LAST_HIM_RUN.json")
        except Exception:
            pass
        return report


def awaken_him(
    mind: Any = None,
    *,
    n_germs: int = 40,
    context_tokens: int = 500_000,
) -> HIM:
    return HIM(mind, n_germs=n_germs, context_tokens=context_tokens)
