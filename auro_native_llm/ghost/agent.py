"""Ghost Agents — lightweight embodiments of computation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from auro_native_llm.ghost.node import MesieGhostNode, NodeResult
from auro_native_llm.ghost.receipts import GhostReceiptChain


@dataclass
class GhostTask:
    intent: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    task_id: str = ""
    timeout_s: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    allow_llm_escalation: bool = True

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = f"gt-{uuid.uuid4().hex[:10]}"
        if not self.actions:
            # default pipeline: spectral → embed → (optional) intelligence
            self.actions = [
                {"engine": "mesie", "action": "spectral", "payload": {"text": self.intent}},
                {"engine": "mesie", "action": "embed", "payload": {"text": self.intent}},
            ]


@dataclass
class GhostOutcome:
    ok: bool
    task_id: str
    agent_id: str
    steps: List[Dict[str, Any]]
    result: Dict[str, Any]
    chain: GhostReceiptChain
    used_llm: bool = False
    escalate_reason: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.ghost.outcome.v1",
            "ok": self.ok,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "steps": self.steps,
            "result": self.result,
            "used_llm": self.used_llm,
            "escalate_reason": self.escalate_reason,
            "latency_ms": self.latency_ms,
            "receipt_chain": self.chain.to_dict(),
            "custody": self.chain.custody_answers(),
        }


class GhostAgentRuntime:
    """Auro Ghost Agent: MESIE bus first, LLM only when escalation justified."""

    def __init__(
        self,
        mind: Any = None,
        *,
        agent_id: Optional[str] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.mind = mind
        self.agent_id = agent_id or f"ghost-{uuid.uuid4().hex[:8]}"
        self.node = MesieGhostNode(mind)
        self.llm_fn = llm_fn
        self.history: List[GhostOutcome] = []

    def _default_llm(self, prompt: str) -> str:
        if self.llm_fn is not None:
            return self.llm_fn(prompt)
        if self.mind is not None and hasattr(self.mind, "generate"):
            try:
                r = self.mind.generate(prompt, max_new_tokens=64)
                if hasattr(r, "output"):
                    out = r.output
                    if hasattr(out, "text"):
                        return str(out.text)[:2000]
                    return str(out)[:2000]
                if hasattr(r, "text"):
                    return str(r.text)[:2000]
                return str(r)[:2000]
            except Exception as exc:
                return f"[llm_error] {exc}"
        return "[no-llm] mesie-only mode"

    def execute(self, task: GhostTask | str, *, chain: Optional[GhostReceiptChain] = None) -> GhostOutcome:
        t0 = time.perf_counter()
        if isinstance(task, str):
            task = GhostTask(intent=task)
        chain = chain or GhostReceiptChain()
        chain.append(
            "ghost_activate",
            {
                "agent_id": self.agent_id,
                "task_id": task.task_id,
                "intent": task.intent[:500],
                "n_actions": len(task.actions),
            },
            actor=self.agent_id,
            model_id=getattr(self.mind, "model_id", None),
        )

        steps: List[Dict[str, Any]] = []
        last: Dict[str, Any] = {}
        mesie_ok = True
        for i, act in enumerate(task.actions):
            engine = str(act.get("engine") or "mesie")
            action = str(act.get("action") or "embed")
            payload = dict(act.get("payload") or {})
            if "text" not in payload and task.intent:
                payload.setdefault("text", task.intent)
            # chain previous embedding into next if present
            if last.get("output", {}).get("embedding_head") and action in ("match", "intelligence"):
                payload.setdefault("prior_embed", last["output"]["embedding_head"])

            if engine in ("mesie", "mesie_ghost", "spectral", "helix"):
                res: NodeResult = self.node.run(action, payload)
                step = res.to_dict()
                step["step"] = i
                steps.append(step)
                chain.append(
                    "mesie",
                    {
                        "engine": res.engine,
                        "action": res.action,
                        "ok": res.ok,
                        "evidence_ref": res.evidence_ref,
                        "latency_ms": res.latency_ms,
                        "change": f"mesie:{res.action}",
                    },
                    actor=self.agent_id,
                    ok=res.ok,
                )
                last = step
                mesie_ok = mesie_ok and res.ok
            elif engine in ("llm", "language"):
                # explicit LLM step
                text = self._default_llm(str(payload.get("prompt") or task.intent))
                step = {"step": i, "engine": "llm", "action": "generate", "ok": True, "output": {"text": text[:1500]}}
                steps.append(step)
                chain.append(
                    "llm",
                    {"tool": "llm", "change": "llm_generate", "chars": len(text)},
                    actor=self.agent_id,
                    model_id=getattr(self.mind, "model_id", None),
                )
                last = step
            else:
                # route unknown engines through mesie node
                res = self.node.run(action, payload)
                step = res.to_dict()
                step["step"] = i
                steps.append(step)
                chain.append("tool", {"tool": engine, "action": action, "ok": res.ok}, actor=self.agent_id, ok=res.ok)
                last = step

        used_llm = False
        escalate_reason = None
        # Hybrid escalation: LLM only when language/planning needed or mesie weak
        need_plan = any(
            k in task.intent.lower()
            for k in ("explain", "plan", "strategy", "write", "why", "summarize", "design")
        )
        if task.allow_llm_escalation and (need_plan or not mesie_ok):
            escalate_reason = "language_or_planning" if need_plan else "mesie_incomplete"
            features = {
                "mesie_steps": len(steps),
                "last_engine": last.get("engine"),
                "spectral": (last.get("output") or {}).get("metrics"),
                "embed_dim": (last.get("output") or {}).get("embedding_dim"),
            }
            prompt = (
                f"[GHOST hybrid escalation — MESIE features first]\n"
                f"Intent: {task.intent}\n"
                f"MESIE features: {features}\n"
                f"Produce a short grounded plan. Label inferences as model_inference.\n"
            )
            text = self._default_llm(prompt)
            steps.append(
                {
                    "step": len(steps),
                    "engine": "llm",
                    "action": "escalate",
                    "ok": True,
                    "output": {"text": text[:1500], "reason": escalate_reason},
                }
            )
            chain.append(
                "llm",
                {
                    "tool": "llm_escalate",
                    "change": "escalation",
                    "reason": escalate_reason,
                    "chars": len(text),
                },
                actor=self.agent_id,
                model_id=getattr(self.mind, "model_id", None),
            )
            used_llm = True
            last = steps[-1]

        # Prefer mesie GhostAgent if available for attestation parity
        try:
            from mesie.agentic.ghost import GhostAgent, TaskSpec

            def _bus(engine: str = "mesie", action: str = "embed", payload: Optional[Dict] = None, **kw: Any) -> Dict:
                return self.node.bus_dispatch(engine, action, payload or kw)

            ga = GhostAgent(agent_id=self.agent_id, bus_dispatch=_bus)
            # already executed; optional parity spawn is skip-heavy — record availability
            chain.append(
                "validate",
                {"check": "mesie.GhostAgent_importable", "ok": True, "change": "parity_ok"},
                actor=self.agent_id,
                ok=True,
            )
            _ = (ga, TaskSpec)  # bound for type presence
        except Exception as exc:
            chain.append(
                "validate",
                {"check": "mesie.GhostAgent_importable", "ok": False, "error": str(exc)[:120]},
                actor=self.agent_id,
                ok=False,
            )

        chain.append(
            "result",
            {
                "ok": mesie_ok or used_llm,
                "used_llm": used_llm,
                "n_steps": len(steps),
                "change": "ghost_complete",
            },
            actor=self.agent_id,
            ok=mesie_ok or used_llm,
        )

        outcome = GhostOutcome(
            ok=mesie_ok or used_llm,
            task_id=task.task_id,
            agent_id=self.agent_id,
            steps=steps,
            result={
                "intent": task.intent,
                "last": last,
                "n_steps": len(steps),
                "hybrid": {"mesie_first": True, "llm": used_llm, "reason": escalate_reason},
            },
            chain=chain,
            used_llm=used_llm,
            escalate_reason=escalate_reason,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        self.history.append(outcome)
        return outcome
