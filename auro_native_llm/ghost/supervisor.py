"""Supervisory layer (MONDAY-light): intent → policy → ghosts → validate → receipts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from auro_native_llm.ghost.agent import GhostAgentRuntime, GhostTask
from auro_native_llm.ghost.pillars import ClaimKind, GHOST_DOCTRINE, label_claim, pillars_health
from auro_native_llm.ghost.policy import PolicyGate, RiskClass
from auro_native_llm.ghost.receipts import GhostReceiptChain


@dataclass
class SupervisoryResult:
    ok: bool
    intent: str
    risk_class: int
    policy: Dict[str, Any]
    outcome: Dict[str, Any]
    claims: List[Dict[str, Any]]
    receipt_chain: Dict[str, Any]
    haunt_flags: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.ghost.supervisor.v1",
            "ok": self.ok,
            "intent": self.intent,
            "risk_class": self.risk_class,
            "policy": self.policy,
            "outcome": self.outcome,
            "claims": self.claims,
            "receipt_chain": self.receipt_chain,
            "haunt_flags": self.haunt_flags,
            "latency_ms": self.latency_ms,
            "pillars": pillars_health(),
            "flow": (
                "intent → policy_gate → ghost_agents/mesie → "
                "[llm escalate] → validate → receipt_chain"
            ),
        }


class GhostSupervisor:
    """Governed interface — persona cannot weaken policy or evidence rules."""

    def __init__(self, mind: Any = None) -> None:
        self.mind = mind
        self.gate = PolicyGate()
        self.agent = GhostAgentRuntime(mind)
        self._orphan_count = 0

    def run(
        self,
        intent: str,
        *,
        requested_scopes: Optional[List[str]] = None,
        human_approved: bool = False,
        actions: Optional[List[Dict[str, Any]]] = None,
        offline: bool = True,
    ) -> SupervisoryResult:
        t0 = time.perf_counter()
        chain = GhostReceiptChain()
        chain.append(
            "intent",
            {
                "initiator": "operator",
                "intent": intent[:1000],
                "change": "intent_received",
            },
            actor="operator",
            model_id=getattr(self.mind, "model_id", None),
        )

        decision = self.gate.decide(
            intent,
            requested_scopes=requested_scopes,
            human_approved=human_approved,
            offline=offline,
        )
        chain.append(
            "policy",
            {
                **decision.to_dict(),
                "change": "policy_decision",
            },
            actor="ghost.policy",
            ok=decision.allowed,
        )

        claims = [
            label_claim(intent[:200], ClaimKind.OPERATOR_FACT, evidence_ref="operator.intent").to_dict()
        ]

        if not decision.allowed:
            chain.append(
                "result",
                {"ok": False, "denied": True, "change": "policy_deny"},
                actor="ghost.policy",
                ok=False,
            )
            return SupervisoryResult(
                ok=False,
                intent=intent,
                risk_class=int(decision.risk_class),
                policy=decision.to_dict(),
                outcome={"denied": True, "reasons": decision.reasons},
                claims=claims,
                receipt_chain=chain.to_dict(),
                haunt_flags=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Build ghost task — MESIE-first default actions by scope
        if actions is None:
            actions = []
            if "mesie_read" in decision.scopes or "embed" in decision.scopes or "mesie_compute" in decision.scopes:
                actions.append({"engine": "mesie", "action": "spectral", "payload": {"text": intent}})
                actions.append({"engine": "mesie", "action": "embed", "payload": {"text": intent}})
            if "mesie_compute" in decision.scopes:
                actions.append({"engine": "mesie", "action": "helix", "payload": {"text": intent}})
            if not actions:
                actions = [{"engine": "mesie", "action": "embed", "payload": {"text": intent}}]

        allow_llm = "generate_local" in decision.scopes or "chat" in decision.scopes
        task = GhostTask(
            intent=intent,
            actions=actions,
            timeout_s=decision.time_limit_s,
            allow_llm_escalation=allow_llm and decision.risk_class <= RiskClass.C3_TOOL_MUTATE,
            metadata={"scopes": decision.scopes, "risk": int(decision.risk_class)},
        )
        outcome = self.agent.execute(task, chain=chain)

        # Grounded claims from steps
        for step in outcome.steps:
            if step.get("engine") in ("mesie_ghost_node", "mesie.helix", "mesie.spectral_fft", "mesie.connectome", "mesie.match_validate", "mesie.intelligence") or str(step.get("engine", "")).startswith("mesie"):
                claims.append(
                    label_claim(
                        f"{step.get('action')}: ok={step.get('ok')}",
                        ClaimKind.DETERMINISTIC_MESIE,
                        evidence_ref=str((step.get("evidence_ref") or step.get("engine"))),
                    ).to_dict()
                )
            if step.get("engine") == "llm":
                claims.append(
                    label_claim(
                        str((step.get("output") or {}).get("text") or "")[:200],
                        ClaimKind.MODEL_INFERENCE,
                        uncertainty="model-inferred-not-attested",
                    ).to_dict()
                )

        # Haunt detector: orphaned = steps without receipts, or LLM without escalation reason when used
        haunt: List[str] = []
        if outcome.used_llm and not outcome.escalate_reason:
            haunt.append("llm_without_escalation_reason")
        if not outcome.chain.verify().get("ok"):
            haunt.append("broken_receipt_chain")
        if not outcome.steps:
            haunt.append("empty_execution")
            self._orphan_count += 1

        chain.append(
            "validate",
            {
                "haunt_flags": haunt,
                "claims_n": len(claims),
                "receipt_verify": outcome.chain.verify(),
                "change": "supervisor_validate",
            },
            actor="ghost.supervisor",
            ok=len(haunt) == 0,
        )

        # Absorb into mind training if present
        if self.mind is not None and getattr(self.mind, "organs", None) and self.mind.organs.trainer:
            try:
                from auro_native_llm.organism.self_train import Experience

                doctrine = "\n".join(GHOST_DOCTRINE[:4])
                self.mind.organs.trainer.absorb(
                    Experience(
                        text=f"GHOST_EXEC\n{intent}\n{doctrine}\nreceipt_tip={outcome.chain.tip_hash}",
                        kind="ghost",
                        model_id=getattr(self.mind, "model_id", "Auro"),
                        reward=0.9 if outcome.ok and not haunt else 0.4,
                        meta={"task_id": outcome.task_id, "haunt": haunt},
                    )
                )
            except Exception:
                pass

        return SupervisoryResult(
            ok=outcome.ok and len(haunt) == 0,
            intent=intent,
            risk_class=int(decision.risk_class),
            policy=decision.to_dict(),
            outcome=outcome.to_dict(),
            claims=claims,
            receipt_chain=outcome.chain.to_dict(),
            haunt_flags=haunt,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


def run_ghost_intent(mind: Any, intent: str, **kw: Any) -> Dict[str, Any]:
    """One-shot supervised ghost execution."""
    return GhostSupervisor(mind).run(intent, **kw).to_dict()
