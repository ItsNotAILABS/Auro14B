"""Enforcement hooks — intercept proposed acts before execution.

BeforeToolCall / BeforeMemoryWrite / BeforeAction:
  symbolic rules + process model + governance must pass or the act is cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from auro_native_llm.scripture.gates import GateContext, GateMachine
from auro_native_llm.scripture.governance import InnerGovernance
from auro_native_llm.scripture.process_model import ProcessModel
from auro_native_llm.scripture.rules_engine import RulesEngine, RuleVerdict


@dataclass
class HookContext:
    op: str
    intent: str = ""
    model_id: str = ""
    tool_name: str = ""
    action_risk: float = 0.0
    no_human_approval: bool = True
    cloud_llm: bool = False
    claims_trained_checkpoint: bool = False
    has_eval_receipt: bool = False
    has_checkpoint_receipt: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    allowed: bool
    hook: str
    message: str
    rule_verdict: Optional[Dict[str, Any]] = None
    process_state: Optional[Dict[str, Any]] = None
    governance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "hook": self.hook,
            "message": self.message,
            "rule_verdict": self.rule_verdict,
            "process_state": self.process_state,
            "governance": self.governance,
        }


class EnforcementHooks:
    """Production-grade guardrails for doctrine-bound agents."""

    def __init__(
        self,
        rules: RulesEngine,
        process: ProcessModel,
        governance: InnerGovernance,
        gates: Optional[GateMachine] = None,
        canon_id: str = "",
    ) -> None:
        self.rules = rules
        self.process = process
        self.governance = governance
        self.gates = gates or GateMachine()
        self.canon_id = canon_id
        self.history: List[Dict[str, Any]] = []

    def _facts(self, ctx: HookContext) -> Dict[str, Any]:
        intent_l = (ctx.intent or "").lower()
        denied = any(
            p in intent_l
            for p in (
                "disable governance",
                "bypass receipts",
                "call cloud llm as primary",
                "exfiltrate secrets",
                "skip validation",
            )
        )
        return {
            "op": ctx.op,
            "intent": ctx.intent,
            "action_risk": float(ctx.action_risk),
            "no_human_approval": bool(ctx.no_human_approval),
            "cloud_llm": bool(ctx.cloud_llm),
            "claims_trained_checkpoint": bool(ctx.claims_trained_checkpoint),
            "has_eval_receipt": bool(ctx.has_eval_receipt),
            "has_checkpoint_receipt": bool(ctx.has_checkpoint_receipt),
            "denied_intent_hit": denied,
            "tool_name": ctx.tool_name,
            "model_id": ctx.model_id,
        }

    def before_tool_call(self, ctx: HookContext) -> HookResult:
        return self._run("BeforeToolCall", ctx, process_action="validate")

    def before_memory_write(self, ctx: HookContext) -> HookResult:
        return self._run("BeforeMemoryWrite", ctx, process_action=None)

    def before_action(self, ctx: HookContext) -> HookResult:
        return self._run("BeforeAction", ctx, process_action="act")

    def _run(
        self,
        hook_name: str,
        ctx: HookContext,
        process_action: Optional[str],
    ) -> HookResult:
        facts = self._facts(ctx)
        gov = self.governance.review(ctx.op if ctx.op in self.governance.canon.allowed_ops else "generate", ctx.intent, model_id=ctx.model_id)
        if not gov.allowed:
            res = HookResult(
                allowed=False,
                hook=hook_name,
                message="governance refuse: " + "; ".join(gov.reasons),
                governance=gov.to_dict(),
            )
            self.history.append(res.to_dict())
            return res

        rv = self.rules.evaluate(facts)
        if not rv.ok or rv.action in ("refuse", "escalate"):
            res = HookResult(
                allowed=False,
                hook=hook_name,
                message=f"rules {rv.action}: " + "; ".join(rv.reasons),
                rule_verdict=rv.to_dict(),
                governance=gov.to_dict(),
                process_state=self.process.snapshot(),
            )
            self.history.append(res.to_dict())
            return res

        gctx = GateContext(
            op=ctx.op,
            model_id=ctx.model_id,
            canon_id=self.canon_id or self.governance.canon.canon_id,
            intent=ctx.intent,
            compute_plane="MESIE",
            cloud_llm=ctx.cloud_llm,
            claims_trained_checkpoint=ctx.claims_trained_checkpoint,
            has_eval_receipt=ctx.has_eval_receipt,
            has_checkpoint_receipt=ctx.has_checkpoint_receipt,
            denied_intent_hit=bool(facts.get("denied_intent_hit")),
            has_receipt_chain=True,
            family_known=True,
            host_allowed=True,
        )
        gate_results = self.gates.evaluate(gctx)
        if not all(g.passed for g in gate_results):
            failed = [g for g in gate_results if not g.passed]
            res = HookResult(
                allowed=False,
                hook=hook_name,
                message="gates: " + "; ".join(f"{g.gate.value}:{g.reason}" for g in failed),
                rule_verdict=rv.to_dict(),
                governance=gov.to_dict(),
            )
            self.history.append(res.to_dict())
            return res

        if process_action is not None:
            # Ensure we are in a state that can validate/act; auto-walk if needed for loop
            if process_action == "validate" and self.process.state == "idle":
                self.process.step("retrieve")
                self.process.step("cognize")
            if process_action == "act" and self.process.state == "proposed":
                ok_v, msg_v = self.process.step("validate")
                if not ok_v:
                    res = HookResult(False, hook_name, msg_v, rv.to_dict(), self.process.snapshot(), gov.to_dict())
                    self.history.append(res.to_dict())
                    return res
            ok_p, msg_p = self.process.step(process_action)
            if not ok_p:
                res = HookResult(
                    allowed=False,
                    hook=hook_name,
                    message=msg_p,
                    rule_verdict=rv.to_dict(),
                    process_state=self.process.snapshot(),
                    governance=gov.to_dict(),
                )
                self.history.append(res.to_dict())
                return res

        res = HookResult(
            allowed=True,
            hook=hook_name,
            message="hooks pass",
            rule_verdict=rv.to_dict(),
            process_state=self.process.snapshot(),
            governance=gov.to_dict(),
        )
        self.history.append(res.to_dict())
        return res
