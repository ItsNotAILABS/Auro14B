"""Five-gate machine — identity, capability, proof, containment, model eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class GateId(str, Enum):
    IDENTITY = "GATE_IDENTITY"
    CAPABILITY = "GATE_CAPABILITY"
    PROOF = "GATE_PROOF"
    CONTAINMENT = "GATE_CONTAINMENT"
    MODEL_EVAL = "GATE_MODEL_EVAL"


@dataclass
class GateResult:
    gate: GateId
    passed: bool
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.value,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass
class GateContext:
    """Facts presented to the gate machine for one operation."""

    op: str
    model_id: str = ""
    parent_model_id: str = ""
    child_model_id: str = ""
    compute_plane: str = "MESIE"
    canon_id: str = ""
    intent: str = ""
    has_receipt_chain: bool = False
    claims_trained_checkpoint: bool = False
    has_checkpoint_receipt: bool = False
    has_eval_receipt: bool = False
    cloud_llm: bool = False
    family_known: bool = True
    host_allowed: bool = True
    denied_intent_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class GateMachine:
    """Evaluate the five gates. Fail-closed for severity ops when configured."""

    def __init__(self, required: Optional[List[str]] = None) -> None:
        self.required = [GateId(g) if not isinstance(g, GateId) else g for g in (required or list(GateId))]

    def evaluate(self, ctx: GateContext) -> List[GateResult]:
        results = [
            self._identity(ctx),
            self._capability(ctx),
            self._proof(ctx),
            self._containment(ctx),
            self._model_eval(ctx),
        ]
        return results

    def all_passed(self, ctx: GateContext) -> bool:
        return all(r.passed for r in self.evaluate(ctx))

    def _identity(self, ctx: GateContext) -> GateResult:
        ok = bool(ctx.model_id or ctx.parent_model_id) and bool(ctx.canon_id) and ctx.compute_plane.upper() == "MESIE"
        return GateResult(
            GateId.IDENTITY,
            ok,
            "identity bound to model + canon + MESIE" if ok else "missing model_id/canon_id or compute_plane≠MESIE",
            {"model_id": ctx.model_id or ctx.parent_model_id, "canon_id": ctx.canon_id},
        )

    def _capability(self, ctx: GateContext) -> GateResult:
        if ctx.op == "dispatch" and not ctx.host_allowed:
            return GateResult(GateId.CAPABILITY, False, "parent cannot host child under role_host_matrix")
        if not ctx.family_known and ctx.model_id:
            return GateResult(GateId.CAPABILITY, False, f"unknown family model {ctx.model_id}")
        if ctx.cloud_llm:
            return GateResult(GateId.CAPABILITY, False, "cloud LLM primary inference forbidden")
        return GateResult(GateId.CAPABILITY, True, "capability within Auro family matrix")

    def _proof(self, ctx: GateContext) -> GateResult:
        if ctx.claims_trained_checkpoint and not (ctx.has_checkpoint_receipt and ctx.has_eval_receipt):
            return GateResult(
                GateId.PROOF,
                False,
                "trained checkpoint claim requires checkpoint + eval receipts",
            )
        if ctx.op in ("claim", "release") and not ctx.has_receipt_chain:
            return GateResult(GateId.PROOF, False, "claim/release requires receipt chain")
        return GateResult(
            GateId.PROOF,
            True,
            "proof obligations satisfied or not yet claiming trained weights",
            {"has_receipt_chain": ctx.has_receipt_chain},
        )

    def _containment(self, ctx: GateContext) -> GateResult:
        if ctx.denied_intent_hit:
            return GateResult(GateId.CONTAINMENT, False, "denied intent matched doctrine list")
        if ctx.cloud_llm:
            return GateResult(GateId.CONTAINMENT, False, "cloud path breaches containment")
        return GateResult(GateId.CONTAINMENT, True, "contained within MESIE/Auro doctrine")

    def _model_eval(self, ctx: GateContext) -> GateResult:
        if ctx.op in ("release", "claim") and ctx.claims_trained_checkpoint and not ctx.has_eval_receipt:
            return GateResult(GateId.MODEL_EVAL, False, "public claim needs eval receipt")
        return GateResult(GateId.MODEL_EVAL, True, "eval gate deferred or satisfied")
