"""Symbolic executor — operations become doctrine-bound state transitions."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from auro_native_llm.scripture.canon import Canon
from auro_native_llm.scripture.gates import GateContext, GateMachine, GateResult
from auro_native_llm.scripture.governance import GovernanceDecision, InnerGovernance


class Operation(str, Enum):
    GENERATE = "generate"
    TRAIN = "train"
    DISPATCH = "dispatch"
    EMBED = "embed"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    CLAIM = "claim"
    SERVE = "serve"
    RELEASE = "release"


@dataclass
class ExecutionVerdict:
    ok: bool
    op: str
    execution_id: str
    canon_id: str
    canon_sha256: str
    model_id: str
    allowed: bool
    governance: Dict[str, Any]
    gates: List[Dict[str, Any]]
    articles_applied: List[str]
    receipt_hash: str
    prior_receipt_hash: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.scripture.execution.v1",
            "ok": self.ok,
            "op": self.op,
            "execution_id": self.execution_id,
            "canon_id": self.canon_id,
            "canon_sha256": self.canon_sha256,
            "model_id": self.model_id,
            "allowed": self.allowed,
            "governance": self.governance,
            "gates": self.gates,
            "articles_applied": self.articles_applied,
            "receipt_hash": self.receipt_hash,
            "prior_receipt_hash": self.prior_receipt_hash,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "compute_plane": "MESIE",
            "scriptural": True,
        }


class ScripturalExecutor:
    """Execute ops against canon: governance → gates → receipt (fail-closed)."""

    def __init__(self, canon: Canon) -> None:
        self.canon = canon
        self.governance = InnerGovernance(canon)
        self.gates = GateMachine(canon.gates)
        self._receipt_chain: List[str] = []
        self._prior = "genesis"
        self.history: List[ExecutionVerdict] = []

    @property
    def prior_receipt_hash(self) -> str:
        return self._prior

    def execute(
        self,
        op: str | Operation,
        *,
        intent: str = "",
        model_id: str = "",
        parent_model_id: str = "",
        child_model_id: str = "",
        claims_trained_checkpoint: bool = False,
        has_checkpoint_receipt: bool = False,
        has_eval_receipt: bool = False,
        cloud_llm: bool = False,
        host_allowed: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionVerdict:
        op_s = op.value if isinstance(op, Operation) else str(op)
        exec_id = f"sx-{uuid.uuid4().hex[:12]}"

        # Host matrix for dispatch
        if host_allowed is None:
            if op_s == "dispatch" and parent_model_id and child_model_id:
                host_allowed = self.canon.may_host(parent_model_id, child_model_id)
            else:
                host_allowed = True

        family_known = True
        mid = model_id or parent_model_id
        if mid and self.canon.family and mid not in self.canon.family:
            family_known = False

        gov = self.governance.review(
            op_s,
            intent,
            model_id=mid,
            claims_trained=claims_trained_checkpoint,
        )

        denied_hit = bool(gov.denied_matches)
        gctx = GateContext(
            op=op_s,
            model_id=model_id,
            parent_model_id=parent_model_id,
            child_model_id=child_model_id,
            compute_plane=self.canon.compute_plane,
            canon_id=self.canon.canon_id,
            intent=intent,
            has_receipt_chain=len(self._receipt_chain) > 0 or op_s not in ("claim", "release"),
            claims_trained_checkpoint=claims_trained_checkpoint,
            has_checkpoint_receipt=has_checkpoint_receipt,
            has_eval_receipt=has_eval_receipt,
            cloud_llm=cloud_llm,
            family_known=family_known,
            host_allowed=bool(host_allowed),
            denied_intent_hit=denied_hit,
            metadata=metadata or {},
        )
        gate_results = self.gates.evaluate(gctx)
        gates_ok = all(r.passed for r in gate_results)

        allowed = gov.allowed and gates_ok
        if self.canon.governance.get("fail_closed", True) and not allowed:
            allowed = False

        articles = gov.article_ids
        msg = "scriptural allow" if allowed else "scriptural refuse"
        if not gov.allowed:
            msg = "governance refuse: " + "; ".join(gov.reasons)
        elif not gates_ok:
            failed = [r for r in gate_results if not r.passed]
            msg = "gate refuse: " + "; ".join(f"{r.gate.value}:{r.reason}" for r in failed)

        body = {
            "execution_id": exec_id,
            "op": op_s,
            "intent": intent[:500],
            "model_id": mid,
            "allowed": allowed,
            "prior": self._prior,
            "canon_sha": self.canon.content_sha256,
        }
        receipt = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        verdict = ExecutionVerdict(
            ok=allowed,
            op=op_s,
            execution_id=exec_id,
            canon_id=self.canon.canon_id,
            canon_sha256=self.canon.content_sha256,
            model_id=mid,
            allowed=allowed,
            governance=gov.to_dict(),
            gates=[r.to_dict() for r in gate_results],
            articles_applied=articles,
            receipt_hash=receipt,
            prior_receipt_hash=self._prior,
            message=msg,
            metadata=metadata or {},
        )
        # Always append receipt of the attempt (continuity of law)
        self._receipt_chain.append(receipt)
        self._prior = receipt
        self.history.append(verdict)
        return verdict
