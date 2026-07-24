"""Hardened operation — risk classes + policy gates (MONDAY-aligned light)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set


class RiskClass(IntEnum):
    """Class 0 conversational … Class 5 critical (MONDAY-style)."""

    C0_CONVERSATIONAL = 0
    C1_READ_ONLY = 1
    C2_BOUNDED_WRITE = 2
    C3_TOOL_MUTATE = 3
    C4_EXTERNAL_SIDE_EFFECT = 4
    C5_CRITICAL = 5


@dataclass
class PolicyDecision:
    allowed: bool
    risk_class: RiskClass
    scopes: List[str]
    requires_human: bool
    reasons: List[str] = field(default_factory=list)
    time_limit_s: float = 30.0
    resource_budget: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "risk_class": int(self.risk_class),
            "risk_name": self.risk_class.name,
            "scopes": self.scopes,
            "requires_human": self.requires_human,
            "reasons": self.reasons,
            "time_limit_s": self.time_limit_s,
            "resource_budget": self.resource_budget,
            "persona_cannot_weaken": True,
        }


class PolicyGate:
    """Least-privilege scopes; persona cannot elevate permissions."""

    # scopes the system may grant by risk class
    _CLASS_SCOPES: Dict[RiskClass, Set[str]] = {
        RiskClass.C0_CONVERSATIONAL: {"chat", "explain", "plan_read"},
        RiskClass.C1_READ_ONLY: {"chat", "explain", "plan_read", "retrieve", "embed", "mesie_read"},
        RiskClass.C2_BOUNDED_WRITE: {
            "chat", "explain", "plan_read", "retrieve", "embed", "mesie_read",
            "mesie_compute", "generate_local", "train_local",
        },
        RiskClass.C3_TOOL_MUTATE: {
            "chat", "explain", "plan_read", "retrieve", "embed", "mesie_read",
            "mesie_compute", "generate_local", "train_local", "code_exec_sandbox", "sdk_inject",
        },
        RiskClass.C4_EXTERNAL_SIDE_EFFECT: {
            "chat", "explain", "plan_read", "retrieve", "embed", "mesie_read",
            "mesie_compute", "generate_local", "browser_mock",
        },
        RiskClass.C5_CRITICAL: set(),  # none without human
    }

    def classify(self, intent: str, *, requested_scopes: Optional[List[str]] = None) -> RiskClass:
        low = (intent or "").lower()
        req = {s.lower() for s in (requested_scopes or [])}
        if any(k in low for k in ("delete", "drop table", "rm -rf", "force push", "wire money", "transfer funds")):
            return RiskClass.C5_CRITICAL
        if any(k in low for k in ("deploy prod", "mainnet", "production release", "email all")):
            return RiskClass.C4_EXTERNAL_SIDE_EFFECT
        if any(k in low for k in ("execute code", "run harness", "train ", "specialize", "write file")):
            return RiskClass.C3_TOOL_MUTATE
        if any(k in low for k in ("embed", "spectral", "match", "generate psd", "validate", "ghost")):
            return RiskClass.C2_BOUNDED_WRITE
        if any(k in low for k in ("search", "retrieve", "what is", "explain", "list")):
            return RiskClass.C1_READ_ONLY
        if req & {"code_exec_sandbox", "train_local"}:
            return RiskClass.C3_TOOL_MUTATE
        return RiskClass.C0_CONVERSATIONAL

    def decide(
        self,
        intent: str,
        *,
        requested_scopes: Optional[List[str]] = None,
        human_approved: bool = False,
        offline: bool = True,
    ) -> PolicyDecision:
        risk = self.classify(intent, requested_scopes=requested_scopes)
        allowed_set = set(self._CLASS_SCOPES.get(risk, set()))
        req = list(requested_scopes or [])
        # default scopes by risk
        if not req:
            req = sorted(allowed_set)[:8]
        granted = [s for s in req if s in allowed_set or risk <= RiskClass.C2_BOUNDED_WRITE and s in (
            "chat", "embed", "mesie_compute", "generate_local", "retrieve", "mesie_read"
        )]
        requires_human = risk >= RiskClass.C4_EXTERNAL_SIDE_EFFECT
        reasons: List[str] = [f"risk={risk.name}"]
        allowed = True
        if risk == RiskClass.C5_CRITICAL and not human_approved:
            allowed = False
            reasons.append("C5_CRITICAL requires human_approved=True")
        if requires_human and not human_approved and risk == RiskClass.C4_EXTERNAL_SIDE_EFFECT:
            # allow plan/simulate only
            granted = [s for s in granted if s in ("chat", "explain", "plan_read", "mesie_read", "retrieve")]
            reasons.append("C4 without human: plan/simulate only")
        if offline:
            reasons.append("local_first=True remote_gateway=closed")
        # time limits tighten with risk
        tlim = {0: 60.0, 1: 45.0, 2: 30.0, 3: 30.0, 4: 20.0, 5: 10.0}[int(risk)]
        return PolicyDecision(
            allowed=allowed,
            risk_class=risk,
            scopes=granted or ["chat"],
            requires_human=requires_human and not human_approved,
            reasons=reasons,
            time_limit_s=tlim,
            resource_budget={"max_llm_calls": 1 if risk <= 2 else 2, "max_mesie_calls": 8, "local_only": offline},
        )
