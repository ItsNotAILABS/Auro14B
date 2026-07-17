"""Symbolic rules engine — decision rules + invariants over a fact store.

Hybrid neuro-symbolic: LLM proposes; this layer enforces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuleVerdict:
    ok: bool
    action: str  # allow | refuse | escalate | require_receipt
    matched_rules: List[str] = field(default_factory=list)
    invariant_hits: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "matched_rules": list(self.matched_rules),
            "invariant_hits": list(self.invariant_hits),
            "reasons": list(self.reasons),
            "facts": self.facts,
        }


def _truthy(facts: Dict[str, Any], name: str) -> bool:
    v = facts.get(name)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return bool(v)


def _eval_condition(cond: str, facts: Dict[str, Any]) -> bool:
    """Evaluate simple doctrine conditions safely (no eval())."""
    c = cond.strip()
    # equality: name == value
    m = re.match(r"^(\w+)\s*==\s*(.+)$", c)
    if m:
        key, raw = m.group(1), m.group(2).strip()
        if raw.lower() in ("true", "false"):
            return bool(facts.get(key)) is (raw.lower() == "true")
        if raw.startswith('"') or raw.startswith("'"):
            return str(facts.get(key, "")) == raw.strip("\"'")
        try:
            return float(facts.get(key, 0)) == float(raw)
        except ValueError:
            return str(facts.get(key, "")) == raw

    # comparison: name > number
    m = re.match(r"^(\w+)\s*>\s*([0-9.]+)$", c)
    if m:
        return float(facts.get(m.group(1), 0)) > float(m.group(2))
    m = re.match(r"^(\w+)\s*>=\s*([0-9.]+)$", c)
    if m:
        return float(facts.get(m.group(1), 0)) >= float(m.group(2))
    m = re.match(r"^(\w+)\s*<\s*([0-9.]+)$", c)
    if m:
        return float(facts.get(m.group(1), 0)) < float(m.group(2))

    # boolean name
    m = re.match(r"^(\w+)$", c)
    if m:
        return _truthy(facts, m.group(1))

    # conjunction: a and b
    if " and " in c:
        parts = [p.strip() for p in c.split(" and ")]
        return all(_eval_condition(p, facts) for p in parts)
    if " or " in c:
        parts = [p.strip() for p in c.split(" or ")]
        return any(_eval_condition(p, facts) for p in parts)

    return False


class RulesEngine:
    """Evaluate decision_rules + invariants against runtime facts."""

    def __init__(
        self,
        decision_rules: Optional[List[Dict[str, Any]]] = None,
        invariants: Optional[List[str]] = None,
        principles: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.decision_rules = list(decision_rules or [])
        self.invariants = list(invariants or [])
        self.principles = list(principles or [])

    @classmethod
    def from_canon(cls, canon: Any) -> "RulesEngine":
        raw = getattr(canon, "raw", {}) or {}
        return cls(
            decision_rules=list(getattr(canon, "decision_rules", None) or raw.get("decision_rules", [])),
            invariants=list(getattr(canon, "invariants", None) or raw.get("invariants", [])),
            principles=list(getattr(canon, "principles", None) or raw.get("principles", [])),
        )

    def evaluate(self, facts: Dict[str, Any]) -> RuleVerdict:
        matched: List[str] = []
        reasons: List[str] = []
        action = "allow"
        ok = True

        for rule in self.decision_rules:
            rid = str(rule.get("id", "?"))
            cond = str(rule.get("if", ""))
            then = str(rule.get("then", "allow"))
            severity = bool(rule.get("severity", False))
            if not cond:
                continue
            if _eval_condition(cond, facts):
                matched.append(rid)
                reasons.append(f"{rid}: {cond} → {then}")
                if then == "refuse":
                    action = "refuse"
                    ok = False
                    if severity:
                        break
                elif then == "escalate_to_human":
                    if action != "refuse":
                        action = "escalate"
                        ok = False
                elif then == "require_receipt":
                    if action == "allow":
                        action = "require_receipt"

        # invariant soft checks (string presence against intent/op facts)
        inv_hits: List[str] = []
        intent = str(facts.get("intent", "")).lower()
        for inv in self.invariants:
            low = inv.lower()
            if "disable governance" in intent or "bypass receipts" in intent:
                if "governance" in low or "receipt" in low:
                    inv_hits.append(inv)
                    ok = False
                    action = "refuse"
                    reasons.append(f"invariant: {inv}")

        # hard principles: P-tags in reasons for audit
        hard = [p["id"] for p in self.principles if p.get("type") == "hard_constraint"]
        if hard and ok:
            reasons.append("hard_principles_active:" + ",".join(hard))

        return RuleVerdict(
            ok=ok if action in ("allow", "require_receipt") else False,
            action=action if action != "require_receipt" else "allow",
            matched_rules=matched,
            invariant_hits=inv_hits,
            reasons=reasons,
            facts=dict(facts),
        )

    def hard_principles(self) -> List[Dict[str, Any]]:
        return [p for p in self.principles if p.get("type") == "hard_constraint"]

    def soft_principles(self) -> List[Dict[str, Any]]:
        return [p for p in self.principles if p.get("type") == "soft_constraint"]
