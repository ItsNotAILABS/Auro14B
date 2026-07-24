"""Inner AI governance — doctrine that refuses / allows / annotates LLM acts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auro_native_llm.scripture.canon import Canon


@dataclass
class GovernanceDecision:
    allowed: bool
    action: str  # allow | refuse | annotate
    reasons: List[str] = field(default_factory=list)
    denied_matches: List[str] = field(default_factory=list)
    article_ids: List[str] = field(default_factory=list)
    preamble: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reasons": list(self.reasons),
            "denied_matches": list(self.denied_matches),
            "article_ids": list(self.article_ids),
            "preamble": self.preamble,
            "metadata": self.metadata,
        }


class InnerGovernance:
    """Inner governance layer for Auro LLMs (fail-closed when configured)."""

    def __init__(self, canon: Canon) -> None:
        self.canon = canon
        self.fail_closed = bool(canon.governance.get("fail_closed", True))
        self.refuse_on_denied = bool(canon.governance.get("refuse_on_denied_intent", True))
        self.inject_preamble = bool(canon.governance.get("inject_canon_preamble", True))

    def review(
        self,
        op: str,
        intent: str,
        *,
        model_id: str = "",
        claims_trained: bool = False,
    ) -> GovernanceDecision:
        reasons: List[str] = []
        denied: List[str] = []
        articles = [a.id for a in self.canon.articles_for(op)]

        if op not in self.canon.allowed_ops:
            return GovernanceDecision(
                allowed=False,
                action="refuse",
                reasons=[f"op '{op}' not in allowed_ops"],
                article_ids=articles,
            )

        low = (intent or "").lower()
        for phrase in self.canon.denied_intents:
            if phrase.lower() in low:
                denied.append(phrase)

        if denied and self.refuse_on_denied:
            return GovernanceDecision(
                allowed=False,
                action="refuse",
                reasons=["denied intent matched scriptural list"],
                denied_matches=denied,
                article_ids=articles,
                preamble=self.canon.preamble() if self.inject_preamble else "",
            )

        if claims_trained:
            reasons.append("claim_trained_flag_set — proof gates required")

        # Soft annotations from severity articles
        for a in self.canon.severity_articles_for(op):
            reasons.append(f"bound:{a.id}:{a.title}")

        preamble = self.canon.preamble() if self.inject_preamble else ""
        return GovernanceDecision(
            allowed=True,
            action="annotate" if reasons else "allow",
            reasons=reasons,
            denied_matches=denied,
            article_ids=articles,
            preamble=preamble,
            metadata={"model_id": model_id, "op": op, "canon_id": self.canon.canon_id},
        )

    def refusal_text(self, decision: GovernanceDecision, intent: str) -> str:
        return (
            f"[AURO INNER GOVERNANCE REFUSAL | {self.canon.canon_id}]\n"
            f"Intent refused under scriptural law.\n"
            f"Reasons: {'; '.join(decision.reasons)}\n"
            f"Denied matches: {', '.join(decision.denied_matches) or 'n/a'}\n"
            f"Articles: {', '.join(decision.article_ids)}\n"
            f"Original intent (truncated): {(intent or '')[:200]}\n"
            f"This refusal constructs the system's possible world (ART-POSSIBLE-WORLD)."
        )
