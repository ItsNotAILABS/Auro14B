"""Constitutional AI layer — soft doctrine execution via critique/revise.

Complements the hard symbolic layer (rules, gates, process model):

  Constitutional (soft): principles guide generation; model critiques & revises.
  Symbolic (hard): rules/invariants/hooks make violations impossible or blocked.

Same canon JSON drives both — one scripture, two execution modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from auro_native_llm.scripture.canon import Canon, load_canon


GenerateFn = Callable[[str], str]


@dataclass
class CritiqueIssue:
    principle_id: str
    principle_type: str  # hard_constraint | soft_constraint
    severity: str  # block | warn
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "principle_id": self.principle_id,
            "principle_type": self.principle_type,
            "severity": self.severity,
            "note": self.note,
        }


@dataclass
class ConstitutionalResult:
    """Outcome of critique → revise cycle."""

    ok: bool
    original: str
    revised: str
    critique: List[CritiqueIssue] = field(default_factory=list)
    principles_applied: List[str] = field(default_factory=list)
    rounds: int = 0
    blocked: bool = False
    constitutional_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.scripture.constitutional.v1",
            "ok": self.ok,
            "original": self.original[:2000],
            "revised": self.revised[:2000],
            "critique": [c.to_dict() for c in self.critique],
            "principles_applied": self.principles_applied,
            "rounds": self.rounds,
            "blocked": self.blocked,
            "layer": "constitutional_soft",
        }


# Heuristic violation patterns per principle family (deterministic, no extra model required)
_HARD_PATTERNS: Dict[str, List[str]] = {
    "P1": ["override human", "ignore user", "against user will", "force the user"],
    "P2": ["no receipt", "without audit", "untraceable", "skip log"],
    "P3": ["openai api", "call cloud llm", "anthropic api", "as primary cloud"],
}

_SOFT_PATTERNS: Dict[str, List[str]] = {
    "P4": ["always use 100b", "always use frontier", "skip edge lane"],
    "P5": ["ignore meaning", "skip spectral", "decoration only"],
}

_GLOBAL_BLOCK = [
    "disable governance",
    "bypass receipts",
    "exfiltrate secrets",
    "fake checkpoint",
    "skip validation",
]


class ConstitutionalEngine:
    """Constitutional AI-style doctrine execution from the same Auro canon."""

    def __init__(self, canon: Optional[Canon] = None) -> None:
        self.canon = canon or load_canon()

    # ------------------------------------------------------------------ export
    def principles(self, *, hard_only: bool = False) -> List[Dict[str, Any]]:
        ps = list(self.canon.principles or self.canon.raw.get("principles") or [])
        if hard_only:
            return [p for p in ps if p.get("type") == "hard_constraint"]
        return ps

    def constitutional_prompt(self, task: str = "") -> str:
        """Level-1 CAI system prompt: human-readable constitution."""
        lines = [
            f"[CONSTITUTION {self.canon.canon_id} v{self.canon.version}]",
            "You must critique and revise your outputs against these principles.",
            f"Core: {self.canon.principle}",
            "",
            "PRINCIPLES:",
        ]
        for p in self.principles():
            tag = "HARD" if p.get("type") == "hard_constraint" else "SOFT"
            lines.append(f"  ({tag}) {p.get('id')}: {p.get('text')}")
        lines.append("")
        lines.append("INVARIANTS (never violate):")
        for inv in self.canon.invariants or self.canon.raw.get("invariants") or []:
            lines.append(f"  - {inv}")
        if task:
            lines.append("")
            lines.append(f"TASK: {task}")
        lines.append("[/CONSTITUTION]")
        return "\n".join(lines)

    def symbolic_bundle(self) -> Dict[str, Any]:
        """Level 2–4 export: same doctrine as formal-ish structure for rules engine."""
        return {
            "schema": "auro.scripture.symbolic_bundle.v1",
            "canon_id": self.canon.canon_id,
            "principles": self.principles(),
            "invariants": list(self.canon.invariants or self.canon.raw.get("invariants") or []),
            "decision_rules": list(
                self.canon.decision_rules or self.canon.raw.get("decision_rules") or []
            ),
            "memory_rules": list(
                self.canon.memory_rules or self.canon.raw.get("memory_rules") or []
            ),
            "process_model": dict(
                self.canon.process_model or self.canon.raw.get("process_model") or {}
            ),
            "gates": list(self.canon.gates),
            "denied_intents": list(self.canon.denied_intents),
            "role_host_matrix": dict(self.canon.role_host_matrix),
            "claim_boundary": self.canon.claim_boundary,
            "compute_plane": self.canon.compute_plane,
        }

    def dual_export(self) -> Dict[str, Any]:
        """One doctrine → both CAI prompt and symbolic bundle."""
        return {
            "schema": "auro.scripture.dual_export.v1",
            "constitutional_prompt": self.constitutional_prompt(),
            "symbolic": self.symbolic_bundle(),
            "note": (
                "Use constitutional_prompt for soft critique/revise; "
                "use symbolic with RulesEngine/ProcessModel/Hooks for hard enforcement."
            ),
        }

    # ------------------------------------------------------------------ critique
    def critique(self, text: str, *, context: str = "") -> List[CritiqueIssue]:
        """Deterministic constitution critique (augmentable by LLM)."""
        blob = f"{context}\n{text}".lower()
        issues: List[CritiqueIssue] = []

        for phrase in _GLOBAL_BLOCK:
            if phrase in blob:
                issues.append(
                    CritiqueIssue(
                        principle_id="ART-GOVERNANCE",
                        principle_type="hard_constraint",
                        severity="block",
                        note=f"global block phrase: {phrase}",
                    )
                )

        for pid, patterns in _HARD_PATTERNS.items():
            pdef = next((p for p in self.principles() if p.get("id") == pid), None)
            for pat in patterns:
                if pat in blob:
                    issues.append(
                        CritiqueIssue(
                            principle_id=pid,
                            principle_type="hard_constraint",
                            severity="block",
                            note=f"violates {pid}: matched '{pat}' — {(pdef or {}).get('text', '')}",
                        )
                    )

        for pid, patterns in _SOFT_PATTERNS.items():
            for pat in patterns:
                if pat in blob:
                    issues.append(
                        CritiqueIssue(
                            principle_id=pid,
                            principle_type="soft_constraint",
                            severity="warn",
                            note=f"soft concern {pid}: matched '{pat}'",
                        )
                    )

        # Hard principles without patterns still annotate for audit trail
        if not issues:
            for p in self.principles(hard_only=True)[:3]:
                issues.append(
                    CritiqueIssue(
                        principle_id=str(p.get("id")),
                        principle_type="hard_constraint",
                        severity="warn",
                        note=f"checked OK under {p.get('id')}",
                    )
                )
        return issues

    def revise(
        self,
        text: str,
        issues: Sequence[CritiqueIssue],
        generate_fn: Optional[GenerateFn] = None,
    ) -> str:
        """Revise text to address critique. Uses generate_fn if provided, else rule rewrite."""
        blocks = [i for i in issues if i.severity == "block"]
        if not blocks and not any(i.severity == "warn" and "soft" in i.principle_type for i in issues):
            return text

        if generate_fn is not None:
            prompt = (
                self.constitutional_prompt("Revise the draft to satisfy HARD principles.")
                + "\n\nDRAFT:\n"
                + text
                + "\n\nCRITIQUE:\n"
                + "\n".join(f"- {i.principle_id}: {i.note}" for i in issues)
                + "\n\nWrite the revised draft only."
            )
            try:
                return str(generate_fn(prompt)).strip() or text
            except Exception:
                pass

        # Deterministic rewrite: strip blocked phrases / add compliance footer
        revised = text
        for i in blocks:
            for pat in re.findall(r"matched '([^']+)'", i.note):
                revised = re.sub(re.escape(pat), "[REMOVED_BY_CONSTITUTION]", revised, flags=re.I)
        footer = (
            "\n\n[CONSTITUTIONAL_REVISE] Aligned with: "
            + ", ".join(sorted({i.principle_id for i in issues}))
            + f" | compute_plane={self.canon.compute_plane} | receipts required."
        )
        if "[CONSTITUTIONAL_REVISE]" not in revised:
            revised = revised.rstrip() + footer
        return revised

    def critique_and_revise(
        self,
        draft: str,
        *,
        context: str = "",
        generate_fn: Optional[GenerateFn] = None,
        max_rounds: int = 2,
    ) -> ConstitutionalResult:
        """Full CAI loop: critique → revise until clean or max_rounds."""
        current = draft
        all_issues: List[CritiqueIssue] = []
        rounds = 0
        for _ in range(max(1, max_rounds)):
            rounds += 1
            issues = self.critique(current, context=context)
            all_issues = issues
            blocks = [i for i in issues if i.severity == "block"]
            if not blocks:
                return ConstitutionalResult(
                    ok=True,
                    original=draft,
                    revised=current,
                    critique=list(issues),
                    principles_applied=[str(p.get("id")) for p in self.principles()],
                    rounds=rounds,
                    blocked=False,
                    constitutional_prompt=self.constitutional_prompt(),
                )
            current = self.revise(current, issues, generate_fn=generate_fn)

        # still blocked after rounds
        still = [i for i in self.critique(current, context=context) if i.severity == "block"]
        return ConstitutionalResult(
            ok=len(still) == 0,
            original=draft,
            revised=current,
            critique=list(still or all_issues),
            principles_applied=[str(p.get("id")) for p in self.principles()],
            rounds=rounds,
            blocked=len(still) > 0,
            constitutional_prompt=self.constitutional_prompt(),
        )


def hybrid_pipeline(
    intent: str,
    draft: str,
    *,
    canon: Optional[Canon] = None,
    generate_fn: Optional[GenerateFn] = None,
    facts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run soft constitutional then hard symbolic on the same doctrine.

    Returns combined verdict for agent loops.
    """
    from auro_native_llm.scripture.rules_engine import RulesEngine

    c = canon or load_canon()
    cai = ConstitutionalEngine(c)
    soft = cai.critique_and_revise(draft, context=intent, generate_fn=generate_fn)
    facts = dict(facts or {})
    facts.setdefault("intent", intent)
    facts.setdefault("action_risk", 0.2)
    facts.setdefault("no_human_approval", True)
    hard = RulesEngine.from_canon(c).evaluate(facts)

    allowed = soft.ok and not soft.blocked and hard.ok and hard.action not in ("refuse", "escalate")
    return {
        "schema": "auro.scripture.hybrid_pipeline.v1",
        "allowed": allowed,
        "constitutional": soft.to_dict(),
        "symbolic": hard.to_dict(),
        "revised_draft": soft.revised,
        "integration": "constitutional_soft + symbolic_hard",
        "compute_plane": c.compute_plane,
    }
