"""Ethics governance — fairness, transparency, accountability, harm prevention."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class EthicsFramework:
    """Top-level ethics framework for MESIE system operations."""

    name: str = "MESIE Ethics Framework"
    version: str = "1.0.0"
    principles: list = field(default_factory=lambda: [
        "fairness",
        "transparency",
        "accountability",
        "harm_prevention",
        "privacy",
        "consent",
        "human_oversight",
    ])
    review_frequency_days: int = 90

    def has_principle(self, principle: str) -> bool:
        return principle.lower() in self.principles

    def principle_count(self) -> int:
        return len(self.principles)

    def add_principle(self, principle: str) -> None:
        if principle.lower() not in self.principles:
            self.principles.append(principle.lower())

    def is_compliant(self, checks: dict) -> bool:
        """Check if all required principles are satisfied."""
        for principle in self.principles:
            if principle in checks and not checks[principle]:
                return False
        return True


@dataclass
class FairnessChecker:
    """Checks for fairness in system outputs and decisions."""

    protected_attributes: list = field(default_factory=lambda: [
        "race", "gender", "age", "disability", "religion", "nationality",
    ])
    max_disparity_ratio: float = 0.8
    min_sample_size: int = 30

    def check_disparity(self, group_a_rate: float, group_b_rate: float) -> dict:
        if group_b_rate == 0:
            ratio = float("inf") if group_a_rate > 0 else 1.0
        else:
            ratio = group_a_rate / group_b_rate
        return {
            "ratio": ratio,
            "passes": self.max_disparity_ratio <= ratio <= (1 / self.max_disparity_ratio),
            "threshold": self.max_disparity_ratio,
        }

    def is_attribute_protected(self, attribute: str) -> bool:
        return attribute.lower() in self.protected_attributes

    def evaluate_outcomes(self, outcomes_by_group: dict) -> dict:
        """Evaluate fairness across multiple groups."""
        results = {}
        groups = list(outcomes_by_group.keys())
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                key = f"{g1}_vs_{g2}"
                results[key] = self.check_disparity(
                    outcomes_by_group[g1], outcomes_by_group[g2]
                )
        overall_pass = all(r["passes"] for r in results.values())
        return {"comparisons": results, "overall_pass": overall_pass}


@dataclass
class TransparencyReport:
    """Generates transparency reports for system operations."""

    system_name: str
    reporting_period_start: Optional[datetime] = None
    reporting_period_end: Optional[datetime] = None
    sections: list = field(default_factory=list)

    def add_section(self, title: str, content: str, data: Optional[dict] = None) -> None:
        self.sections.append({
            "title": title,
            "content": content,
            "data": data or {},
            "added_at": datetime.now(timezone.utc).isoformat(),
        })

    def section_count(self) -> int:
        return len(self.sections)

    def generate_summary(self) -> dict:
        return {
            "system": self.system_name,
            "period_start": self.reporting_period_start.isoformat() if self.reporting_period_start else None,
            "period_end": self.reporting_period_end.isoformat() if self.reporting_period_end else None,
            "sections": len(self.sections),
            "section_titles": [s["title"] for s in self.sections],
        }

    def is_complete(self) -> bool:
        return (
            self.reporting_period_start is not None
            and self.reporting_period_end is not None
            and len(self.sections) > 0
        )


@dataclass
class HarmPreventionGate:
    """Gate that blocks operations that could cause harm."""

    categories: list = field(default_factory=lambda: [
        "physical_harm",
        "data_breach",
        "discrimination",
        "misinformation",
        "privacy_violation",
        "environmental_damage",
    ])
    blocked_operations: list = field(default_factory=list)
    override_requires: str = "ethics_officer"

    def evaluate(self, operation: str, risk_scores: dict) -> dict:
        blocked = []
        for category in self.categories:
            score = risk_scores.get(category, 0.0)
            if score > 0.7:
                blocked.append(category)
        result = {
            "operation": operation,
            "allowed": len(blocked) == 0,
            "blocked_categories": blocked,
            "risk_scores": risk_scores,
        }
        if blocked:
            self.blocked_operations.append(result)
        return result

    def block_count(self) -> int:
        return len(self.blocked_operations)

    def is_category_tracked(self, category: str) -> bool:
        return category in self.categories


@dataclass
class AccountabilityLedger:
    """Immutable ledger for accountability tracking."""

    entries: list = field(default_factory=list)
    signers: list = field(default_factory=list)

    def record_decision(
        self,
        decision: str,
        made_by: str,
        rationale: str,
        affected_parties: Optional[list] = None,
    ) -> dict:
        entry = {
            "id": len(self.entries) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "made_by": made_by,
            "rationale": rationale,
            "affected_parties": affected_parties or [],
        }
        self.entries.append(entry)
        return entry

    def get_decisions_by(self, actor: str) -> list:
        return [e for e in self.entries if e["made_by"] == actor]

    def decision_count(self) -> int:
        return len(self.entries)

    def add_signer(self, signer: str) -> None:
        if signer not in self.signers:
            self.signers.append(signer)
