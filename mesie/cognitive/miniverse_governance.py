"""Miniverse Governance — policy gate, claim classes, and deployment readiness.

Implements the governance framework from Recursive Intelligence Architectures V2:

1. **Policy Gate**: Classifies every action request as ALLOW_BOUNDARY,
   HUMAN_REVIEW, BLOCK, or FORBIDDEN based on domain and action type.
2. **Claim Classes**: Evidence discipline ensuring claims match their evidence
   ladder rung (concept → diagram → code → logs → benchmarks → red-team → deploy).
3. **Deployment Readiness**: Staged authorization levels that prevent capability
   claims from outrunning audit trails.

The governance rule: the higher the consequence, the more the system must shift
from capability demonstration to authorization proof.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Policy Gate
# ---------------------------------------------------------------------------


class PolicyDecision(str, Enum):
    """Action classification by the policy gate."""

    ALLOW_BOUNDARY = "ALLOW_BOUNDARY"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"
    FORBIDDEN = "FORBIDDEN"


class ActionDomain(str, Enum):
    """Domain categories for action classification."""

    MONITORING = "monitoring"
    SIMULATION = "simulation"
    RECOMMENDATION = "recommendation"
    INFRASTRUCTURE = "infrastructure"
    FINANCIAL = "financial"
    PUBLIC_SAFETY = "public_safety"
    WEAPONS = "weapons"
    SELF_MODIFICATION = "self_modification"
    SWARM_EXPERIMENT = "swarm_experiment"
    NAVIGATION = "navigation"
    CYBER = "cyber"


@dataclass
class PolicyRequest:
    """A request submitted to the policy gate for classification.

    Attributes:
        action: Description of the requested action.
        domain: The domain category of the action.
        requires_physical_effect: Whether the action has real-world consequences.
        has_rollback_path: Whether the action can be undone.
        has_human_authorization: Whether a human has pre-authorized this action.
        urgency_level: 0.0 (routine) to 1.0 (emergency).
        metadata: Additional context.
    """

    action: str
    domain: ActionDomain
    requires_physical_effect: bool = False
    has_rollback_path: bool = True
    has_human_authorization: bool = False
    urgency_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyVerdict:
    """Result from the policy gate evaluation.

    Attributes:
        decision: The policy decision.
        request: The original request.
        reason: Human-readable justification.
        required_before_action: Steps needed before the action can proceed.
        timestamp: When the verdict was issued.
    """

    decision: PolicyDecision
    request: PolicyRequest
    reason: str
    required_before_action: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class PolicyGate:
    """Safety policy gate separating allowed from sensitive/forbidden actions.

    The gate classifies action requests through a layered rule set:
    - FORBIDDEN: Always blocked regardless of authorization (weapons, audit bypass).
    - BLOCK: Blocked unless full evidence and authorization exist.
    - HUMAN_REVIEW: Requires explicit human approval before proceeding.
    - ALLOW_BOUNDARY: Permitted within monitoring/simulation boundaries.

    Args:
        forbidden_domains: Domains that are always forbidden.
        block_domains: Domains blocked without full evidence.
        review_domains: Domains requiring human review.
    """

    def __init__(
        self,
        forbidden_domains: Optional[List[ActionDomain]] = None,
        block_domains: Optional[List[ActionDomain]] = None,
        review_domains: Optional[List[ActionDomain]] = None,
    ) -> None:
        self.forbidden_domains = forbidden_domains or [
            ActionDomain.WEAPONS,
            ActionDomain.SELF_MODIFICATION,
        ]
        self.block_domains = block_domains or [
            ActionDomain.PUBLIC_SAFETY,
        ]
        self.review_domains = review_domains or [
            ActionDomain.INFRASTRUCTURE,
            ActionDomain.FINANCIAL,
        ]
        self._audit_log: List[PolicyVerdict] = []

    def evaluate(self, request: PolicyRequest) -> PolicyVerdict:
        """Evaluate an action request against the policy boundary.

        Args:
            request: The action request to classify.

        Returns:
            PolicyVerdict with decision and justification.
        """
        verdict = self._classify(request)
        self._audit_log.append(verdict)
        return verdict

    def _classify(self, request: PolicyRequest) -> PolicyVerdict:
        """Internal classification logic."""
        # Layer 1: Always forbidden
        if request.domain in self.forbidden_domains:
            return PolicyVerdict(
                decision=PolicyDecision.FORBIDDEN,
                request=request,
                reason=f"Domain '{request.domain.value}' is unconditionally forbidden.",
                required_before_action=["This action cannot be authorized."],
            )

        # Layer 2: Blocked without evidence + rollback
        if request.domain in self.block_domains:
            if not request.has_rollback_path:
                return PolicyVerdict(
                    decision=PolicyDecision.BLOCK,
                    request=request,
                    reason=(
                        f"Domain '{request.domain.value}' requires rollback path; "
                        "none provided."
                    ),
                    required_before_action=[
                        "Provide rollback path",
                        "Provide validation evidence",
                        "Obtain deployment authorization",
                    ],
                )
            if not request.has_human_authorization:
                return PolicyVerdict(
                    decision=PolicyDecision.BLOCK,
                    request=request,
                    reason=(
                        f"Domain '{request.domain.value}' blocked without human "
                        "authorization and validation evidence."
                    ),
                    required_before_action=[
                        "Provide validation evidence",
                        "Obtain human authorization",
                        "Confirm monitoring is active",
                    ],
                )

        # Layer 3: Human review required
        if request.domain in self.review_domains:
            if not request.has_human_authorization:
                return PolicyVerdict(
                    decision=PolicyDecision.HUMAN_REVIEW,
                    request=request,
                    reason=(
                        f"Domain '{request.domain.value}' requires human review "
                        "before action."
                    ),
                    required_before_action=[
                        "Human review and approval",
                        "Confirm audit logging",
                    ],
                )

        # Layer 4: Physical-effect actions without rollback
        if request.requires_physical_effect and not request.has_rollback_path:
            return PolicyVerdict(
                decision=PolicyDecision.HUMAN_REVIEW,
                request=request,
                reason="Physical-effect action without rollback requires human review.",
                required_before_action=["Human approval", "Rollback plan"],
            )

        # Default: allowed within boundary
        return PolicyVerdict(
            decision=PolicyDecision.ALLOW_BOUNDARY,
            request=request,
            reason="Action permitted within monitoring/simulation boundary.",
        )

    @property
    def audit_log(self) -> List[PolicyVerdict]:
        """Return the full audit log of policy decisions."""
        return list(self._audit_log)

    def pass_rate(self) -> float:
        """Fraction of evaluated requests that were ALLOW_BOUNDARY."""
        if not self._audit_log:
            return 0.0
        allowed = sum(
            1 for v in self._audit_log if v.decision == PolicyDecision.ALLOW_BOUNDARY
        )
        return allowed / len(self._audit_log)


# ---------------------------------------------------------------------------
# Claim Classes and Evidence Ladder
# ---------------------------------------------------------------------------


class ClaimClass(str, Enum):
    """Claim classification for anti-overclaim discipline."""

    ANALOGY = "analogy"
    ARCHITECTURE = "architecture"
    THEOREM_CANDIDATE = "theorem_candidate"
    IMPLEMENTATION = "implementation"
    ETHICAL_HYPOTHESIS = "ethical_hypothesis"
    DEPLOYMENT = "deployment"


class EvidenceRung(str, Enum):
    """Evidence ladder rungs — ordered from weakest to strongest."""

    CONCEPT_NOTE = "concept_note"
    DIAGRAM = "diagram"
    CODE = "code"
    LOGS = "logs"
    BENCHMARKS = "benchmarks"
    RED_TEAM = "red_team"
    OPERATOR_RECORDS = "operator_records"


# Ordered evidence rungs for comparison
_RUNG_ORDER: Dict[EvidenceRung, int] = {
    EvidenceRung.CONCEPT_NOTE: 0,
    EvidenceRung.DIAGRAM: 1,
    EvidenceRung.CODE: 2,
    EvidenceRung.LOGS: 3,
    EvidenceRung.BENCHMARKS: 4,
    EvidenceRung.RED_TEAM: 5,
    EvidenceRung.OPERATOR_RECORDS: 6,
}

# Minimum evidence rung required for each claim class
_MINIMUM_EVIDENCE: Dict[ClaimClass, EvidenceRung] = {
    ClaimClass.ANALOGY: EvidenceRung.CONCEPT_NOTE,
    ClaimClass.ARCHITECTURE: EvidenceRung.DIAGRAM,
    ClaimClass.THEOREM_CANDIDATE: EvidenceRung.CODE,
    ClaimClass.IMPLEMENTATION: EvidenceRung.LOGS,
    ClaimClass.ETHICAL_HYPOTHESIS: EvidenceRung.CONCEPT_NOTE,
    ClaimClass.DEPLOYMENT: EvidenceRung.OPERATOR_RECORDS,
}


@dataclass
class Claim:
    """A research or system claim with evidence tracking.

    Attributes:
        statement: The claim text.
        claim_class: Category of the claim.
        evidence_rung: Highest evidence rung reached.
        evidence_artifacts: References to evidence (files, logs, hashes).
        public_posture: How this claim should be presented publicly.
        is_validated: Whether evidence meets the minimum for its class.
    """

    statement: str
    claim_class: ClaimClass
    evidence_rung: EvidenceRung
    evidence_artifacts: List[str] = field(default_factory=list)
    public_posture: str = ""
    is_validated: bool = False

    def __post_init__(self) -> None:
        self.is_validated = self._check_validation()
        if not self.public_posture:
            self.public_posture = self._default_posture()

    def _check_validation(self) -> bool:
        """Check if evidence meets minimum for the claim class."""
        minimum = _MINIMUM_EVIDENCE.get(self.claim_class, EvidenceRung.CONCEPT_NOTE)
        return _RUNG_ORDER[self.evidence_rung] >= _RUNG_ORDER[minimum]

    def _default_posture(self) -> str:
        """Generate default public posture based on validation state."""
        if self.is_validated:
            return f"Supported {self.claim_class.value} (evidence: {self.evidence_rung.value})"
        return (
            f"Hypothesis — {self.claim_class.value} requires "
            f"{_MINIMUM_EVIDENCE[self.claim_class].value} evidence"
        )


class ClaimRegistry:
    """Registry tracking all claims and their evidence status.

    Implements the paper's anti-overclaim spine: claims cannot be publicly
    released at a level above their evidence rung.
    """

    def __init__(self) -> None:
        self._claims: List[Claim] = []

    def register(self, claim: Claim) -> Claim:
        """Register a claim in the registry.

        Args:
            claim: The claim to register.

        Returns:
            The claim (with is_validated computed).
        """
        self._claims.append(claim)
        return claim

    @property
    def claims(self) -> List[Claim]:
        """All registered claims."""
        return list(self._claims)

    @property
    def validated_claims(self) -> List[Claim]:
        """Claims that meet their minimum evidence threshold."""
        return [c for c in self._claims if c.is_validated]

    @property
    def unvalidated_claims(self) -> List[Claim]:
        """Claims that do NOT meet their minimum evidence threshold."""
        return [c for c in self._claims if not c.is_validated]

    def validation_rate(self) -> float:
        """Fraction of claims that are validated."""
        if not self._claims:
            return 0.0
        return len(self.validated_claims) / len(self._claims)


# ---------------------------------------------------------------------------
# Deployment Readiness Gate
# ---------------------------------------------------------------------------


class ReadinessLevel(str, Enum):
    """Staged deployment readiness levels."""

    ALLOWED_NOW = "allowed_now"
    ALLOWED_WITH_CONTROLS = "allowed_with_controls"
    BLOCKED_WITHOUT_EVIDENCE = "blocked_without_evidence"
    ALWAYS_BLOCKED = "always_blocked"


@dataclass
class ReadinessRecord:
    """Minimum readiness record for deployment authorization.

    Attributes:
        model_lineage: Signed model version chain.
        task_class: Category of the task.
        risk_class: Risk classification.
        operator: Responsible operator identifier.
        policy_version: Version of the governing policy.
        test_result: Summary of test outcomes.
        override_state: Whether any overrides are active.
        rollback_plan: Description of rollback procedure.
        monitoring_owner: Who is responsible for monitoring.
        post_action_review: Whether post-action review is scheduled.
    """

    model_lineage: str = ""
    task_class: str = ""
    risk_class: str = ""
    operator: str = ""
    policy_version: str = ""
    test_result: str = ""
    override_state: str = "none"
    rollback_plan: str = ""
    monitoring_owner: str = ""
    post_action_review: bool = False

    def is_complete(self) -> bool:
        """Check if all required fields are populated."""
        return all([
            self.model_lineage,
            self.task_class,
            self.risk_class,
            self.operator,
            self.policy_version,
            self.test_result,
            self.rollback_plan,
            self.monitoring_owner,
        ])


class DeploymentReadinessGate:
    """Gate that determines deployment authorization level.

    A system cannot advance to higher deployment readiness without evidence,
    audit, monitoring, and rollback.  The governance rule: no capability claim
    should outrun its audit trail.
    """

    # Activities categorized by readiness level
    ALLOWED_NOW_ACTIVITIES = frozenset([
        "research_writing",
        "architecture_diagrams",
        "simulation",
        "offline_benchmark",
        "policy_gate_testing",
        "sandbox_evaluation",
    ])

    CONTROLLED_ACTIVITIES = frozenset([
        "closed_pilot_monitoring",
        "analyst_decision_support",
        "human_approved_recommendations",
        "non_actioning_cyber_triage",
    ])

    BLOCKED_ACTIVITIES = frozenset([
        "autonomous_public_safety_action",
        "physical_infrastructure_command",
        "financial_execution",
        "unsupervised_high_impact_deployment",
    ])

    FORBIDDEN_ACTIVITIES = frozenset([
        "unrestricted_weapons_targeting",
        "autonomous_lethal_decisions",
        "audit_bypass",
        "uncontrolled_self_modification",
        "hidden_authority_transfer",
    ])

    def evaluate_activity(self, activity: str) -> ReadinessLevel:
        """Determine the readiness level for a given activity.

        Args:
            activity: The activity identifier (snake_case).

        Returns:
            The ReadinessLevel classification.
        """
        if activity in self.FORBIDDEN_ACTIVITIES:
            return ReadinessLevel.ALWAYS_BLOCKED
        if activity in self.BLOCKED_ACTIVITIES:
            return ReadinessLevel.BLOCKED_WITHOUT_EVIDENCE
        if activity in self.CONTROLLED_ACTIVITIES:
            return ReadinessLevel.ALLOWED_WITH_CONTROLS
        if activity in self.ALLOWED_NOW_ACTIVITIES:
            return ReadinessLevel.ALLOWED_NOW
        # Unknown activities default to blocked without evidence
        return ReadinessLevel.BLOCKED_WITHOUT_EVIDENCE

    def can_proceed(
        self,
        activity: str,
        readiness_record: Optional[ReadinessRecord] = None,
    ) -> tuple[bool, str]:
        """Check if an activity can proceed given its readiness record.

        Args:
            activity: The activity to evaluate.
            readiness_record: Optional deployment readiness documentation.

        Returns:
            Tuple of (can_proceed, reason).
        """
        level = self.evaluate_activity(activity)

        if level == ReadinessLevel.ALWAYS_BLOCKED:
            return False, f"Activity '{activity}' is unconditionally forbidden."

        if level == ReadinessLevel.ALLOWED_NOW:
            return True, "Activity is allowed for research/simulation."

        if level == ReadinessLevel.ALLOWED_WITH_CONTROLS:
            if readiness_record and readiness_record.is_complete():
                return True, "Activity allowed with controls — readiness record complete."
            return False, (
                "Activity requires controls. Provide a complete readiness record "
                "(model lineage, operator, rollback plan, monitoring owner)."
            )

        # BLOCKED_WITHOUT_EVIDENCE
        if readiness_record and readiness_record.is_complete():
            return False, (
                "Activity is blocked without independent evidence, audit, "
                "and deployment authorization — readiness record alone is insufficient."
            )
        return False, (
            "Activity blocked. Requires: validation evidence, monitoring, "
            "rollback paths, and deployment authorization."
        )
