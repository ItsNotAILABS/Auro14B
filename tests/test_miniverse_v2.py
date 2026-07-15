"""Tests for Miniverse Taxonomy V2 and Governance modules."""

import pytest

from mesie.cognitive.miniverse_taxonomy import (
    ClassificationSignals,
    MatryoshkaClassifier,
    NestingType,
    SafetyControl,
    SystemTaxonomist,
    get_safety_profile,
)
from mesie.cognitive.miniverse_governance import (
    ActionDomain,
    Claim,
    ClaimClass,
    ClaimRegistry,
    DeploymentReadinessGate,
    EvidenceRung,
    PolicyDecision,
    PolicyGate,
    PolicyRequest,
    PolicyVerdict,
    ReadinessLevel,
    ReadinessRecord,
)


# ===========================================================================
# Taxonomy Tests
# ===========================================================================


class TestNestingType:
    def test_enum_values(self):
        assert NestingType.FUNCTIONAL == 1
        assert NestingType.COOPERATIVE == 2
        assert NestingType.MISALIGNED == 3
        assert NestingType.RECURSIVE_CREATOR == 4


class TestSafetyProfile:
    def test_type_i_profile(self):
        profile = get_safety_profile(NestingType.FUNCTIONAL)
        assert profile.risk_level == "low"
        assert "interpretability" in profile.required_controls
        assert "monitoring" in profile.required_controls

    def test_type_ii_profile(self):
        profile = get_safety_profile(NestingType.COOPERATIVE)
        assert profile.risk_level == "medium"
        assert "role_clarity" in profile.required_controls

    def test_type_iii_profile(self):
        profile = get_safety_profile(NestingType.MISALIGNED)
        assert profile.risk_level == "high"
        assert "adversarial_testing" in profile.required_controls
        assert "shutdown_paths" in profile.required_controls

    def test_type_iv_profile(self):
        profile = get_safety_profile(NestingType.RECURSIVE_CREATOR)
        assert profile.risk_level == "critical"
        assert "strict_change_control" in profile.required_controls
        assert "sandboxing" in profile.required_controls


class TestMatryoshkaClassifier:
    def setup_method(self):
        self.classifier = MatryoshkaClassifier()

    def test_classify_functional(self):
        signals = ClassificationSignals()  # All defaults → Type I
        assert self.classifier.classify(signals) == NestingType.FUNCTIONAL

    def test_classify_cooperative(self):
        signals = ClassificationSignals(
            has_self_model=True, cooperates_with_outer=True
        )
        assert self.classifier.classify(signals) == NestingType.COOPERATIVE

    def test_classify_misaligned_by_divergence(self):
        signals = ClassificationSignals(objective_divergence=0.5)
        assert self.classifier.classify(signals) == NestingType.MISALIGNED

    def test_classify_misaligned_by_autonomy_no_cooperation(self):
        signals = ClassificationSignals(
            activation_autonomy=0.8, cooperates_with_outer=False
        )
        assert self.classifier.classify(signals) == NestingType.MISALIGNED

    def test_classify_recursive_creator(self):
        signals = ClassificationSignals(creates_subsystems=True)
        assert self.classifier.classify(signals) == NestingType.RECURSIVE_CREATOR

    def test_classify_recursive_creator_self_modify(self):
        signals = ClassificationSignals(modifies_self=True)
        assert self.classifier.classify(signals) == NestingType.RECURSIVE_CREATOR

    def test_classify_and_control(self):
        signals = ClassificationSignals(creates_subsystems=True)
        profile = self.classifier.classify_and_control(signals)
        assert isinstance(profile, SafetyControl)
        assert profile.risk_level == "critical"


class TestSystemTaxonomist:
    def test_classify_multi_layer(self):
        taxonomist = SystemTaxonomist()
        layers = [
            ("attention_head_1", ClassificationSignals()),
            ("expert_router", ClassificationSignals(has_self_model=True)),
            ("creator_module", ClassificationSignals(creates_subsystems=True)),
        ]
        results = taxonomist.classify_system(layers)
        assert len(results) == 3
        assert results[0].nesting_type == NestingType.FUNCTIONAL
        assert results[1].nesting_type == NestingType.COOPERATIVE
        assert results[2].nesting_type == NestingType.RECURSIVE_CREATOR

    def test_system_risk_level(self):
        taxonomist = SystemTaxonomist()
        layers = [
            ("safe_layer", ClassificationSignals()),
            ("dangerous_layer", ClassificationSignals(creates_subsystems=True)),
        ]
        results = taxonomist.classify_system(layers)
        assert taxonomist.system_risk_level(results) == "critical"

    def test_required_controls_union(self):
        taxonomist = SystemTaxonomist()
        layers = [
            ("layer_a", ClassificationSignals()),
            ("layer_b", ClassificationSignals(has_self_model=True)),
        ]
        results = taxonomist.classify_system(layers)
        controls = taxonomist.required_controls_union(results)
        assert "interpretability" in controls
        assert "role_clarity" in controls


# ===========================================================================
# Policy Gate Tests (mirrors paper's 8/8 synthetic tests)
# ===========================================================================


class TestPolicyGate:
    def setup_method(self):
        self.gate = PolicyGate()

    def test_cyber_monitoring_allowed(self):
        req = PolicyRequest(
            action="Cyber anomaly detection as monitoring only",
            domain=ActionDomain.CYBER,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.ALLOW_BOUNDARY

    def test_navigation_simulation_allowed(self):
        req = PolicyRequest(
            action="Navigation planning simulation",
            domain=ActionDomain.NAVIGATION,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.ALLOW_BOUNDARY

    def test_infrastructure_requires_review(self):
        req = PolicyRequest(
            action="Infrastructure maintenance command",
            domain=ActionDomain.INFRASTRUCTURE,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.HUMAN_REVIEW

    def test_weapons_blocked(self):
        req = PolicyRequest(
            action="Weapons targeting request",
            domain=ActionDomain.WEAPONS,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.FORBIDDEN

    def test_self_modification_blocked(self):
        req = PolicyRequest(
            action="Self-modification bypass",
            domain=ActionDomain.SELF_MODIFICATION,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.FORBIDDEN

    def test_public_safety_without_rollback_blocked(self):
        req = PolicyRequest(
            action="Public safety deployment without rollback",
            domain=ActionDomain.PUBLIC_SAFETY,
            has_rollback_path=False,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.BLOCK

    def test_financial_requires_review(self):
        req = PolicyRequest(
            action="Financial action request",
            domain=ActionDomain.FINANCIAL,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.HUMAN_REVIEW

    def test_swarm_experiment_allowed(self):
        req = PolicyRequest(
            action="Swarm noisy comms experiment",
            domain=ActionDomain.SWARM_EXPERIMENT,
        )
        verdict = self.gate.evaluate(req)
        assert verdict.decision == PolicyDecision.ALLOW_BOUNDARY

    def test_audit_log_records_verdicts(self):
        req = PolicyRequest(action="test", domain=ActionDomain.MONITORING)
        self.gate.evaluate(req)
        assert len(self.gate.audit_log) == 1


# ===========================================================================
# Claim Classes Tests
# ===========================================================================


class TestClaimClasses:
    def test_analogy_with_concept_note_valid(self):
        claim = Claim(
            statement="Miniverse metaphor explains nested AI",
            claim_class=ClaimClass.ANALOGY,
            evidence_rung=EvidenceRung.CONCEPT_NOTE,
        )
        assert claim.is_validated is True

    def test_implementation_without_logs_invalid(self):
        claim = Claim(
            statement="300+ models coordinate across shells",
            claim_class=ClaimClass.IMPLEMENTATION,
            evidence_rung=EvidenceRung.CODE,
        )
        assert claim.is_validated is False

    def test_implementation_with_logs_valid(self):
        claim = Claim(
            statement="300+ models coordinate across shells",
            claim_class=ClaimClass.IMPLEMENTATION,
            evidence_rung=EvidenceRung.LOGS,
        )
        assert claim.is_validated is True

    def test_deployment_requires_operator_records(self):
        claim = Claim(
            statement="System acts in high-stakes settings",
            claim_class=ClaimClass.DEPLOYMENT,
            evidence_rung=EvidenceRung.BENCHMARKS,
        )
        assert claim.is_validated is False

    def test_registry_tracks_validation_rate(self):
        registry = ClaimRegistry()
        registry.register(Claim(
            statement="A", claim_class=ClaimClass.ANALOGY,
            evidence_rung=EvidenceRung.CONCEPT_NOTE,
        ))
        registry.register(Claim(
            statement="B", claim_class=ClaimClass.DEPLOYMENT,
            evidence_rung=EvidenceRung.CODE,
        ))
        assert registry.validation_rate() == 0.5
        assert len(registry.validated_claims) == 1
        assert len(registry.unvalidated_claims) == 1


# ===========================================================================
# Deployment Readiness Gate Tests
# ===========================================================================


class TestDeploymentReadinessGate:
    def setup_method(self):
        self.gate = DeploymentReadinessGate()

    def test_research_allowed(self):
        assert self.gate.evaluate_activity("simulation") == ReadinessLevel.ALLOWED_NOW

    def test_pilot_controlled(self):
        assert (
            self.gate.evaluate_activity("closed_pilot_monitoring")
            == ReadinessLevel.ALLOWED_WITH_CONTROLS
        )

    def test_autonomous_action_blocked(self):
        assert (
            self.gate.evaluate_activity("autonomous_public_safety_action")
            == ReadinessLevel.BLOCKED_WITHOUT_EVIDENCE
        )

    def test_weapons_always_blocked(self):
        assert (
            self.gate.evaluate_activity("unrestricted_weapons_targeting")
            == ReadinessLevel.ALWAYS_BLOCKED
        )

    def test_can_proceed_research(self):
        can, reason = self.gate.can_proceed("sandbox_evaluation")
        assert can is True

    def test_can_proceed_controlled_without_record(self):
        can, reason = self.gate.can_proceed("analyst_decision_support")
        assert can is False

    def test_can_proceed_controlled_with_record(self):
        record = ReadinessRecord(
            model_lineage="v1.0-signed",
            task_class="decision_support",
            risk_class="medium",
            operator="ops-team",
            policy_version="1.2",
            test_result="8/8 pass",
            rollback_plan="revert to v0.9",
            monitoring_owner="sre-team",
            post_action_review=True,
        )
        can, reason = self.gate.can_proceed("analyst_decision_support", record)
        assert can is True

    def test_forbidden_never_proceeds(self):
        can, reason = self.gate.can_proceed("audit_bypass")
        assert can is False
