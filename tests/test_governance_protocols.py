"""
Test suite for MESIE Governance Protocols — 200 tests covering:
- Temporal governance (versioning, deprecation, time policies)
- User governance (roles, access, multi-tenant, audit trails)
- Ethics (fairness, transparency, accountability, harm prevention)
- Policies (data, usage, compliance, retention, consent)
- Audit (immutable logs, provenance, integrity)
- HTTP Service governance (contracts, rate limits, SLA, API versioning)
"""

import pytest
from datetime import datetime, timezone, timedelta

from mesie.governance.temporal import TemporalPolicy, VersionGovernor, DeprecationSchedule
from mesie.governance.users import UserRole, AccessPolicy, MultiTenantGovernor, UserAuditTrail
from mesie.governance.ethics import (
    EthicsFramework,
    FairnessChecker,
    TransparencyReport,
    HarmPreventionGate,
    AccountabilityLedger,
)
from mesie.governance.policies import (
    DataPolicy,
    UsagePolicy,
    ComplianceChecker,
    RetentionPolicy,
    ConsentRegistry,
)
from mesie.governance.audit import AuditLog, ProvenanceTracker, ImmutableRecord, AuditQuery
from mesie.governance.http_service import (
    ServiceContract,
    RateLimitPolicy,
    SLADefinition,
    EndpointGovernance,
    APIVersionPolicy,
)


# ============================================================
# TEMPORAL GOVERNANCE TESTS (1-35)
# ============================================================

class TestTemporalPolicy:
    def test_policy_creation(self):
        policy = TemporalPolicy(name="data-retention")
        assert policy.name == "data-retention"
        assert policy.review_interval_days == 90

    def test_policy_active_by_default(self):
        policy = TemporalPolicy(name="default")
        assert policy.is_active() is True

    def test_policy_not_yet_effective(self):
        future = datetime.now(timezone.utc) + timedelta(days=30)
        policy = TemporalPolicy(name="future", effective_from=future)
        assert policy.is_active() is False

    def test_policy_expired(self):
        past = datetime.now(timezone.utc) - timedelta(days=60)
        expired = datetime.now(timezone.utc) - timedelta(days=1)
        policy = TemporalPolicy(name="old", effective_from=past, effective_until=expired)
        assert policy.is_active() is False

    def test_policy_within_window(self):
        past = datetime.now(timezone.utc) - timedelta(days=10)
        future = datetime.now(timezone.utc) + timedelta(days=10)
        policy = TemporalPolicy(name="active", effective_from=past, effective_until=future)
        assert policy.is_active() is True

    def test_days_until_expiry(self):
        future = datetime.now(timezone.utc) + timedelta(days=30)
        policy = TemporalPolicy(name="expiring", effective_until=future)
        days = policy.days_until_expiry()
        assert days is not None
        assert 29 <= days <= 30

    def test_days_until_expiry_none_when_no_expiry(self):
        policy = TemporalPolicy(name="eternal")
        assert policy.days_until_expiry() is None

    def test_auto_expire_flag(self):
        policy = TemporalPolicy(name="auto", auto_expire=True)
        assert policy.auto_expire is True

    def test_review_interval_custom(self):
        policy = TemporalPolicy(name="quarterly", review_interval_days=30)
        assert policy.review_interval_days == 30


class TestVersionGovernor:
    def test_creation(self):
        gov = VersionGovernor(current_version="0.4.0", supported_versions=["0.3.0", "0.4.0"])
        assert gov.current_version == "0.4.0"

    def test_is_supported(self):
        gov = VersionGovernor(current_version="1.0", supported_versions=["0.9", "1.0"])
        assert gov.is_supported("1.0") is True
        assert gov.is_supported("0.8") is False

    def test_deprecate_version(self):
        gov = VersionGovernor(current_version="2.0", supported_versions=["1.0", "2.0"])
        gov.deprecate("1.0")
        assert gov.is_deprecated("1.0") is True
        assert gov.is_supported("1.0") is False

    def test_sunset_version(self):
        gov = VersionGovernor(current_version="2.0", deprecated_versions=["1.0"])
        gov.sunset("1.0")
        assert gov.is_sunset("1.0") is True
        assert gov.is_deprecated("1.0") is False

    def test_promote_version(self):
        gov = VersionGovernor(current_version="2.0", supported_versions=["2.0"])
        gov.promote("2.1")
        assert gov.is_supported("2.1") is True

    def test_full_lifecycle(self):
        gov = VersionGovernor(current_version="1.0", supported_versions=["1.0"])
        gov.promote("2.0")
        gov.deprecate("1.0")
        gov.sunset("1.0")
        assert gov.is_supported("2.0")
        assert gov.is_sunset("1.0")
        assert not gov.is_supported("1.0")

    def test_min_support_window(self):
        gov = VersionGovernor(current_version="1.0", min_support_window_days=180)
        assert gov.min_support_window_days == 180


class TestDeprecationSchedule:
    def test_creation(self):
        dep = DeprecationSchedule(component="old-api")
        assert dep.component == "old-api"

    def test_days_until_sunset(self):
        future = datetime.now(timezone.utc) + timedelta(days=60)
        dep = DeprecationSchedule(component="v1", sunset_at=future)
        days = dep.days_until_sunset()
        assert days is not None
        assert 59 <= days <= 60

    def test_past_sunset(self):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        dep = DeprecationSchedule(component="v0", sunset_at=past)
        assert dep.is_past_sunset() is True

    def test_not_past_sunset(self):
        future = datetime.now(timezone.utc) + timedelta(days=30)
        dep = DeprecationSchedule(component="v1", sunset_at=future)
        assert dep.is_past_sunset() is False

    def test_has_replacement(self):
        dep = DeprecationSchedule(component="v1", replacement="v2")
        assert dep.has_replacement() is True

    def test_no_replacement(self):
        dep = DeprecationSchedule(component="v1")
        assert dep.has_replacement() is False

    def test_migration_guide(self):
        dep = DeprecationSchedule(component="v1", migration_guide="docs/migrate-v2.md")
        assert dep.migration_guide == "docs/migrate-v2.md"

    def test_no_sunset_date_returns_none(self):
        dep = DeprecationSchedule(component="v1")
        assert dep.days_until_sunset() is None


# ============================================================
# USER GOVERNANCE TESTS (36-70)
# ============================================================

class TestUserRole:
    def test_viewer_can_read(self):
        assert UserRole.VIEWER.can_read() is True

    def test_viewer_cannot_write(self):
        assert UserRole.VIEWER.can_write() is False

    def test_contributor_can_write(self):
        assert UserRole.CONTRIBUTOR.can_write() is True

    def test_admin_can_admin(self):
        assert UserRole.ADMINISTRATOR.can_admin() is True

    def test_analyst_cannot_admin(self):
        assert UserRole.ANALYST.can_admin() is False

    def test_auditor_can_audit(self):
        assert UserRole.AUDITOR.can_audit() is True

    def test_contributor_cannot_audit(self):
        assert UserRole.CONTRIBUTOR.can_audit() is False

    def test_system_owner_all_permissions(self):
        role = UserRole.SYSTEM_OWNER
        assert role.can_read() and role.can_write() and role.can_admin() and role.can_audit()

    def test_ethics_officer_can_audit(self):
        assert UserRole.ETHICS_OFFICER.can_audit() is True

    def test_role_values(self):
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.ADMINISTRATOR.value == "administrator"


class TestAccessPolicy:
    def test_creation(self):
        policy = AccessPolicy(resource="spectral-data")
        assert policy.resource == "spectral-data"

    def test_grants_access_empty_roles(self):
        policy = AccessPolicy(resource="public")
        assert policy.grants_access(UserRole.VIEWER) is True

    def test_restricts_access(self):
        policy = AccessPolicy(resource="admin-panel", allowed_roles=[UserRole.ADMINISTRATOR])
        assert policy.grants_access(UserRole.ADMINISTRATOR) is True
        assert policy.grants_access(UserRole.VIEWER) is False

    def test_mfa_required(self):
        policy = AccessPolicy(resource="secrets", require_mfa=True)
        assert policy.requires_elevated() is True

    def test_not_restricted(self):
        policy = AccessPolicy(resource="docs")
        assert policy.is_restricted() is False

    def test_restricted_with_roles(self):
        policy = AccessPolicy(resource="data", allowed_roles=[UserRole.ANALYST])
        assert policy.is_restricted() is True

    def test_session_hours(self):
        policy = AccessPolicy(resource="api", max_session_hours=8)
        assert policy.max_session_hours == 8


class TestMultiTenantGovernor:
    def test_creation(self):
        gov = MultiTenantGovernor(tenant_id="org-1", namespace="research")
        assert gov.tenant_id == "org-1"

    def test_namespace(self):
        gov = MultiTenantGovernor(tenant_id="org-1", namespace="prod")
        assert gov.get_namespace() == "org-1/prod"

    def test_within_quota(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns", resource_quota={"cpu": 100})
        assert gov.is_within_quota("cpu", 50) is True
        assert gov.is_within_quota("cpu", 150) is False

    def test_no_quota_always_within(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns")
        assert gov.is_within_quota("anything", 99999) is True

    def test_cross_tenant_denied_by_default(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns")
        assert gov.can_access_cross_tenant() is False

    def test_cross_tenant_allowed(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns", cross_tenant_allowed=True)
        assert gov.can_access_cross_tenant() is True

    def test_quota_utilization(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns", resource_quota={"mem": 1000})
        assert gov.quota_utilization("mem", 500) == 0.5

    def test_quota_utilization_no_limit(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns")
        assert gov.quota_utilization("cpu", 100) is None

    def test_isolation_level(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns", isolation_level="physical")
        assert gov.isolation_level == "physical"

    def test_data_residency(self):
        gov = MultiTenantGovernor(tenant_id="t1", namespace="ns", data_residency="EU")
        assert gov.data_residency == "EU"


class TestUserAuditTrail:
    def test_creation(self):
        trail = UserAuditTrail(user_id="user-1")
        assert trail.user_id == "user-1"
        assert trail.action_count() == 0

    def test_log_action(self):
        trail = UserAuditTrail(user_id="user-1")
        entry = trail.log_action("read", "dataset-a")
        assert entry["action"] == "read"
        assert entry["user_id"] == "user-1"

    def test_action_count(self):
        trail = UserAuditTrail(user_id="user-1")
        trail.log_action("read", "a")
        trail.log_action("write", "b")
        assert trail.action_count() == 2

    def test_filter_by_action(self):
        trail = UserAuditTrail(user_id="user-1")
        trail.log_action("read", "a")
        trail.log_action("write", "b")
        trail.log_action("read", "c")
        reads = trail.get_actions("read")
        assert len(reads) == 2

    def test_last_action(self):
        trail = UserAuditTrail(user_id="user-1")
        trail.log_action("login", "system")
        trail.log_action("read", "data")
        assert trail.last_action()["action"] == "read"

    def test_last_action_empty(self):
        trail = UserAuditTrail(user_id="user-1")
        assert trail.last_action() is None


# ============================================================
# ETHICS GOVERNANCE TESTS (71-110)
# ============================================================

class TestEthicsFramework:
    def test_creation(self):
        fw = EthicsFramework()
        assert fw.name == "MESIE Ethics Framework"

    def test_has_principle(self):
        fw = EthicsFramework()
        assert fw.has_principle("fairness") is True
        assert fw.has_principle("profit_maximization") is False

    def test_principle_count(self):
        fw = EthicsFramework()
        assert fw.principle_count() == 7

    def test_add_principle(self):
        fw = EthicsFramework()
        fw.add_principle("sustainability")
        assert fw.has_principle("sustainability") is True

    def test_add_duplicate_principle(self):
        fw = EthicsFramework()
        count_before = fw.principle_count()
        fw.add_principle("fairness")
        assert fw.principle_count() == count_before

    def test_is_compliant_all_pass(self):
        fw = EthicsFramework()
        checks = {"fairness": True, "transparency": True}
        assert fw.is_compliant(checks) is True

    def test_is_not_compliant(self):
        fw = EthicsFramework()
        checks = {"fairness": False}
        assert fw.is_compliant(checks) is False

    def test_compliance_unchecked_principles_pass(self):
        fw = EthicsFramework()
        checks = {}  # nothing checked = compliant
        assert fw.is_compliant(checks) is True


class TestFairnessChecker:
    def test_creation(self):
        fc = FairnessChecker()
        assert fc.max_disparity_ratio == 0.8

    def test_check_disparity_passes(self):
        fc = FairnessChecker()
        result = fc.check_disparity(0.9, 1.0)
        assert result["passes"] is True

    def test_check_disparity_fails(self):
        fc = FairnessChecker()
        result = fc.check_disparity(0.5, 1.0)
        assert result["passes"] is False

    def test_protected_attribute(self):
        fc = FairnessChecker()
        assert fc.is_attribute_protected("gender") is True
        assert fc.is_attribute_protected("favorite_color") is False

    def test_evaluate_outcomes_fair(self):
        fc = FairnessChecker()
        outcomes = {"group_a": 0.9, "group_b": 0.95}
        result = fc.evaluate_outcomes(outcomes)
        assert result["overall_pass"] is True

    def test_evaluate_outcomes_unfair(self):
        fc = FairnessChecker()
        outcomes = {"group_a": 0.3, "group_b": 1.0}
        result = fc.evaluate_outcomes(outcomes)
        assert result["overall_pass"] is False

    def test_equal_rates_pass(self):
        fc = FairnessChecker()
        result = fc.check_disparity(1.0, 1.0)
        assert result["passes"] is True

    def test_zero_denominator(self):
        fc = FairnessChecker()
        result = fc.check_disparity(0.5, 0.0)
        assert result["ratio"] == float("inf")


class TestTransparencyReport:
    def test_creation(self):
        tr = TransparencyReport(system_name="MESIE")
        assert tr.system_name == "MESIE"

    def test_add_section(self):
        tr = TransparencyReport(system_name="MESIE")
        tr.add_section("Usage Stats", "Monthly usage data")
        assert tr.section_count() == 1

    def test_generate_summary(self):
        tr = TransparencyReport(system_name="MESIE")
        tr.add_section("Operations", "Ops summary")
        summary = tr.generate_summary()
        assert summary["system"] == "MESIE"
        assert summary["sections"] == 1

    def test_is_complete(self):
        tr = TransparencyReport(
            system_name="MESIE",
            reporting_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            reporting_period_end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        tr.add_section("Summary", "content")
        assert tr.is_complete() is True

    def test_is_incomplete(self):
        tr = TransparencyReport(system_name="MESIE")
        assert tr.is_complete() is False


class TestHarmPreventionGate:
    def test_creation(self):
        gate = HarmPreventionGate()
        assert len(gate.categories) == 6

    def test_allows_safe_operation(self):
        gate = HarmPreventionGate()
        result = gate.evaluate("analyze", {"physical_harm": 0.1, "data_breach": 0.2})
        assert result["allowed"] is True

    def test_blocks_harmful_operation(self):
        gate = HarmPreventionGate()
        result = gate.evaluate("deploy", {"physical_harm": 0.9})
        assert result["allowed"] is False
        assert "physical_harm" in result["blocked_categories"]

    def test_block_count(self):
        gate = HarmPreventionGate()
        gate.evaluate("bad-op", {"discrimination": 0.95})
        assert gate.block_count() == 1

    def test_category_tracked(self):
        gate = HarmPreventionGate()
        assert gate.is_category_tracked("data_breach") is True
        assert gate.is_category_tracked("financial_loss") is False

    def test_multiple_blocks(self):
        gate = HarmPreventionGate()
        result = gate.evaluate("risky", {"physical_harm": 0.8, "data_breach": 0.9})
        assert len(result["blocked_categories"]) == 2


class TestAccountabilityLedger:
    def test_creation(self):
        ledger = AccountabilityLedger()
        assert ledger.decision_count() == 0

    def test_record_decision(self):
        ledger = AccountabilityLedger()
        entry = ledger.record_decision("deploy-v2", "admin", "Performance improvement")
        assert entry["decision"] == "deploy-v2"

    def test_decisions_by_actor(self):
        ledger = AccountabilityLedger()
        ledger.record_decision("d1", "alice", "reason1")
        ledger.record_decision("d2", "bob", "reason2")
        ledger.record_decision("d3", "alice", "reason3")
        assert len(ledger.get_decisions_by("alice")) == 2

    def test_add_signer(self):
        ledger = AccountabilityLedger()
        ledger.add_signer("CEO")
        assert "CEO" in ledger.signers

    def test_no_duplicate_signers(self):
        ledger = AccountabilityLedger()
        ledger.add_signer("CEO")
        ledger.add_signer("CEO")
        assert len(ledger.signers) == 1


# ============================================================
# POLICY GOVERNANCE TESTS (111-145)
# ============================================================

class TestDataPolicy:
    def test_creation(self):
        dp = DataPolicy(name="spectral-records")
        assert dp.name == "spectral-records"

    def test_allows_purpose(self):
        dp = DataPolicy(name="test", allowed_purposes=["research", "analysis"])
        assert dp.allows_purpose("research") is True
        assert dp.allows_purpose("advertising") is False

    def test_prohibits_use(self):
        dp = DataPolicy(name="test", prohibited_uses=["surveillance"])
        assert dp.prohibits_use("surveillance") is True

    def test_is_restricted(self):
        dp = DataPolicy(name="secret", classification="confidential")
        assert dp.is_restricted() is True

    def test_not_restricted(self):
        dp = DataPolicy(name="open", classification="public")
        assert dp.is_restricted() is False

    def test_requires_protection(self):
        dp = DataPolicy(name="pii", encryption_required=True)
        assert dp.requires_protection() is True

    def test_no_protection_needed(self):
        dp = DataPolicy(name="open")
        assert dp.requires_protection() is False


class TestUsagePolicy:
    def test_creation(self):
        up = UsagePolicy(name="standard")
        assert up.max_requests_per_hour == 1000

    def test_within_rate_limit(self):
        up = UsagePolicy(name="test", max_requests_per_hour=100)
        assert up.is_within_rate_limit(50) is True
        assert up.is_within_rate_limit(150) is False

    def test_within_volume_limit(self):
        up = UsagePolicy(name="test", max_data_volume_mb=500)
        assert up.is_within_volume_limit(250) is True
        assert up.is_within_volume_limit(750) is False

    def test_allows_operation(self):
        up = UsagePolicy(name="test", allowed_operations=["read", "compute"])
        assert up.allows_operation("read") is True
        assert up.allows_operation("delete") is False

    def test_utilization(self):
        up = UsagePolicy(name="test", max_requests_per_hour=100)
        assert up.utilization(50) == 0.5

    def test_require_attribution(self):
        up = UsagePolicy(name="academic", require_attribution=True)
        assert up.require_attribution is True


class TestComplianceChecker:
    def test_creation(self):
        cc = ComplianceChecker()
        assert "GDPR" in cc.frameworks

    def test_check_passes(self):
        cc = ComplianceChecker()
        cc.check("data_encryption", True)
        assert cc.is_compliant() is True

    def test_check_fails(self):
        cc = ComplianceChecker()
        cc.check("data_encryption", False, "Not encrypted")
        assert cc.is_compliant() is False

    def test_compliance_score(self):
        cc = ComplianceChecker()
        cc.check("req1", True)
        cc.check("req2", True)
        cc.check("req3", False)
        assert abs(cc.compliance_score() - 2/3) < 0.01

    def test_supports_framework(self):
        cc = ComplianceChecker()
        assert cc.supports_framework("GDPR") is True
        assert cc.supports_framework("PCI-DSS") is False

    def test_empty_compliance_score(self):
        cc = ComplianceChecker()
        assert cc.compliance_score() == 1.0


class TestRetentionPolicy:
    def test_creation(self):
        rp = RetentionPolicy(data_type="logs")
        assert rp.retention_days == 365

    def test_within_retention(self):
        rp = RetentionPolicy(data_type="logs", retention_days=90)
        assert rp.is_within_retention(30) is True
        assert rp.is_within_retention(100) is False

    def test_should_archive(self):
        rp = RetentionPolicy(data_type="logs", archive_after_days=180)
        assert rp.should_archive(200) is True
        assert rp.should_archive(100) is False

    def test_should_delete(self):
        rp = RetentionPolicy(data_type="temp", delete_after_days=30)
        assert rp.should_delete(40) is True
        assert rp.should_delete(20) is False

    def test_legal_hold_prevents_archive(self):
        rp = RetentionPolicy(data_type="evidence", archive_after_days=30, legal_hold=True)
        assert rp.should_archive(100) is False

    def test_legal_hold_prevents_delete(self):
        rp = RetentionPolicy(data_type="evidence", delete_after_days=30, legal_hold=True)
        assert rp.should_delete(100) is False

    def test_days_until_action(self):
        rp = RetentionPolicy(data_type="logs", archive_after_days=90)
        assert rp.days_until_action(60) == 30


class TestConsentRegistry:
    def test_creation(self):
        cr = ConsentRegistry()
        assert cr.consent_count() == 0

    def test_record_consent(self):
        cr = ConsentRegistry()
        cr.record_consent("user-1", "analytics", True)
        assert cr.has_consent("user-1", "analytics") is True

    def test_no_consent(self):
        cr = ConsentRegistry()
        assert cr.has_consent("user-1", "marketing") is False

    def test_revoke_consent(self):
        cr = ConsentRegistry()
        cr.record_consent("user-1", "tracking", True)
        cr.revoke_consent("user-1", "tracking")
        assert cr.has_consent("user-1", "tracking") is False

    def test_active_consents(self):
        cr = ConsentRegistry()
        cr.record_consent("u1", "analytics", True)
        cr.record_consent("u2", "marketing", False)
        assert len(cr.active_consents()) == 1

    def test_consent_count(self):
        cr = ConsentRegistry()
        cr.record_consent("u1", "a", True)
        cr.record_consent("u2", "b", True)
        assert cr.consent_count() == 2


# ============================================================
# AUDIT GOVERNANCE TESTS (146-175)
# ============================================================

class TestImmutableRecord:
    def test_creation(self):
        record = ImmutableRecord(sequence=0, event_type="access", actor="user-1", resource="data", action="read")
        assert record.sequence == 0

    def test_seal_and_verify(self):
        record = ImmutableRecord(sequence=0, event_type="access", actor="user-1", resource="data", action="read")
        record.seal()
        assert record.verify_integrity() is True

    def test_tampered_record_fails_verify(self):
        record = ImmutableRecord(sequence=0, event_type="access", actor="user-1", resource="data", action="read")
        record.seal()
        record.action = "delete"
        assert record.verify_integrity() is False

    def test_hash_not_empty_after_seal(self):
        record = ImmutableRecord(sequence=0, event_type="test", actor="a", resource="b", action="c")
        record.seal()
        assert len(record.record_hash) == 64


class TestAuditLog:
    def test_creation(self):
        log = AuditLog()
        assert log.length() == 0

    def test_append(self):
        log = AuditLog()
        log.append("access", "user-1", "dataset", "read")
        assert log.length() == 1

    def test_chain_integrity(self):
        log = AuditLog()
        log.append("access", "user-1", "data", "read")
        log.append("modify", "user-2", "data", "write")
        log.append("export", "user-1", "data", "export")
        assert log.verify_chain() is True

    def test_chain_tamper_detection(self):
        log = AuditLog()
        log.append("access", "user-1", "data", "read")
        log.append("modify", "user-2", "data", "write")
        log.records[0].action = "tampered"
        assert log.verify_chain() is False

    def test_get_by_actor(self):
        log = AuditLog()
        log.append("a", "alice", "r1", "read")
        log.append("b", "bob", "r2", "write")
        log.append("c", "alice", "r3", "read")
        assert len(log.get_by_actor("alice")) == 2

    def test_get_by_resource(self):
        log = AuditLog()
        log.append("a", "user", "dataset-A", "read")
        log.append("b", "user", "dataset-B", "read")
        assert len(log.get_by_resource("dataset-A")) == 1


class TestProvenanceTracker:
    def test_creation(self):
        pt = ProvenanceTracker()
        assert pt.artifact_count() == 0

    def test_start_chain(self):
        pt = ProvenanceTracker()
        pt.start_chain("artifact-1", "upload")
        assert pt.artifact_count() == 1

    def test_add_transformation(self):
        pt = ProvenanceTracker()
        pt.start_chain("a1", "raw")
        pt.add_transformation("a1", "normalize", "system")
        assert pt.lineage_depth("a1") == 1

    def test_get_lineage(self):
        pt = ProvenanceTracker()
        pt.start_chain("a1", "sensor-data")
        lineage = pt.get_lineage("a1")
        assert lineage["origin"] == "sensor-data"

    def test_nonexistent_artifact(self):
        pt = ProvenanceTracker()
        assert pt.get_lineage("missing") is None
        assert pt.lineage_depth("missing") == 0

    def test_multiple_transformations(self):
        pt = ProvenanceTracker()
        pt.start_chain("a1", "raw")
        pt.add_transformation("a1", "filter", "pipeline")
        pt.add_transformation("a1", "embed", "model")
        pt.add_transformation("a1", "index", "store")
        assert pt.lineage_depth("a1") == 3


class TestAuditQuery:
    def test_by_actor(self):
        log = AuditLog()
        log.append("ev", "alice", "r", "read")
        log.append("ev", "bob", "r", "write")
        query = AuditQuery(log=log)
        assert len(query.by_actor("alice")) == 1

    def test_by_event_type(self):
        log = AuditLog()
        log.append("security", "user", "r", "login")
        log.append("data", "user", "r", "read")
        query = AuditQuery(log=log)
        assert len(query.by_event_type("security")) == 1

    def test_by_action(self):
        log = AuditLog()
        log.append("ev", "user", "r", "read")
        log.append("ev", "user", "r", "write")
        log.append("ev", "user", "r", "read")
        query = AuditQuery(log=log)
        assert len(query.by_action("read")) == 2

    def test_count(self):
        log = AuditLog()
        log.append("a", "u", "r", "x")
        log.append("b", "u", "r", "y")
        query = AuditQuery(log=log)
        assert query.count() == 2

    def test_latest(self):
        log = AuditLog()
        for i in range(20):
            log.append("ev", f"user-{i}", "r", "act")
        query = AuditQuery(log=log)
        latest = query.latest(5)
        assert len(latest) == 5


# ============================================================
# HTTP SERVICE GOVERNANCE TESTS (176-200)
# ============================================================

class TestServiceContract:
    def test_creation(self):
        sc = ServiceContract(service_name="mesie-api", version="0.2.0", base_url="https://api.mesie.dev")
        assert sc.service_name == "mesie-api"

    def test_add_endpoint(self):
        sc = ServiceContract(service_name="api", version="1.0", base_url="/")
        sc.add_endpoint("GET", "/health", "Health check", auth_required=False)
        assert sc.endpoint_count() == 1

    def test_get_endpoint(self):
        sc = ServiceContract(service_name="api", version="1.0", base_url="/")
        sc.add_endpoint("POST", "/v1/validate", "Validate record")
        ep = sc.get_endpoint("POST", "/v1/validate")
        assert ep is not None
        assert ep["description"] == "Validate record"

    def test_content_type_support(self):
        sc = ServiceContract(service_name="api", version="1.0", base_url="/")
        assert sc.supports_content_type("application/json") is True
        assert sc.supports_content_type("text/xml") is False

    def test_payload_limit(self):
        sc = ServiceContract(service_name="api", version="1.0", base_url="/", max_payload_bytes=1024)
        assert sc.is_payload_within_limit(500) is True
        assert sc.is_payload_within_limit(2000) is False


class TestRateLimitPolicy:
    def test_creation(self):
        rl = RateLimitPolicy()
        assert rl.requests_per_minute == 100

    def test_within_limit(self):
        rl = RateLimitPolicy(requests_per_minute=60)
        assert rl.is_within_limit(30, "minute") is True
        assert rl.is_within_limit(70, "minute") is False

    def test_burst_exceeded(self):
        rl = RateLimitPolicy(burst_limit=20)
        assert rl.is_burst_exceeded(25) is True
        assert rl.is_burst_exceeded(15) is False

    def test_effective_limit(self):
        rl = RateLimitPolicy(requests_per_second=5, requests_per_minute=100, requests_per_hour=500)
        assert rl.effective_limit("second") == 5
        assert rl.effective_limit("hour") == 500


class TestSLADefinition:
    def test_creation(self):
        sla = SLADefinition(service_name="mesie-api")
        assert sla.availability_target == 99.9

    def test_meets_availability(self):
        sla = SLADefinition(service_name="api", availability_target=99.9)
        assert sla.meets_availability(99.95) is True
        assert sla.meets_availability(99.0) is False

    def test_meets_latency(self):
        sla = SLADefinition(service_name="api", max_response_time_ms=200)
        assert sla.meets_latency(150) is True
        assert sla.meets_latency(300) is False

    def test_meets_error_rate(self):
        sla = SLADefinition(service_name="api", max_error_rate=0.01)
        assert sla.meets_error_rate(0.005) is True
        assert sla.meets_error_rate(0.05) is False

    def test_is_compliant(self):
        sla = SLADefinition(service_name="api")
        result = sla.is_compliant(99.99, 100, 0.001)
        assert result["overall"] is True

    def test_not_compliant(self):
        sla = SLADefinition(service_name="api")
        result = sla.is_compliant(95.0, 1000, 0.1)
        assert result["overall"] is False


class TestEndpointGovernance:
    def test_creation(self):
        eg = EndpointGovernance(path="/v1/match", method="POST")
        assert eg.path == "/v1/match"

    def test_deprecated(self):
        eg = EndpointGovernance(path="/v0/old", method="GET", deprecated=True)
        assert eg.is_deprecated() is True

    def test_has_cache(self):
        eg = EndpointGovernance(path="/data", method="GET", response_cache_seconds=60)
        assert eg.has_cache() is True

    def test_validates_headers(self):
        eg = EndpointGovernance(path="/api", method="POST", required_headers=["Authorization", "Content-Type"])
        result = eg.validates_headers({"Authorization": "******", "Content-Type": "application/json"})
        assert result["valid"] is True

    def test_missing_headers(self):
        eg = EndpointGovernance(path="/api", method="POST", required_headers=["Authorization"])
        result = eg.validates_headers({"Content-Type": "application/json"})
        assert result["valid"] is False
        assert "Authorization" in result["missing"]

    def test_body_too_large(self):
        eg = EndpointGovernance(path="/upload", method="POST", max_request_body_bytes=1024)
        assert eg.is_body_too_large(2048) is True
        assert eg.is_body_too_large(512) is False


class TestAPIVersionPolicy:
    def test_creation(self):
        avp = APIVersionPolicy()
        assert avp.current_version == "v1"

    def test_is_supported(self):
        avp = APIVersionPolicy(supported_versions=["v1", "v2"])
        assert avp.is_supported("v1") is True
        assert avp.is_supported("v3") is False

    def test_is_current(self):
        avp = APIVersionPolicy(current_version="v2")
        assert avp.is_current("v2") is True
        assert avp.is_current("v1") is False

    def test_deprecate_version(self):
        avp = APIVersionPolicy(supported_versions=["v1", "v2"])
        avp.deprecate_version("v1")
        assert avp.is_deprecated("v1") is True
        assert avp.is_supported("v1") is False

    def test_add_version(self):
        avp = APIVersionPolicy(supported_versions=["v1"])
        avp.add_version("v2")
        assert avp.is_supported("v2") is True


# ============================================================
# INTEGRATION & CROSS-CUTTING GOVERNANCE TESTS (201-232)
# ============================================================

class TestGovernanceIntegration:
    """Cross-cutting tests verifying governance components work together."""

    def test_temporal_policy_with_version_governor(self):
        policy = TemporalPolicy(name="version-support", review_interval_days=180)
        gov = VersionGovernor(current_version="2.0", supported_versions=["1.0", "2.0"])
        assert policy.is_active()
        assert gov.is_supported("1.0")

    def test_access_policy_with_audit_trail(self):
        policy = AccessPolicy(resource="spectral-data", allowed_roles=[UserRole.ANALYST])
        trail = UserAuditTrail(user_id="analyst-1")
        if policy.grants_access(UserRole.ANALYST):
            trail.log_action("access_granted", "spectral-data")
        assert trail.action_count() == 1

    def test_ethics_with_harm_gate(self):
        fw = EthicsFramework()
        gate = HarmPreventionGate()
        result = gate.evaluate("train_model", {"discrimination": 0.2})
        checks = {"fairness": result["allowed"]}
        assert fw.is_compliant(checks) is True

    def test_data_policy_with_consent(self):
        dp = DataPolicy(name="user-analytics", allowed_purposes=["analytics"])
        cr = ConsentRegistry()
        cr.record_consent("user-42", "analytics", True)
        assert dp.allows_purpose("analytics")
        assert cr.has_consent("user-42", "analytics")

    def test_audit_log_with_provenance(self):
        log = AuditLog()
        pt = ProvenanceTracker()
        pt.start_chain("spectrum-001", "sensor")
        log.append("create", "pipeline", "spectrum-001", "ingest")
        pt.add_transformation("spectrum-001", "normalize", "pipeline")
        log.append("transform", "pipeline", "spectrum-001", "normalize")
        assert log.verify_chain()
        assert pt.lineage_depth("spectrum-001") == 1

    def test_service_contract_with_rate_limit(self):
        sc = ServiceContract(service_name="mesie-api", version="1.0", base_url="/")
        rl = RateLimitPolicy(requests_per_minute=100)
        sc.add_endpoint("POST", "/v1/match", "Match spectra")
        assert sc.endpoint_count() == 1
        assert rl.is_within_limit(50, "minute")

    def test_sla_with_endpoint_governance(self):
        sla = SLADefinition(service_name="api", max_response_time_ms=200)
        eg = EndpointGovernance(path="/v1/validate", method="POST", response_cache_seconds=30)
        assert sla.meets_latency(150)
        assert eg.has_cache()

    def test_multi_tenant_with_data_policy(self):
        gov = MultiTenantGovernor(tenant_id="org-a", namespace="prod", resource_quota={"storage_gb": 100})
        dp = DataPolicy(name="tenant-data", classification="confidential")
        assert gov.is_within_quota("storage_gb", 50)
        assert dp.is_restricted()

    def test_compliance_with_retention(self):
        cc = ComplianceChecker(frameworks=["GDPR"])
        rp = RetentionPolicy(data_type="pii", retention_days=730, delete_after_days=730)
        cc.check("retention_compliant", rp.is_within_retention(365))
        assert cc.is_compliant()

    def test_full_governance_pipeline(self):
        """End-to-end: user requests data, governance checks apply."""
        # 1. Check role
        role = UserRole.ANALYST
        policy = AccessPolicy(resource="dataset-X", allowed_roles=[UserRole.ANALYST, UserRole.ADMINISTRATOR])
        assert policy.grants_access(role)

        # 2. Check consent
        consent = ConsentRegistry()
        consent.record_consent("analyst-1", "research", True)
        assert consent.has_consent("analyst-1", "research")

        # 3. Check data policy
        dp = DataPolicy(name="dataset-X", allowed_purposes=["research"])
        assert dp.allows_purpose("research")

        # 4. Ethics gate
        gate = HarmPreventionGate()
        result = gate.evaluate("analyze", {"discrimination": 0.1, "privacy_violation": 0.05})
        assert result["allowed"]

        # 5. Audit
        trail = UserAuditTrail(user_id="analyst-1")
        trail.log_action("data_access", "dataset-X", {"purpose": "research"})
        assert trail.action_count() == 1

    def test_version_deprecation_with_api_policy(self):
        dep = DeprecationSchedule(component="v1-api", replacement="v2-api")
        avp = APIVersionPolicy(current_version="v2", supported_versions=["v1", "v2"])
        assert dep.has_replacement()
        avp.deprecate_version("v1")
        assert avp.is_deprecated("v1")

    def test_transparency_report_generation(self):
        tr = TransparencyReport(
            system_name="MESIE",
            reporting_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            reporting_period_end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        tr.add_section("Governance Summary", "All policies active")
        tr.add_section("Ethics Audit", "No violations detected")
        tr.add_section("Service SLA", "99.95% availability achieved")
        assert tr.is_complete()
        assert tr.section_count() == 3

    def test_accountability_with_audit(self):
        ledger = AccountabilityLedger()
        log = AuditLog()
        decision = ledger.record_decision("enable-feature-X", "admin", "User demand")
        log.append("governance", "admin", "feature-X", "enable")
        assert ledger.decision_count() == 1
        assert log.verify_chain()

    def test_rate_limit_across_windows(self):
        rl = RateLimitPolicy(requests_per_second=10, requests_per_minute=100, requests_per_hour=1000)
        assert rl.is_within_limit(8, "second")
        assert rl.is_within_limit(90, "minute")
        assert rl.is_within_limit(900, "hour")
        assert not rl.is_within_limit(15, "second")

    def test_consent_lifecycle(self):
        cr = ConsentRegistry()
        cr.record_consent("user-1", "marketing", True)
        assert cr.has_consent("user-1", "marketing")
        cr.revoke_consent("user-1", "marketing")
        assert not cr.has_consent("user-1", "marketing")
        cr.record_consent("user-1", "marketing", True)
        assert cr.has_consent("user-1", "marketing")

    def test_immutable_audit_chain_long(self):
        log = AuditLog()
        for i in range(50):
            log.append("event", f"actor-{i % 5}", f"resource-{i % 10}", "action")
        assert log.length() == 50
        assert log.verify_chain()

    def test_fairness_multiple_groups(self):
        fc = FairnessChecker()
        outcomes = {"a": 0.85, "b": 0.90, "c": 0.88}
        result = fc.evaluate_outcomes(outcomes)
        assert result["overall_pass"] is True

    def test_governance_module_imports(self):
        """Verify the governance module exposes all expected classes."""
        from mesie.governance import (
            TemporalPolicy, VersionGovernor, DeprecationSchedule,
            UserRole, AccessPolicy, MultiTenantGovernor, UserAuditTrail,
            EthicsFramework, FairnessChecker, TransparencyReport,
            HarmPreventionGate, AccountabilityLedger,
            DataPolicy, UsagePolicy, ComplianceChecker, RetentionPolicy, ConsentRegistry,
            AuditLog, ProvenanceTracker, ImmutableRecord, AuditQuery,
            ServiceContract, RateLimitPolicy, SLADefinition, EndpointGovernance, APIVersionPolicy,
        )
        assert TemporalPolicy is not None
        assert ServiceContract is not None

    def test_endpoint_sunset_with_deprecation_schedule(self):
        future = datetime.now(timezone.utc) + timedelta(days=90)
        eg = EndpointGovernance(path="/v0/old", method="GET", deprecated=True, sunset_date=future.isoformat())
        dep = DeprecationSchedule(component="/v0/old", sunset_at=future, replacement="/v1/new")
        assert eg.is_deprecated()
        assert dep.has_replacement()
        assert not dep.is_past_sunset()

    def test_data_policy_classification_levels(self):
        for level in ("public", "internal", "confidential", "restricted"):
            dp = DataPolicy(name=f"{level}-data", classification=level)
            if level in ("confidential", "restricted"):
                assert dp.is_restricted()
            else:
                assert not dp.is_restricted()

    def test_multi_tenant_isolation_levels(self):
        for level in ("logical", "physical", "hybrid"):
            gov = MultiTenantGovernor(tenant_id="t", namespace="ns", isolation_level=level)
            assert gov.isolation_level == level

    def test_usage_policy_commercial_flag(self):
        academic = UsagePolicy(name="academic", commercial_use_allowed=False)
        enterprise = UsagePolicy(name="enterprise", commercial_use_allowed=True)
        assert not academic.commercial_use_allowed
        assert enterprise.commercial_use_allowed

    def test_provenance_multi_artifact(self):
        pt = ProvenanceTracker()
        pt.start_chain("a1", "sensor-1")
        pt.start_chain("a2", "sensor-2")
        pt.start_chain("a3", "sensor-3")
        pt.add_transformation("a1", "fft", "engine")
        pt.add_transformation("a2", "psd", "engine")
        assert pt.artifact_count() == 3
        assert pt.lineage_depth("a1") == 1
        assert pt.lineage_depth("a3") == 0

    def test_service_contract_cors(self):
        sc = ServiceContract(service_name="api", version="1.0", base_url="/", cors_origins=["https://mesie.dev"])
        assert "https://mesie.dev" in sc.cors_origins

    def test_audit_query_by_resource(self):
        log = AuditLog()
        log.append("access", "user-1", "dataset-A", "read")
        log.append("access", "user-2", "dataset-B", "read")
        log.append("modify", "user-1", "dataset-A", "write")
        query = AuditQuery(log=log)
        assert len(query.by_resource("dataset-A")) == 2

    def test_ethics_framework_custom(self):
        fw = EthicsFramework(
            name="Custom Framework",
            principles=["fairness", "sustainability", "inclusivity"]
        )
        assert fw.principle_count() == 3
        assert fw.has_principle("sustainability")
        assert not fw.has_principle("transparency")

    def test_sla_partial_compliance(self):
        sla = SLADefinition(service_name="api", availability_target=99.9, max_response_time_ms=200, max_error_rate=0.01)
        result = sla.is_compliant(99.95, 300, 0.005)
        assert result["availability"] is True
        assert result["latency"] is False
        assert result["error_rate"] is True
        assert result["overall"] is False

    def test_retention_no_action_needed(self):
        rp = RetentionPolicy(data_type="logs", retention_days=365, archive_after_days=180, delete_after_days=730)
        assert rp.days_until_action(100) == 80  # 180 - 100

    def test_version_governor_multiple_promotions(self):
        gov = VersionGovernor(current_version="3.0", supported_versions=["3.0"])
        for v in ["3.1", "3.2", "3.3"]:
            gov.promote(v)
        assert len(gov.supported_versions) == 4

    def test_harm_gate_all_safe(self):
        gate = HarmPreventionGate()
        result = gate.evaluate("routine_analysis", {
            "physical_harm": 0.0,
            "data_breach": 0.1,
            "discrimination": 0.05,
            "misinformation": 0.0,
            "privacy_violation": 0.1,
            "environmental_damage": 0.0,
        })
        assert result["allowed"] is True
        assert gate.block_count() == 0

    def test_temporal_policy_review_interval_customizable(self):
        for days in (30, 60, 90, 180, 365):
            policy = TemporalPolicy(name=f"review-{days}", review_interval_days=days)
            assert policy.review_interval_days == days

    def test_user_role_enum_completeness(self):
        expected = {"viewer", "contributor", "analyst", "administrator", "auditor", "ethics_officer", "system_owner"}
        actual = {r.value for r in UserRole}
        assert actual == expected
