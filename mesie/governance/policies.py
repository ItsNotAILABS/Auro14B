"""Policy governance — data policies, usage, compliance, retention, consent."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DataPolicy:
    """Defines data governance rules for the system."""

    name: str
    classification: str = "internal"  # public, internal, confidential, restricted
    allowed_purposes: list = field(default_factory=lambda: ["research", "analysis"])
    prohibited_uses: list = field(default_factory=list)
    retention_days: Optional[int] = None
    encryption_required: bool = False
    anonymization_required: bool = False

    def allows_purpose(self, purpose: str) -> bool:
        return purpose in self.allowed_purposes

    def prohibits_use(self, use: str) -> bool:
        return use in self.prohibited_uses

    def is_restricted(self) -> bool:
        return self.classification in ("confidential", "restricted")

    def requires_protection(self) -> bool:
        return self.encryption_required or self.anonymization_required


@dataclass
class UsagePolicy:
    """Defines acceptable use policies for system resources."""

    name: str
    max_requests_per_hour: int = 1000
    max_data_volume_mb: float = 1000.0
    allowed_operations: list = field(default_factory=lambda: ["read", "compute", "export"])
    require_attribution: bool = True
    commercial_use_allowed: bool = True

    def is_within_rate_limit(self, current_requests: int) -> bool:
        return current_requests <= self.max_requests_per_hour

    def is_within_volume_limit(self, current_mb: float) -> bool:
        return current_mb <= self.max_data_volume_mb

    def allows_operation(self, operation: str) -> bool:
        return operation in self.allowed_operations

    def utilization(self, current_requests: int) -> float:
        if self.max_requests_per_hour == 0:
            return float("inf")
        return current_requests / self.max_requests_per_hour


@dataclass
class ComplianceChecker:
    """Validates compliance with governance policies."""

    frameworks: list = field(default_factory=lambda: ["GDPR", "CCPA", "SOC2"])
    checks_passed: list = field(default_factory=list)
    checks_failed: list = field(default_factory=list)

    def check(self, requirement: str, result: bool, details: str = "") -> dict:
        entry = {
            "requirement": requirement,
            "passed": result,
            "details": details,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        if result:
            self.checks_passed.append(entry)
        else:
            self.checks_failed.append(entry)
        return entry

    def is_compliant(self) -> bool:
        return len(self.checks_failed) == 0

    def compliance_score(self) -> float:
        total = len(self.checks_passed) + len(self.checks_failed)
        if total == 0:
            return 1.0
        return len(self.checks_passed) / total

    def supports_framework(self, framework: str) -> bool:
        return framework in self.frameworks


@dataclass
class RetentionPolicy:
    """Defines data retention and deletion schedules."""

    data_type: str
    retention_days: int = 365
    archive_after_days: Optional[int] = None
    delete_after_days: Optional[int] = None
    legal_hold: bool = False

    def should_archive(self, age_days: int) -> bool:
        if self.legal_hold:
            return False
        if self.archive_after_days is None:
            return False
        return age_days >= self.archive_after_days

    def should_delete(self, age_days: int) -> bool:
        if self.legal_hold:
            return False
        if self.delete_after_days is None:
            return False
        return age_days >= self.delete_after_days

    def is_within_retention(self, age_days: int) -> bool:
        return age_days <= self.retention_days

    def days_until_action(self, age_days: int) -> Optional[int]:
        if self.archive_after_days and age_days < self.archive_after_days:
            return self.archive_after_days - age_days
        if self.delete_after_days and age_days < self.delete_after_days:
            return self.delete_after_days - age_days
        return None


@dataclass
class ConsentRegistry:
    """Tracks user consent for data processing activities."""

    consents: dict = field(default_factory=dict)

    def record_consent(
        self, user_id: str, purpose: str, granted: bool, expires_at: Optional[datetime] = None
    ) -> dict:
        key = f"{user_id}:{purpose}"
        entry = {
            "user_id": user_id,
            "purpose": purpose,
            "granted": granted,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
        }
        self.consents[key] = entry
        return entry

    def has_consent(self, user_id: str, purpose: str) -> bool:
        key = f"{user_id}:{purpose}"
        entry = self.consents.get(key)
        if entry is None:
            return False
        return entry["granted"]

    def revoke_consent(self, user_id: str, purpose: str) -> bool:
        key = f"{user_id}:{purpose}"
        if key in self.consents:
            self.consents[key]["granted"] = False
            return True
        return False

    def consent_count(self) -> int:
        return len(self.consents)

    def active_consents(self) -> list:
        return [e for e in self.consents.values() if e["granted"]]
