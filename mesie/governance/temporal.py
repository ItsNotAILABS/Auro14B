"""Temporal governance — versioning, deprecation, time-based policies."""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional


@dataclass
class TemporalPolicy:
    """Defines time-based governance rules for system operations."""

    name: str
    effective_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    effective_until: Optional[datetime] = None
    review_interval_days: int = 90
    auto_expire: bool = False

    def is_active(self, at: Optional[datetime] = None) -> bool:
        now = at or datetime.now(timezone.utc)
        if now < self.effective_from:
            return False
        if self.effective_until and now > self.effective_until:
            return False
        return True

    def needs_review(self, at: Optional[datetime] = None) -> bool:
        now = at or datetime.now(timezone.utc)
        elapsed = (now - self.effective_from).days
        return elapsed > 0 and elapsed % self.review_interval_days == 0

    def days_until_expiry(self, at: Optional[datetime] = None) -> Optional[int]:
        if not self.effective_until:
            return None
        now = at or datetime.now(timezone.utc)
        delta = (self.effective_until - now).days
        return max(0, delta)


@dataclass
class VersionGovernor:
    """Governs version lifecycle across the system."""

    current_version: str
    supported_versions: list = field(default_factory=list)
    deprecated_versions: list = field(default_factory=list)
    sunset_versions: list = field(default_factory=list)
    min_support_window_days: int = 180

    def is_supported(self, version: str) -> bool:
        return version in self.supported_versions

    def is_deprecated(self, version: str) -> bool:
        return version in self.deprecated_versions

    def is_sunset(self, version: str) -> bool:
        return version in self.sunset_versions

    def deprecate(self, version: str) -> None:
        if version in self.supported_versions:
            self.supported_versions.remove(version)
        if version not in self.deprecated_versions:
            self.deprecated_versions.append(version)

    def sunset(self, version: str) -> None:
        if version in self.deprecated_versions:
            self.deprecated_versions.remove(version)
        if version not in self.sunset_versions:
            self.sunset_versions.append(version)

    def promote(self, version: str) -> None:
        if version not in self.supported_versions:
            self.supported_versions.append(version)


@dataclass
class DeprecationSchedule:
    """Tracks deprecation timelines for system components."""

    component: str
    deprecated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sunset_at: Optional[datetime] = None
    replacement: Optional[str] = None
    migration_guide: Optional[str] = None

    def days_until_sunset(self, at: Optional[datetime] = None) -> Optional[int]:
        if not self.sunset_at:
            return None
        now = at or datetime.now(timezone.utc)
        return max(0, (self.sunset_at - now).days)

    def is_past_sunset(self, at: Optional[datetime] = None) -> bool:
        if not self.sunset_at:
            return False
        now = at or datetime.now(timezone.utc)
        return now > self.sunset_at

    def has_replacement(self) -> bool:
        return self.replacement is not None
