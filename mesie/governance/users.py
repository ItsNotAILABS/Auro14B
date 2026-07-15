"""User governance — roles, access control, multi-tenant, audit trails."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class UserRole(Enum):
    """Standard governance roles for system users."""

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    ANALYST = "analyst"
    ADMINISTRATOR = "administrator"
    AUDITOR = "auditor"
    ETHICS_OFFICER = "ethics_officer"
    SYSTEM_OWNER = "system_owner"

    def can_read(self) -> bool:
        return True

    def can_write(self) -> bool:
        return self in (
            UserRole.CONTRIBUTOR,
            UserRole.ANALYST,
            UserRole.ADMINISTRATOR,
            UserRole.SYSTEM_OWNER,
        )

    def can_admin(self) -> bool:
        return self in (UserRole.ADMINISTRATOR, UserRole.SYSTEM_OWNER)

    def can_audit(self) -> bool:
        return self in (UserRole.AUDITOR, UserRole.ETHICS_OFFICER, UserRole.SYSTEM_OWNER)


@dataclass
class AccessPolicy:
    """Defines access control policies for resources."""

    resource: str
    allowed_roles: list = field(default_factory=list)
    require_mfa: bool = False
    max_session_hours: int = 24
    ip_allowlist: Optional[list] = None
    geo_restrictions: Optional[list] = None

    def grants_access(self, role: UserRole) -> bool:
        if not self.allowed_roles:
            return True
        return role in self.allowed_roles or role.value in self.allowed_roles

    def is_restricted(self) -> bool:
        return bool(self.allowed_roles) or self.require_mfa

    def requires_elevated(self) -> bool:
        return self.require_mfa


@dataclass
class MultiTenantGovernor:
    """Governs multi-tenant separation and resource isolation."""

    tenant_id: str
    namespace: str
    resource_quota: dict = field(default_factory=dict)
    isolation_level: str = "logical"  # logical, physical, hybrid
    data_residency: Optional[str] = None
    cross_tenant_allowed: bool = False

    def is_within_quota(self, resource: str, current_usage: float) -> bool:
        limit = self.resource_quota.get(resource)
        if limit is None:
            return True
        return current_usage <= limit

    def can_access_cross_tenant(self) -> bool:
        return self.cross_tenant_allowed

    def get_namespace(self) -> str:
        return f"{self.tenant_id}/{self.namespace}"

    def quota_utilization(self, resource: str, current_usage: float) -> Optional[float]:
        limit = self.resource_quota.get(resource)
        if limit is None or limit == 0:
            return None
        return current_usage / limit


@dataclass
class UserAuditTrail:
    """Tracks user actions for governance and compliance."""

    user_id: str
    entries: list = field(default_factory=list)

    def log_action(self, action: str, resource: str, details: Optional[dict] = None) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": self.user_id,
            "action": action,
            "resource": resource,
            "details": details or {},
        }
        self.entries.append(entry)
        return entry

    def get_actions(self, action_type: Optional[str] = None) -> list:
        if action_type is None:
            return self.entries
        return [e for e in self.entries if e["action"] == action_type]

    def action_count(self) -> int:
        return len(self.entries)

    def last_action(self) -> Optional[dict]:
        return self.entries[-1] if self.entries else None
