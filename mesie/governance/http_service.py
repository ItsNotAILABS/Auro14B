"""HTTP Service governance — contracts, rate limits, SLA, API versioning."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ServiceContract:
    """Defines the governance contract for an HTTP service."""

    service_name: str
    version: str
    base_url: str
    endpoints: list = field(default_factory=list)
    authentication_required: bool = True
    max_payload_bytes: int = 10_485_760  # 10MB
    supported_content_types: list = field(default_factory=lambda: ["application/json"])
    cors_origins: list = field(default_factory=lambda: ["*"])

    def add_endpoint(self, method: str, path: str, description: str, auth_required: bool = True) -> dict:
        endpoint = {
            "method": method.upper(),
            "path": path,
            "description": description,
            "auth_required": auth_required,
        }
        self.endpoints.append(endpoint)
        return endpoint

    def endpoint_count(self) -> int:
        return len(self.endpoints)

    def get_endpoint(self, method: str, path: str) -> Optional[dict]:
        for ep in self.endpoints:
            if ep["method"] == method.upper() and ep["path"] == path:
                return ep
        return None

    def supports_content_type(self, content_type: str) -> bool:
        return content_type in self.supported_content_types

    def is_payload_within_limit(self, size_bytes: int) -> bool:
        return size_bytes <= self.max_payload_bytes


@dataclass
class RateLimitPolicy:
    """Defines rate limiting rules for HTTP services."""

    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_limit: int = 50
    per_user: bool = True
    per_ip: bool = True

    def is_within_limit(self, current_count: int, window: str = "minute") -> bool:
        limits = {
            "second": self.requests_per_second,
            "minute": self.requests_per_minute,
            "hour": self.requests_per_hour,
        }
        limit = limits.get(window, self.requests_per_minute)
        return current_count <= limit

    def is_burst_exceeded(self, concurrent: int) -> bool:
        return concurrent > self.burst_limit

    def effective_limit(self, window: str = "minute") -> int:
        limits = {
            "second": self.requests_per_second,
            "minute": self.requests_per_minute,
            "hour": self.requests_per_hour,
        }
        return limits.get(window, self.requests_per_minute)


@dataclass
class SLADefinition:
    """Service Level Agreement definition for HTTP services."""

    service_name: str
    availability_target: float = 99.9  # percentage
    max_response_time_ms: int = 500
    max_error_rate: float = 0.01  # 1%
    support_hours: str = "24x7"
    incident_response_minutes: int = 15
    data_durability: float = 99.999

    def meets_availability(self, actual_availability: float) -> bool:
        return actual_availability >= self.availability_target

    def meets_latency(self, actual_ms: float) -> bool:
        return actual_ms <= self.max_response_time_ms

    def meets_error_rate(self, actual_rate: float) -> bool:
        return actual_rate <= self.max_error_rate

    def is_compliant(self, availability: float, latency_ms: float, error_rate: float) -> dict:
        return {
            "availability": self.meets_availability(availability),
            "latency": self.meets_latency(latency_ms),
            "error_rate": self.meets_error_rate(error_rate),
            "overall": (
                self.meets_availability(availability)
                and self.meets_latency(latency_ms)
                and self.meets_error_rate(error_rate)
            ),
        }


@dataclass
class EndpointGovernance:
    """Governance rules for individual HTTP endpoints."""

    path: str
    method: str
    rate_limit: Optional[RateLimitPolicy] = None
    required_headers: list = field(default_factory=list)
    max_request_body_bytes: int = 1_048_576  # 1MB
    response_cache_seconds: int = 0
    deprecated: bool = False
    sunset_date: Optional[str] = None

    def is_deprecated(self) -> bool:
        return self.deprecated

    def has_cache(self) -> bool:
        return self.response_cache_seconds > 0

    def validates_headers(self, headers: dict) -> dict:
        missing = [h for h in self.required_headers if h not in headers]
        return {"valid": len(missing) == 0, "missing": missing}

    def is_body_too_large(self, size_bytes: int) -> bool:
        return size_bytes > self.max_request_body_bytes


@dataclass
class APIVersionPolicy:
    """Governs API versioning strategy for HTTP services."""

    strategy: str = "url_prefix"  # url_prefix, header, query_param
    current_version: str = "v1"
    supported_versions: list = field(default_factory=lambda: ["v1"])
    deprecated_versions: list = field(default_factory=list)
    version_header: str = "X-API-Version"

    def is_supported(self, version: str) -> bool:
        return version in self.supported_versions

    def is_deprecated(self, version: str) -> bool:
        return version in self.deprecated_versions

    def is_current(self, version: str) -> bool:
        return version == self.current_version

    def add_version(self, version: str) -> None:
        if version not in self.supported_versions:
            self.supported_versions.append(version)

    def deprecate_version(self, version: str) -> None:
        if version in self.supported_versions:
            self.supported_versions.remove(version)
        if version not in self.deprecated_versions:
            self.deprecated_versions.append(version)
