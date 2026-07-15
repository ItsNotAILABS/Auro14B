"""Audit governance — immutable logs, provenance tracking, querying."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Optional


@dataclass
class ImmutableRecord:
    """A single immutable audit record with integrity hash."""

    sequence: int
    event_type: str
    actor: str
    resource: str
    action: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict = field(default_factory=dict)
    previous_hash: str = ""
    record_hash: str = ""

    def compute_hash(self) -> str:
        content = f"{self.sequence}|{self.event_type}|{self.actor}|{self.resource}|{self.action}|{self.timestamp}|{self.details}|{self.previous_hash}"
        return sha256(content.encode()).hexdigest()

    def seal(self) -> None:
        self.record_hash = self.compute_hash()

    def verify_integrity(self) -> bool:
        return self.record_hash == self.compute_hash()


@dataclass
class AuditLog:
    """Append-only audit log with hash chain integrity."""

    records: list = field(default_factory=list)
    name: str = "default"

    def append(self, event_type: str, actor: str, resource: str, action: str, details: Optional[dict] = None) -> ImmutableRecord:
        previous_hash = self.records[-1].record_hash if self.records else "genesis"
        record = ImmutableRecord(
            sequence=len(self.records),
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            details=details or {},
            previous_hash=previous_hash,
        )
        record.seal()
        self.records.append(record)
        return record

    def verify_chain(self) -> bool:
        for i, record in enumerate(self.records):
            if not record.verify_integrity():
                return False
            if i > 0 and record.previous_hash != self.records[i - 1].record_hash:
                return False
        return True

    def length(self) -> int:
        return len(self.records)

    def get_by_actor(self, actor: str) -> list:
        return [r for r in self.records if r.actor == actor]

    def get_by_resource(self, resource: str) -> list:
        return [r for r in self.records if r.resource == resource]


@dataclass
class ProvenanceTracker:
    """Tracks provenance chains for data and computations."""

    chains: dict = field(default_factory=dict)

    def start_chain(self, artifact_id: str, origin: str, metadata: Optional[dict] = None) -> dict:
        chain = {
            "artifact_id": artifact_id,
            "origin": origin,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "transformations": [],
        }
        self.chains[artifact_id] = chain
        return chain

    def add_transformation(self, artifact_id: str, operation: str, actor: str, params: Optional[dict] = None) -> Optional[dict]:
        chain = self.chains.get(artifact_id)
        if chain is None:
            return None
        entry = {
            "operation": operation,
            "actor": actor,
            "params": params or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        chain["transformations"].append(entry)
        return entry

    def get_lineage(self, artifact_id: str) -> Optional[dict]:
        return self.chains.get(artifact_id)

    def lineage_depth(self, artifact_id: str) -> int:
        chain = self.chains.get(artifact_id)
        if chain is None:
            return 0
        return len(chain["transformations"])

    def artifact_count(self) -> int:
        return len(self.chains)


@dataclass
class AuditQuery:
    """Query interface for audit logs."""

    log: AuditLog

    def by_actor(self, actor: str) -> list:
        return self.log.get_by_actor(actor)

    def by_resource(self, resource: str) -> list:
        return self.log.get_by_resource(resource)

    def by_event_type(self, event_type: str) -> list:
        return [r for r in self.log.records if r.event_type == event_type]

    def by_action(self, action: str) -> list:
        return [r for r in self.log.records if r.action == action]

    def count(self) -> int:
        return self.log.length()

    def latest(self, n: int = 10) -> list:
        return self.log.records[-n:]
