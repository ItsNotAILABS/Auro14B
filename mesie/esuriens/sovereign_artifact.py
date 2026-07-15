"""Sovereign Artifact — archived completed task chains."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskChain:
    """An ordered chain of linked tasks representing a completed workflow.

    Attributes:
        chain_id: Unique identifier for the chain.
        task_ids: Ordered list of task IDs forming the chain.
        root_task_id: The originating task.
        horizon_levels: Set of horizon levels spanned by this chain.
        created_at: When the chain was first started.
        completed_at: When the final task in the chain completed.
        total_duration: Total elapsed time from creation to completion.
    """

    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_ids: List[str] = field(default_factory=list)
    root_task_id: Optional[str] = None
    horizon_levels: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "task_ids": self.task_ids,
            "root_task_id": self.root_task_id,
            "horizon_levels": self.horizon_levels,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_duration": self.total_duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskChain":
        """Deserialize chain from dictionary."""
        return cls(
            chain_id=data["chain_id"],
            task_ids=data.get("task_ids", []),
            root_task_id=data.get("root_task_id"),
            horizon_levels=data.get("horizon_levels", []),
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at"),
            total_duration=data.get("total_duration", 0.0),
        )


@dataclass
class SovereignArtifact:
    """A sovereign artifact representing an archived task chain.

    Sovereign artifacts are immutable records of completed task chains
    that persist in the Memory Temple as permanent knowledge.

    Attributes:
        artifact_id: Unique identifier.
        chain: The completed task chain.
        sovereignty_level: Level of authority (0=basic, 1=standard, 2=sovereign).
        digest: Content hash for integrity verification.
        insights: Extracted insights or learnings from the chain.
        spectral_fingerprint: Combined spectral signature of all tasks.
        archived_at: When the artifact was archived.
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chain: TaskChain = field(default_factory=TaskChain)
    sovereignty_level: int = 0
    digest: str = ""
    insights: List[str] = field(default_factory=list)
    spectral_fingerprint: Optional[List[float]] = None
    archived_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize artifact to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "chain": self.chain.to_dict(),
            "sovereignty_level": self.sovereignty_level,
            "digest": self.digest,
            "insights": self.insights,
            "spectral_fingerprint": self.spectral_fingerprint,
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SovereignArtifact":
        """Deserialize artifact from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            chain=TaskChain.from_dict(data["chain"]),
            sovereignty_level=data.get("sovereignty_level", 0),
            digest=data.get("digest", ""),
            insights=data.get("insights", []),
            spectral_fingerprint=data.get("spectral_fingerprint"),
            archived_at=data.get("archived_at", time.time()),
        )
