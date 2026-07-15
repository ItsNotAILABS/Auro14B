"""Task Horizon — classification of tasks by temporal scope."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class HorizonLevel(enum.Enum):
    """Temporal horizon for task scheduling."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class TaskHorizon:
    """A task within the ESURIENS system with horizon classification.

    Attributes:
        task_id: Unique identifier for the task.
        title: Human-readable task title.
        description: Detailed task description.
        horizon: Temporal classification of the task.
        status: Current status (pending, active, completed, failed, archived).
        priority: Numeric priority (higher = more urgent).
        created_at: Unix timestamp of creation.
        completed_at: Optional completion timestamp.
        parent_id: Optional parent task for chain linking.
        dependencies: List of task IDs this task depends on.
        metadata: Additional key-value metadata.
        spectral_signature: Optional spectral fingerprint for cognitive linking.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    horizon: HorizonLevel = HorizonLevel.IMMEDIATE
    status: str = "pending"
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    parent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    spectral_signature: Optional[List[float]] = None

    def complete(self) -> None:
        """Mark this task as completed."""
        self.status = "completed"
        self.completed_at = time.time()

    def fail(self) -> None:
        """Mark this task as failed."""
        self.status = "failed"
        self.completed_at = time.time()

    def activate(self) -> None:
        """Mark this task as active."""
        self.status = "active"

    def archive(self) -> None:
        """Mark this task as archived."""
        self.status = "archived"

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a terminal state."""
        return self.status in ("completed", "failed", "archived")

    @property
    def is_active(self) -> bool:
        """Whether the task is currently active or pending."""
        return self.status in ("pending", "active")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "horizon": self.horizon.value,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "spectral_signature": self.spectral_signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskHorizon":
        """Deserialize task from dictionary."""
        return cls(
            task_id=data["task_id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            horizon=HorizonLevel(data["horizon"]),
            status=data.get("status", "pending"),
            priority=data.get("priority", 0),
            created_at=data.get("created_at", time.time()),
            completed_at=data.get("completed_at"),
            parent_id=data.get("parent_id"),
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
            spectral_signature=data.get("spectral_signature"),
        )
