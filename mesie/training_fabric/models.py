from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List


class NodeRole(str, Enum):
    TRAINER = "trainer"
    DATA = "data"
    EVALUATOR = "evaluator"
    INFERENCE = "inference"
    ORCHESTRATOR = "orchestrator"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    hostname: str
    roles: List[NodeRole]
    gpu_count: int
    gpu_memory_gb: int
    system_memory_gb: int
    storage_free_gb: int
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def total_gpu_memory_gb(self) -> int:
        return self.gpu_count * self.gpu_memory_gb

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["roles"] = [role.value for role in self.roles]
        return data


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    version: str
    root_uri: str
    sha256: str
    token_count: int
    ownership_basis: str
    tokenizer_id: str


@dataclass
class TrainingJob:
    job_id: str
    model_id: str
    dataset: DatasetManifest
    required_gpus: int
    min_gpu_memory_gb: int
    command: List[str]
    state: JobState = JobState.QUEUED
    assigned_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
