from __future__ import annotations

from typing import List

from .models import ComputeNode, JobState, TrainingJob
from .registry import NodeRegistry


class InsufficientCapacity(RuntimeError):
    pass


class SovereignScheduler:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def assign(self, job: TrainingJob) -> List[ComputeNode]:
        remaining = job.required_gpus
        selected: List[ComputeNode] = []
        for node in self.registry.eligible(job.required_gpus, job.min_gpu_memory_gb):
            selected.append(node)
            remaining -= node.gpu_count
            if remaining <= 0:
                break
        if remaining > 0:
            raise InsufficientCapacity(
                f"Need {job.required_gpus} GPUs with >= {job.min_gpu_memory_gb}GB each"
            )
        job.assigned_nodes = [node.node_id for node in selected]
        return selected

    def launch_plan(self, job: TrainingJob) -> dict:
        nodes = self.assign(job)
        job.state = JobState.RUNNING
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "dataset_id": job.dataset.dataset_id,
            "nodes": [node.to_dict() for node in nodes],
            "command": job.command,
            "state": job.state.value,
        }
