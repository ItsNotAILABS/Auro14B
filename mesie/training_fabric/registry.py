from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from .models import ComputeNode, NodeRole


class NodeRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.nodes: Dict[str, ComputeNode] = {}
        if self.path.exists():
            self.load()

    def register(self, node: ComputeNode) -> None:
        if node.gpu_count < 0 or node.gpu_memory_gb < 0:
            raise ValueError("GPU values must be non-negative")
        self.nodes[node.node_id] = node
        self.save()

    def eligible(self, required_gpus: int, min_gpu_memory_gb: int) -> Iterable[ComputeNode]:
        del required_gpus
        return sorted(
            (
                node
                for node in self.nodes.values()
                if node.gpu_count > 0
                and node.gpu_memory_gb >= min_gpu_memory_gb
                and NodeRole.TRAINER in node.roles
            ),
            key=lambda node: (node.gpu_memory_gb, node.gpu_count),
            reverse=True,
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({key: value.to_dict() for key, value in self.nodes.items()}, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {
            key: ComputeNode(
                node_id=value["node_id"],
                hostname=value["hostname"],
                roles=[NodeRole(role) for role in value["roles"]],
                gpu_count=value["gpu_count"],
                gpu_memory_gb=value["gpu_memory_gb"],
                system_memory_gb=value["system_memory_gb"],
                storage_free_gb=value["storage_free_gb"],
                labels=value.get("labels", {}),
            )
            for key, value in raw.items()
        }
