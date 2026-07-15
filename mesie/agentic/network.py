"""Agent network — multi-agent topology for parallel ghost computation.

Enables ghost agents to form networks, share results, and coordinate
across multiple processing threads simultaneously. This is what gives
MESIE multi-network speed — many ghosts working in parallel.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from mesie.agentic.ghost import GhostAgent, GhostConfig, GhostResult, TaskSpec
from mesie.agentic.spawner import AgentSpawner, SpawnerConfig


class NetworkTopology(Enum):
    """Network topologies for agent coordination."""

    STAR = "star"          # One coordinator, many workers
    MESH = "mesh"          # All agents can communicate
    PIPELINE = "pipeline"  # Sequential chain
    TREE = "tree"          # Hierarchical delegation
    SWARM = "swarm"        # Emergent coordination


@dataclass
class NetworkNode:
    """A node in the agent network."""

    node_id: str
    agent: GhostAgent
    connections: Set[str] = field(default_factory=set)
    role: str = "worker"


@dataclass
class NetworkResult:
    """Aggregated result from a network computation."""

    network_id: str
    topology: NetworkTopology
    node_results: Dict[str, GhostResult] = field(default_factory=dict)
    aggregated_embedding: Optional[List[float]] = None
    elapsed_s: float = 0.0
    success: bool = True


class AgentNetwork:
    """Multi-agent network for coordinated ghost computation.

    Creates a topology of ghost agents that can process tasks in parallel,
    share intermediate results, and aggregate outputs. This is the
    multi-network speed layer.

    Args:
        topology: Network topology type.
        spawner: Agent spawner for creating ghosts.
        network_id: Unique network identifier.
    """

    def __init__(
        self,
        topology: NetworkTopology = NetworkTopology.STAR,
        spawner: Optional[AgentSpawner] = None,
        network_id: Optional[str] = None,
    ) -> None:
        self.network_id = network_id or f"net-{uuid.uuid4().hex[:8]}"
        self.topology = topology
        self._spawner = spawner or AgentSpawner()
        self._nodes: Dict[str, NetworkNode] = {}

    @property
    def size(self) -> int:
        return len(self._nodes)

    def add_node(
        self,
        agent: GhostAgent,
        role: str = "worker",
        connections: Optional[Set[str]] = None,
    ) -> str:
        """Add a ghost agent as a network node."""
        node = NetworkNode(
            node_id=agent.agent_id,
            agent=agent,
            connections=connections or set(),
            role=role,
        )
        self._nodes[node.node_id] = node
        return node.node_id

    def connect(self, node_a: str, node_b: str) -> None:
        """Establish bidirectional connection between nodes."""
        if node_a in self._nodes and node_b in self._nodes:
            self._nodes[node_a].connections.add(node_b)
            self._nodes[node_b].connections.add(node_a)

    def execute_parallel(self, tasks: List[TaskSpec]) -> NetworkResult:
        """Execute tasks across network nodes in parallel (simulated).

        In STAR topology: coordinator distributes, workers execute.
        In PIPELINE: each task feeds into the next node.
        In MESH/SWARM: all nodes get all tasks.

        Args:
            tasks: Tasks to distribute across the network.

        Returns:
            Aggregated NetworkResult.
        """
        t0 = time.time()
        node_results: Dict[str, GhostResult] = {}

        if self.topology == NetworkTopology.PIPELINE:
            node_results = self._execute_pipeline(tasks)
        elif self.topology == NetworkTopology.STAR:
            node_results = self._execute_star(tasks)
        else:
            node_results = self._execute_broadcast(tasks)

        # Aggregate embeddings if available
        embeddings = [
            r.embedding
            for r in node_results.values()
            if r.embedding is not None
        ]
        aggregated = None
        if embeddings:
            import numpy as np
            aggregated = np.mean(embeddings, axis=0).tolist()

        success = all(r.success for r in node_results.values())

        return NetworkResult(
            network_id=self.network_id,
            topology=self.topology,
            node_results=node_results,
            aggregated_embedding=aggregated,
            elapsed_s=time.time() - t0,
            success=success,
        )

    def _execute_pipeline(self, tasks: List[TaskSpec]) -> Dict[str, GhostResult]:
        """Pipeline: each node processes one task, chaining outputs."""
        results: Dict[str, GhostResult] = {}
        nodes = list(self._nodes.values())
        context: Dict[str, Any] = {}

        for i, task in enumerate(tasks):
            if i >= len(nodes):
                break
            node = nodes[i]
            # Inject previous context into task actions
            if context:
                for action in task.actions:
                    action.setdefault("payload", {}).update(context)
            result = node.agent.execute(task)
            results[node.node_id] = result
            # Chain: accumulate step data
            if result.success:
                for step in result.steps:
                    context.update(step.get("data", {}))
        return results

    def _execute_star(self, tasks: List[TaskSpec]) -> Dict[str, GhostResult]:
        """Star: distribute tasks round-robin to worker nodes."""
        results: Dict[str, GhostResult] = {}
        workers = [n for n in self._nodes.values() if n.role == "worker"]
        if not workers:
            workers = list(self._nodes.values())

        for i, task in enumerate(tasks):
            node = workers[i % len(workers)]
            result = node.agent.execute(task)
            results[node.node_id] = result
        return results

    def _execute_broadcast(self, tasks: List[TaskSpec]) -> Dict[str, GhostResult]:
        """Broadcast: every node executes every task."""
        results: Dict[str, GhostResult] = {}
        for node in self._nodes.values():
            for task in tasks:
                result = node.agent.execute(task)
                results[node.node_id] = result
        return results

    def __repr__(self) -> str:
        return (
            f"AgentNetwork(id={self.network_id!r}, topology={self.topology.value}, "
            f"nodes={self.size})"
        )
