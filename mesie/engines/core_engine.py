"""Core Engine — the inner nucleus that ties all MESIE engines together.

The CoreEngine is the central nervous system: it owns the bus, registry,
octopus controller, and agent spawner. It provides multi-network dispatch,
workflow embedding, and unified task execution. Everything flows through here.

This is what activates the ghost in the machine.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from mesie.agentic.ghost import GhostAgent, GhostConfig, GhostResult, TaskSpec
from mesie.agentic.network import AgentNetwork, NetworkResult, NetworkTopology
from mesie.agentic.spawner import AgentSpawner, SpawnerConfig
from mesie.engines.base import Engine, EngineRegistry
from mesie.engines.registry import build_default_registry
from mesie.internal_api.bus import InternalBus
from mesie.internal_api.messages import EngineResponse, MessageEnvelope, MessageTopic
from mesie.internal_api.router import InternalRouter


@dataclass
class CoreConfig:
    """Configuration for the inner core engine.

    Attributes:
        max_agents: Maximum ghost agents.
        default_topology: Default network topology for multi-agent tasks.
        embed_workflows: Embed workflow results into vector space.
        embed_dim: Dimensionality for workflow embeddings.
        enable_reasoning: Enable reasoning chains in agents.
        network_width: Number of parallel agents in network tasks.
    """

    max_agents: int = 128
    default_topology: NetworkTopology = NetworkTopology.STAR
    embed_workflows: bool = True
    embed_dim: int = 64
    enable_reasoning: bool = True
    network_width: int = 4


class CoreEngine(Engine):
    """The inner core — ties all engines, spawns ghosts, runs networks.

    This is the beating heart of MESIE V0.3. It:
    - Owns the internal bus and engine registry
    - Dispatches actions across all engines
    - Spawns ghost agents to do actual computational work
    - Creates multi-agent networks for parallel processing
    - Embeds workflows and datasets into vector space
    - Chains reasoning across engines

    Args:
        config: Core configuration.
        bus: Optional pre-existing bus (creates one if None).
    """

    name = "core"
    capabilities = [
        "dispatch",
        "spawn_agent",
        "run_network",
        "embed_workflow",
        "embed_dataset",
        "reason",
        "status",
    ]

    def __init__(
        self,
        config: Optional[CoreConfig] = None,
        bus: Optional[InternalBus] = None,
    ) -> None:
        self.config = config or CoreConfig()
        self.bus = bus or InternalBus()
        self.registry = build_default_registry(self.bus)
        self.router = InternalRouter(bus=self.bus, registry=self.registry)

        # Register self
        self.registry.register(self)

        # Agent infrastructure
        self._spawner = AgentSpawner(
            config=SpawnerConfig(
                max_agents=self.config.max_agents,
                default_config=GhostConfig(embed_results=self.config.embed_workflows),
            ),
            bus_dispatch=self._bus_dispatch,
            embed_fn=self._embed_result if self.config.embed_workflows else None,
        )

        # Workflow embedding state
        self._workflow_embeddings: List[np.ndarray] = []
        self._dataset_embeddings: Dict[str, np.ndarray] = {}
        self._task_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        """Handle messages on the internal bus."""
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown: {action}")

        try:
            if action == "dispatch":
                return self._handle_dispatch(message.payload)
            elif action == "spawn_agent":
                return self._handle_spawn(message.payload)
            elif action == "run_network":
                return self._handle_network(message.payload)
            elif action == "embed_workflow":
                return self._handle_embed_workflow(message.payload)
            elif action == "embed_dataset":
                return self._handle_embed_dataset(message.payload)
            elif action == "reason":
                return self._handle_reason(message.payload)
            elif action == "status":
                return self._handle_status()
        except Exception as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")

    # ------------------------------------------------------------------
    # Public API (direct calls, not just bus)
    # ------------------------------------------------------------------

    def dispatch(self, engine: str, action: str, payload: Dict[str, Any]) -> EngineResponse:
        """Dispatch an action to any engine on the bus."""
        return self.router.call(engine, action, payload)

    def spawn_ghost(self, task: TaskSpec, **kwargs) -> GhostResult:
        """Spawn a ghost agent and execute a task."""
        result = self._spawner.spawn(task, **kwargs)
        self._task_log.append({
            "type": "ghost",
            "task_id": task.task_id,
            "intent": task.intent,
            "success": result.success,
            "elapsed_s": result.elapsed_s,
        })
        return result

    def spawn_many(self, tasks: List[TaskSpec]) -> List[GhostResult]:
        """Spawn multiple ghosts for parallel task execution."""
        results = self._spawner.spawn_many(tasks)
        for task, result in zip(tasks, results):
            self._task_log.append({
                "type": "ghost_batch",
                "task_id": task.task_id,
                "intent": task.intent,
                "success": result.success,
            })
        return results

    def create_network(
        self,
        n_agents: Optional[int] = None,
        topology: Optional[NetworkTopology] = None,
    ) -> AgentNetwork:
        """Create a multi-agent network with ghost nodes.

        Args:
            n_agents: Number of agents in the network.
            topology: Network topology (defaults to config).

        Returns:
            Configured AgentNetwork ready for task execution.
        """
        n = n_agents or self.config.network_width
        topo = topology or self.config.default_topology
        network = AgentNetwork(topology=topo, spawner=self._spawner)

        for i in range(n):
            agent = GhostAgent(
                agent_id=f"net-ghost-{i}",
                config=GhostConfig(embed_results=self.config.embed_workflows),
                bus_dispatch=self._bus_dispatch,
                embed_fn=self._embed_result if self.config.embed_workflows else None,
            )
            role = "coordinator" if i == 0 and topo == NetworkTopology.STAR else "worker"
            network.add_node(agent, role=role)

        return network

    def run_network(
        self,
        tasks: List[TaskSpec],
        topology: Optional[NetworkTopology] = None,
        n_agents: Optional[int] = None,
    ) -> NetworkResult:
        """Create a network and run tasks through it.

        Args:
            tasks: Tasks to execute across the network.
            topology: Network topology.
            n_agents: Number of network nodes.

        Returns:
            Aggregated NetworkResult.
        """
        network = self.create_network(n_agents=n_agents, topology=topology)
        result = network.execute_parallel(tasks)
        self._task_log.append({
            "type": "network",
            "network_id": network.network_id,
            "topology": result.topology.value,
            "success": result.success,
            "elapsed_s": result.elapsed_s,
        })
        return result

    def embed_workflow(self, workflow_steps: List[Dict[str, Any]]) -> np.ndarray:
        """Embed a workflow definition into vector space.

        Converts workflow structure (engine names, actions, parameters)
        into a fixed-dimension embedding for similarity search and
        workflow recommendation.

        Args:
            workflow_steps: List of step dicts with engine/action/payload.

        Returns:
            Embedding vector of shape (embed_dim,).
        """
        rng = np.random.default_rng(hash(str(workflow_steps)) % (2**32))
        # Deterministic embedding based on workflow content
        raw = np.zeros(self.config.embed_dim, dtype=np.float64)
        for i, step in enumerate(workflow_steps):
            engine_hash = hash(step.get("engine", "")) % 1000 / 1000.0
            action_hash = hash(step.get("action", "")) % 1000 / 1000.0
            position_signal = (i + 1) / max(len(workflow_steps), 1)
            raw += rng.normal(0, 0.1, self.config.embed_dim) * position_signal
            raw[i % self.config.embed_dim] += engine_hash
            raw[(i + 1) % self.config.embed_dim] += action_hash

        # L2 normalize
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw /= norm

        self._workflow_embeddings.append(raw)
        return raw

    def embed_dataset(
        self,
        dataset_id: str,
        records: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        """Embed a dataset into a single vector representation.

        Creates a centroid embedding from dataset records for
        dataset-level similarity and retrieval.

        Args:
            dataset_id: Identifier for the dataset.
            records: Sequence of record dicts.

        Returns:
            Dataset embedding vector.
        """
        rng = np.random.default_rng(hash(dataset_id) % (2**32))
        embeddings = []
        for rec in records:
            vec = rng.normal(0, 0.3, self.config.embed_dim)
            # Incorporate actual numeric data if present
            if "amplitude" in rec:
                amp = np.array(rec["amplitude"], dtype=np.float64)
                # Spectral features → embedding dimensions
                n = min(len(amp), self.config.embed_dim)
                vec[:n] += amp[:n] * 0.5
            if "frequency" in rec:
                freq = np.array(rec["frequency"], dtype=np.float64)
                n = min(len(freq), self.config.embed_dim)
                vec[:n] += np.log1p(freq[:n]) * 0.1
            embeddings.append(vec)

        centroid = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.config.embed_dim)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        self._dataset_embeddings[dataset_id] = centroid
        return centroid

    def reason_chain(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reasoning chain through the intelligence engine.

        Args:
            query: What to reason about.
            context: Supporting context data.

        Returns:
            Reasoning result dict with conclusion, confidence, evidence.
        """
        # Route through intelligence engine
        from mesie.core.records import MultiElementRecord, SpectralComponent
        # Create a synthetic record from context if needed
        if "record" in context:
            rec = context["record"]
        else:
            amp = context.get("amplitude", np.random.default_rng(42).random(32).tolist())
            freq = context.get("frequency", np.linspace(0.1, 50, 32).tolist())
            rec = MultiElementRecord(
                record_id=f"reason-{uuid.uuid4().hex[:6]}",
                components=[SpectralComponent(
                    name="reasoning_input",
                    frequency=np.array(freq),
                    amplitude=np.array(amp),
                )],
            )

        resp = self.router.call("intelligence", "reason", {"record": rec})
        return {
            "query": query,
            "conclusion": resp.data.get("conclusion", "unknown"),
            "confidence": resp.data.get("confidence", 0.0),
            "evidence": resp.data.get("evidence", {}),
            "engine_ok": resp.ok,
        }

    @property
    def task_log(self) -> List[Dict[str, Any]]:
        return list(self._task_log)

    @property
    def workflow_embedding_count(self) -> int:
        return len(self._workflow_embeddings)

    @property
    def dataset_embedding_count(self) -> int:
        return len(self._dataset_embeddings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bus_dispatch(
        self, engine: str, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch function given to ghost agents."""
        resp = self.router.call(engine, action, payload)
        return {"ok": resp.ok, "data": resp.data, "error": resp.error}

    def _embed_result(self, data: Dict[str, Any]) -> List[float]:
        """Embed a ghost result into a vector."""
        rng = np.random.default_rng(hash(str(data)) % (2**32))
        vec = rng.normal(0, 0.2, self.config.embed_dim)
        # Incorporate numeric data
        for key, val in data.items():
            if isinstance(val, (int, float)):
                idx = hash(key) % self.config.embed_dim
                vec[idx] += float(val) * 0.1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def _handle_dispatch(self, payload: Dict[str, Any]) -> EngineResponse:
        engine = payload["engine"]
        action = payload["action"]
        inner = payload.get("payload", {})
        resp = self.router.call(engine, action, inner)
        return EngineResponse(resp.ok, self.name, "dispatch", resp.data, error=resp.error)

    def _handle_spawn(self, payload: Dict[str, Any]) -> EngineResponse:
        task = TaskSpec(
            intent=payload.get("intent", "ghost_task"),
            actions=payload.get("actions", []),
            chain=payload.get("chain", True),
        )
        result = self.spawn_ghost(task)
        return EngineResponse(
            result.success,
            self.name,
            "spawn_agent",
            {"task_id": result.task_id, "steps": len(result.steps), "elapsed_s": result.elapsed_s},
            error=result.error,
        )

    def _handle_network(self, payload: Dict[str, Any]) -> EngineResponse:
        tasks = [
            TaskSpec(intent=t.get("intent", "net_task"), actions=t.get("actions", []))
            for t in payload.get("tasks", [])
        ]
        topo = NetworkTopology(payload.get("topology", self.config.default_topology.value))
        n = payload.get("n_agents", self.config.network_width)
        result = self.run_network(tasks, topology=topo, n_agents=n)
        return EngineResponse(
            result.success,
            self.name,
            "run_network",
            {
                "network_id": result.network_id,
                "topology": result.topology.value,
                "nodes": len(result.node_results),
                "elapsed_s": result.elapsed_s,
            },
        )

    def _handle_embed_workflow(self, payload: Dict[str, Any]) -> EngineResponse:
        steps = payload.get("steps", [])
        emb = self.embed_workflow(steps)
        return EngineResponse(True, self.name, "embed_workflow", {
            "dim": len(emb),
            "norm": float(np.linalg.norm(emb)),
        })

    def _handle_embed_dataset(self, payload: Dict[str, Any]) -> EngineResponse:
        dataset_id = payload.get("dataset_id", "unnamed")
        records = payload.get("records", [])
        emb = self.embed_dataset(dataset_id, records)
        return EngineResponse(True, self.name, "embed_dataset", {
            "dataset_id": dataset_id,
            "n_records": len(records),
            "dim": len(emb),
        })

    def _handle_reason(self, payload: Dict[str, Any]) -> EngineResponse:
        query = payload.get("query", "")
        context = payload.get("context", {})
        result = self.reason_chain(query, context)
        return EngineResponse(True, self.name, "reason", result)

    def _handle_status(self) -> EngineResponse:
        return EngineResponse(True, self.name, "status", {
            "engines": self.registry.names(),
            "active_agents": self._spawner.active_count,
            "total_tasks": len(self._task_log),
            "workflow_embeddings": self.workflow_embedding_count,
            "dataset_embeddings": self.dataset_embedding_count,
        })
