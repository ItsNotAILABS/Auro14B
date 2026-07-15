"""Agent spawner — creates and manages fleets of ghost agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mesie.agentic.ghost import AgentState, GhostAgent, GhostConfig, GhostResult, TaskSpec


@dataclass
class SpawnerConfig:
    """Configuration for the agent spawner.

    Attributes:
        max_agents: Maximum concurrent ghost agents.
        default_config: Default GhostConfig for spawned agents.
        auto_embed: Embed all task results by default.
        recycle_completed: Allow reuse of completed agent slots.
    """

    max_agents: int = 64
    default_config: GhostConfig = field(default_factory=GhostConfig)
    auto_embed: bool = True
    recycle_completed: bool = True


class AgentSpawner:
    """Factory and pool manager for ghost agents.

    The spawner creates ghost agents on demand, tracks their lifecycles,
    and collects results. It's the workforce management layer.

    Args:
        config: Spawner configuration.
        bus_dispatch: Engine bus dispatch function shared by all agents.
        embed_fn: Embedding function shared by all agents.
    """

    def __init__(
        self,
        config: Optional[SpawnerConfig] = None,
        bus_dispatch: Optional[Callable[..., Dict[str, Any]]] = None,
        embed_fn: Optional[Callable[[Dict[str, Any]], List[float]]] = None,
    ) -> None:
        self.config = config or SpawnerConfig()
        self._dispatch = bus_dispatch
        self._embed_fn = embed_fn
        self._agents: Dict[str, GhostAgent] = {}
        self._results: List[GhostResult] = []

    @property
    def active_count(self) -> int:
        """Number of currently active (non-completed/failed) agents."""
        return sum(
            1
            for a in self._agents.values()
            if a.state in (AgentState.ACTIVE, AgentState.SPAWNING, AgentState.WAITING)
        )

    @property
    def total_spawned(self) -> int:
        return len(self._agents)

    def spawn(
        self,
        task: TaskSpec,
        *,
        agent_id: Optional[str] = None,
        config: Optional[GhostConfig] = None,
    ) -> GhostResult:
        """Spawn a ghost agent and immediately execute the task.

        Args:
            task: Task specification to execute.
            agent_id: Optional custom agent ID.
            config: Override agent config.

        Returns:
            GhostResult from execution.

        Raises:
            RuntimeError: If max agents exceeded.
        """
        if self.config.recycle_completed:
            self._recycle()

        if self.active_count >= self.config.max_agents:
            raise RuntimeError(
                f"Max agents ({self.config.max_agents}) reached. "
                "Wait for completions or increase limit."
            )

        cfg = config or self.config.default_config
        agent = GhostAgent(
            agent_id=agent_id or f"ghost-{uuid.uuid4().hex[:8]}",
            config=cfg,
            bus_dispatch=self._dispatch,
            embed_fn=self._embed_fn,
        )
        self._agents[agent.agent_id] = agent
        result = agent.execute(task)
        self._results.append(result)
        return result

    def spawn_many(self, tasks: List[TaskSpec]) -> List[GhostResult]:
        """Spawn multiple ghosts, one per task, and collect results.

        Args:
            tasks: List of task specifications.

        Returns:
            List of GhostResults in task order.
        """
        return [self.spawn(t) for t in tasks]

    def get_agent(self, agent_id: str) -> Optional[GhostAgent]:
        return self._agents.get(agent_id)

    def all_results(self) -> List[GhostResult]:
        return list(self._results)

    def _recycle(self) -> None:
        """Remove completed/failed agents to free slots."""
        done = [
            aid
            for aid, a in self._agents.items()
            if a.state in (AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED)
        ]
        for aid in done:
            del self._agents[aid]

    def __repr__(self) -> str:
        return (
            f"AgentSpawner(active={self.active_count}, "
            f"total={self.total_spawned}, results={len(self._results)})"
        )
