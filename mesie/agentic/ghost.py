"""Ghost agents — lightweight agentic embodiments of computation.

A GhostAgent receives a TaskSpec, activates engines on the internal bus,
chains results, and delivers structured output. Ghosts are the "spirit"
that animates the machine — they do actual work autonomously.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence


class AgentState(Enum):
    """Lifecycle states of a ghost agent."""

    DORMANT = "dormant"
    SPAWNING = "spawning"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskSpec:
    """Specification for work a ghost agent must perform.

    Attributes:
        task_id: Unique identifier.
        intent: Human-readable goal description.
        actions: Ordered list of (engine, action, payload) triples.
        chain: If True, each step's output feeds into the next step's payload.
        timeout_s: Maximum wall-clock seconds before auto-cancel.
        metadata: Arbitrary metadata attached to the task.
    """

    intent: str
    actions: List[Dict[str, Any]]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    chain: bool = True
    timeout_s: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GhostConfig:
    """Configuration for ghost agent behavior."""

    max_retries: int = 2
    retry_delay_s: float = 0.1
    collect_telemetry: bool = True
    embed_results: bool = True


@dataclass
class GhostResult:
    """Outcome of a ghost agent's task execution.

    Attributes:
        task_id: Which task was executed.
        success: Whether all steps completed.
        steps: Per-step results.
        elapsed_s: Total wall-clock time.
        embedding: Optional embedding of the full workflow result.
        error: Error message if failed.
    """

    task_id: str
    success: bool
    steps: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    embedding: Optional[List[float]] = None
    error: Optional[str] = None


class GhostAgent:
    """A ghost in the machine — an agentic embodiment that does work.

    Each ghost is spawned with a config, receives tasks, and executes them
    by dispatching actions through the engine bus.

    Args:
        agent_id: Unique agent identifier.
        config: Behavioral configuration.
        bus_dispatch: Callable(engine, action, payload) -> response dict.
        embed_fn: Optional callable to embed result data into vectors.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[GhostConfig] = None,
        bus_dispatch: Optional[Callable[..., Dict[str, Any]]] = None,
        embed_fn: Optional[Callable[[Dict[str, Any]], List[float]]] = None,
    ) -> None:
        self.agent_id = agent_id or f"ghost-{uuid.uuid4().hex[:8]}"
        self.config = config or GhostConfig()
        self._dispatch = bus_dispatch
        self._embed_fn = embed_fn
        self._state = AgentState.DORMANT
        self._history: List[GhostResult] = []

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def history(self) -> List[GhostResult]:
        return list(self._history)

    def execute(self, task: TaskSpec) -> GhostResult:
        """Execute a task specification, chaining steps through engines.

        Args:
            task: The task to execute.

        Returns:
            GhostResult with per-step outcomes and optional embedding.
        """
        self._state = AgentState.SPAWNING
        t0 = time.time()
        steps: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {}

        self._state = AgentState.ACTIVE
        try:
            for i, action_spec in enumerate(task.actions):
                engine = action_spec["engine"]
                action = action_spec["action"]
                payload = dict(action_spec.get("payload", {}))

                # Chain: merge previous results into payload
                if task.chain and context:
                    payload.update(context)

                # Check timeout
                if (time.time() - t0) > task.timeout_s:
                    self._state = AgentState.FAILED
                    result = GhostResult(
                        task_id=task.task_id,
                        success=False,
                        steps=steps,
                        elapsed_s=time.time() - t0,
                        error="Timeout exceeded",
                    )
                    self._history.append(result)
                    return result

                # Dispatch to engine
                step_result = self._dispatch_with_retry(engine, action, payload)
                steps.append({
                    "step": i,
                    "engine": engine,
                    "action": action,
                    "ok": step_result.get("ok", False),
                    "data": step_result.get("data", {}),
                })

                if not step_result.get("ok", False):
                    self._state = AgentState.FAILED
                    result = GhostResult(
                        task_id=task.task_id,
                        success=False,
                        steps=steps,
                        elapsed_s=time.time() - t0,
                        error=step_result.get("error", f"Step {i} failed"),
                    )
                    self._history.append(result)
                    return result

                # Update chaining context
                context.update(step_result.get("data", {}))

        except Exception as exc:
            self._state = AgentState.FAILED
            result = GhostResult(
                task_id=task.task_id,
                success=False,
                steps=steps,
                elapsed_s=time.time() - t0,
                error=str(exc),
            )
            self._history.append(result)
            return result

        # Embed the result if configured
        embedding = None
        if self.config.embed_results and self._embed_fn is not None:
            try:
                embedding = self._embed_fn(context)
            except Exception:
                pass

        self._state = AgentState.COMPLETED
        result = GhostResult(
            task_id=task.task_id,
            success=True,
            steps=steps,
            elapsed_s=time.time() - t0,
            embedding=embedding,
        )
        self._history.append(result)
        return result

    def _dispatch_with_retry(
        self, engine: str, action: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch with retry logic."""
        last_err = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                if self._dispatch is None:
                    return {"ok": True, "data": payload}
                resp = self._dispatch(engine, action, payload)
                if resp.get("ok", False) or attempt == self.config.max_retries:
                    return resp
                last_err = resp.get("error", "unknown")
            except Exception as exc:
                last_err = str(exc)
                if attempt == self.config.max_retries:
                    return {"ok": False, "error": last_err}
            time.sleep(self.config.retry_delay_s)
        return {"ok": False, "error": last_err}

    def reset(self) -> None:
        """Reset agent to dormant state."""
        self._state = AgentState.DORMANT

    def __repr__(self) -> str:
        return f"GhostAgent(id={self.agent_id!r}, state={self._state.value})"
