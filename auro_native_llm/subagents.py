"""Multi-embedded sub-agent router for the Auro model family.

Larger Auro lanes host smaller lanes as embedded sub-agents. Routing is
role → preferred tier → parent capacity check, with optional MESIE ghost
spawn when the agentic bus is available.

Scaffold only: dispatch receipts are not inference results.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from auro_native_llm.family import load_family
from auro_native_llm.types import (
    FAMILY_CONTRACT_VERSION,
    ROLE_DEFAULT_TIER,
    TIER_RANK,
    TIER_TO_MODEL_ID,
    FamilyManifest,
    ModelLane,
    ModelTier,
    SubAgentDispatch,
    SubAgentRole,
    SubAgentSpec,
)


def _simple_embed(text: str, dim: int = 32) -> List[float]:
    """Deterministic bag-of-hashes embedding for scaffold receipts (no torch)."""
    vec = [0.0] * dim
    tokens = text.lower().split() or [text]
    for i, tok in enumerate(tokens):
        h = hashlib.sha256(f"{i}:{tok}".encode()).digest()
        for j in range(dim):
            vec[j] += (h[j % len(h)] / 255.0) - 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class MultiEmbeddedSubAgentRouter:
    """Routes tasks to multi-embedded Auro sub-agents under a parent lane.

    Example:
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-14B")
        result = router.dispatch(SubAgentRole.SPECTRAL_MATCH, "match two PSDs")
    """

    def __init__(
        self,
        parent_model_id: str = "Auro-14B",
        family: Optional[FamilyManifest] = None,
        spawner_factory: Optional[Callable[[], Any]] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.family = family or load_family()
        self.parent_model_id = parent_model_id
        self._spawner_factory = spawner_factory
        self._embed_fn = embed_fn or _simple_embed
        self._history: List[SubAgentDispatch] = []

        parent = self.family.get_lane(parent_model_id)
        if parent is None:
            raise ValueError(f"unknown parent model_id: {parent_model_id}")
        self.parent: ModelLane = parent

    @property
    def history(self) -> List[SubAgentDispatch]:
        return list(self._history)

    def can_host(self, child_tier: ModelTier) -> bool:
        if not self.parent.can_embed_subagents:
            return False
        if child_tier not in self.parent.embeddable_tiers:
            return False
        # Parent tier must be strictly larger than child
        return TIER_RANK[self.parent.tier] > TIER_RANK[child_tier]

    def resolve_child(
        self,
        role: SubAgentRole,
        preferred_model_id: Optional[str] = None,
    ) -> ModelLane:
        """Pick the smallest capable child lane the parent can host."""
        if preferred_model_id:
            lane = self.family.get_lane(preferred_model_id)
            if lane is None:
                raise ValueError(f"unknown preferred model_id: {preferred_model_id}")
            if not self.can_host(lane.tier):
                raise ValueError(
                    f"{self.parent_model_id} cannot embed {preferred_model_id} "
                    f"(tier={lane.tier.value})"
                )
            return lane

        preferred_tier = ROLE_DEFAULT_TIER.get(role, ModelTier.SPECIALIST)
        # Walk from preferred tier upward until hostable
        ordered = sorted(ModelTier, key=lambda t: TIER_RANK[t])
        candidates = [t for t in ordered if TIER_RANK[t] >= TIER_RANK[preferred_tier]]
        for tier in candidates:
            if self.can_host(tier):
                model_id = TIER_TO_MODEL_ID[tier]
                lane = self.family.get_lane(model_id)
                if lane is not None:
                    return lane

        # Same-tier self-dispatch only for orchestrator roles on parent itself
        if role in self.parent.subagent_roles:
            return self.parent

        raise ValueError(
            f"no embeddable lane for role={role.value} under parent={self.parent_model_id}"
        )

    def build_spec(
        self,
        role: SubAgentRole,
        intent: str,
        *,
        preferred_model_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SubAgentSpec:
        child = self.resolve_child(role, preferred_model_id)
        embedding = self._embed_fn(f"{role.value}:{intent}")
        return SubAgentSpec(
            agent_id=f"auro-sa-{uuid.uuid4().hex[:10]}",
            role=role,
            child_model_id=child.model_id,
            parent_model_id=self.parent_model_id,
            task_id=task_id or uuid.uuid4().hex[:12],
            intent=intent,
            embedding=embedding,
            metadata={
                "contract_version": FAMILY_CONTRACT_VERSION,
                "child_tier": child.tier.value,
                "parent_tier": self.parent.tier.value,
                **(metadata or {}),
            },
        )

    def dispatch(
        self,
        role: SubAgentRole | str,
        intent: str,
        *,
        preferred_model_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_ghost: bool = False,
    ) -> SubAgentDispatch:
        """Route a role/intent to an embedded sub-agent and emit a dispatch result.

        When use_ghost=True and MESIE agentic is importable, spawn a GhostAgent
        scaffold task. Otherwise return a typed scaffold dispatch receipt.
        """
        if isinstance(role, str):
            role = SubAgentRole(role)

        try:
            spec = self.build_spec(
                role,
                intent,
                preferred_model_id=preferred_model_id,
                task_id=task_id,
                metadata=metadata,
            )
        except ValueError as exc:
            fail = SubAgentDispatch(
                ok=False,
                parent_model_id=self.parent_model_id,
                child_model_id="",
                role=role if isinstance(role, SubAgentRole) else SubAgentRole.ROUTER,
                agent_id="",
                task_id=task_id or "",
                message="dispatch failed",
                error=str(exc),
            )
            self._history.append(fail)
            return fail

        ghost_note = "scaffold-dispatch-no-checkpoint"
        if use_ghost:
            ghost_note = self._try_ghost_spawn(spec)

        result = SubAgentDispatch(
            ok=True,
            parent_model_id=spec.parent_model_id,
            child_model_id=spec.child_model_id,
            role=spec.role,
            agent_id=spec.agent_id,
            task_id=spec.task_id,
            message=f"[{spec.child_model_id}/{spec.role.value}] {intent} :: {ghost_note}",
            embedding=spec.embedding,
            error=None,
        )
        self._history.append(result)
        return result

    def dispatch_council(
        self,
        intent: str,
        roles: Optional[Sequence[SubAgentRole | str]] = None,
    ) -> List[SubAgentDispatch]:
        """Spawn a multi-role council under the parent orchestrator/frontier lane."""
        default_roles: List[SubAgentRole] = [
            SubAgentRole.PLAN,
            SubAgentRole.SPECTRAL_MATCH,
            SubAgentRole.CRITIQUE,
            SubAgentRole.TOOL_CALL,
        ]
        if roles is None:
            selected = default_roles
        else:
            selected = [SubAgentRole(r) if isinstance(r, str) else r for r in roles]
        return [self.dispatch(r, f"{intent} :: role={r.value}") for r in selected]

    def _try_ghost_spawn(self, spec: SubAgentSpec) -> str:
        try:
            if self._spawner_factory is not None:
                spawner = self._spawner_factory()
            else:
                from mesie.agentic import AgentSpawner, SpawnerConfig
                from mesie.agentic.ghost import TaskSpec

                spawner = AgentSpawner(config=SpawnerConfig(max_agents=32, auto_embed=True))
                task = TaskSpec(
                    intent=spec.intent,
                    actions=[
                        {
                            "engine": "auro_subagent",
                            "action": spec.role.value,
                            "payload": {
                                "child_model_id": spec.child_model_id,
                                "parent_model_id": spec.parent_model_id,
                                "agent_id": spec.agent_id,
                            },
                        }
                    ],
                    task_id=spec.task_id,
                    metadata=spec.metadata,
                )
                # Without bus_dispatch, ghost records a structured scaffold step
                result = spawner.spawn(task, agent_id=spec.agent_id)
                return f"ghost-spawned success={result.success}"
            return "custom-spawner"
        except Exception as exc:  # pragma: no cover - optional path
            return f"ghost-unavailable ({type(exc).__name__})"


def route_role(
    role: str | SubAgentRole,
    intent: str,
    parent_model_id: str = "Auro-14B",
) -> Dict[str, Any]:
    """Convenience: one-shot dispatch to dict."""
    if isinstance(role, str):
        role = SubAgentRole(role)
    router = MultiEmbeddedSubAgentRouter(parent_model_id=parent_model_id)
    return router.dispatch(role, intent).to_dict()
