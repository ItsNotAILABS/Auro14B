"""Capability-first, capacity-aware AURO embedded sub-agent router.

Routing prefers the smallest checkpoint lane that declares the requested role,
fits under the parent parameter target, and is permitted by the parent's
embeddable tier policy. Optional MESIE GhostAgent dispatch remains available.
A dispatch receipt is not an inference or training result.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from auro_native_llm.family import load_family
from auro_native_llm.types import FAMILY_CONTRACT_VERSION, ROLE_DEFAULT_TIER, TIER_RANK, FamilyManifest, ModelLane, ModelTier, SubAgentDispatch, SubAgentRole, SubAgentSpec


def _simple_embed(text: str, dim: int = 32) -> List[float]:
    vec = [0.0] * dim
    tokens = text.lower().split() or [text]
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(f"{index}:{token}".encode()).digest()
        for offset in range(dim):
            vec[offset] += (digest[offset % len(digest)] / 255.0) - 0.5
    norm = math.sqrt(sum(value * value for value in vec)) or 1.0
    return [value / norm for value in vec]


class MultiEmbeddedSubAgentRouter:
    def __init__(self, parent_model_id: str = "Auro-14B", family: Optional[FamilyManifest] = None, spawner_factory: Optional[Callable[[], Any]] = None, embed_fn: Optional[Callable[[str], List[float]]] = None) -> None:
        self.family = family or load_family()
        self.parent_model_id = parent_model_id
        self._spawner_factory = spawner_factory
        self._embed_fn = embed_fn or _simple_embed
        self._history: List[SubAgentDispatch] = []
        parent = self.family.get_lane(parent_model_id)
        if parent is None:
            raise ValueError(f"unknown parent model_id: {parent_model_id}")
        self.parent = parent

    @property
    def history(self) -> List[SubAgentDispatch]:
        return list(self._history)

    def can_host_lane(self, child: ModelLane) -> bool:
        if child.model_id == self.parent.model_id or not self.parent.can_embed_subagents:
            return False
        if child.tier not in self.parent.embeddable_tiers:
            return False
        return int(self.parent.parameter_target) > int(child.parameter_target)

    def can_host(self, child_tier: ModelTier) -> bool:
        return any(lane.tier == child_tier and self.can_host_lane(lane) for lane in self.family.lanes)

    def resolve_child(self, role: SubAgentRole, preferred_model_id: Optional[str] = None) -> ModelLane:
        if preferred_model_id:
            lane = self.family.get_lane(preferred_model_id)
            if lane is None:
                raise ValueError(f"unknown preferred model_id: {preferred_model_id}")
            if role not in lane.subagent_roles:
                raise ValueError(f"{preferred_model_id} does not declare role={role.value}")
            if not self.can_host_lane(lane):
                raise ValueError(f"{self.parent_model_id} cannot embed {preferred_model_id}")
            return lane
        capable = [lane for lane in self.family.lanes if role in lane.subagent_roles and self.can_host_lane(lane)]
        if capable:
            return sorted(capable, key=lambda lane: (lane.parameter_target, lane.model_id))[0]
        preferred_tier = ROLE_DEFAULT_TIER.get(role, ModelTier.SPECIALIST)
        fallback = [lane for lane in self.family.lanes if self.can_host_lane(lane) and TIER_RANK[lane.tier] >= TIER_RANK[preferred_tier]]
        if fallback:
            return sorted(fallback, key=lambda lane: (TIER_RANK[lane.tier], lane.parameter_target, lane.model_id))[0]
        if role in self.parent.subagent_roles:
            return self.parent
        raise ValueError(f"no embeddable lane for role={role.value} under parent={self.parent_model_id}")

    def build_spec(self, role: SubAgentRole, intent: str, *, preferred_model_id: Optional[str] = None, task_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SubAgentSpec:
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
            metadata={"contract_version": FAMILY_CONTRACT_VERSION, "child_tier": child.tier.value, "child_parameter_target": child.parameter_target, "parent_tier": self.parent.tier.value, "parent_parameter_target": self.parent.parameter_target, "routing_policy": "capability-then-smallest-capacity", **(metadata or {})},
        )

    def dispatch(self, role: SubAgentRole | str, intent: str, *, preferred_model_id: Optional[str] = None, task_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, use_ghost: bool = False) -> SubAgentDispatch:
        parsed_role = SubAgentRole(role) if isinstance(role, str) else role
        try:
            spec = self.build_spec(parsed_role, intent, preferred_model_id=preferred_model_id, task_id=task_id, metadata=metadata)
        except ValueError as exc:
            result = SubAgentDispatch(False, self.parent_model_id, "", parsed_role, "", task_id or "", "dispatch failed", error=str(exc))
            self._history.append(result)
            return result
        ghost_note = self._try_ghost_spawn(spec) if use_ghost else "typed-dispatch-no-inference"
        result = SubAgentDispatch(True, spec.parent_model_id, spec.child_model_id, spec.role, spec.agent_id, spec.task_id, f"[{spec.child_model_id}/{spec.role.value}] {intent} :: {ghost_note}", spec.embedding)
        self._history.append(result)
        return result

    def dispatch_council(self, intent: str, roles: Optional[Sequence[SubAgentRole | str]] = None, *, use_ghost: bool = False) -> List[SubAgentDispatch]:
        selected = roles or [SubAgentRole.PLAN, SubAgentRole.SPECTRAL_MATCH, SubAgentRole.CRITIQUE, SubAgentRole.TOOL_CALL]
        parsed = [SubAgentRole(item) if isinstance(item, str) else item for item in selected]
        return [self.dispatch(role, f"{intent} :: role={role.value}", use_ghost=use_ghost) for role in parsed]

    def _try_ghost_spawn(self, spec: SubAgentSpec) -> str:
        try:
            if self._spawner_factory is not None:
                self._spawner_factory()
                return "custom-spawner"
            from mesie.agentic import AgentSpawner, SpawnerConfig
            from mesie.agentic.ghost import TaskSpec
            spawner = AgentSpawner(config=SpawnerConfig(max_agents=32, auto_embed=True))
            task = TaskSpec(intent=spec.intent, actions=[{"engine": "auro_subagent", "action": spec.role.value, "payload": {"child_model_id": spec.child_model_id, "parent_model_id": spec.parent_model_id, "agent_id": spec.agent_id}}], task_id=spec.task_id, metadata=spec.metadata)
            result = spawner.spawn(task, agent_id=spec.agent_id)
            return f"ghost-spawned success={result.success}"
        except Exception as exc:
            return f"ghost-unavailable ({type(exc).__name__})"


def route_role(role: str | SubAgentRole, intent: str, parent_model_id: str = "Auro-14B") -> Dict[str, Any]:
    parsed = SubAgentRole(role) if isinstance(role, str) else role
    return MultiEmbeddedSubAgentRouter(parent_model_id=parent_model_id).dispatch(parsed, intent).to_dict()
