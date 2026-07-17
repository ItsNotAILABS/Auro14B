"""Shared Auro model-family types (Python contract).

Mirrored in:
  - bindings/julia/AuroFamily/src/AuroFamily.jl
  - bindings/haskell/AuroFamily.hs

These types are the polyglot wire schema for multi-embedded sub-agents.
They do not claim trained checkpoints exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


FAMILY_CONTRACT_VERSION = "1.0.0"
FAMILY_ID = "Auro"


class ModelTier(str, Enum):
    """Capacity tier for an Auro lane."""

    EDGE = "edge"  # 2B
    SPECIALIST = "specialist"  # 4B
    GENERAL = "general"  # 8B
    ORCHESTRATOR = "orchestrator"  # 14B
    FRONTIER = "frontier"  # 100B


class SubAgentRole(str, Enum):
    """Canonical multi-embedded sub-agent roles across the family."""

    # Edge (2B)
    ROUTER = "router"
    TOOL_CALL = "tool_call"
    EMBED_FAST = "embed_fast"
    SPECTRAL_TRIAGE = "spectral_triage"
    # Specialist (4B)
    CODE_EDIT = "code_edit"
    SPECTRAL_MATCH = "spectral_match"
    JSON_STRUCT = "json_struct"
    TOOL_PLAN = "tool_plan"
    # General (8B)
    REASON = "reason"
    PLAN = "plan"
    CRITIQUE = "critique"
    SPECTRAL_EXPLAIN = "spectral_explain"
    # Orchestrator (14B)
    ORCHESTRATOR = "orchestrator"
    COUNCIL_CHAIR = "council_chair"
    INSTRUCT_DEV = "instruct_dev"
    MULTI_AGENT_ROUTER = "multi_agent_router"
    # Frontier (100B)
    FRONTIER_RESEARCH = "frontier_research"
    LONG_HORIZON = "long_horizon"
    SAFETY_REVIEW = "safety_review"
    DEEP_COUNCIL = "deep_council"


# Default role → preferred tier (smallest capable lane)
ROLE_DEFAULT_TIER: Dict[SubAgentRole, ModelTier] = {
    SubAgentRole.ROUTER: ModelTier.EDGE,
    SubAgentRole.TOOL_CALL: ModelTier.EDGE,
    SubAgentRole.EMBED_FAST: ModelTier.EDGE,
    SubAgentRole.SPECTRAL_TRIAGE: ModelTier.EDGE,
    SubAgentRole.CODE_EDIT: ModelTier.SPECIALIST,
    SubAgentRole.SPECTRAL_MATCH: ModelTier.SPECIALIST,
    SubAgentRole.JSON_STRUCT: ModelTier.SPECIALIST,
    SubAgentRole.TOOL_PLAN: ModelTier.SPECIALIST,
    SubAgentRole.REASON: ModelTier.GENERAL,
    SubAgentRole.PLAN: ModelTier.GENERAL,
    SubAgentRole.CRITIQUE: ModelTier.GENERAL,
    SubAgentRole.SPECTRAL_EXPLAIN: ModelTier.GENERAL,
    SubAgentRole.ORCHESTRATOR: ModelTier.ORCHESTRATOR,
    SubAgentRole.COUNCIL_CHAIR: ModelTier.ORCHESTRATOR,
    SubAgentRole.INSTRUCT_DEV: ModelTier.ORCHESTRATOR,
    SubAgentRole.MULTI_AGENT_ROUTER: ModelTier.ORCHESTRATOR,
    SubAgentRole.FRONTIER_RESEARCH: ModelTier.FRONTIER,
    SubAgentRole.LONG_HORIZON: ModelTier.FRONTIER,
    SubAgentRole.SAFETY_REVIEW: ModelTier.FRONTIER,
    SubAgentRole.DEEP_COUNCIL: ModelTier.FRONTIER,
}

TIER_RANK: Dict[ModelTier, int] = {
    ModelTier.EDGE: 0,
    ModelTier.SPECIALIST: 1,
    ModelTier.GENERAL: 2,
    ModelTier.ORCHESTRATOR: 3,
    ModelTier.FRONTIER: 4,
}

TIER_TO_MODEL_ID: Dict[ModelTier, str] = {
    ModelTier.EDGE: "Auro-2B",
    ModelTier.SPECIALIST: "Auro-4B",
    ModelTier.GENERAL: "Auro-8B",
    ModelTier.ORCHESTRATOR: "Auro-14B",
    ModelTier.FRONTIER: "Auro-100B",
}

MODEL_ID_TO_TIER: Dict[str, ModelTier] = {v: k for k, v in TIER_TO_MODEL_ID.items()}

FAMILY_PARAMETER_TARGETS: Dict[str, int] = {
    "Auro-2B": 2_000_000_000,
    "Auro-4B": 4_000_000_000,
    "Auro-8B": 8_000_000_000,
    "Auro-14B": 14_000_000_000,
    "Auro-100B": 100_000_000_000,
}


@dataclass
class ArchitectureSpec:
    """Decoder-only transformer architecture (scaffold, not weights)."""

    hidden_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    intermediate_size: int
    context_window_tokens_target: int
    vocab_size_target: int
    family: str = "decoder-only-transformer"
    objective: str = "causal-language-modeling"
    activation: str = "silu"
    normalization: str = "rmsnorm"
    position_encoding: str = "rope"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelLane:
    """One size lane in the Auro family."""

    model_id: str
    parameter_target: int
    tier: ModelTier
    status: str
    architecture: ArchitectureSpec
    subagent_roles: List[SubAgentRole] = field(default_factory=list)
    can_embed_subagents: bool = False
    embeddable_tiers: List[ModelTier] = field(default_factory=list)
    purpose: str = ""
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_target": self.parameter_target,
            "tier": self.tier.value,
            "status": self.status,
            "architecture": self.architecture.to_dict(),
            "subagent_roles": [r.value for r in self.subagent_roles],
            "can_embed_subagents": self.can_embed_subagents,
            "embeddable_tiers": [t.value for t in self.embeddable_tiers],
            "purpose": self.purpose,
            "config_path": self.config_path,
            "family_id": FAMILY_ID,
            "contract_version": FAMILY_CONTRACT_VERSION,
        }


@dataclass
class SubAgentSpec:
    """Embedded sub-agent instance hosted by a parent model lane."""

    agent_id: str
    role: SubAgentRole
    child_model_id: str
    parent_model_id: str
    task_id: str
    intent: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.native_llm.subagent_spec.v1",
            "contract_version": FAMILY_CONTRACT_VERSION,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "child_model_id": self.child_model_id,
            "parent_model_id": self.parent_model_id,
            "task_id": self.task_id,
            "intent": self.intent,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


@dataclass
class SubAgentDispatch:
    """Result of routing a role to a multi-embedded sub-agent."""

    ok: bool
    parent_model_id: str
    child_model_id: str
    role: SubAgentRole
    agent_id: str
    task_id: str
    message: str
    embedding: Optional[List[float]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.native_llm.subagent_dispatch.v1",
            "contract_version": FAMILY_CONTRACT_VERSION,
            "ok": self.ok,
            "parent_model_id": self.parent_model_id,
            "child_model_id": self.child_model_id,
            "role": self.role.value,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "message": self.message,
            "embedding": self.embedding,
            "error": self.error,
        }


@dataclass
class FamilyManifest:
    """Full Auro family charter in typed form."""

    family_id: str
    family_name: str
    status: str
    lanes: List[ModelLane]
    polyglot_types: Sequence[str] = ("python", "julia", "haskell")
    claim_boundary: str = (
        "defines architecture and multi-embedded sub-agent contracts only; "
        "no trained weights claimed"
    )

    def model_ids(self) -> List[str]:
        return [lane.model_id for lane in self.lanes]

    def get_lane(self, model_id: str) -> Optional[ModelLane]:
        for lane in self.lanes:
            if lane.model_id == model_id:
                return lane
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.native_llm.family_charter.v1",
            "contract_version": FAMILY_CONTRACT_VERSION,
            "family_id": self.family_id,
            "family_name": self.family_name,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "polyglot_types": list(self.polyglot_types),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }
