"""Shared Auro model-family types (Python contract).

Mirrored in Julia and Haskell bindings. These types describe architecture,
routing and receipts; they do not prove that a trained checkpoint exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

FAMILY_CONTRACT_VERSION = "2.0.0"
FAMILY_ID = "Auro"


class ModelTier(str, Enum):
    ATOMIC = "atomic"  # 156K / 250M / 500M
    EDGE = "edge"  # 2B
    SPECIALIST = "specialist"  # 4B
    GENERAL = "general"  # 8B
    ORCHESTRATOR = "orchestrator"  # 14B
    FRONTIER = "frontier"  # 100B


class SubAgentRole(str, Enum):
    ROUTING_SEED = "routing_seed"
    CLASSIFIER = "classifier"
    JSON_REPAIR = "json_repair"
    TOOL_SELECTION = "tool_selection"
    STYLE_GUARD = "style_guard"
    INTENT_EXTRACT = "intent_extract"
    RETRIEVAL_FILTER = "retrieval_filter"
    STRUCTURED_TRANSFORM = "structured_transform"
    CODE_TRIAGE = "code_triage"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    SEMANTIC_OUTLINE = "semantic_outline"
    TOOL_EXECUTION_PLAN = "tool_execution_plan"
    CODE_PATCH = "code_patch"
    EVIDENCE_REVIEW = "evidence_review"
    LOCAL_WORKER = "local_worker"
    EXPERT_CONSENSUS = "expert_consensus"
    TEXT_EXPANSION = "text_expansion"
    CREATIVE_BRANCH = "creative_branch"
    ROUTER = "router"
    TOOL_CALL = "tool_call"
    EMBED_FAST = "embed_fast"
    SPECTRAL_TRIAGE = "spectral_triage"
    CODE_EDIT = "code_edit"
    SPECTRAL_MATCH = "spectral_match"
    JSON_STRUCT = "json_struct"
    TOOL_PLAN = "tool_plan"
    REASON = "reason"
    PLAN = "plan"
    CRITIQUE = "critique"
    SPECTRAL_EXPLAIN = "spectral_explain"
    ORCHESTRATOR = "orchestrator"
    COUNCIL_CHAIR = "council_chair"
    INSTRUCT_DEV = "instruct_dev"
    MULTI_AGENT_ROUTER = "multi_agent_router"
    FRONTIER_RESEARCH = "frontier_research"
    LONG_HORIZON = "long_horizon"
    SAFETY_REVIEW = "safety_review"
    DEEP_COUNCIL = "deep_council"


ATOMIC_ROLES = {
    SubAgentRole.ROUTING_SEED,
    SubAgentRole.CLASSIFIER,
    SubAgentRole.JSON_REPAIR,
    SubAgentRole.TOOL_SELECTION,
    SubAgentRole.STYLE_GUARD,
    SubAgentRole.INTENT_EXTRACT,
    SubAgentRole.RETRIEVAL_FILTER,
    SubAgentRole.STRUCTURED_TRANSFORM,
    SubAgentRole.CODE_TRIAGE,
    SubAgentRole.MEMORY_CONSOLIDATION,
    SubAgentRole.SEMANTIC_OUTLINE,
    SubAgentRole.TOOL_EXECUTION_PLAN,
    SubAgentRole.CODE_PATCH,
    SubAgentRole.EVIDENCE_REVIEW,
    SubAgentRole.LOCAL_WORKER,
    SubAgentRole.EXPERT_CONSENSUS,
    SubAgentRole.TEXT_EXPANSION,
    SubAgentRole.CREATIVE_BRANCH,
}

ROLE_DEFAULT_TIER: Dict[SubAgentRole, ModelTier] = {
    **{role: ModelTier.ATOMIC for role in ATOMIC_ROLES},
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
    ModelTier.ATOMIC: 0,
    ModelTier.EDGE: 1,
    ModelTier.SPECIALIST: 2,
    ModelTier.GENERAL: 3,
    ModelTier.ORCHESTRATOR: 4,
    ModelTier.FRONTIER: 5,
}

TIER_TO_MODEL_ID: Dict[ModelTier, str] = {
    ModelTier.ATOMIC: "Auro-500M",
    ModelTier.EDGE: "Auro-2B",
    ModelTier.SPECIALIST: "Auro-4B",
    ModelTier.GENERAL: "Auro-8B",
    ModelTier.ORCHESTRATOR: "Auro-14B",
    ModelTier.FRONTIER: "Auro-100B",
}

MODEL_ID_TO_TIER: Dict[str, ModelTier] = {
    "Auro-156K": ModelTier.ATOMIC,
    "Auro-250M": ModelTier.ATOMIC,
    "Auro-500M": ModelTier.ATOMIC,
    "Auro-500M-SENSUS": ModelTier.ATOMIC,
    "Auro-500M-PRAXIS": ModelTier.ATOMIC,
    "Auro-500M-VERBUM": ModelTier.ATOMIC,
    "Auro-2B": ModelTier.EDGE,
    "Auro-4B": ModelTier.SPECIALIST,
    "Auro-8B": ModelTier.GENERAL,
    "Auro-14B": ModelTier.ORCHESTRATOR,
    "Auro-100B": ModelTier.FRONTIER,
}

FAMILY_PARAMETER_TARGETS: Dict[str, int] = {
    "Auro-156K": 156_000,
    "Auro-250M": 250_000_000,
    "Auro-500M": 500_000_000,
    "Auro-500M-SENSUS": 500_000_000,
    "Auro-500M-PRAXIS": 500_000_000,
    "Auro-500M-VERBUM": 500_000_000,
    "Auro-2B": 2_000_000_000,
    "Auro-4B": 4_000_000_000,
    "Auro-8B": 8_000_000_000,
    "Auro-14B": 14_000_000_000,
    "Auro-100B": 100_000_000_000,
}


@dataclass
class ArchitectureSpec:
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
            "subagent_roles": [role.value for role in self.subagent_roles],
            "can_embed_subagents": self.can_embed_subagents,
            "embeddable_tiers": [tier.value for tier in self.embeddable_tiers],
            "purpose": self.purpose,
            "config_path": self.config_path,
            "family_id": FAMILY_ID,
            "contract_version": FAMILY_CONTRACT_VERSION,
        }


@dataclass
class SubAgentSpec:
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
            "schema": "auro.native_llm.subagent_spec.v2",
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
            "schema": "auro.native_llm.subagent_dispatch.v2",
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
    family_id: str
    family_name: str
    status: str
    lanes: List[ModelLane]
    polyglot_types: Sequence[str] = ("python", "julia", "haskell")
    claim_boundary: str = "defines architecture and multi-embedded sub-agent contracts only; no trained weights claimed"

    def model_ids(self) -> List[str]:
        return [lane.model_id for lane in self.lanes]

    def get_lane(self, model_id: str) -> Optional[ModelLane]:
        for lane in self.lanes:
            if lane.model_id == model_id:
                return lane
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.native_llm.family_charter.v2",
            "contract_version": FAMILY_CONTRACT_VERSION,
            "family_id": self.family_id,
            "family_name": self.family_name,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "polyglot_types": list(self.polyglot_types),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }
