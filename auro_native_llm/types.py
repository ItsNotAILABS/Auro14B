"""Shared AURO model-family types.

Mirrored in:
  - bindings/julia/AuroFamily/src/AuroFamily.jl
  - bindings/haskell/AuroFamily.hs

The contract describes architecture, composition, routing, and evidence
boundaries. It does not claim that a trained checkpoint exists for every lane.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


FAMILY_CONTRACT_VERSION = "2.1.0"
FAMILY_ID = "Auro"


class ModelTier(str, Enum):
    """Capacity and composition tier for an AURO lane."""

    ATOMIC = "atomic"
    EDGE = "edge"
    SPECIALIST = "specialist"
    GENERAL = "general"
    ORCHESTRATOR = "orchestrator"
    FRONTIER = "frontier"


class SubAgentRole(str, Enum):
    """Canonical roles across the capacity x capability family."""

    ROUTING_SEED = "routing_seed"
    CLASSIFIER = "classifier"
    JSON_REPAIR = "json_repair"
    TOOL_SELECTION = "tool_selection"
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

# Tier-only callers receive the largest atomic lane by default. Capability-aware
# routers should use ROLE_DEFAULT_MODEL_ID.
TIER_TO_MODEL_ID: Dict[ModelTier, str] = {
    ModelTier.ATOMIC: "Auro-500M",
    ModelTier.EDGE: "Auro-2B",
    ModelTier.SPECIALIST: "Auro-4B",
    ModelTier.GENERAL: "Auro-8B",
    ModelTier.ORCHESTRATOR: "Auro-14B",
    ModelTier.FRONTIER: "Auro-100B",
}

ATOMIC_MODEL_IDS = ("Auro-156K", "Auro-250M", "Auro-500M")

MODEL_ID_TO_TIER: Dict[str, ModelTier] = {
    "Auro-156K": ModelTier.ATOMIC,
    "Auro-250M": ModelTier.ATOMIC,
    "Auro-500M": ModelTier.ATOMIC,
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
    "Auro-2B": 2_000_000_000,
    "Auro-4B": 4_000_000_000,
    "Auro-8B": 8_000_000_000,
    "Auro-14B": 14_000_000_000,
    "Auro-100B": 100_000_000_000,
}

ROLE_DEFAULT_MODEL_ID: Dict[SubAgentRole, str] = {
    SubAgentRole.ROUTING_SEED: "Auro-156K",
    SubAgentRole.CLASSIFIER: "Auro-156K",
    SubAgentRole.JSON_REPAIR: "Auro-156K",
    SubAgentRole.TOOL_SELECTION: "Auro-156K",
    SubAgentRole.INTENT_EXTRACT: "Auro-250M",
    SubAgentRole.RETRIEVAL_FILTER: "Auro-250M",
    SubAgentRole.STRUCTURED_TRANSFORM: "Auro-250M",
    SubAgentRole.CODE_TRIAGE: "Auro-250M",
    SubAgentRole.MEMORY_CONSOLIDATION: "Auro-250M",
    SubAgentRole.SEMANTIC_OUTLINE: "Auro-250M",
    SubAgentRole.TOOL_EXECUTION_PLAN: "Auro-500M",
    SubAgentRole.CODE_PATCH: "Auro-500M",
    SubAgentRole.EVIDENCE_REVIEW: "Auro-500M",
    SubAgentRole.LOCAL_WORKER: "Auro-500M",
    SubAgentRole.EXPERT_CONSENSUS: "Auro-500M",
    SubAgentRole.TEXT_EXPANSION: "Auro-500M",
    **{
        role: TIER_TO_MODEL_ID[tier]
        for role, tier in ROLE_DEFAULT_TIER.items()
        if tier != ModelTier.ATOMIC
    },
}

AURO_2B_SPECIALIST_TRIAD = (
    "Auro-500M-SENSUS",
    "Auro-500M-PRAXIS",
    "Auro-500M-VERBUM",
)

CANONICAL_CLAIM_BOUNDARIES = (
    "architecture-configuration-is-not-a-trained-checkpoint",
    "accepted-context-is-not-dense-attention",
    "named-agent-is-not-a-separately-trained-model",
    "source-code-is-not-deployment-evidence",
    "local-hash-chain-is-not-external-custody",
    "same-session-recall-is-not-persistent-memory",
    "successful-build-is-not-clean-install-proof",
    "generated-answer-is-not-experimental-validation",
)


@dataclass
class ArchitectureSpec:
    """Decoder-only transformer architecture contract, not weights."""

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
    attention_type: str = "gqa"
    experts: int = 1
    top_k: int = 1
    moe_every: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelLane:
    """One capacity x capability lane in the AURO family."""

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
    model_class: str = ""
    capabilities: List[str] = field(default_factory=list)
    deploy_profiles: List[str] = field(default_factory=list)
    checkpoint_evidence_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_target": self.parameter_target,
            "tier": self.tier.value,
            "model_class": self.model_class or self.tier.value,
            "status": self.status,
            "architecture": self.architecture.to_dict(),
            "subagent_roles": [r.value for r in self.subagent_roles],
            "capabilities": list(self.capabilities or [r.value for r in self.subagent_roles]),
            "can_embed_subagents": self.can_embed_subagents,
            "embeddable_tiers": [t.value for t in self.embeddable_tiers],
            "deploy_profiles": list(self.deploy_profiles),
            "checkpoint_evidence_required": self.checkpoint_evidence_required,
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
    evidence_refs: List[str] = field(default_factory=list)
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
            "evidence_refs": list(self.evidence_refs),
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
    """Full AURO family charter in typed form."""

    family_id: str
    family_name: str
    status: str
    lanes: List[ModelLane]
    polyglot_types: Sequence[str] = ("python", "julia", "haskell")
    claim_boundary: str = (
        "defines architecture, composition, and sub-agent contracts only; "
        "exact promoted checkpoint evidence is required for capability claims"
    )
    composition: Dict[str, Any] = field(default_factory=dict)
    claim_boundaries: Sequence[str] = CANONICAL_CLAIM_BOUNDARIES

    def model_ids(self) -> List[str]:
        return [lane.model_id for lane in self.lanes]

    def get_lane(self, model_id: str) -> Optional[ModelLane]:
        for lane in self.lanes:
            if lane.model_id == model_id:
                return lane
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.model-family.v2",
            "contract_version": FAMILY_CONTRACT_VERSION,
            "family_id": self.family_id,
            "family_name": self.family_name,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "claim_boundaries": list(self.claim_boundaries),
            "polyglot_types": list(self.polyglot_types),
            "composition": dict(self.composition),
            "lanes": [lane.to_dict() for lane in self.lanes],
        }
