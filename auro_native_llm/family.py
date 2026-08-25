"""Canonical AURO family registry and composition contract.

The registry extends the existing model family rather than inventing a second
ladder. It exposes atomic, micro, core, orchestrator, and frontier lanes while
keeping architecture targets, local artifacts, promoted checkpoints, and
verified capabilities as separate evidence states.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.receipt import emit_receipt, load_json_config
from auro_native_llm.types import (
    ATOMIC_MODEL_IDS,
    AURO_2B_SPECIALIST_TRIAD,
    CANONICAL_CLAIM_BOUNDARIES,
    FAMILY_ID,
    FAMILY_PARAMETER_TARGETS,
    ROLE_DEFAULT_MODEL_ID,
    ArchitectureSpec,
    FamilyManifest,
    ModelLane,
    ModelTier,
    SubAgentRole,
    TIER_TO_MODEL_ID,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FAMILY_CONFIG = _REPO_ROOT / "native_llm" / "configs" / "auro_family.json"
_FAMILY_DIR = _REPO_ROOT / "native_llm" / "configs" / "family"

CANONICAL_MODEL_ORDER = (
    "Auro-156K",
    "Auro-250M",
    "Auro-500M",
    "Auro-2B",
    "Auro-4B",
    "Auro-8B",
    "Auro-14B",
    "Auro-100B",
)

_BUILTIN_ARCHITECTURE: Dict[str, ArchitectureSpec] = {
    "Auro-156K": ArchitectureSpec(64, 2, 4, 2, 64, 1024, 1024, experts=8, top_k=2, moe_every=2),
    "Auro-250M": ArchitectureSpec(768, 16, 12, 4, 2048, 4096, 64000, experts=8, top_k=2, moe_every=2),
    "Auro-500M": ArchitectureSpec(1024, 24, 16, 4, 4096, 8192, 64000, experts=8, top_k=2, moe_every=2),
    "Auro-2B": ArchitectureSpec(2048, 24, 16, 4, 5632, 8192, 128000, experts=8, top_k=2, moe_every=2),
    "Auro-4B": ArchitectureSpec(3072, 28, 24, 4, 8192, 16384, 128000, experts=8, top_k=2, moe_every=2),
    "Auro-8B": ArchitectureSpec(4096, 32, 32, 8, 14336, 32768, 128000, experts=16, top_k=4, moe_every=2),
    "Auro-14B": ArchitectureSpec(5120, 48, 40, 8, 13824, 65536, 128000, experts=16, top_k=4, moe_every=2),
    "Auro-100B": ArchitectureSpec(12288, 80, 96, 16, 32768, 131072, 256000, experts=32, top_k=4, moe_every=2),
}

_BUILTIN_ROLES: Dict[str, List[str]] = {
    "Auro-156K": ["routing_seed", "classifier", "json_repair", "tool_selection"],
    "Auro-250M": ["intent_extract", "retrieval_filter", "structured_transform", "code_triage", "memory_consolidation", "semantic_outline"],
    "Auro-500M": ["tool_execution_plan", "code_patch", "evidence_review", "local_worker", "expert_consensus", "text_expansion"],
    "Auro-2B": ["router", "tool_call", "embed_fast", "spectral_triage"],
    "Auro-4B": ["code_edit", "spectral_match", "json_struct", "tool_plan"],
    "Auro-8B": ["reason", "plan", "critique", "spectral_explain"],
    "Auro-14B": ["orchestrator", "council_chair", "instruct_dev", "multi_agent_router"],
    "Auro-100B": ["frontier_research", "long_horizon", "safety_review", "deep_council"],
}

_BUILTIN_EMBEDDABLE: Dict[str, List[str]] = {
    "Auro-156K": [],
    "Auro-250M": [],
    "Auro-500M": [],
    "Auro-2B": ["atomic"],
    "Auro-4B": ["atomic", "edge"],
    "Auro-8B": ["atomic", "edge", "specialist"],
    "Auro-14B": ["atomic", "edge", "specialist", "general"],
    "Auro-100B": ["atomic", "edge", "specialist", "general", "orchestrator"],
}

_DEPLOY_PROFILES: Dict[str, List[str]] = {
    "Auro-156K": ["wasm", "embedded", "high-multiplicity-swarm"],
    "Auro-250M": ["phone", "browser-wasm", "cpu", "embedded-expert"],
    "Auro-500M": ["phone-high-memory", "laptop", "edge-gpu", "embedded-expert"],
    "Auro-2B": ["phone-high-memory", "laptop", "private-edge-server"],
    "Auro-4B": ["laptop", "workstation", "private-api"],
    "Auro-8B": ["workstation", "server"],
    "Auro-14B": ["gpu-server", "distributed-private-runtime"],
    "Auro-100B": ["distributed-training", "distributed-inference"],
}


def _default_composition() -> Dict[str, Any]:
    return {
        "schema": "nexus.model-council.v1",
        "routing_policy": "capability-first-then-smallest-capable-lane",
        "atomic_lanes": list(ATOMIC_MODEL_IDS),
        "parent_model": "Auro-2B",
        "specialist_triad": list(AURO_2B_SPECIALIST_TRIAD),
        "task_capsules": {
            "bounded": True,
            "full_parent_context_broadcast": False,
            "required": ["task_id", "parent_model_id", "expert_model_id", "role", "objective", "constraints", "evidence_refs", "max_output_tokens", "capsule_hash"],
        },
        "mesie_offload": "ingress, specialist, atomic-worker, consensus, and egress stages",
        "conversational_renderer": "python-wasm-fluidizer",
        "claim_boundary": "specialist identities are routing contracts until exact checkpoint or adapter evidence exists",
    }


def model_id_to_tier_safe(model_id: str) -> ModelTier:
    from auro_native_llm.types import MODEL_ID_TO_TIER

    if model_id in MODEL_ID_TO_TIER:
        return MODEL_ID_TO_TIER[model_id]
    lowered = model_id.lower()
    if any(token in lowered for token in ("156k", "250m", "500m")):
        return ModelTier.ATOMIC
    if "2b" in lowered:
        return ModelTier.EDGE
    if "4b" in lowered:
        return ModelTier.SPECIALIST
    if "8b" in lowered:
        return ModelTier.GENERAL
    if "14b" in lowered:
        return ModelTier.ORCHESTRATOR
    if "100b" in lowered or "200b" in lowered:
        return ModelTier.FRONTIER
    return ModelTier.GENERAL


def _parse_roles(raw: List[str]) -> List[SubAgentRole]:
    roles: List[SubAgentRole] = []
    for item in raw:
        try:
            roles.append(SubAgentRole(item))
        except ValueError:
            continue
    return roles


def _parse_tiers(raw: List[str]) -> List[ModelTier]:
    tiers: List[ModelTier] = []
    for item in raw:
        try:
            tiers.append(ModelTier(item))
        except ValueError:
            continue
    return tiers


def _arch_from_dict(data: Dict[str, Any]) -> ArchitectureSpec:
    return ArchitectureSpec(
        hidden_size=int(data["hidden_size"]),
        layers=int(data["layers"]),
        attention_heads=int(data["attention_heads"]),
        kv_heads=int(data["kv_heads"]),
        intermediate_size=int(data.get("intermediate_size", data.get("hidden_size", 0) * 4)),
        context_window_tokens_target=int(data.get("context_window_tokens_target", 8192)),
        vocab_size_target=int(data.get("vocab_size_target", 128000)),
        family=str(data.get("family", "decoder-only-transformer")),
        objective=str(data.get("objective", "causal-language-modeling")),
        activation=str(data.get("activation", "silu")),
        normalization=str(data.get("normalization", "rmsnorm")),
        position_encoding=str(data.get("position_encoding", "rope")),
        attention_type=str(data.get("attention_type", "gqa")),
        experts=int(data.get("experts", data.get("num_experts", 1))),
        top_k=int(data.get("top_k", data.get("top_k_experts", 1))),
        moe_every=int(data.get("moe_every", 0)),
    )


def _lane_from_config(path: Path) -> ModelLane:
    cfg = load_json_config(path)
    model_id = str(cfg["model_id"])
    tier = ModelTier(str(cfg.get("tier", model_id_to_tier_safe(model_id).value)))
    roles = _parse_roles(list(cfg.get("subagent_roles", _BUILTIN_ROLES.get(model_id, []))))
    embeddable = _parse_tiers(list(cfg.get("embeddable_tiers", _BUILTIN_EMBEDDABLE.get(model_id, []))))
    arch_data = cfg.get("architecture") or {}
    architecture = _BUILTIN_ARCHITECTURE[model_id] if not arch_data and model_id in _BUILTIN_ARCHITECTURE else _arch_from_dict(arch_data)
    return ModelLane(
        model_id=model_id,
        parameter_target=int(cfg.get("parameter_target", FAMILY_PARAMETER_TARGETS.get(model_id, 0))),
        tier=tier,
        model_class=str(cfg.get("model_class", tier.value)),
        status=str(cfg.get("status", "development-target-not-trained-checkpoint")),
        architecture=architecture,
        subagent_roles=roles,
        capabilities=list(cfg.get("capabilities", [role.value for role in roles])),
        can_embed_subagents=bool(cfg.get("can_embed_subagents", bool(embeddable))),
        embeddable_tiers=embeddable,
        deploy_profiles=list(cfg.get("deploy_profiles", _DEPLOY_PROFILES.get(model_id, []))),
        checkpoint_evidence_required=bool(cfg.get("checkpoint_evidence_required", True)),
        purpose=str(cfg.get("purpose", "")),
        config_path=str(path),
    )


def builtin_family() -> FamilyManifest:
    lanes: List[ModelLane] = []
    for model_id in CANONICAL_MODEL_ORDER:
        arch = _BUILTIN_ARCHITECTURE[model_id]
        tier = model_id_to_tier_safe(model_id)
        roles = _parse_roles(_BUILTIN_ROLES[model_id])
        embeddable = _parse_tiers(_BUILTIN_EMBEDDABLE[model_id])
        lanes.append(
            ModelLane(
                model_id=model_id,
                parameter_target=FAMILY_PARAMETER_TARGETS[model_id],
                tier=tier,
                model_class="atomic" if tier == ModelTier.ATOMIC else tier.value,
                status="architecture-target-not-trained-checkpoint",
                architecture=arch,
                subagent_roles=roles,
                capabilities=[role.value for role in roles],
                can_embed_subagents=bool(embeddable),
                embeddable_tiers=embeddable,
                deploy_profiles=_DEPLOY_PROFILES[model_id],
                purpose=f"{model_id} capacity x capability lane",
                config_path=str(_FAMILY_DIR / f"{model_id.lower().replace('-', '_')}.json"),
            )
        )
    return FamilyManifest(
        family_id=FAMILY_ID,
        family_name="AURO Native LLM Family",
        status="production-scaffold-not-trained-checkpoint",
        lanes=lanes,
        composition=_default_composition(),
    )


def load_family(config_path: Optional[str | Path] = None) -> FamilyManifest:
    path = Path(config_path) if config_path else _DEFAULT_FAMILY_CONFIG
    if not path.exists():
        return builtin_family()

    charter = load_json_config(path)
    lanes: List[ModelLane] = []
    for entry in charter.get("lanes", []):
        rel = entry.get("config")
        if rel:
            lane_path = _REPO_ROOT / rel if not Path(rel).is_absolute() else Path(rel)
            if lane_path.exists():
                lanes.append(_lane_from_config(lane_path))
                continue
        model_id = str(entry["model_id"])
        tier = ModelTier(str(entry.get("tier", model_id_to_tier_safe(model_id).value)))
        roles = _parse_roles(list(entry.get("default_roles", _BUILTIN_ROLES.get(model_id, []))))
        embeddable = _parse_tiers(list(entry.get("embeddable_tiers", _BUILTIN_EMBEDDABLE.get(model_id, []))))
        arch = _BUILTIN_ARCHITECTURE.get(model_id, ArchitectureSpec(1024, 12, 8, 2, 2816, 4096, 32000))
        lanes.append(
            ModelLane(
                model_id=model_id,
                parameter_target=int(entry.get("parameter_target", FAMILY_PARAMETER_TARGETS.get(model_id, 0))),
                tier=tier,
                model_class=str(entry.get("model_class", tier.value)),
                status=str(entry.get("status", "development-target-not-trained-checkpoint")),
                architecture=arch,
                subagent_roles=roles,
                capabilities=list(entry.get("capabilities", [role.value for role in roles])),
                can_embed_subagents=bool(entry.get("can_embed_subagents", bool(embeddable))),
                embeddable_tiers=embeddable,
                deploy_profiles=list(entry.get("deploy_profiles", _DEPLOY_PROFILES.get(model_id, []))),
                purpose=str(entry.get("deploy_profile", "")),
                config_path=str(rel) if rel else None,
            )
        )

    if not lanes:
        return builtin_family()

    return FamilyManifest(
        family_id=str(charter.get("family_id", FAMILY_ID)),
        family_name=str(charter.get("family_name", "AURO Native LLM Family")),
        status=str(charter.get("status", "production-scaffold-not-trained-checkpoint")),
        lanes=lanes,
        polyglot_types=tuple(charter.get("polyglot_types", ["python", "julia", "haskell"])),
        claim_boundary=str(charter.get("claim_boundary", "exact promoted checkpoint evidence is required for capability claims")),
        composition=dict(charter.get("composition", _default_composition())),
        claim_boundaries=tuple(charter.get("claim_boundaries", CANONICAL_CLAIM_BOUNDARIES)),
    )


def validate_family(manifest: FamilyManifest) -> List[str]:
    errors: List[str] = []
    found = manifest.model_ids()
    if found != list(CANONICAL_MODEL_ORDER):
        errors.append(f"family lanes must match canonical order: {list(CANONICAL_MODEL_ORDER)}; got {found}")
    for lane in manifest.lanes:
        if "not-trained" not in lane.status and "architecture-target" not in lane.status:
            errors.append(f"{lane.model_id}: status must preserve an architecture/checkpoint truth boundary")
        if lane.parameter_target <= 0:
            errors.append(f"{lane.model_id}: parameter_target must be positive")
        if lane.architecture.hidden_size <= 0 or lane.architecture.layers <= 0:
            errors.append(f"{lane.model_id}: invalid architecture dimensions")
        if lane.architecture.experts < lane.architecture.top_k or lane.architecture.top_k <= 0:
            errors.append(f"{lane.model_id}: invalid MoE expert routing geometry")
        if lane.tier == ModelTier.ATOMIC and lane.parameter_target >= 1_000_000_000:
            errors.append(f"{lane.model_id}: atomic lanes must be below one billion target parameters")
    two_b = manifest.get_lane("Auro-2B")
    if not two_b or not two_b.can_embed_subagents or ModelTier.ATOMIC not in two_b.embeddable_tiers:
        errors.append("Auro-2B must embed atomic sub-agent lanes")
    triad = list((manifest.composition or {}).get("specialist_triad", []))
    if triad != list(AURO_2B_SPECIALIST_TRIAD):
        errors.append(f"Auro-2B specialist triad mismatch: {triad}")
    missing_boundaries = sorted(set(CANONICAL_CLAIM_BOUNDARIES) - set(manifest.claim_boundaries))
    if missing_boundaries:
        errors.append(f"missing claim boundaries: {missing_boundaries}")
    if "python" not in manifest.polyglot_types:
        errors.append("polyglot_types must include python")
    return errors


def list_model_ids(config_path: Optional[str | Path] = None) -> List[str]:
    return load_family(config_path).model_ids()


def get_lane(model_id: str, config_path: Optional[str | Path] = None) -> Optional[ModelLane]:
    return load_family(config_path).get_lane(model_id)


def tier_model_id(tier: ModelTier) -> str:
    return TIER_TO_MODEL_ID[tier]


def preferred_model_for_role(role: SubAgentRole) -> str:
    return ROLE_DEFAULT_MODEL_ID[role]


def emit_family_receipt(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else _DEFAULT_FAMILY_CONFIG
    manifest = load_family(path if path.exists() else None)
    errors = validate_family(manifest)
    if errors:
        raise SystemExit("family validation failed: " + "; ".join(errors))
    payload = manifest.to_dict()
    return emit_receipt("family_charter", path if path.exists() else "builtin", payload)
