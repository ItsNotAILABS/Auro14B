"""Auro model-family registry from 156K through 100B.

The registry preserves architecture and checkpoint claim boundaries while
exposing typed lanes for capability-first, capacity-aware sub-agent routing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.receipt import emit_receipt, load_json_config
from auro_native_llm.types import (
    FAMILY_ID,
    FAMILY_PARAMETER_TARGETS,
    ArchitectureSpec,
    FamilyManifest,
    MODEL_ID_TO_TIER,
    ModelLane,
    ModelTier,
    SubAgentRole,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FAMILY_CONFIG = _REPO_ROOT / "native_llm" / "configs" / "auro_family.json"
_FAMILY_DIR = _REPO_ROOT / "native_llm" / "configs" / "family"

_BUILTIN_ARCHITECTURE: Dict[str, ArchitectureSpec] = {
    "Auro-156K": ArchitectureSpec(64, 2, 4, 2, 64, 1_024, 1_024),
    "Auro-250M": ArchitectureSpec(768, 18, 12, 4, 2_560, 4_096, 64_000),
    "Auro-500M": ArchitectureSpec(1_024, 24, 16, 4, 3_584, 8_192, 64_000),
    "Auro-2B": ArchitectureSpec(2_048, 24, 16, 4, 5_632, 8_192, 128_000),
    "Auro-4B": ArchitectureSpec(3_072, 28, 24, 4, 8_192, 16_384, 128_000),
    "Auro-8B": ArchitectureSpec(4_096, 32, 32, 8, 14_336, 32_768, 128_000),
    "Auro-14B": ArchitectureSpec(5_120, 48, 40, 8, 13_824, 32_768, 128_000),
    "Auro-100B": ArchitectureSpec(12_288, 80, 96, 16, 32_768, 131_072, 256_000),
}

_BUILTIN_ROLES: Dict[str, List[str]] = {
    "Auro-156K": ["routing_seed", "classifier", "json_repair", "tool_selection", "style_guard"],
    "Auro-250M": ["intent_extract", "retrieval_filter", "structured_transform", "code_triage", "memory_consolidation", "semantic_outline"],
    "Auro-500M": ["tool_execution_plan", "code_patch", "evidence_review", "local_worker", "expert_consensus", "text_expansion", "creative_branch"],
    "Auro-2B": ["router", "tool_call", "embed_fast", "spectral_triage"],
    "Auro-4B": ["code_edit", "spectral_match", "json_struct", "tool_plan"],
    "Auro-8B": ["reason", "plan", "critique", "spectral_explain"],
    "Auro-14B": ["orchestrator", "council_chair", "instruct_dev", "multi_agent_router"],
    "Auro-100B": ["frontier_research", "long_horizon", "safety_review", "deep_council"],
}

_BUILTIN_EMBEDDABLE: Dict[str, List[str]] = {
    "Auro-156K": [],
    "Auro-250M": ["atomic"],
    "Auro-500M": ["atomic"],
    "Auro-2B": ["atomic"],
    "Auro-4B": ["atomic", "edge"],
    "Auro-8B": ["atomic", "edge", "specialist"],
    "Auro-14B": ["atomic", "edge", "specialist", "general"],
    "Auro-100B": ["atomic", "edge", "specialist", "general", "orchestrator"],
}


def model_id_to_tier_safe(model_id: str) -> ModelTier:
    if model_id in MODEL_ID_TO_TIER:
        return MODEL_ID_TO_TIER[model_id]
    upper = model_id.upper()
    if upper.endswith("K") or "250M" in upper or "500M" in upper:
        return ModelTier.ATOMIC
    if "2B" in upper:
        return ModelTier.EDGE
    if "4B" in upper:
        return ModelTier.SPECIALIST
    if "8B" in upper:
        return ModelTier.GENERAL
    if "14B" in upper:
        return ModelTier.ORCHESTRATOR
    if "100B" in upper or "200B" in upper:
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
        context_window_tokens_target=int(data.get("context_window_tokens_target", 8_192)),
        vocab_size_target=int(data.get("vocab_size_target", 128_000)),
        family=str(data.get("family", "decoder-only-transformer")),
        objective=str(data.get("objective", "causal-language-modeling")),
        activation=str(data.get("activation", "silu")),
        normalization=str(data.get("normalization", "rmsnorm")),
        position_encoding=str(data.get("position_encoding", "rope")),
    )


def _lane_from_config(path: Path) -> ModelLane:
    cfg = load_json_config(path)
    model_id = str(cfg["model_id"])
    tier = ModelTier(str(cfg.get("tier") or model_id_to_tier_safe(model_id).value))
    roles = _parse_roles(list(cfg.get("subagent_roles", _BUILTIN_ROLES.get(model_id, []))))
    embeddable = _parse_tiers(list(cfg.get("embeddable_tiers", _BUILTIN_EMBEDDABLE.get(model_id, []))))
    arch_data = cfg.get("architecture") or {}
    architecture = _BUILTIN_ARCHITECTURE[model_id] if not arch_data and model_id in _BUILTIN_ARCHITECTURE else _arch_from_dict(arch_data)
    return ModelLane(
        model_id=model_id,
        parameter_target=int(cfg.get("parameter_target", FAMILY_PARAMETER_TARGETS.get(model_id, 0))),
        tier=tier,
        status=str(cfg.get("status", "development-target-not-trained-checkpoint")),
        architecture=architecture,
        subagent_roles=roles,
        can_embed_subagents=bool(cfg.get("can_embed_subagents", bool(embeddable))),
        embeddable_tiers=embeddable,
        purpose=str(cfg.get("purpose", "")),
        config_path=str(path),
    )


def builtin_family() -> FamilyManifest:
    lanes: List[ModelLane] = []
    for model_id, arch in _BUILTIN_ARCHITECTURE.items():
        tier = model_id_to_tier_safe(model_id)
        roles = _parse_roles(_BUILTIN_ROLES[model_id])
        embeddable = _parse_tiers(_BUILTIN_EMBEDDABLE[model_id])
        lanes.append(
            ModelLane(
                model_id=model_id,
                parameter_target=FAMILY_PARAMETER_TARGETS[model_id],
                tier=tier,
                status="architecture-target-not-trained-checkpoint" if model_id == "Auro-100B" else "development-target-not-trained-checkpoint",
                architecture=arch,
                subagent_roles=roles,
                can_embed_subagents=bool(embeddable),
                embeddable_tiers=embeddable,
                purpose=f"{model_id} multi-embedded sub-agent lane",
                config_path=str(_FAMILY_DIR / f"{model_id.lower().replace('-', '_')}.json"),
            )
        )
    return FamilyManifest(family_id=FAMILY_ID, family_name="Auro Native LLM Family", status="production-scaffold-not-trained-checkpoint", lanes=lanes)


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
        arch = _BUILTIN_ARCHITECTURE.get(model_id, ArchitectureSpec(1_024, 12, 8, 2, 2_816, 4_096, 32_000))
        lanes.append(
            ModelLane(
                model_id=model_id,
                parameter_target=int(entry.get("parameter_target", FAMILY_PARAMETER_TARGETS.get(model_id, 0))),
                tier=tier,
                status=str(entry.get("status", "development-target-not-trained-checkpoint")),
                architecture=arch,
                subagent_roles=roles,
                can_embed_subagents=bool(entry.get("can_embed_subagents", bool(embeddable))),
                embeddable_tiers=embeddable,
                purpose=str(entry.get("deploy_profile", "")),
                config_path=str(rel) if rel else None,
            )
        )
    if not lanes:
        return builtin_family()
    return FamilyManifest(
        family_id=str(charter.get("family_id", FAMILY_ID)),
        family_name=str(charter.get("family_name", "Auro Native LLM Family")),
        status=str(charter.get("status", "production-scaffold-not-trained-checkpoint")),
        lanes=lanes,
        polyglot_types=tuple(charter.get("polyglot_types", ["python", "julia", "haskell"])),
        claim_boundary=str(charter.get("claim_boundary", "defines architecture and multi-embedded sub-agent contracts only; no trained weights claimed")),
    )


def validate_family(manifest: FamilyManifest) -> List[str]:
    errors: List[str] = []
    expected = {"Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"}
    missing = expected - set(manifest.model_ids())
    if missing:
        errors.append(f"missing family lanes: {sorted(missing)}")
    for lane in manifest.lanes:
        if not any(marker in lane.status for marker in ("not-trained", "architecture-target", "reference")):
            errors.append(f"{lane.model_id}: status must declare its checkpoint boundary")
        if lane.parameter_target <= 0:
            errors.append(f"{lane.model_id}: parameter_target must be positive")
        if lane.architecture.hidden_size <= 0 or lane.architecture.layers <= 0:
            errors.append(f"{lane.model_id}: invalid architecture dimensions")
        if lane.can_embed_subagents and not lane.embeddable_tiers:
            errors.append(f"{lane.model_id}: can_embed_subagents but empty embeddable_tiers")
    if "python" not in manifest.polyglot_types:
        errors.append("polyglot_types must include python")
    return errors


def list_model_ids(config_path: Optional[str | Path] = None) -> List[str]:
    return load_family(config_path).model_ids()


def get_lane(model_id: str, config_path: Optional[str | Path] = None) -> Optional[ModelLane]:
    return load_family(config_path).get_lane(model_id)


def tier_model_id(tier: ModelTier) -> str:
    from auro_native_llm.types import TIER_TO_MODEL_ID
    return TIER_TO_MODEL_ID[tier]


def emit_family_receipt(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else _DEFAULT_FAMILY_CONFIG
    manifest = load_family(path if path.exists() else None)
    errors = validate_family(manifest)
    if errors:
        raise SystemExit("family validation failed: " + "; ".join(errors))
    return emit_receipt("family_charter", path if path.exists() else "builtin", manifest.to_dict())
