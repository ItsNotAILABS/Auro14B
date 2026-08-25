import json
from pathlib import Path

from auro_native_llm.family import (
    CANONICAL_MODEL_ORDER,
    load_family,
    preferred_model_for_role,
    validate_family,
)
from auro_native_llm.model.auro4b_architecture import FULL_ARCHITECTURE
from auro_native_llm.types import (
    AURO_2B_SPECIALIST_TRIAD,
    CANONICAL_CLAIM_BOUNDARIES,
    FAMILY_CONTRACT_VERSION,
    ModelTier,
    SubAgentRole,
)

ROOT = Path(__file__).resolve().parents[1]


def test_family_contract_is_versioned_and_complete():
    manifest = load_family()
    assert FAMILY_CONTRACT_VERSION == "2.1.0"
    assert manifest.model_ids() == list(CANONICAL_MODEL_ORDER)
    assert validate_family(manifest) == []


def test_atomic_lanes_and_two_b_parent_are_first_class():
    manifest = load_family()
    atomic = [lane for lane in manifest.lanes if lane.tier == ModelTier.ATOMIC]
    assert [lane.model_id for lane in atomic] == ["Auro-156K", "Auro-250M", "Auro-500M"]
    assert all(lane.parameter_target < 1_000_000_000 for lane in atomic)
    two_b = manifest.get_lane("Auro-2B")
    assert two_b is not None
    assert two_b.can_embed_subagents is True
    assert ModelTier.ATOMIC in two_b.embeddable_tiers
    assert manifest.composition["specialist_triad"] == list(AURO_2B_SPECIALIST_TRIAD)
    assert manifest.composition["task_capsules"]["full_parent_context_broadcast"] is False


def test_capability_first_routing_selects_smallest_declared_lane():
    assert preferred_model_for_role(SubAgentRole.JSON_REPAIR) == "Auro-156K"
    assert preferred_model_for_role(SubAgentRole.RETRIEVAL_FILTER) == "Auro-250M"
    assert preferred_model_for_role(SubAgentRole.EVIDENCE_REVIEW) == "Auro-500M"
    assert preferred_model_for_role(SubAgentRole.ROUTER) == "Auro-2B"
    assert preferred_model_for_role(SubAgentRole.REASON) == "Auro-8B"


def test_all_truth_boundaries_remain_machine_readable():
    manifest = load_family()
    assert set(CANONICAL_CLAIM_BOUNDARIES).issubset(set(manifest.claim_boundaries))
    assert "architecture-configuration-is-not-a-trained-checkpoint" in manifest.claim_boundaries
    assert "named-agent-is-not-a-separately-trained-model" in manifest.claim_boundaries
    assert "generated-answer-is-not-experimental-validation" in manifest.claim_boundaries


def test_four_b_active_and_stored_parameter_accounting_is_not_conflated():
    estimate = FULL_ARCHITECTURE.parameter_estimate()
    assert estimate["active_total"] == 4_026_977_280
    assert estimate["stored_total"] == 7_650_855_936
    assert estimate["stored_total"] > estimate["active_total"]
    assert FULL_ARCHITECTURE.top_k_experts == 2
    assert FULL_ARCHITECTURE.num_experts == 8


def test_json_charter_matches_python_contract():
    raw = json.loads((ROOT / "native_llm" / "configs" / "auro_family.json").read_text(encoding="utf-8"))
    assert raw["schema"] == "auro.model-family.v2"
    assert raw["contract_version"] == "2.1.0"
    assert [lane["model_id"] for lane in raw["lanes"]] == list(CANONICAL_MODEL_ORDER)
    assert raw["composition"]["specialist_triad"] == list(AURO_2B_SPECIALIST_TRIAD)
    assert raw["composition"]["mesie_offload"]
    assert raw["composition"]["conversational_renderer"] == "python-wasm-fluidizer"
