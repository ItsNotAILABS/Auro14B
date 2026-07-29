from auro_native_llm.family import load_family, validate_family
from auro_native_llm.model import family_config
from auro_native_llm.model.atomic_family import ATOMIC_LADDER, AURO_500M_TRIAD, architecture_for, sub2b_manifest
from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter
from auro_native_llm.types import ModelTier, SubAgentRole


def test_canonical_family_contains_every_sub2b_lane():
    family = load_family()
    assert {"Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B"} <= set(family.model_ids())
    assert validate_family(family) == []
    assert family.get_lane("Auro-2B").can_embed_subagents is True
    assert ModelTier.ATOMIC in family.get_lane("Auro-2B").embeddable_tiers


def test_atomic_architecture_accounting_separates_active_and_stored_capacity():
    for model_id in ("Auro-250M", "Auro-500M"):
        arch = architecture_for(model_id)
        accounting = arch.parameter_accounting()
        ratio = accounting["active_parameters_per_token_estimate"] / arch.parameter_target
        assert 0.90 <= ratio <= 1.10
        assert accounting["stored_parameters_estimate"] > accounting["active_parameters_per_token_estimate"]
        assert accounting["moe_layers"] > 0


def test_250m_and_500m_build_through_existing_moe_family_policy():
    config_250 = family_config("Auro-250M", mode="full")
    config_500 = family_config("Auro-500M", mode="full")
    assert config_250.use_moe and config_500.use_moe
    assert config_250.top_k_experts == config_500.top_k_experts == 2
    assert config_250.num_experts == config_500.num_experts == 8
    assert config_250.max_seq_len == ATOMIC_LADDER["Auro-250M"].context_window_tokens_target
    assert config_500.max_seq_len == ATOMIC_LADDER["Auro-500M"].context_window_tokens_target
    variant = family_config("Auro-500M-SENSUS", mode="full")
    assert variant.model_id == "Auro-500M-SENSUS"
    assert variant.parameter_target == 500_000_000


def test_2b_routes_atomic_roles_to_the_smallest_capable_lane():
    router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-2B")
    assert router.resolve_child(SubAgentRole.TOOL_SELECTION).model_id == "Auro-156K"
    assert router.resolve_child(SubAgentRole.INTENT_EXTRACT).model_id == "Auro-250M"
    assert router.resolve_child(SubAgentRole.EVIDENCE_REVIEW).model_id == "Auro-500M"


def test_triad_identity_is_three_distinct_specialization_contracts():
    assert [item.variant_id for item in AURO_500M_TRIAD] == [
        "Auro-500M-SENSUS",
        "Auro-500M-PRAXIS",
        "Auro-500M-VERBUM",
    ]
    manifest = sub2b_manifest()
    assert manifest["checkpoint_release_required"] is True
    assert manifest["parameter_accounting"].startswith("do not add agent instances")
