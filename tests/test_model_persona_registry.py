from __future__ import annotations

from pathlib import Path

from auro_native_llm.model.registry import MODEL_BY_ID, MODELS, NEURO_FEATURE, model_manifest
from auro_native_llm.production_fleet.capabilities import BUILTINS
from auro_native_llm.production_fleet.personas import PERSONAS, get_persona, persona_manifest

DOC = Path("docs/MODELS_PERSONAS_AND_FEATURES.md")


def test_model_registry_is_unique_and_complete():
    assert len(MODELS) == len(MODEL_BY_ID)
    assert {item.id for item in MODELS} == set(MODEL_BY_ID)
    for model in MODELS:
        assert model.id
        assert model.intended_use
        assert model.features
        assert model.checkpoint_status
        assert model.claim_boundary
        assert model.declared_context_tokens > 0


def test_every_persona_has_valid_models_and_capabilities():
    capability_names = {item.name for item in BUILTINS}
    for persona in PERSONAS:
        assert persona.preferred_models
        assert all(model_id in MODEL_BY_ID for model_id in persona.preferred_models)
        assert persona.capability_prefixes
        for prefix in persona.capability_prefixes:
            assert any(name == prefix or name.startswith(prefix + ".") for name in capability_names), prefix


def test_execution_personas_are_server_authoritative():
    for persona in PERSONAS:
        if any(prefix in {"build", "office", "browser.task", "wallet"} for prefix in persona.capability_prefixes):
            assert persona.execution_mode == "server-approved-only"


def test_persona_parameter_and_memory_boundaries():
    manifest = persona_manifest()
    assert manifest["parameter_accounting"].startswith("personas share")
    assert manifest["execution_authority"] == "server"
    assert manifest["memory_authority"] == "untrusted evidence only"
    assert all(item.memory_mode == "privacy-filtered-retrieval" for item in PERSONAS)


def test_registry_manifests_are_serializable_and_truthful():
    models = model_manifest()
    personas = persona_manifest()
    assert models["schema"] == "auro.model-feature-registry.v2"
    assert personas["schema"] == "auro.persona-registry.v1"
    assert models["rules"]["architecture_is_not_training_evidence"] is True
    assert models["rules"]["declared_context_is_not_verified_quality"] is True
    assert models["rules"]["neuromorphic_residual_is_not_quality_evidence"] is True
    assert models["rules"]["neuromorphic_ceu_is_not_physical_energy"] is True
    assert get_persona("browser_brain").execution_mode == "server-approved-only"
    assert MODEL_BY_ID["HIM-native-v0"].checkpoint_status == "fixture-only"
    assert "not assistant quality" in MODEL_BY_ID["HIM-native-v0"].claim_boundary


def test_neuromorphic_residual_is_only_registered_on_standard_auro_moe_family():
    expected = {"Auro-156K", "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"}
    actual = {model.id for model in MODELS if NEURO_FEATURE in model.features}
    assert actual == expected
    assert NEURO_FEATURE not in MODEL_BY_ID["AURO-ST-14B"].features
    assert "not claimed wired here" in MODEL_BY_ID["AURO-ST-14B"].claim_boundary


def test_documentation_covers_every_model_and_persona():
    text = DOC.read_text(encoding="utf-8")
    for model in MODELS:
        assert model.id in text
    for persona in PERSONAS:
        assert persona.name in text
    assert "server-authoritative" in text
    assert "untrusted evidence" in text
    assert "architecture targets" in text
