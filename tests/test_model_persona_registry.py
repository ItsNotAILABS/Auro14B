from __future__ import annotations

from auro_native_llm.model.registry import MODEL_BY_ID, MODELS, model_manifest
from auro_native_llm.production_fleet.capabilities import BUILTINS
from auro_native_llm.production_fleet.personas import PERSONAS, get_persona, persona_manifest


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
    assert models["schema"] == "auro.model-feature-registry.v1"
    assert personas["schema"] == "auro.persona-registry.v1"
    assert models["rules"]["architecture_is_not_training_evidence"] is True
    assert models["rules"]["declared_context_is_not_verified_quality"] is True
    assert get_persona("browser_brain").execution_mode == "server-approved-only"
    assert MODEL_BY_ID["HIM-native-v0"].checkpoint_status == "fixture-only"
    assert "not assistant quality" in MODEL_BY_ID["HIM-native-v0"].claim_boundary
