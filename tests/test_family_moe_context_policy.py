from auro_native_llm.model import (
    CONTEXT_MULTIPLIER,
    POLICY_VERSION,
    family_config,
)


EXPECTED_DEV_CONTEXTS = {
    "Auro-156K": 1024,
    "Auro-2B": 2048,
    "Auro-4B": 8192,
    "Auro-8B": 16384,
    "Auro-14B": 16384,
    "Auro-100B": 32768,
}

EXPECTED_FULL_CONTEXTS = {
    "Auro-2B": 8192,
    "Auro-4B": 32768,
    "Auro-8B": 32768,
    "Auro-14B": 65536,
    "Auro-100B": 131072,
}


def assert_upgraded(config, expected_context):
    assert config.use_moe is True
    assert config.num_experts >= 8
    assert 2 <= config.top_k_experts <= config.num_experts
    assert config.max_seq_len == expected_context
    assert config.extra["family_upgrade_policy"] == POLICY_VERSION
    assert config.extra["context_multiplier"] == CONTEXT_MULTIPLIER
    assert config.extra["declared_max_seq_len"] == expected_context
    assert config.extra["all_family_members_moe"] is True
    assert config.extra["long_context_quality_verified"] is False
    assert config.extra["long_context_training_required"] is True


def test_every_dev_family_member_is_moe_and_fourfold_context():
    for model_id, expected_context in EXPECTED_DEV_CONTEXTS.items():
        assert_upgraded(family_config(model_id, mode="dev"), expected_context)


def test_every_full_family_member_is_moe_and_fourfold_context():
    for model_id, expected_context in EXPECTED_FULL_CONTEXTS.items():
        assert_upgraded(family_config(model_id, mode="full"), expected_context)


def test_policy_is_idempotent():
    first = family_config("Auro-2B", mode="dev")
    second_context = first.max_seq_len
    from auro_native_llm.model import apply_family_upgrade
    second = apply_family_upgrade(first)
    assert second.max_seq_len == second_context


def test_seed_model_is_now_executable_configuration():
    config = family_config("Auro-156K")
    assert config.model_id == "Auro-156K"
    assert config.parameter_target == 156_000
    assert config.hidden_dim == 64
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.num_kv_heads == 2
    assert config.max_seq_len == 1024
