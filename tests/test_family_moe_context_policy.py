from auro_native_llm.model import (
    ATOMIC_POLICY_VERSION,
    CONTEXT_MULTIPLIER,
    POLICY_VERSION,
    family_config,
)


EXPECTED_DEV_CONTEXTS = {
    "Auro-156K": 1024,
    "Auro-250M": 4096,
    "Auro-500M": 8192,
    "Auro-2B": 2048,
    "Auro-4B": 8192,
    "Auro-8B": 16384,
    "Auro-14B": 16384,
    "Auro-100B": 32768,
}

EXPECTED_FULL_CONTEXTS = {
    "Auro-156K": 1024,
    "Auro-250M": 4096,
    "Auro-500M": 8192,
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
    assert config.extra["architecture_configuration_is_not_checkpoint"] is True
    assert config.extra["checkpoint_evidence_required"] is True


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


def test_atomic_lanes_are_executable_configuration_contracts():
    expected = {
        "Auro-156K": (156_000, 64, 2, 4, 2),
        "Auro-250M": (250_000_000, 192, 6, 3, 1),
        "Auro-500M": (500_000_000, 256, 8, 4, 1),
    }
    for model_id, geometry in expected.items():
        config = family_config(model_id, mode="dev")
        target, hidden, layers, heads, kv_heads = geometry
        assert config.model_id == model_id
        assert config.parameter_target == target
        assert config.hidden_dim == hidden
        assert config.num_layers == layers
        assert config.num_heads == heads
        assert config.num_kv_heads == kv_heads
        assert config.extra["atomic_family_policy"] == ATOMIC_POLICY_VERSION
        assert config.extra["checkpoint_quality_verified"] is False
        assert config.extra["checkpoint_promotion_verified"] is False


def test_atomic_full_geometry_matches_declared_targets():
    two_fifty = family_config("Auro-250M", mode="full")
    five_hundred = family_config("Auro-500M", mode="full")
    assert (two_fifty.hidden_dim, two_fifty.num_layers, two_fifty.num_heads, two_fifty.num_kv_heads) == (768, 16, 12, 4)
    assert (five_hundred.hidden_dim, five_hundred.num_layers, five_hundred.num_heads, five_hundred.num_kv_heads) == (1024, 24, 16, 4)
