import json

import pytest

from auro_native_llm.model.auro4b import (
    architecture_to_overrides,
    build_auro4b_config,
)
from auro_native_llm.model.auro4b_architecture import (
    Auro4BArchitecture,
    FULL_ARCHITECTURE,
    PROXY_ARCHITECTURE,
)


def test_full_geometry_is_internally_consistent_and_near_four_billion_active():
    architecture = FULL_ARCHITECTURE
    architecture.validate()
    estimate = architecture.parameter_estimate()
    assert architecture.hidden_dim == architecture.num_attention_heads * architecture.head_dim
    assert architecture.num_attention_heads // architecture.num_kv_heads == 3
    assert architecture.use_moe is True
    assert architecture.num_experts == 8
    assert architecture.top_k_experts == 2
    assert architecture.max_seq_len == 65536
    assert 3_900_000_000 <= estimate["active_total"] <= 4_150_000_000
    assert estimate["total"] == estimate["active_total"]
    assert estimate["stored_total"] > estimate["active_total"]
    assert abs(estimate["delta_from_target"]) / architecture.parameter_target < 0.04


def test_full_contract_matches_backbone_without_unsupported_claims():
    payload = build_auro4b_config("full")
    assert payload["attention"] == "grouped_query_attention"
    assert payload["positional_encoding"] == "rotary"
    assert payload["normalization"] == "rms_norm"
    assert payload["activation"] == "swiglu"
    assert payload["use_moe"] is True
    assert payload["context_multiplier_from_v1"] == 4
    assert payload["claims"]["trained_general_knowledge"] is False
    assert payload["claims"]["hallucination_reduction"] is False
    assert payload["claims"]["long_context_extrapolation_verified"] is False
    assert payload["claims"]["moe_routing_quality_verified"] is False


def test_proxy_preserves_gqa_moe_and_fourfold_context():
    PROXY_ARCHITECTURE.validate()
    assert PROXY_ARCHITECTURE.num_attention_heads // PROXY_ARCHITECTURE.num_kv_heads == 3
    assert PROXY_ARCHITECTURE.use_moe is True
    assert PROXY_ARCHITECTURE.num_experts == 8
    assert PROXY_ARCHITECTURE.max_seq_len == 4096
    assert PROXY_ARCHITECTURE.parameter_estimate()["active_total"] < 100_000_000


def test_overrides_enable_moe_gqa_and_long_context():
    overrides = architecture_to_overrides(FULL_ARCHITECTURE)
    assert overrides["use_moe"] is True
    assert overrides["num_experts"] == 8
    assert overrides["top_k_experts"] == 2
    assert overrides["moe_every"] == 4
    assert overrides["num_kv_heads"] == 8
    assert overrides["ffn_dim"] == 8192
    assert overrides["max_seq_len"] == 65536
    assert overrides["extra"]["checkpoint_constitution"] == "auro.substrate.checkpoint.v1"
    assert overrides["extra"]["long_context_quality_verified"] is False


def test_invalid_head_geometry_is_rejected():
    with pytest.raises(ValueError):
        Auro4BArchitecture(hidden_dim=3000).validate()


def test_non_moe_architecture_is_rejected():
    with pytest.raises(ValueError):
        Auro4BArchitecture(use_moe=False).validate()


def test_config_is_json_serializable():
    json.dumps(build_auro4b_config("full"), sort_keys=True)
