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


def test_full_geometry_is_internally_consistent_and_near_four_billion():
    architecture = FULL_ARCHITECTURE
    architecture.validate()
    estimate = architecture.parameter_estimate()
    assert architecture.hidden_dim == architecture.num_attention_heads * architecture.head_dim
    assert architecture.num_attention_heads // architecture.num_kv_heads == 3
    assert 3_900_000_000 <= estimate["total"] <= 4_150_000_000
    assert abs(estimate["delta_from_target"]) / architecture.parameter_target < 0.04


def test_full_contract_matches_paper_backbone_without_unsupported_claims():
    payload = build_auro4b_config("full")
    assert payload["attention"] == "grouped_query_attention"
    assert payload["positional_encoding"] == "rotary"
    assert payload["normalization"] == "rms_norm"
    assert payload["activation"] == "swiglu"
    assert payload["claims"]["trained_general_knowledge"] is False
    assert payload["claims"]["hallucination_reduction"] is False
    assert payload["claims"]["long_context_extrapolation_verified"] is False


def test_proxy_preserves_gqa_ratio_and_is_executable_scale():
    PROXY_ARCHITECTURE.validate()
    assert PROXY_ARCHITECTURE.num_attention_heads // PROXY_ARCHITECTURE.num_kv_heads == 3
    assert PROXY_ARCHITECTURE.parameter_estimate()["total"] < 100_000_000


def test_overrides_disable_generic_moe_and_enable_gqa():
    overrides = architecture_to_overrides(FULL_ARCHITECTURE)
    assert overrides["use_moe"] is False
    assert overrides["num_kv_heads"] == 8
    assert overrides["ffn_dim"] == 10240
    assert overrides["extra"]["checkpoint_constitution"] == "auro.substrate.checkpoint.v1"


def test_invalid_head_geometry_is_rejected():
    with pytest.raises(ValueError):
        Auro4BArchitecture(hidden_dim=3000).validate()


def test_config_is_json_serializable():
    json.dumps(build_auro4b_config("full"), sort_keys=True)
