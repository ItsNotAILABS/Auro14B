import pytest

from auro_native_llm.evaluation.long_context import (
    build_evidence_receipt,
    geometric_curriculum,
    perplexity_by_position,
    retrieval_report,
    routing_balance,
)
from auro_native_llm.evaluation.promotion import (
    constitutional_evidence_from_receipt,
    verify_long_context_receipt,
)
from auro_native_llm.substrate.checkpoint_constitution import ConstitutionalGateError


def valid_receipt(exact=True):
    curriculum = geometric_curriculum("Auro-4B", 16384, 65536, 1_000_000)
    retrieval = retrieval_report([
        {"needle_position": 0.1, "expected": "a", "observed": "a"},
        {"needle_position": 0.5, "expected": "b", "observed": "b"},
        {"needle_position": 0.9, "expected": "c", "observed": "c"},
    ])
    perplexity = perplexity_by_position([2.0] * 800)
    routing = routing_balance([[i % 8, (i + 1) % 8] for i in range(128)], 8)
    return build_evidence_receipt(
        model_id="Auro-4B",
        checkpoint_sha256="a" * 64,
        curriculum=curriculum,
        retrieval=retrieval,
        perplexity=perplexity,
        routing=routing,
        regression={"passed": True, "decision": "eligible"},
        runner={"runtime": "exact-test"},
        exact_checkpoint=exact,
    )


def test_verified_receipt_maps_all_required_constitutional_evidence():
    evidence = constitutional_evidence_from_receipt(valid_receipt())
    assert evidence["long_context_curriculum_pass"]
    assert evidence["retrieval_position_pass"]
    assert evidence["perplexity_position_pass"]
    assert evidence["moe_routing_balance_pass"]
    assert evidence["regression_receipt_pass"]
    assert evidence["protected_capabilities_pass"]


def test_proxy_or_smoke_receipt_cannot_promote():
    with pytest.raises(ConstitutionalGateError):
        verify_long_context_receipt(valid_receipt(exact=False))


def test_tampered_receipt_is_rejected():
    receipt = valid_receipt()
    receipt["routing"]["passed"] = False
    with pytest.raises(ConstitutionalGateError):
        verify_long_context_receipt(receipt)
