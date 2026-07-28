import json

import pytest

from auro_native_llm.evaluation.long_context import (
    build_evidence_receipt,
    geometric_curriculum,
    perplexity_by_position,
    regression_report,
    retrieval_report,
    routing_balance,
)


def test_curriculum_reaches_target_and_advances_by_tokens():
    curriculum = geometric_curriculum("Auro-4B", 16384, 65536, 8_000_000, stages=4)
    curriculum.validate()
    assert curriculum.stages[-1].context_length == 65536
    assert curriculum.stage_for_tokens(0).context_length == 16384
    assert curriculum.stage_for_tokens(7_999_999).context_length == 65536
    assert sum(stage.token_budget for stage in curriculum.stages) == 8_000_000


def test_retrieval_scores_early_middle_and_late_positions():
    cases = [
        {"needle_position": 0.1, "expected": "a", "observed": "a"},
        {"needle_position": 0.5, "expected": "b", "observed": "b"},
        {"needle_position": 0.9, "expected": "c", "observed": "c"},
    ]
    report = retrieval_report(cases)
    assert report["passed"]
    assert set(report["position_buckets"]) == {"early", "middle", "late"}


def test_perplexity_by_position_detects_tail_degradation():
    stable = perplexity_by_position([2.0] * 800, 8)
    degraded = perplexity_by_position([2.0] * 700 + [2.5] * 100, 8)
    assert stable["passed"]
    assert not degraded["passed"]
    assert degraded["first_to_last_ratio"] > 1.20


def test_routing_balance_accepts_uniform_and_rejects_dead_experts():
    uniform = routing_balance([[i % 8, (i + 1) % 8] for i in range(128)], 8)
    assert uniform["passed"]
    assert not uniform["dead_experts"]
    collapsed = routing_balance([[0, 1] for _ in range(64)], 8)
    assert not collapsed["passed"]
    assert collapsed["dead_experts"]


def test_routing_rejects_duplicate_expert_per_token():
    with pytest.raises(ValueError):
        routing_balance([[1, 1]], 8)


def test_regression_report_quarantines_protected_metric_drop():
    baseline = {
        "retrieval": {"accuracy": 0.95},
        "perplexity": {"first_to_last_ratio": 1.05},
        "routing": {"coefficient_of_variation": 0.10},
        "protected_metrics": {"coding": 0.80},
    }
    candidate = {
        "retrieval": {"accuracy": 0.95},
        "perplexity": {"first_to_last_ratio": 1.06},
        "routing": {"coefficient_of_variation": 0.11},
        "protected_metrics": {"coding": 0.70},
    }
    report = regression_report(baseline, candidate)
    assert not report["passed"]
    assert report["decision"] == "quarantine"


def test_exact_checkpoint_receipt_can_promote_but_smoke_cannot():
    curriculum = geometric_curriculum("Auro-4B", 16384, 65536, 1_000_000)
    retrieval = retrieval_report([
        {"needle_position": 0.1, "expected": "a", "observed": "a"},
        {"needle_position": 0.5, "expected": "b", "observed": "b"},
        {"needle_position": 0.9, "expected": "c", "observed": "c"},
    ])
    perplexity = perplexity_by_position([2.0] * 800)
    routing = routing_balance([[i % 8, (i + 1) % 8] for i in range(128)], 8)
    regression = {"passed": True, "decision": "eligible"}
    exact = build_evidence_receipt(
        model_id="Auro-4B",
        checkpoint_sha256="a" * 64,
        curriculum=curriculum,
        retrieval=retrieval,
        perplexity=perplexity,
        routing=routing,
        regression=regression,
        runner={"hardware": "test"},
        exact_checkpoint=True,
    )
    smoke = build_evidence_receipt(
        model_id="Auro-4B",
        checkpoint_sha256="smoke",
        curriculum=curriculum,
        retrieval=retrieval,
        perplexity=perplexity,
        routing=routing,
        regression=regression,
        runner={"kind": "smoke"},
        exact_checkpoint=False,
    )
    assert exact["promotion"]["eligible"]
    assert smoke["promotion"]["decision"] == "quarantine"
    json.dumps(exact, sort_keys=True)
