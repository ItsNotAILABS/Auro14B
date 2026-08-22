from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from readiness_score import CRITICAL_GATES, score_readiness
from tokenizer_audit import REQUIRED_CONTROL_TOKENS, audit_manifest


def test_readiness_requires_every_critical_gate_and_no_blockers():
    gates = {name: 1.0 for name in (
        "checkpoint_integrity", "training_provenance", "tokenizer_integrity",
        "corpus_provenance", "official_benchmarks", "coding_execution",
        "governed_execution", "api_chat_smoke", "browser_chat_smoke",
        "portability", "clean_install", "model_card_claims",
    )}
    passed = score_readiness({"gates": gates, "unresolved_blockers": []}, 0.85)
    assert passed["ready"] is True

    failed_gates = dict(gates)
    failed_gates["official_benchmarks"] = 0.84
    failed = score_readiness({"gates": failed_gates, "unresolved_blockers": []}, 0.85)
    assert failed["ready"] is False
    assert "official_benchmarks" in failed["failed_critical_gates"]

    blocked = score_readiness({"gates": gates, "unresolved_blockers": ["runner unavailable"]}, 0.85)
    assert blocked["ready"] is False
    assert blocked["unresolved_blockers"] == ["runner unavailable"]


def test_readiness_score_is_not_accuracy_claim():
    result = score_readiness({"gates": {name: 1.0 for name in CRITICAL_GATES}}, 0.85)
    assert result["claim_boundary"]["readiness_is_benchmark_accuracy"] is False
    assert result["claim_boundary"]["readiness_is_intelligence_percentage"] is False


def test_tokenizer_audit_requires_all_load_bearing_controls():
    complete = {
        "schema": "test.tokenizer.v1",
        "vocab_size": 300,
        "unknown_token": None,
        "byte_round_trip": True,
        "control_tokens": ["<pad>", "<bos>", "<eos>", *REQUIRED_CONTROL_TOKENS],
    }
    result = audit_manifest(complete)
    assert result["ready"] is True
    assert result["missing_control_tokens"] == []

    incomplete = dict(complete)
    incomplete["control_tokens"] = [token for token in complete["control_tokens"] if token != "<mathesis>"]
    result = audit_manifest(incomplete)
    assert result["ready"] is False
    assert result["missing_control_tokens"] == ["<mathesis>"]
