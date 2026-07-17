"""Measured intelligence — coding harness + promotion gate must not be theater."""

from __future__ import annotations

from auro_native_llm.intelligence.coding import CodingOrchestrator, run_coding_harness
from auro_native_llm.intelligence.reasoning import ReasoningOrchestrator
from auro_native_llm.intelligence.promotion import PromotionGate, run_readiness
from auro_native_llm.organism.family import build_mind
from auro_foundry.coding_harness import built_in_smoke_tasks


def test_coding_harness_non_zero_pass():
    mind = build_mind("Auro-2B", lite=True)
    report = run_coding_harness(mind)
    assert report["summary"]["tasks"] >= 3
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["usable"] is True
    assert all(r["passed"] for r in report["results"])


def test_reasoning_probes_pass():
    mind = build_mind("Auro-2B", lite=True)
    r = ReasoningOrchestrator(mind).run_probes()
    assert r["summary"]["accuracy"] >= 0.75
    assert r["summary"]["usable"] is True


def test_promotion_m1_when_measured():
    mind = build_mind("Auro-2B", lite=True)
    payload = run_readiness(mind, output_dir="artifacts/auro-readiness-test")
    assert payload["schema"] == "auro.nova_promotion_receipt.v1"
    ready = payload["readiness"]
    assert ready["coding_pass_rate"] == 1.0, payload["coding_receipt"]["summary"]
    assert ready["reasoning_accuracy"] >= 0.75, payload["reasoning_receipt"]["summary"]
    assert ready["generation_usable"] is True
    # Real harness + probes → at least M1 (M2 if organ receipts present)
    assert ready["tier"] in ("M1", "M2"), ready
    assert payload["expansion_allowed"] is True
    assert len(payload["receipt_sha256"]) == 64
