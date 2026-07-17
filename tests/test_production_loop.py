"""Production loop contract: train→measure→save→load→work→learn."""

from __future__ import annotations

from pathlib import Path

from auro_native_llm.organism.production import (
    CLAIM_BOUNDARY,
    PRODUCTION_LOOP,
    ProductionConfig,
    run_production_loop,
)
from auro_native_llm.organism.family import build_mind


def test_claim_boundary_in_mind_info():
    mind = build_mind("Auro-2B", lite=True)
    info = mind.info()
    assert info["live_is_running_model"] is True
    assert info["target_is_architecture_label"] is True
    assert "not marketing" in info["claim_boundary"]
    assert info["production_loop"] == PRODUCTION_LOOP
    assert info["num_params_live"] > 0
    assert info["parameter_target"] >= info["num_params_live"]


def test_production_loop_short(tmp_path: Path):
    report = run_production_loop(
        ProductionConfig(
            model_id="Auro-2B",
            steps=20,
            batch_size=2,
            seq_len=64,
            vocab_size=512,
            output_dir=str(tmp_path / "prod"),
            lite=True,
            keep_learning_pulses=2,
        )
    )
    assert report["loop"] == PRODUCTION_LOOP
    assert CLAIM_BOUNDARY in report["claim_boundary"] or report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["live_is_running_model"] is True
    assert report["target_is_architecture_label"] is True
    assert report["num_params_live"] > 0
    assert report["value_proof"]["durable"] is True
    assert Path(report["value_proof"]["checkpoint"]).exists()
    assert (Path(report["value_proof"]["checkpoint"]) / "PRODUCTION_LOOP.json").exists()
    # improved training OR still valuable path: work after load
    assert report["value_proof"]["work_ok"] is True
    assert report["keep_learning"]["sample_generate_ok"] is True
