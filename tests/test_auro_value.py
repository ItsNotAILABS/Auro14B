"""Prove value: loss improves, checkpoint roundtrips, work still functions."""

from __future__ import annotations

from pathlib import Path

from auro_native_llm.organism.checkpoint import load_mind, save_mind
from auro_native_llm.organism.family import build_mind
from auro_native_llm.organism.value_train import (
    ValueTrainConfig,
    measure_loss,
    run_value_training,
)


class TestValueTrain:
    def test_value_train_improves_or_runs(self, tmp_path: Path):
        report = run_value_training(
            ValueTrainConfig(
                model_id="Auro-2B",
                steps=25,
                batch_size=2,
                seq_len=64,
                vocab_size=512,
                output_dir=str(tmp_path / "minds"),
                lite=True,
                report_every=5,
            )
        )
        assert report["ok"] is True
        assert report["loss_before"]["ce"] > 0
        assert report["loss_after"]["ce"] > 0
        # Must improve holdout CE to count as valuable training signal
        assert report["improved"] is True, report
        assert report["loss_delta_ce"] > 0
        assert Path(report["checkpoint"]).exists()
        assert (Path(report["checkpoint"]) / "VALUE_REPORT.json").exists()
        assert report["probes"]["work_ok"] is True
        assert report["num_params_live"] > 1000

    def test_checkpoint_roundtrip(self, tmp_path: Path):
        mind = build_mind("Auro-2B", lite=True)
        mind.generate("train me spectral", max_new_tokens=6)
        meta = save_mind(mind, tmp_path / "m")
        assert meta["schema"] == "auro.mind.checkpoint.v1"
        loaded = load_mind(tmp_path / "m")
        assert loaded.model_id == "Auro-2B"
        assert loaded.organs.trainer is not None
        assert loaded.organs.memory is not None
        r = loaded.generate("hello after load", max_new_tokens=6)
        assert r.ok is True
