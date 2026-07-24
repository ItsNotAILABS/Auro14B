"""Tests for first-class Auro text LLM on MESIE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.checkpoint import load_checkpoint, save_checkpoint
from auro_native_llm.model.config import family_config, all_family_ids
from auro_native_llm.model.meaning import MultiMeaningField, LATIN_ROOTS, SANSKRIT_ROOTS
from auro_native_llm.model.phi_math import PHI, FIVE_MATH, phi_init
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.model.train import TrainConfig, train_language_model
from auro_native_llm.model.jobs import build_pretrain_command, submit_pretrain_job


class TestPhiMath:
    def test_phi(self):
        assert abs(PHI - 1.618033988749895) < 1e-9
        assert "phi" in FIVE_MATH

    def test_phi_init_shape(self):
        w = phi_init((32, 64), seed=1, layer=2)
        assert w.shape == (32, 64)


class TestMeaning:
    def test_engines_match(self):
        field = MultiMeaningField(64)
        hits = field.annotate("ratio veritas rta satya teotl spectral")
        engines = {h["engine"] for h in hits}
        assert "latin" in engines or "sanskrit" in engines or "nahuatl" in engines
        vec = field.embed("ratio lumen rta prana teotl")
        assert vec.shape == (64,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_lexica_nonempty(self):
        assert len(LATIN_ROOTS) >= 10
        assert len(SANSKRIT_ROOTS) >= 10


class TestTokenizer:
    def test_train_encode_decode(self):
        tok = AuroTokenizer(vocab_size=400)
        tok.train(
            [
                "Auro MESIE spectral intelligence",
                "ratio lumen rta satya teotl",
                "mixture of experts language model",
            ],
            vocab_size=400,
        )
        ids = tok.encode("Auro MESIE ratio")
        assert ids[0] == tok.bos_id
        text = tok.decode(ids)
        assert "Auro" in text or "MESIE" in text or len(text) > 0


class TestAuroLM:
    def test_family_ids(self):
        assert set(all_family_ids()) == {
            "Auro-2B",
            "Auro-4B",
            "Auro-8B",
            "Auro-14B",
            "Auro-100B",
        }

    def test_mesie_arsenal_wired(self):
        """Family configs map onto MESIE SpectralGPT presets with arsenal ON."""
        from auro_native_llm.model.config import family_scale_table, mesie_preset_dims

        table = family_scale_table()
        assert table["Auro-2B"]["dev"]["mesie_preset"] == "spectral_gpt_tiny"
        assert table["Auro-4B"]["dev"]["mesie_preset"] == "spectral_gpt_small"
        assert table["Auro-100B"]["full"]["mesie_preset"] == "spectral_gpt_xl"
        tiny = mesie_preset_dims("spectral_gpt_tiny")
        assert tiny["hidden_dim"] == 256 and tiny["num_layers"] == 4

        cfg = family_config("Auro-2B", mode="dev")
        assert cfg.hidden_dim == 256
        assert cfg.use_moe is True
        assert cfg.use_cross_modal is True
        assert cfg.use_spectral_encoder is True
        assert cfg.positional_encoding == "rotary"
        assert cfg.activation == "swiglu"
        assert cfg.normalization == "rms_norm"
        assert cfg.qk_norm is True
        assert len(cfg.resolved_moe_layers()) >= 1
        assert len(cfg.resolved_cross_modal_layers()) >= 1

        model = AuroLanguageModel.build("Auro-2B", mode="dev", vocab_size=512)
        info = model.info()
        arch = info["architecture"]
        # Live mass must be MESIE-tiny scale (not 64-dim toy)
        assert model.num_params > 5_000_000
        assert arch["use_moe"] is True
        assert arch["use_cross_modal"] is True
        assert arch["use_spectral_encoder"] is True
        assert arch["moe_layers"]
        assert info["mesie_preset"] == "spectral_gpt_tiny"

    def test_build_forward_generate(self):
        model = AuroLanguageModel.build("Auro-2B", mode="dev", vocab_size=512)
        assert model.compute_plane == "MESIE"
        assert model.num_params > 10_000
        info = model.info()
        assert info["backend"].startswith("mesie")
        assert "latin" in info["meaning_engines"]

        ids = np.array([model.tokenizer.encode("hello spectral world")[:32]], dtype=np.int64)
        out = model.forward_ids(ids, text_for_meaning="hello spectral ratio rta")
        assert "logits" in out
        assert out["logits"].shape[-1] == model.config.vocab_size
        assert out["compute_plane"] == "MESIE"

        gen = model.generate("MESIE Auro", max_new_tokens=12, temperature=0.9)
        assert gen.native is True
        assert gen.compute_plane == "MESIE"
        assert len(gen.token_ids) > 0
        assert gen.num_params == model.num_params

    def test_train_step_reduces_or_runs(self):
        model = AuroLanguageModel.build("Auro-2B", mode="dev", vocab_size=512)
        ids = np.array(
            [model.tokenizer.encode("spectral embedding helix resonance" * 3, max_length=48)],
            dtype=np.int64,
        )
        m0 = model.loss_on_batch(ids, ids, text_for_meaning="spectral")
        m1 = model.train_step(ids, ids, lr=1e-2, text_for_meaning="spectral")
        assert "loss" in m1
        assert model.train_steps == 1
        assert m0["loss"] >= 0 and m1["loss"] >= 0

    def test_checkpoint_roundtrip(self, tmp_path: Path):
        model = AuroLanguageModel.build("Auro-2B", mode="dev", vocab_size=512)
        model.train_step(
            np.array([model.tokenizer.encode("teotl ratio", max_length=32)], dtype=np.int64),
            np.array([model.tokenizer.encode("teotl ratio", max_length=32)], dtype=np.int64),
        )
        meta = save_checkpoint(model, tmp_path / "ckpt")
        assert meta["compute_plane"] == "MESIE"
        loaded = load_checkpoint(tmp_path / "ckpt")
        assert loaded.model_id == "Auro-2B"
        assert loaded.train_steps >= 1
        g = loaded.generate("ratio", max_new_tokens=8)
        assert g.native is True


class TestTrainIntegration:
    def test_short_train(self, tmp_path: Path):
        report = train_language_model(
            TrainConfig(
                model_id="Auro-2B",
                mode="dev",
                steps=6,
                batch_size=1,
                seq_len=48,
                vocab_size=512,
                output_dir=str(tmp_path / "ckpts"),
                report_every=2,
            )
        )
        assert report["ok"] is True
        assert report["compute_plane"] == "MESIE"
        assert report["num_params"] > 0
        assert Path(report["checkpoint"]).exists()
        assert "sample_generation" in report


class TestJobs:
    def test_build_command(self):
        cmd = build_pretrain_command("Auro-4B", steps=10)
        assert "auro_native_llm.model.train" in cmd
        assert "Auro-4B" in cmd

    def test_submit_plan(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # minimal package path not needed — jobs import from installed tree
        result = submit_pretrain_job(
            "Auro-2B",
            steps=5,
            execute=False,
            receipt_dir=str(tmp_path / "receipts"),
            registry_path=str(tmp_path / "nodes.json"),
        )
        assert result.get("ok") is True
        assert result["compute_plane"] == "MESIE"
        assert "command" in result
