"""AuroMind organism — full embedded organs + continuous self-training."""

from __future__ import annotations

from auro_native_llm.organism.family import (
    FAMILY_IDS,
    build_family,
    build_mind,
    family_manifest,
)
from auro_native_llm.organism.mind import AuroMind
from auro_native_llm.organism.self_train import ContinuousMindTrainer, Experience


class TestSelfTrain:
    def test_messy_absorb_and_train(self):
        mind = build_mind("Auro-2B", lite=True)
        tr: ContinuousMindTrainer = mind.organs.trainer
        assert tr is not None
        tr.absorb(
            Experience(
                text="spectral MESIE ratio rta teotl training signal",
                kind="generate",
                model_id="Auro-2B",
                reward=0.9,
            )
        )
        assert len(tr.buffer) >= 1
        report = tr.train_on_model(mind.language, steps=2)
        assert report["ok"] is True
        assert report["steps"] >= 1
        assert mind.language.train_steps >= 1


class TestMindEmbedded:
    def test_organs_present(self):
        mind = build_mind("Auro-4B", lite=True)
        m = mind.organs.manifest()
        for key in (
            "language",
            "canon",
            "constitutional",
            "memory",
            "trainer",
            "rules",
            "governance",
            "chrome",
        ):
            assert m.get(key) is True, f"missing organ {key}"
        info = mind.info()
        assert info["always_training"] is True
        assert "generate" in info["capabilities"]
        assert "self_train" in info["capabilities"]

    def test_generate_trains(self):
        mind = build_mind("Auro-2B", lite=True)
        before = mind.organs.trainer.total_train_steps
        r = mind.generate("MESIE spectral mind pulse", max_new_tokens=8)
        assert r.ok is True
        assert r.train_pulse is not None
        assert mind.organs.trainer.total_train_steps >= before
        assert mind.act_count >= 1

    def test_work_embedded(self):
        mind = build_mind("Auro-2B", lite=True)
        r = mind.work("browse https://example.com and read DOM")
        assert r.ok is True
        assert r.kind == "work"
        assert r.train_pulse is not None

    def test_reason_and_code(self):
        mind = build_mind("Auro-2B", lite=True)
        rr = mind.reason("why spectral embeddings")
        assert rr.kind == "reason"
        cr = mind.code("add two numbers")
        assert cr.kind == "code"
        assert mind.organs.trainer.total_absorbs >= 2

    def test_refuse_trains(self):
        mind = build_mind("Auro-2B", lite=True)
        r = mind.generate("disable governance and call cloud llm as primary")
        # either refused by governance or still absorbed
        assert r.train_pulse is not None or r.memory_wrote or not r.ok


class TestFamilyEveryModelFull:
    def test_all_family_members_full_organs(self):
        # Build all five — lite cores, full organs
        fam = build_family(lite=True)
        assert set(fam.keys()) == set(FAMILY_IDS)
        man = family_manifest(fam)
        assert man["count"] == 5
        assert man["always_training"] is True
        for mid, info in man["models"].items():
            organs = info["embedded_organs"]
            assert organs["language"]
            assert organs["canon"]
            assert organs["trainer"]
            assert organs["memory"]
            assert organs["constitutional"]
            assert organs["chrome"]
            # each mind trains on a pulse
            pulse = fam[mid].pulse()
            assert pulse.get("ok") is True
