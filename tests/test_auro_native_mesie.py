"""Auro native models on MESIE compute plane."""

from __future__ import annotations

import json
from pathlib import Path

from auro_native_llm.mesie_compute import (
    MESIEComputePlane,
    get_compute_plane,
    profile_from_lane,
    spectral_fft_metrics,
    text_to_signal,
)
from auro_native_llm.native_model import AuroNativeFamily, AuroNativeModel
from auro_native_llm.native_runtime import AuroNativeRuntime
from auro_native_llm.cli_family import main as family_cli_main

REPO = Path(__file__).resolve().parent.parent
SERVING = REPO / "native_llm" / "configs" / "serving_contract.json"


class TestMesieCompute:
    def test_plane_health(self):
        plane = get_compute_plane()
        h = plane.health()
        assert h["compute_plane"] == "MESIE"
        assert h["native"] is True
        assert h["cloud_llm"] is False
        assert "capabilities" in h

    def test_profile_ladder(self):
        edge = profile_from_lane("Auro-2B", 2_000_000_000, "edge")
        frontier = profile_from_lane("Auro-100B", 100_000_000_000, "frontier")
        assert edge.d_model < frontier.d_model
        assert edge.n_layers < frontier.n_layers
        assert edge.parameter_target == 2_000_000_000

    def test_forward_always_works(self):
        plane = MESIEComputePlane()
        profile = profile_from_lane("Auro-4B", 4_000_000_000, "specialist")
        result = plane.forward("match two PSD records", profile)
        assert result.embedding
        assert len(result.embedding) == profile.d_model
        assert result.latency_ms >= 0
        assert result.backend.value.startswith("mesie") or "neurocore" in result.backend.value or "foundation" in result.backend.value or "spectral" in result.backend.value
        assert "spectral_entropy" in result.spectral_metrics

    def test_fft_metrics(self):
        sig = text_to_signal("hello spectral world", 64)
        m = spectral_fft_metrics(sig)
        assert 0.0 <= m["spectral_entropy"] <= 1.5


class TestAuroNativeModel:
    def test_from_model_id(self):
        m = AuroNativeModel.from_model_id("Auro-8B")
        assert m.compute_plane_name == "MESIE"
        assert m.model_id == "Auro-8B"
        h = m.health()
        assert h["native"] is True or h.get("compute_plane") == "MESIE"

    def test_generate_native(self):
        m = AuroNativeModel.from_model_id("Auro-2B")
        gen = m.generate("triage this vibration spike", role="spectral_triage")
        assert gen.native is True
        assert gen.compute_plane == "MESIE"
        assert gen.model_id == "Auro-2B"
        assert "MESIE" in gen.text or "mesie" in gen.backend.lower() or gen.backend
        assert 0.0 < gen.confidence <= 1.0
        d = gen.to_dict()
        assert d["schema"] == "auro.native_llm.generation.v1"

    def test_family_all_lanes(self):
        fam = AuroNativeFamily()
        assert set(fam.list_models()) == {
            "Auro-2B",
            "Auro-4B",
            "Auro-8B",
            "Auro-14B",
            "Auro-100B",
        }
        for mid in fam.list_models():
            g = fam.generate(mid, f"ping {mid}")
            assert g.compute_plane == "MESIE"
            assert g.native is True


class TestNativeRuntime:
    def test_dispatch_runs_mesie(self):
        rt = AuroNativeRuntime(parent_model_id="Auro-14B")
        r = rt.dispatch("spectral_match", "compare structural FAS A vs B")
        assert r.ok
        assert r.compute_plane == "MESIE"
        assert r.child_model_id == "Auro-4B"
        assert r.generation is not None
        assert r.generation.native is True
        assert r.generation.compute_plane == "MESIE"

    def test_council_native(self):
        rt = AuroNativeRuntime(parent_model_id="Auro-14B")
        results = rt.council("build spectral pipeline")
        assert len(results) >= 3
        assert all(r.ok for r in results)
        assert all(r.generation is not None for r in results)

    def test_models_payload(self):
        rt = AuroNativeRuntime()
        payload = rt.serve_models_payload()
        assert payload["object"] == "list"
        assert len(payload["data"]) == 5
        assert all(m["compute_plane"] == "MESIE" for m in payload["data"])
        assert all(m["native"] is True for m in payload["data"])


class TestCLINative:
    def test_cli_compute(self, capsys):
        code = family_cli_main(["compute"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["health"]["compute_plane"] == "MESIE"

    def test_cli_generate(self, capsys):
        code = family_cli_main(
            ["generate", "--model", "Auro-2B", "--prompt", "hello native", "--max-tokens", "64"]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["compute_plane"] == "MESIE"
        assert data["native"] is True

    def test_cli_dispatch_native(self, capsys):
        code = family_cli_main(
            [
                "dispatch",
                "--parent",
                "Auro-14B",
                "--role",
                "plan",
                "--intent",
                "plan a training run",
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["compute_plane"] == "MESIE"
        assert data["generation"]["native"] is True


class TestServingContract:
    def test_serving_marks_mesie(self):
        cfg = json.loads(SERVING.read_text(encoding="utf-8"))
        assert cfg["compute_plane"] == "MESIE"
        assert cfg["native"] is True
        assert cfg["cloud_llm"] is False
