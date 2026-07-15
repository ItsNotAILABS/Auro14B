"""Tests for internal API, engines, and octopus controller."""

import numpy as np
import pytest

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.internal_api import InternalBus, InternalRouter, MessageTopic
from mesie.engines import build_default_registry
from mesie.octopus import ArmId, OctopusController, OctopusConfig


def _record(rid: str = "t-001") -> MultiElementRecord:
    f = np.linspace(0.1, 10.0, 32)
    a = np.sin(f) + 0.3
    return MultiElementRecord(record_id=rid, components=[SpectralComponent(name="c", frequency=f, amplitude=a)])


class TestInternalBus:
    def test_request_embedding(self):
        bus = InternalBus()
        build_default_registry(bus)
        resp = bus.request("test", "embedding", "transform", {"record": _record()})
        assert resp.ok
        assert "embedding" in resp.data
        assert len(resp.data["embedding"]) > 0

    def test_matching_rank(self):
        bus = InternalBus()
        build_default_registry(bus)
        r0 = _record("a")
        r1 = _record("b")
        resp = bus.request(
            "test",
            "matching",
            "rank",
            {"query": r0, "candidates": [r0, r1], "top_k": 2},
        )
        assert resp.ok
        assert len(resp.data["ranked"]) == 2


class TestEngines:
    def test_registry_has_eleven_engines(self):
        reg = build_default_registry()
        names = reg.names()
        assert "embedding" in names
        assert "fingerprint" in names
        assert "polyglot" in names
        assert "workflow" in names
        assert len(names) >= 11

    def test_control_evaluate(self):
        router = InternalRouter()
        router.call("control", "set_setpoint", {"similarity": 0.8})
        r = router.call("control", "evaluate", {"similarity": 0.5, "anomaly": 1.0})
        assert r.ok
        assert "investigate_match" in r.data.get("commands", [])

    def test_logic_rules(self):
        router = InternalRouter()
        router.call("logic", "clear")
        router.call(
            "logic",
            "add_rule",
            {"name": "a", "condition": "similarity_low", "target_engine": "control", "target_action": "evaluate"},
        )
        r = router.call("logic", "evaluate", {"context": {"similarity": 0.3, "similarity_threshold": 0.65}})
        assert r.ok
        assert r.data["count"] >= 1

    def test_workflow_define_and_status(self):
        router = InternalRouter()
        d = router.call(
            "workflow",
            "define",
            {
                "workflow_id": "test-wf",
                "steps": [{"name": "v", "engine": "validation", "action": "validate"}],
            },
        )
        assert d.ok
        s = router.call("workflow", "status")
        assert s.data["active"]


class TestOctopus:
    def test_controller_lists_engines(self):
        oc = OctopusController()
        assert "polyglot" in oc.list_engines()
        assert len(oc._arms) == 8

    def test_embed_match_arms_use_polyglot(self):
        from mesie.octopus.arms import ARM_ENGINE_MAP, ArmId

        assert ARM_ENGINE_MAP[ArmId.EMBED] == "polyglot"
        assert ARM_ENGINE_MAP[ArmId.MATCH] == "polyglot"

    def test_standard_cycle(self):
        oc = OctopusController(config=OctopusConfig(movement_steps=2))
        report = oc.run_standard_cycle(_record(), candidate=_record("t-002"))
        assert report.validation["ok"]
        assert report.match["ok"]
        assert report.embedding["ok"]
        assert report.polyglot["suite"] == "AISVectorPolyglot"
        assert "sense" in report.arms_used
        assert len(report.plain_summary) > 20

    def test_arm_reach_disabled(self):
        oc = OctopusController()
        oc.arm(ArmId.EMBED).enable(False)
        r = oc.arm(ArmId.EMBED).reach("transform", {"record": _record()})
        assert not r.ok

    def test_user_library_via_reference_folder(self, tmp_path):
        from pathlib import Path

        from mesie.internal_api import InternalRouter
        from mesie.library.user_corpus import _load_spectral_file
        from mesie.octopus import OctopusConfig

        root = Path(__file__).resolve().parents[1]
        ref_dir = root / "data" / "reference"
        if not ref_dir.exists():
            pytest.skip("reference data missing")
        out = tmp_path / "my_spectral_index.json"
        router = InternalRouter()
        r = router.call(
            "embedding",
            "embed_user_paths",
            {"paths": [str(ref_dir)], "save_to": str(out)},
        )
        assert r.ok
        assert r.data["embedded"] >= 4
        ref_files = list(ref_dir.glob("*.json"))
        oc = OctopusController(config=OctopusConfig(user_index_path=str(out), movement_steps=1))
        report = oc.run_standard_cycle(_load_spectral_file(ref_files[0]))
        assert report.user_library["status"]["data"]["user_entries"] >= 4
        assert report.polyglot["vector_indexed"] >= 4