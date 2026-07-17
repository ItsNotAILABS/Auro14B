"""Tests for Auro model family (2B–100B) and multi-embedded sub-agents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auro_native_llm.family import (
    builtin_family,
    load_family,
    validate_family,
    list_model_ids,
    get_lane,
)
from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter, route_role
from auro_native_llm.types import (
    FAMILY_ID,
    FAMILY_PARAMETER_TARGETS,
    ModelTier,
    SubAgentRole,
    TIER_TO_MODEL_ID,
)
from auro_native_llm.cli_family import main as family_cli_main

REPO = Path(__file__).resolve().parent.parent
FAMILY_CONFIG = REPO / "native_llm" / "configs" / "auro_family.json"
FAMILY_DIR = REPO / "native_llm" / "configs" / "family"


class TestFamilyCharter:
    def test_charter_json_exists(self):
        assert FAMILY_CONFIG.exists()
        data = json.loads(FAMILY_CONFIG.read_text(encoding="utf-8"))
        assert data["family_id"] == "Auro"
        assert len(data["lanes"]) == 5

    def test_all_lane_configs_exist(self):
        for name in ("auro_2b", "auro_4b", "auro_8b", "auro_14b", "auro_100b"):
            path = FAMILY_DIR / f"{name}.json"
            assert path.exists(), path
            cfg = json.loads(path.read_text(encoding="utf-8"))
            assert "not-trained" in cfg["status"] or "architecture-target" in cfg["status"]
            assert cfg["parameter_target"] > 0

    def test_load_family_from_disk(self):
        family = load_family(FAMILY_CONFIG)
        assert family.family_id == FAMILY_ID
        assert set(family.model_ids()) == {
            "Auro-2B",
            "Auro-4B",
            "Auro-8B",
            "Auro-14B",
            "Auro-100B",
        }

    def test_validate_family_ok(self):
        family = load_family(FAMILY_CONFIG)
        assert validate_family(family) == []

    def test_builtin_family(self):
        family = builtin_family()
        assert validate_family(family) == []
        assert len(family.lanes) == 5

    def test_parameter_targets(self):
        family = load_family(FAMILY_CONFIG)
        for lane in family.lanes:
            assert lane.parameter_target == FAMILY_PARAMETER_TARGETS[lane.model_id]

    def test_tier_mapping(self):
        assert TIER_TO_MODEL_ID[ModelTier.EDGE] == "Auro-2B"
        assert TIER_TO_MODEL_ID[ModelTier.FRONTIER] == "Auro-100B"
        assert get_lane("Auro-14B").tier == ModelTier.ORCHESTRATOR

    def test_polyglot_types(self):
        family = load_family(FAMILY_CONFIG)
        assert "python" in family.polyglot_types
        assert "julia" in family.polyglot_types
        assert "haskell" in family.polyglot_types


class TestMultiEmbeddedSubAgents:
    def test_14b_hosts_specialist(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-14B")
        result = router.dispatch(SubAgentRole.SPECTRAL_MATCH, "match PSD A vs B")
        assert result.ok
        assert result.child_model_id == "Auro-4B"
        assert result.parent_model_id == "Auro-14B"
        assert result.embedding is not None
        assert len(result.embedding) == 32

    def test_14b_hosts_edge_router(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-14B")
        result = router.dispatch(SubAgentRole.ROUTER, "route this request")
        assert result.ok
        assert result.child_model_id == "Auro-2B"

    def test_8b_hosts_edge_not_orchestrator(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-8B")
        result = router.dispatch(SubAgentRole.TOOL_CALL, "call tool X")
        assert result.ok
        assert result.child_model_id == "Auro-2B"

    def test_2b_cannot_embed(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-2B")
        result = router.dispatch(SubAgentRole.CODE_EDIT, "edit file")
        assert not result.ok
        assert result.error

    def test_100b_council(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-100B")
        results = router.dispatch_council("deep spectral research")
        assert len(results) >= 3
        assert all(r.ok for r in results)

    def test_route_role_convenience(self):
        payload = route_role("plan", "plan training run", parent_model_id="Auro-14B")
        assert payload["ok"] is True
        assert payload["child_model_id"] == "Auro-8B"
        assert payload["schema"] == "auro.native_llm.subagent_dispatch.v1"

    def test_preferred_model_id(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-14B")
        result = router.dispatch(
            SubAgentRole.REASON,
            "think hard",
            preferred_model_id="Auro-8B",
        )
        assert result.ok
        assert result.child_model_id == "Auro-8B"

    def test_ghost_dispatch_optional(self):
        router = MultiEmbeddedSubAgentRouter(parent_model_id="Auro-14B")
        result = router.dispatch(SubAgentRole.PLAN, "ghost path", use_ghost=True)
        assert result.ok
        assert "ghost" in result.message or "scaffold" in result.message


class TestFamilyCLI:
    def test_cli_list(self, capsys):
        code = family_cli_main(["--config", str(FAMILY_CONFIG), "list"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Auro-2B" in out
        assert "Auro-100B" in out

    def test_cli_validate(self, capsys):
        code = family_cli_main(["--config", str(FAMILY_CONFIG), "validate"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True

    def test_cli_dispatch(self, capsys):
        code = family_cli_main(
            [
                "--config",
                str(FAMILY_CONFIG),
                "dispatch",
                "--parent",
                "Auro-14B",
                "--role",
                "spectral_match",
                "--intent",
                "test match",
                "--route-only",
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["child_model_id"] == "Auro-4B"


class TestPolyglotBindingsExist:
    def test_julia_binding(self):
        path = REPO / "bindings" / "julia" / "AuroFamily" / "src" / "AuroFamily.jl"
        text = path.read_text(encoding="utf-8")
        assert "Auro-2B" in text
        assert "Auro-100B" in text
        assert "CONTRACT_VERSION" in text

    def test_haskell_binding(self):
        path = REPO / "bindings" / "haskell" / "AuroFamily.hs"
        text = path.read_text(encoding="utf-8")
        assert "Auro-2B" in text
        assert "Auro-100B" in text
        assert "contractVersion" in text
        assert "canHost" in text
