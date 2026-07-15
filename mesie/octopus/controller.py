"""Octopus engineering controller — multi-arm coordination via internal API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]

from mesie.engines.registry import build_default_registry
from mesie.internal_api.bus import InternalBus
from mesie.internal_api.messages import EngineResponse
from mesie.internal_api.router import InternalRouter
from mesie.io.loaders import RecordInput, load_record
from mesie.octopus.arms import ArmId, OctopusArm
from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter
from mesie.polyglot.contract import SUITE_NAME
from mesie.polyglot.suite import AISVectorPolyglotSuite


@dataclass
class OctopusConfig:
    """Runtime config for octopus workflows."""

    enable_all_arms: bool = True
    default_workflow_id: str = "octopus_standard"
    movement_steps: int = 3
    movement_delta: float = 0.15
    user_library_paths: Sequence[PathLike] = field(default_factory=list)
    user_index_path: Optional[PathLike] = None
    user_index_save: Optional[PathLike] = None
    use_polyglot_arms: bool = True


@dataclass
class OctopusRunReport:
    workflow_id: str
    arms_used: List[str]
    validation: Dict[str, Any]
    embedding: Dict[str, Any]
    match: Dict[str, Any]
    movement: Dict[str, Any]
    control: Dict[str, Any]
    logic: Dict[str, Any]
    memory: Dict[str, Any]
    workflow: Dict[str, Any]
    user_library: Dict[str, Any]
    polyglot: Dict[str, Any]
    plain_summary: str


class OctopusController:
    """Central body — eight arms, one internal bus, AISVectorPolyglot on EMBED/MATCH."""

    def __init__(
        self,
        bus: Optional[InternalBus] = None,
        config: Optional[OctopusConfig] = None,
        polyglot_suite: Optional[AISVectorPolyglotSuite] = None,
    ) -> None:
        self.config = config or OctopusConfig()
        self.bus = bus or InternalBus()
        self.polyglot_suite = polyglot_suite or AISVectorPolyglotSuite()
        self.registry = build_default_registry(self.bus, polyglot_suite=self.polyglot_suite)
        self.router = InternalRouter(bus=self.bus, registry=self.registry)
        self._arms: Dict[ArmId, OctopusArm] = {
            aid: OctopusArm(aid, self.bus) for aid in ArmId
        }
        if not self.config.enable_all_arms:
            for arm in self._arms.values():
                arm.enable(False)
        self._bootstrap_user_library()

    def arm(self, arm_id: ArmId) -> OctopusArm:
        return self._arms[arm_id]

    def list_engines(self) -> List[str]:
        return self.registry.names()

    def _bootstrap_user_library(self) -> None:
        """Wire user spectral folder/index into EMBED arm (AISVectorPolyglot vector bridge)."""
        target = "polyglot" if self.config.use_polyglot_arms else "embedding"
        if self.config.user_index_path:
            self.router.call(
                target,
                "load_user_library",
                {"index_path": str(self.config.user_index_path)},
            )
            return
        if self.config.user_library_paths:
            save = self.config.user_index_save
            if save is None and self.config.user_library_paths:
                root = Path(self.config.user_library_paths[0])
                if root.is_dir():
                    save = root.parent / "library" / "my_spectral_index.json"
                else:
                    save = Path(__file__).resolve().parents[2] / "library" / "my_spectral_index.json"
            self.router.call(
                target,
                "embed_user_paths",
                {
                    "paths": [str(p) for p in self.config.user_library_paths],
                    "save_to": str(save) if save else None,
                },
            )

    def run_standard_cycle(
        self,
        record: RecordInput,
        *,
        candidate: Optional[RecordInput] = None,
    ) -> OctopusRunReport:
        """Full octopus cycle: sense → embed → match → move → control → logic → memory → workflow."""
        rec = load_record(record)
        cand = load_record(candidate) if candidate is not None else rec

        arms_used: List[str] = []
        polyglot_meta: Dict[str, Any] = {"suite": SUITE_NAME}

        v = self.arm(ArmId.SENSE).reach("validate", {"record": rec})
        arms_used.append("sense")

        e = self.arm(ArmId.EMBED).reach("embed", {"record": rec})
        polyglot_meta["embed_runtime"] = e.data.get("runtime")
        polyglot_meta["embed_mode"] = e.data.get("mode")
        user_lib = self.arm(ArmId.EMBED).reach("user_library_status", {})
        neighbors_user: List = []
        vector_neighbors: List = []
        if user_lib.ok and user_lib.data.get("user_entries", 0) > 0:
            q = self.arm(ArmId.EMBED).reach("query", {"record": rec, "top_k": 3})
            if q.ok:
                neighbors_user = q.data.get("neighbors", [])
                vector_neighbors = q.data.get("fingerprint_hits", [])
                polyglot_meta["vector_query_ms"] = q.data.get("elapsed_ms")
        arms_used.append("embed")

        m = self.arm(ArmId.MATCH).reach("match", {"record_a": rec, "record_b": cand})
        polyglot_meta["match_runtime"] = m.data.get("runtime")
        polyglot_meta["match_mode"] = m.data.get("mode")
        arms_used.append("match")

        move_results = []
        for _ in range(self.config.movement_steps):
            mv = self.arm(ArmId.MOVE).reach(
                "advance",
                {"record": rec, "delta": self.config.movement_delta},
            )
            move_results.append(mv.to_dict())
        arms_used.append("move")

        similarity = float(m.data.get("composite_score", 0))
        anomaly_adapter = SpectralAnomalyAdapter(threshold=2.5)
        anomaly_adapter.fit_baseline([rec])
        anomaly_score = float(anomaly_adapter.score_anomaly(cand))

        self.arm(ArmId.CONTROL).reach("set_mode", {"mode": "active"})
        self.arm(ArmId.CONTROL).reach("arm_enable", {"arm": "move", "enabled": True})
        c = self.arm(ArmId.CONTROL).reach(
            "evaluate",
            {"similarity": similarity, "anomaly": anomaly_score},
        )
        arms_used.append("control")

        self.router.call("logic", "add_rule", {
            "name": "low_similarity",
            "condition": "similarity_low",
            "target_engine": "control",
            "target_action": "evaluate",
            "similarity_threshold": 0.65,
        })
        self.router.call("logic", "add_rule", {
            "name": "high_anomaly",
            "condition": "anomaly_high",
            "target_engine": "control",
            "target_action": "evaluate",
            "anomaly_threshold": 2.5,
        })
        lg = self.arm(ArmId.LOGIC).reach(
            "evaluate",
            {
                "context": {
                    "similarity": similarity,
                    "anomaly": anomaly_score,
                    "similarity_threshold": 0.65,
                    "anomaly_threshold": 2.5,
                }
            },
        )
        arms_used.append("logic")

        mem = self.arm(ArmId.MEMORY).reach("memory", {"record": rec})
        reason = self.arm(ArmId.MEMORY).reach("reason", {"record": rec})
        arms_used.append("memory")

        wf_define = self.router.call(
            "workflow",
            "define",
            {
                "workflow_id": self.config.default_workflow_id,
                "steps": [
                    {"name": "validate", "engine": "validation", "action": "validate"},
                    {"name": "embed", "engine": "polyglot", "action": "embed"},
                    {"name": "reason", "engine": "intelligence", "action": "reason"},
                ],
            },
        )
        wf_run = self.router.call(
            "workflow",
            "run",
            {"context": {"record": rec}},
        )
        arms_used.append("workflow")

        polyglot_meta["vector_indexed"] = self.polyglot_suite.vector.state.n_indexed
        polyglot_meta["technical_hits"] = vector_neighbors or neighbors_user

        summary = self._plain_summary(v, m, c, lg, reason, wf_run, polyglot_meta)

        return OctopusRunReport(
            workflow_id=self.config.default_workflow_id,
            arms_used=arms_used,
            validation=v.to_dict(),
            embedding=e.to_dict(),
            match=m.to_dict(),
            movement={"steps": move_results},
            control=c.to_dict(),
            logic=lg.to_dict(),
            memory={"memory": mem.to_dict(), "reason": reason.to_dict()},
            workflow={"define": wf_define.to_dict(), "run": wf_run.to_dict()},
            user_library={
                "status": user_lib.to_dict(),
                "nearest_in_user_library": neighbors_user,
                "vector_fingerprint_hits": vector_neighbors,
            },
            polyglot=polyglot_meta,
            plain_summary=summary,
        )

    @staticmethod
    def _plain_summary(
        validation: EngineResponse,
        match: EngineResponse,
        control: EngineResponse,
        logic: EngineResponse,
        reason: EngineResponse,
        workflow: EngineResponse,
        polyglot: Dict[str, Any],
    ) -> str:
        valid = validation.data.get("is_valid", False)
        score = match.data.get("composite_score", 0)
        cmds = control.data.get("commands", [])
        rules = logic.data.get("count", 0)
        conclusion = reason.data.get("conclusion", "unknown")
        done = workflow.data.get("completed", False)
        embed_rt = polyglot.get("embed_runtime", "python")
        match_rt = polyglot.get("match_runtime", "rust")
        return (
            f"Record valid={valid}. Match score={score:.3f} via {match_rt}. "
            f"Embed via {embed_rt} ({SUITE_NAME}). "
            f"Control issued {cmds or ['none']}. Logic fired {rules} rule(s). "
            f"Intelligence says {conclusion}. Workflow complete={done}."
        )