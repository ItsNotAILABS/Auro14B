"""MESIE bridge to AURO's durable independent harness fabric.

MESIE remains the model compute plane; this bridge lets MESIE-native runtimes
fan out complete persistent work harnesses without duplicating orchestration
state inside model layers.
"""
from __future__ import annotations

from typing import Any, Iterable


class MesieHarnessBridge:
    def __init__(self, fabric=None) -> None:
        if fabric is None:
            from auro_native_llm.work.harness import IndependentHarnessFabric
            fabric = IndependentHarnessFabric()
        self.fabric = fabric

    def fan_out(self, objective: str, subproblems: Iterable[str], *, model_id: str = "Auro-2B") -> dict[str, Any]:
        parent = self.fabric.create_harness(objective, model_id=model_id)
        children = self.fabric.fan_out(parent.id, subproblems, model_id=model_id)
        return {
            "schema": "mesie.harness-bridge.v1",
            "compute_plane": "MESIE",
            "parent_id": parent.id,
            "child_ids": [child.id for child in children],
            "independent_state": True,
            "persistent_resume": True,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "mesie.harness-bridge.v1",
            "compute_plane": "MESIE",
            "fabric": self.fabric.manifest(),
            "complete_harness_fanout": True,
            "subagent_only": False,
        }


__all__ = ["MesieHarnessBridge"]
