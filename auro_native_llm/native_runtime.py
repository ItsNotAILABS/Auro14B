"""Auro native family runtime — multi-embedded sub-agents on MESIE compute.

This is the execution surface: load family → native models → route roles →
MESIE forward/generate → receipts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auro_native_llm.mesie_compute import MESIEComputePlane, get_compute_plane
from auro_native_llm.native_model import AuroNativeFamily, AuroNativeModel, NativeGeneration
from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter
from auro_native_llm.types import SubAgentDispatch, SubAgentRole


@dataclass
class NativeDispatchResult:
    """Sub-agent dispatch that actually ran on MESIE compute."""

    ok: bool
    parent_model_id: str
    child_model_id: str
    role: str
    agent_id: str
    task_id: str
    generation: Optional[NativeGeneration] = None
    route: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    compute_plane: str = "MESIE"
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.native_llm.native_dispatch.v1",
            "ok": self.ok,
            "parent_model_id": self.parent_model_id,
            "child_model_id": self.child_model_id,
            "role": self.role,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "compute_plane": self.compute_plane,
            "latency_ms": self.latency_ms,
            "generation": self.generation.to_dict() if self.generation else None,
            "route": self.route,
            "error": self.error,
            "native": True,
        }


class AuroNativeRuntime:
    """Family runtime: native Auro models + multi-embedded sub-agents + MESIE."""

    def __init__(
        self,
        parent_model_id: str = "Auro-14B",
        plane: Optional[MESIEComputePlane] = None,
    ) -> None:
        self.plane = plane or get_compute_plane()
        self.family_models = AuroNativeFamily(plane=self.plane)
        self.parent_model_id = parent_model_id
        self.router = MultiEmbeddedSubAgentRouter(
            parent_model_id=parent_model_id,
            family=self.family_models.family,
            embed_fn=lambda text: self.family_models.get(parent_model_id).embed(text),
        )
        self._history: List[NativeDispatchResult] = []

    @property
    def history(self) -> List[NativeDispatchResult]:
        return list(self._history)

    def health(self) -> Dict[str, Any]:
        return {
            "runtime": "AuroNativeRuntime",
            "parent_model_id": self.parent_model_id,
            "compute_plane": "MESIE",
            "native": True,
            "cloud_llm": False,
            "family": self.family_models.health(),
            "node": self.plane.discover_node(),
        }

    def get_model(self, model_id: str) -> AuroNativeModel:
        return self.family_models.get(model_id)

    def generate(
        self,
        prompt: str,
        *,
        model_id: Optional[str] = None,
        role: Optional[str] = None,
        spectral_context: Optional[str] = None,
        max_tokens: int = 256,
    ) -> NativeGeneration:
        mid = model_id or self.parent_model_id
        return self.family_models.generate(
            mid,
            prompt,
            role=role,
            spectral_context=spectral_context,
            max_tokens=max_tokens,
        )

    def dispatch(
        self,
        role: str | SubAgentRole,
        intent: str,
        *,
        preferred_model_id: Optional[str] = None,
        spectral_context: Optional[str] = None,
        max_tokens: int = 256,
    ) -> NativeDispatchResult:
        """Route role to embedded child lane and run MESIE-native generation."""
        t0 = time.perf_counter()
        route = self.router.dispatch(
            role,
            intent,
            preferred_model_id=preferred_model_id,
            use_ghost=False,
        )
        if not route.ok:
            out = NativeDispatchResult(
                ok=False,
                parent_model_id=self.parent_model_id,
                child_model_id=route.child_model_id or "",
                role=route.role.value if isinstance(route.role, SubAgentRole) else str(route.role),
                agent_id=route.agent_id,
                task_id=route.task_id,
                route=route.to_dict(),
                error=route.error,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
            self._history.append(out)
            return out

        try:
            child = self.family_models.get(route.child_model_id)
            gen = child.generate(
                intent,
                role=route.role,
                spectral_context=spectral_context,
                max_tokens=max_tokens,
            )
            out = NativeDispatchResult(
                ok=True,
                parent_model_id=route.parent_model_id,
                child_model_id=route.child_model_id,
                role=route.role.value,
                agent_id=route.agent_id,
                task_id=route.task_id,
                generation=gen,
                route=route.to_dict(),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            out = NativeDispatchResult(
                ok=False,
                parent_model_id=route.parent_model_id,
                child_model_id=route.child_model_id,
                role=route.role.value,
                agent_id=route.agent_id,
                task_id=route.task_id,
                route=route.to_dict(),
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        self._history.append(out)
        return out

    def council(
        self,
        intent: str,
        *,
        roles: Optional[List[str]] = None,
        spectral_context: Optional[str] = None,
    ) -> List[NativeDispatchResult]:
        default = ["plan", "spectral_match", "critique", "tool_call"]
        selected = roles or default
        return [
            self.dispatch(r, f"{intent} :: role={r}", spectral_context=spectral_context)
            for r in selected
        ]

    def serve_models_payload(self) -> Dict[str, Any]:
        """OpenAI-ish /v1/models payload for native serving."""
        data = []
        for mid in self.family_models.list_models():
            m = self.family_models.get(mid)
            data.append(
                {
                    "id": mid,
                    "object": "model",
                    "owned_by": "ItsNotAILABS/Auro",
                    "compute_plane": "MESIE",
                    "native": True,
                    "parameter_target": m.lane.parameter_target,
                    "tier": m.lane.tier.value,
                    "profile": m.profile.to_dict(),
                }
            )
        return {"object": "list", "data": data}


def bootstrap_runtime(parent_model_id: str = "Auro-14B") -> AuroNativeRuntime:
    return AuroNativeRuntime(parent_model_id=parent_model_id)
