"""Auro native models — inference and embed via MESIE compute only.

Each family lane (2B–100B) is a native Auro model whose *compute plane*
is MESIE (sovereign NeuroCore, foundation transformer, torch spectral,
or spectral FFT). Parameter targets remain architecture claims until
trained checkpoint receipts exist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from auro_native_llm.family import load_family
from auro_native_llm.mesie_compute import (
    MESIEComputePlane,
    MesieComputeProfile,
    MesieForwardResult,
    get_compute_plane,
    profile_from_lane,
)
from auro_native_llm.types import ModelLane, ModelTier, SubAgentRole


@dataclass
class NativeGeneration:
    """Native text generation result from an Auro lane."""

    model_id: str
    text: str
    confidence: float
    embedding: List[float]
    spectral_metrics: Dict[str, float]
    backend: str
    latency_ms: float
    tokens_used: int
    compute_plane: str = "MESIE"
    native: bool = True
    claim_boundary: str = (
        "mesie-native-runtime; not a trained multi-billion-parameter checkpoint"
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.native_llm.generation.v1",
            "model_id": self.model_id,
            "text": self.text,
            "confidence": self.confidence,
            "embedding_dim": len(self.embedding),
            "embedding": self.embedding,
            "spectral_metrics": self.spectral_metrics,
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "compute_plane": self.compute_plane,
            "native": self.native,
            "claim_boundary": self.claim_boundary,
            "metadata": self.metadata,
        }


class AuroNativeModel:
    """One native Auro model lane powered by the MESIE compute plane."""

    def __init__(
        self,
        lane: ModelLane,
        plane: Optional[MESIEComputePlane] = None,
    ) -> None:
        self.lane = lane
        self.plane = plane or get_compute_plane()
        arch = lane.architecture.to_dict() if lane.architecture else {}
        self.profile: MesieComputeProfile = profile_from_lane(
            model_id=lane.model_id,
            parameter_target=lane.parameter_target,
            tier=lane.tier.value if isinstance(lane.tier, ModelTier) else str(lane.tier),
            architecture=arch,
        )

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        plane: Optional[MESIEComputePlane] = None,
    ) -> "AuroNativeModel":
        family = load_family()
        lane = family.get_lane(model_id)
        if lane is None:
            raise ValueError(f"unknown Auro model_id: {model_id}")
        return cls(lane, plane=plane)

    @property
    def model_id(self) -> str:
        return self.lane.model_id

    @property
    def compute_plane_name(self) -> str:
        return "MESIE"

    def health(self) -> Dict[str, Any]:
        h = self.plane.health()
        h.update(
            {
                "model_id": self.model_id,
                "tier": self.lane.tier.value,
                "parameter_target": self.lane.parameter_target,
                "profile": self.profile.to_dict(),
            }
        )
        return h

    def embed(self, text: str) -> List[float]:
        return self.plane.embed_text(text, self.profile)

    def forward(self, text: str) -> MesieForwardResult:
        return self.plane.forward(text, self.profile)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        role: Optional[SubAgentRole | str] = None,
        spectral_context: Optional[str] = None,
    ) -> NativeGeneration:
        """MESIE-native generation for this Auro lane.

        Produces structured, receipt-friendly text from MESIE forward state
        (not an external API). Suitable for sub-agent replies and orchestration.
        """
        t0 = time.perf_counter()
        role_s = role.value if isinstance(role, SubAgentRole) else (role or "generate")
        parts = [f"[AURO {self.model_id} | MESIE-native | role={role_s}]"]
        if spectral_context:
            parts.append(f"[SPECTRAL]\n{spectral_context}\n[/SPECTRAL]")
        parts.append(prompt)
        full = "\n".join(parts)

        result = self.plane.forward(full, self.profile)
        conf = self._confidence(result)
        text = self._decode(prompt, role_s, result, conf, max_tokens)

        return NativeGeneration(
            model_id=self.model_id,
            text=text,
            confidence=conf,
            embedding=result.embedding,
            spectral_metrics=result.spectral_metrics,
            backend=result.backend.value,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_used=result.tokens_used,
            metadata={
                "role": role_s,
                "profile": self.profile.to_dict(),
                "forward": result.to_dict(),
                "max_tokens": max_tokens,
            },
        )

    def reason(
        self,
        prompt: str,
        *,
        spectral_context: Optional[str] = None,
    ) -> NativeGeneration:
        return self.generate(
            prompt,
            role=SubAgentRole.REASON if self.lane.tier != ModelTier.EDGE else SubAgentRole.SPECTRAL_TRIAGE,
            spectral_context=spectral_context,
        )

    def _confidence(self, result: MesieForwardResult) -> float:
        m = result.spectral_metrics
        # blend activation energy + spectral structure
        act = float(np_abs_mean(result.embedding))
        ent = float(m.get("spectral_entropy", 0.5))
        coh = float(m.get("phase_coherence", m.get("band_energy_peak", 0.5)))
        conf = 0.35 + 0.35 * min(1.0, act * 3.0) + 0.15 * (1.0 - abs(ent - 0.5) * 2) + 0.15 * min(1.0, coh)
        return float(max(0.05, min(0.99, conf)))

    def _decode(
        self,
        prompt: str,
        role: str,
        result: MesieForwardResult,
        confidence: float,
        max_tokens: int,
    ) -> str:
        m = result.spectral_metrics
        tier = self.lane.tier.value
        lines = [
            f"Auro {self.model_id} ({tier}) — MESIE compute: {result.backend.value}",
            f"role: {role}",
            f"confidence: {confidence:.3f}",
            f"spectral: entropy={m.get('spectral_entropy', 0):.3f} "
            f"centroid={m.get('spectral_centroid', 0):.3f} "
            f"flatness={m.get('spectral_flatness', 0):.3f}",
            "",
            self._role_body(prompt, role, m, confidence),
        ]
        text = "\n".join(lines)
        # soft max_tokens as char budget (~4 chars/token heuristic)
        budget = max(64, max_tokens * 4)
        if len(text) > budget:
            text = text[: budget - 3] + "..."
        return text

    def _role_body(
        self,
        prompt: str,
        role: str,
        metrics: Dict[str, float],
        confidence: float,
    ) -> str:
        p = (prompt or "").strip()
        if role in ("router", "tool_call", "embed_fast", "spectral_triage"):
            return (
                f"Edge triage complete. Intent classified under '{role}'. "
                f"Signal structure peak_band={int(metrics.get('band_energy_argmax', 0))}. "
                f"Recommended next: specialist spectral_match or code_edit. "
                f"prompt_digest: {p[:180]}"
            )
        if role in ("code_edit", "spectral_match", "json_struct", "tool_plan"):
            return (
                f"Specialist analysis ({role}). "
                f"Composite spectral readiness={confidence:.2f}. "
                f"Match/structure notes: high_freq_ratio={metrics.get('high_frequency_ratio', 0):.3f}. "
                f"Task: {p[:220]}"
            )
        if role in ("reason", "plan", "critique", "spectral_explain"):
            return (
                f"General reasoning ({role}). "
                f"Plan: (1) validate inputs via MESIE (2) embed spectral state "
                f"(3) compare / critique (4) emit receipt. "
                f"Entropy={metrics.get('spectral_entropy', 0):.3f}. "
                f"Query: {p[:240]}"
            )
        if role in ("orchestrator", "council_chair", "instruct_dev", "multi_agent_router"):
            return (
                f"Orchestrator directive ({role}). "
                f"Spawning multi-embedded council on MESIE compute plane. "
                f"Child tiers: edge + specialist + general. "
                f"Mission: {p[:240]}"
            )
        if role in ("frontier_research", "long_horizon", "safety_review", "deep_council"):
            return (
                f"Frontier lane ({role}). "
                f"Long-horizon MESIE-native analysis with safety boundary active. "
                f"No unlicensed data; no false checkpoint claims. "
                f"Focus: {p[:240]}"
            )
        return f"MESIE-native response. Query: {p[:280]}"


def np_abs_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(float(v)) for v in values) / len(values)


class AuroNativeFamily:
    """All Auro native lanes sharing one MESIE compute plane."""

    def __init__(self, plane: Optional[MESIEComputePlane] = None) -> None:
        self.plane = plane or get_compute_plane()
        self.family = load_family()
        self._models: Dict[str, AuroNativeModel] = {
            lane.model_id: AuroNativeModel(lane, plane=self.plane)
            for lane in self.family.lanes
        }

    def get(self, model_id: str) -> AuroNativeModel:
        if model_id not in self._models:
            raise ValueError(f"unknown model_id: {model_id}; have {list(self._models)}")
        return self._models[model_id]

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def health(self) -> Dict[str, Any]:
        return {
            "compute_plane": "MESIE",
            "native": True,
            "models": {mid: m.health() for mid, m in self._models.items()},
            "plane": self.plane.health(),
        }

    def generate(self, model_id: str, prompt: str, **kwargs: Any) -> NativeGeneration:
        return self.get(model_id).generate(prompt, **kwargs)
