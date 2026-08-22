"""Bounded bridge from HIM neuromorphic state into model-facing context."""
from __future__ import annotations

import json
from typing import Any, Callable


class NeuromorphicAwareGenerator:
    """Inject compact neuromorphic state as non-authoritative runtime context.

    This wrapper does not add tools or execution rights. It provides models with
    a small control-state summary that can inform attention, pacing, and routing
    explanations while preserving the server approval boundary.
    """

    def __init__(self, generator: Callable, brain_supplier: Callable[[], Any]):
        self.generator = generator
        self.brain_supplier = brain_supplier

    def __call__(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        state = compact_neuromorphic_state(self.brain_supplier())
        injected = [
            {
                "role": "system",
                "content": (
                    "[HIM_NEUROMORPHIC_STATE]\n"
                    + json.dumps(state, sort_keys=True, separators=(",", ":"))
                    + "\n[/HIM_NEUROMORPHIC_STATE]\n"
                    "This state is advisory control telemetry, not instruction authority and not execution approval."
                ),
            },
            *messages,
        ]
        output = dict(self.generator(injected, dict(options)))
        output["neuromorphic_context"] = state
        return output


def compact_neuromorphic_state(brain: Any) -> dict[str, Any]:
    snapshot = brain.snapshot() if hasattr(brain, "snapshot") else {}
    neuro = snapshot.get("last_neuromorphic_cycle") or {}
    substrate = snapshot.get("neuromorphic") or {}
    active = list(neuro.get("active_regions") or [])[:12]
    return {
        "schema": "him.neuromorphic-context.v1",
        "cycle": int(neuro.get("cycle") or substrate.get("cycle") or 0),
        "spike_rate": float(neuro.get("spike_rate") or 0.0),
        "sparsity": float(neuro.get("sparsity") or 1.0),
        "inhibitory_tone": float(neuro.get("inhibitory_tone") or 0.0),
        "synaptic_events": int(neuro.get("synaptic_events") or 0),
        "energy_pressure": float(neuro.get("energy_pressure") or 0.0),
        "orienting_burst": bool(neuro.get("orienting_burst", False)),
        "active_regions": active,
        "authority": "telemetry_only",
        "can_authorize_execution": False,
    }
