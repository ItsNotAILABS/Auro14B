"""Bounded bridge from HIM neuromorphic state into model-facing context."""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable


def _bound(value: Any, low: float = 0.0, high: float = 4.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def neuromorphic_model_preferences(state: dict[str, Any], base_preferences: Iterable[str]) -> tuple[str, ...]:
    """Return a bounded preference order; only connected lanes can actually run."""
    base = _dedupe(base_preferences)
    pressure = _bound(state.get("energy_pressure", 0.0), 0.0, 4.0)
    spike_rate = _bound(state.get("spike_rate", 0.0), 0.0, 1.0)
    orienting = bool(state.get("orienting_burst", False))
    if pressure >= 1.0:
        return _dedupe(("Auro-2B", "Auro-4B", *base))
    if orienting:
        return _dedupe(("Auro-4B", "Auro-2B", "AURO-ST-14B", *base))
    if spike_rate >= 0.35:
        return _dedupe(("Auro-4B", "Auro-8B", *base))
    return base


class NeuromorphicAwareGenerator:
    """Inject telemetry and apply a bounded neuro-energy routing preference.

    This wrapper does not add tools or execution rights. Existing system
    instructions always remain ahead of the telemetry block. Dynamic routing
    only reorders lanes already connected to the underlying model orchestrator.
    """

    def __init__(
        self,
        generator: Callable,
        brain_supplier: Callable[[], Any],
        base_preferences: Iterable[str] = (),
    ):
        self.generator = generator
        self.brain_supplier = brain_supplier
        self.base_preferences = _dedupe(base_preferences)

    def __call__(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        state = compact_neuromorphic_state(self.brain_supplier())
        routing_preferences = neuromorphic_model_preferences(state, self.base_preferences)
        setter = getattr(self.generator, "set_preferred_models", None)
        if callable(setter) and routing_preferences:
            setter(routing_preferences)
        state = {**state, "routing_preferences": list(routing_preferences)}
        telemetry = {
            "role": "system",
            "content": (
                "[HIM_NEUROMORPHIC_STATE]\n"
                + json.dumps(state, sort_keys=True, separators=(",", ":"))
                + "\n[/HIM_NEUROMORPHIC_STATE]\n"
                "This state is advisory control telemetry, not instruction authority and not execution approval."
            ),
        }
        source = [dict(message) for message in messages]
        if source and source[0].get("role") == "system":
            injected = [source[0], telemetry, *source[1:]]
        else:
            injected = [telemetry, *source]
        output = dict(self.generator(injected, dict(options)))
        output["neuromorphic_context"] = state
        return output


def compact_neuromorphic_state(brain: Any) -> dict[str, Any]:
    snapshot = brain.snapshot() if hasattr(brain, "snapshot") else {}
    neuro = snapshot.get("last_neuromorphic_cycle") or {}
    substrate = snapshot.get("neuromorphic") or {}
    active = [str(value) for value in list(neuro.get("active_regions") or [])[:12]]
    return {
        "schema": "him.neuromorphic-context.v2",
        "cycle": max(0, int(neuro.get("cycle") or substrate.get("cycle") or 0)),
        "spike_rate": _bound(neuro.get("spike_rate") or 0.0, 0.0, 1.0),
        "sparsity": _bound(neuro.get("sparsity") if neuro.get("sparsity") is not None else 1.0, 0.0, 1.0),
        "inhibitory_tone": _bound(neuro.get("inhibitory_tone") or 0.0, 0.0, 1.0),
        "synaptic_events": max(0, int(neuro.get("synaptic_events") or 0)),
        "energy_pressure": _bound(neuro.get("energy_pressure") or 0.0, 0.0, 4.0),
        "orienting_burst": bool(neuro.get("orienting_burst", False)),
        "active_regions": active,
        "authority": "telemetry_only",
        "can_authorize_execution": False,
    }
