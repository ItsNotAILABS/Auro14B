"""Verified multi-model routing for HIM/NOVA.

A model lane is an actual generator plus identity metadata. Agents do not count
as models and parameter totals are never summed unless weights are truly loaded.
Persona preferences are routing priors only: a preferred model must still exist,
be enabled, satisfy the requested capability, and pass the same failure policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Callable, Iterable

Generator = Callable[[list[dict[str, str]], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ModelLane:
    id: str
    model: str
    role: str
    provider: str
    generator: Generator = field(repr=False, compare=False)
    parameter_count: int | None = None
    capabilities: tuple[str, ...] = ("general",)
    priority: int = 100
    local: bool = False
    enabled: bool = True
    checkpoint_hash: str | None = None

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("generator", None)
        value["parameter_count_verified"] = self.parameter_count is not None
        value["identity_verified"] = bool(self.checkpoint_hash or (self.model and self.provider))
        return value


@dataclass(frozen=True)
class RouteDecision:
    task: str
    strategy: str
    selected_lane: str
    candidates: tuple[str, ...]
    reason: str


class MultiModelOrchestrator:
    """Task router with observable attempts, bounded failover and no identity blur."""

    def __init__(self, lanes: Iterable[ModelLane], allow_hosted_fallback: bool = False):
        self.lanes = {lane.id: lane for lane in lanes if lane.enabled}
        if not self.lanes:
            raise ValueError("at least one enabled model lane is required")
        self.allow_hosted_fallback = allow_hosted_fallback
        self.preferred_models: tuple[str, ...] = ()
        self.traces: list[dict[str, Any]] = []

    def set_preferred_models(self, preferred_models: Iterable[str]) -> tuple[str, ...]:
        requested = tuple(dict.fromkeys(str(value) for value in preferred_models if str(value)))
        # Preferences may name registry models that are not connected runtime lanes.
        # Preserve them for receipts, but routing only rewards connected matches.
        self.preferred_models = requested
        return requested

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "him.model_fleet.v2",
            "model_count": len(self.lanes),
            "allow_hosted_fallback": self.allow_hosted_fallback,
            "preferred_models": list(self.preferred_models),
            "parameter_accounting": "per-lane weights; never agents or token counts",
            "models": [lane.public() for lane in self.lanes.values()],
        }

    def classify(self, text: str) -> str:
        lower = text.lower()
        scores = {
            "code": sum(value in lower for value in ("code", "python", "typescript", "function", "debug", "test", "contract", "solidity")),
            "math": sum(value in lower for value in ("calculate", "equation", "proof", "risk", "monte carlo", "optimize", "math")),
            "research": sum(value in lower for value in ("research", "source", "evidence", "compare", "investigate", "latest")),
            "tool": sum(value in lower for value in ("deploy", "execute", "run", "build", "manage", "worker", "agent")),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] else "general"

    def _preference_rank(self, lane: ModelLane) -> int:
        for index, preferred in enumerate(self.preferred_models):
            if preferred in {lane.id, lane.model}:
                return index
        return len(self.preferred_models) + 1

    def route(self, messages: list[dict[str, str]], strategy: str = "single") -> RouteDecision:
        text = "\n".join(str(item.get("content", "")) for item in messages)
        task = self.classify(text)
        ranked = sorted(
            self.lanes.values(),
            key=lambda lane: (
                task not in lane.capabilities,
                self._preference_rank(lane),
                not lane.local,
                lane.priority,
                lane.id,
            ),
        )
        selected = ranked[0]
        preferred_match = selected.id in self.preferred_models or selected.model in self.preferred_models
        return RouteDecision(
            task,
            strategy,
            selected.id,
            tuple(lane.id for lane in ranked),
            f"selected {selected.id}: capability={task}; persona_preferred={preferred_match}; local and priority break remaining ties",
        )

    def __call__(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        options = dict(options)
        strategy = str(options.pop("auro_strategy", "single"))
        decision = self.route(messages, strategy)
        ordered = [self.lanes[lane_id] for lane_id in decision.candidates]
        attempts: list[dict[str, Any]] = []
        started = time.perf_counter()
        last: Exception | None = None
        for lane in ordered:
            if lane.id != decision.selected_lane and lane.provider != "repository-native-open-weights" and not self.allow_hosted_fallback:
                continue
            attempt_started = time.perf_counter()
            try:
                output = lane.generator(messages, dict(options))
                attempts.append({
                    "lane_id": lane.id,
                    "model": lane.model,
                    "provider": lane.provider,
                    "task": decision.task,
                    "ok": True,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                result = dict(output)
                result["routed_model"] = lane.public()
                result["route"] = asdict(decision)
                self._record(decision, attempts, started)
                return result
            except Exception as exc:
                last = exc
                attempts.append({
                    "lane_id": lane.id,
                    "model": lane.model,
                    "provider": lane.provider,
                    "task": decision.task,
                    "ok": False,
                    "error": type(exc).__name__,
                    "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
        self._record(decision, attempts, started)
        raise RuntimeError(f"all authorized model lanes failed ({len(attempts)} attempts)") from last

    def _record(self, decision: RouteDecision, attempts: list[dict[str, Any]], started: float) -> None:
        body = {
            "decision": asdict(decision),
            "attempts": attempts,
            "preferred_models": list(self.preferred_models),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        body["receipt_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.traces.append(body)

    def drain_traces(self) -> list[dict[str, Any]]:
        value = self.traces[:]
        self.traces.clear()
        return value
