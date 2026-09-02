from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass
class BrainRuntimeState:
    schema: str = "auro.brain_runtime_state.v1"
    identity: str = "AURO"
    goals: list[str] = field(default_factory=list)
    trust: float = 0.5
    stress: float = 0.0
    overload: float = 0.0
    salience: float = 0.5
    recurrence_depth: int = 0
    active_drive: str = "stability"
    unresolved_tensions: list[str] = field(default_factory=list)
    active_tasks: list[str] = field(default_factory=list)
    consequence_history: list[dict[str, Any]] = field(default_factory=list)
    episodic_memory: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    sequence: int = 0
    previous_hash: str = "GENESIS"
    state_hash: str = ""

    def normalized(self) -> dict[str, Any]:
        data = asdict(self)
        data["state_hash"] = ""
        return data

    def seal(self) -> str:
        raw = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":")).encode()
        self.state_hash = hashlib.sha256(raw).hexdigest()
        return self.state_hash

    def validate(self) -> None:
        for name in ("trust", "stress", "overload", "salience", "confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.recurrence_depth < 0 or self.sequence < 0:
            raise ValueError("recurrence_depth and sequence must be non-negative")


class PersistentBrainState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> BrainRuntimeState:
        if not self.path.exists():
            state = BrainRuntimeState()
            state.seal()
            return state
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        state = BrainRuntimeState(**payload)
        expected = state.state_hash
        actual = state.seal()
        if expected and expected != actual:
            raise RuntimeError("brain runtime state hash mismatch")
        state.state_hash = expected or actual
        state.validate()
        return state

    def save(self, state: BrainRuntimeState) -> BrainRuntimeState:
        state.validate()
        state.seal()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        return state

    def transition(self, *, observation: str, consequence: dict[str, Any] | None = None, task: str | None = None) -> BrainRuntimeState:
        state = self.load()
        previous = state.state_hash
        state.previous_hash = previous
        state.sequence += 1
        state.recurrence_depth += 1
        if task and task not in state.active_tasks:
            state.active_tasks.append(task)
        if observation:
            state.episodic_memory.append({"sequence": state.sequence, "observation": observation, "parent_hash": previous})
            state.episodic_memory = state.episodic_memory[-256:]
        if consequence:
            state.consequence_history.append({"sequence": state.sequence, **consequence})
            state.consequence_history = state.consequence_history[-256:]
        return self.save(state)

    @staticmethod
    def inference_context(state: BrainRuntimeState) -> str:
        payload = {
            "identity": state.identity,
            "goals": state.goals[-8:],
            "trust": state.trust,
            "stress": state.stress,
            "overload": state.overload,
            "salience": state.salience,
            "recurrence_depth": state.recurrence_depth,
            "active_drive": state.active_drive,
            "unresolved_tensions": state.unresolved_tensions[-8:],
            "active_tasks": state.active_tasks[-8:],
            "confidence": state.confidence,
            "sequence": state.sequence,
            "state_hash": state.state_hash,
        }
        return "<brain_state>" + json.dumps(payload, sort_keys=True) + "</brain_state>"
