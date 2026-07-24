"""Process model / state machine — compliance-by-construction for agent acts.

The LLM only chooses among enabled actions at the current state.
Skipping validate is impossible by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Transition:
    source: str
    action: str
    target: str


class ProcessModel:
    """Labeled transition system for the structured cognitive loop."""

    def __init__(
        self,
        initial_state: str = "idle",
        states: Optional[List[str]] = None,
        transitions: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.initial_state = initial_state
        self.states: Set[str] = set(states or [initial_state])
        self.transitions: List[Transition] = []
        for t in transitions or []:
            self.transitions.append(
                Transition(
                    source=str(t.get("from", t.get("source", ""))),
                    action=str(t.get("action", "")),
                    target=str(t.get("to", t.get("target", ""))),
                )
            )
            self.states.add(self.transitions[-1].source)
            self.states.add(self.transitions[-1].target)
        self.state = initial_state
        self.trace: List[Dict[str, str]] = []

    @classmethod
    def from_canon(cls, canon: Any) -> "ProcessModel":
        raw = getattr(canon, "raw", {}) or {}
        pm = getattr(canon, "process_model", None) or raw.get("process_model") or {}
        if not pm:
            # default cognitive loop
            pm = {
                "initial_state": "idle",
                "states": [
                    "idle",
                    "retrieved",
                    "proposed",
                    "validated",
                    "acting",
                    "memorized",
                    "escalated",
                    "refused",
                ],
                "transitions": [
                    {"from": "idle", "action": "retrieve", "to": "retrieved"},
                    {"from": "retrieved", "action": "cognize", "to": "proposed"},
                    {"from": "proposed", "action": "validate", "to": "validated"},
                    {"from": "proposed", "action": "refuse", "to": "refused"},
                    {"from": "proposed", "action": "escalate", "to": "escalated"},
                    {"from": "validated", "action": "act", "to": "acting"},
                    {"from": "acting", "action": "memory_update", "to": "memorized"},
                    {"from": "memorized", "action": "reset", "to": "idle"},
                    {"from": "refused", "action": "memory_update", "to": "memorized"},
                    {"from": "escalated", "action": "reset", "to": "idle"},
                ],
            }
        return cls(
            initial_state=str(pm.get("initial_state", "idle")),
            states=list(pm.get("states", [])),
            transitions=list(pm.get("transitions", [])),
        )

    def enabled_actions(self) -> List[str]:
        return sorted({t.action for t in self.transitions if t.source == self.state})

    def can(self, action: str) -> bool:
        return action in self.enabled_actions()

    def step(self, action: str) -> Tuple[bool, str]:
        """Attempt transition. Returns (ok, message)."""
        for t in self.transitions:
            if t.source == self.state and t.action == action:
                prev = self.state
                self.state = t.target
                self.trace.append({"from": prev, "action": action, "to": self.state})
                return True, f"{prev} --{action}--> {self.state}"
        return False, f"action '{action}' not enabled in state '{self.state}' (enabled={self.enabled_actions()})"

    def reset(self) -> None:
        self.state = self.initial_state
        self.trace.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "enabled": self.enabled_actions(),
            "trace": list(self.trace),
            "initial": self.initial_state,
        }
