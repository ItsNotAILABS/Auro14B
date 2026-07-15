"""Octopus arms — each arm maps to engines and internal API topics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mesie.internal_api.bus import InternalBus
from mesie.internal_api.messages import EngineResponse, MessageTopic


class ArmId(str, Enum):
    """Eight arms — octopus engineering layout."""

    SENSE = "sense"
    EMBED = "embed"
    MATCH = "match"
    MOVE = "move"
    CONTROL = "control"
    WORKFLOW = "workflow"
    LOGIC = "logic"
    MEMORY = "memory"


ARM_ENGINE_MAP: Dict[ArmId, str] = {
    ArmId.SENSE: "validation",
    ArmId.EMBED: "polyglot",
    ArmId.MATCH: "polyglot",
    ArmId.MOVE: "movement",
    ArmId.CONTROL: "control",
    ArmId.WORKFLOW: "workflow",
    ArmId.LOGIC: "logic",
    ArmId.MEMORY: "intelligence",
}


ARM_DEFAULT_ACTIONS: Dict[ArmId, str] = {
    ArmId.SENSE: "validate",
    ArmId.EMBED: "embed",
    ArmId.MATCH: "match",
    ArmId.MOVE: "advance",
    ArmId.CONTROL: "status",
    ArmId.WORKFLOW: "status",
    ArmId.LOGIC: "evaluate",
    ArmId.MEMORY: "memory",
}


@dataclass
class ArmState:
    arm: ArmId
    enabled: bool = True
    last_ok: bool = True
    last_action: Optional[str] = None
    ticks: int = 0


@dataclass
class OctopusArm:
    """One tentacle — routes commands to a backing engine via the bus."""

    arm_id: ArmId
    bus: InternalBus
    state: ArmState = field(init=False)

    def __post_init__(self) -> None:
        self.state = ArmState(arm=self.arm_id)

    @property
    def engine_name(self) -> str:
        return ARM_ENGINE_MAP[self.arm_id]

    def reach(self, action: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> EngineResponse:
        """Extend arm: send request to mapped engine."""
        if not self.state.enabled:
            return EngineResponse(False, self.engine_name, action or "", error=f"Arm {self.arm_id.value} disabled")

        act = action or ARM_DEFAULT_ACTIONS[self.arm_id]
        topic = MessageTopic.ARM_COMMAND
        if self.arm_id == ArmId.WORKFLOW:
            topic = MessageTopic.WORKFLOW_STEP
        elif self.arm_id == ArmId.MOVE:
            topic = MessageTopic.MOVEMENT_TICK
        elif self.arm_id == ArmId.CONTROL:
            topic = MessageTopic.CONTROL_SETPOINT
        elif self.arm_id == ArmId.LOGIC:
            topic = MessageTopic.LOGIC_RULE

        resp = self.bus.request(
            source=f"arm:{self.arm_id.value}",
            target=self.engine_name,
            action=act,
            payload=payload or {},
            topic=topic,
        )
        self.state.ticks += 1
        self.state.last_ok = resp.ok
        self.state.last_action = act
        return resp

    def enable(self, on: bool = True) -> None:
        self.state.enabled = on