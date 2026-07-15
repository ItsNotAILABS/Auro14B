"""Control engine — setpoints, modes, and arm coordination for octopus robotics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope


@dataclass
class ControlState:
    mode: str = "idle"
    similarity_setpoint: float = 0.75
    anomaly_limit: float = 2.5
    active_arms: Dict[str, bool] = field(default_factory=dict)
    emergency_stop: bool = False


class ControlEngine(Engine):
    name = "control"
    capabilities = ["set_mode", "set_setpoint", "arm_enable", "status", "evaluate"]

    def __init__(self) -> None:
        self._state = ControlState()

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        p = message.payload

        if action == "set_mode":
            self._state.mode = str(p.get("mode", "idle"))
            return EngineResponse(True, self.name, action, {"mode": self._state.mode})

        if action == "set_setpoint":
            if "similarity" in p:
                self._state.similarity_setpoint = float(p["similarity"])
            if "anomaly_limit" in p:
                self._state.anomaly_limit = float(p["anomaly_limit"])
            return EngineResponse(
                True,
                self.name,
                action,
                {
                    "similarity_setpoint": self._state.similarity_setpoint,
                    "anomaly_limit": self._state.anomaly_limit,
                },
            )

        if action == "arm_enable":
            arm = str(p.get("arm", ""))
            enabled = bool(p.get("enabled", True))
            self._state.active_arms[arm] = enabled
            return EngineResponse(True, self.name, action, {"arm": arm, "enabled": enabled})

        if action == "status":
            return EngineResponse(True, self.name, action, self._state_dict())

        if action == "evaluate":
            similarity = float(p.get("similarity", 0.0))
            anomaly = float(p.get("anomaly", 0.0))
            commands = []
            if self._state.emergency_stop:
                commands.append("halt_all")
            elif similarity < self._state.similarity_setpoint:
                commands.append("investigate_match")
            if anomaly > self._state.anomaly_limit:
                commands.append("trigger_alert")
            return EngineResponse(
                True,
                self.name,
                action,
                {"commands": commands, "mode": self._state.mode},
            )

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _state_dict(self) -> Dict:
        return {
            "mode": self._state.mode,
            "similarity_setpoint": self._state.similarity_setpoint,
            "anomaly_limit": self._state.anomaly_limit,
            "active_arms": dict(self._state.active_arms),
            "emergency_stop": self._state.emergency_stop,
        }