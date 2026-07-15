"""Internal API message types for cross-engine communication."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MessageTopic(str, Enum):
    """Standard bus topics."""

    ENGINE_REQUEST = "engine.request"
    ENGINE_RESPONSE = "engine.response"
    ARM_COMMAND = "arm.command"
    ARM_STATUS = "arm.status"
    WORKFLOW_STEP = "workflow.step"
    WORKFLOW_COMPLETE = "workflow.complete"
    CONTROL_SETPOINT = "control.setpoint"
    MOVEMENT_TICK = "movement.tick"
    LOGIC_RULE = "logic.rule"
    EMBED_INDEX = "embed.index"
    BROADCAST = "bus.broadcast"


@dataclass
class MessageEnvelope:
    """Packet passed between engines and octopus arms."""

    topic: MessageTopic
    source: str
    target: str
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def reply_to(self, source: str, action: str, payload: Dict[str, Any]) -> MessageEnvelope:
        return MessageEnvelope(
            topic=MessageTopic.ENGINE_RESPONSE,
            source=source,
            target=self.source,
            action=action,
            payload=payload,
            correlation_id=self.message_id,
        )


@dataclass
class EngineResponse:
    """Normalized engine output."""

    ok: bool
    engine: str
    action: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "engine": self.engine,
            "action": self.action,
            "data": self.data,
            "error": self.error,
        }