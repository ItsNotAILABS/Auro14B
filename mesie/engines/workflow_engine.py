"""Workflow engine — multi-step pipelines with embedded state for octopus arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mesie.engines.base import Engine
from mesie.internal_api.bus import InternalBus
from mesie.internal_api.messages import EngineResponse, MessageEnvelope, MessageTopic


@dataclass
class WorkflowStep:
    name: str
    engine: str
    action: str
    payload_key: str = "record"
    done: bool = False
    result: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowDefinition:
    workflow_id: str
    steps: List[WorkflowStep] = field(default_factory=list)


class WorkflowEngine(Engine):
    name = "workflow"
    capabilities = ["define", "run", "status", "reset"]

    def __init__(self, bus: Optional[InternalBus] = None) -> None:
        self._bus = bus
        self._active: Optional[WorkflowDefinition] = None
        self._run_log: List[Dict[str, Any]] = []

    def attach_bus(self, bus: InternalBus) -> None:
        self._bus = bus

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        p = message.payload

        if action == "define":
            steps = [
                WorkflowStep(
                    name=s["name"],
                    engine=s["engine"],
                    action=s["action"],
                    payload_key=s.get("payload_key", "record"),
                )
                for s in p.get("steps", [])
            ]
            self._active = WorkflowDefinition(workflow_id=str(p.get("workflow_id", "default")), steps=steps)
            return EngineResponse(True, self.name, action, {"workflow_id": self._active.workflow_id, "steps": len(steps)})

        if action == "run":
            if self._active is None:
                return EngineResponse(False, self.name, action, error="No workflow defined")
            if self._bus is None:
                return EngineResponse(False, self.name, action, error="Bus not attached")
            context = dict(p.get("context", {}))
            for step in self._active.steps:
                payload = {step.payload_key: context.get("record"), **context}
                resp = self._bus.request("workflow", step.engine, step.action, payload, topic=MessageTopic.WORKFLOW_STEP)
                step.done = resp.ok
                step.result = resp.to_dict()
                self._run_log.append({"step": step.name, "ok": resp.ok, "engine": step.engine})
                if not resp.ok:
                    break
                context.update(resp.data)
            embed_resp = self._bus.request(
                "workflow",
                "embedding",
                "workflow_embed",
                {"steps": [{"name": s.name, "done": s.done, "priority": 1.0} for s in self._active.steps]},
            )
            return EngineResponse(
                True,
                self.name,
                action,
                {
                    "workflow_id": self._active.workflow_id,
                    "completed": all(s.done for s in self._active.steps),
                    "log": self._run_log[-len(self._active.steps) :],
                    "workflow_embedding": embed_resp.data.get("workflow_embedding"),
                },
            )

        if action == "status":
            if self._active is None:
                return EngineResponse(True, self.name, action, {"active": False})
            return EngineResponse(
                True,
                self.name,
                action,
                {
                    "active": True,
                    "workflow_id": self._active.workflow_id,
                    "steps": [{"name": s.name, "done": s.done} for s in self._active.steps],
                },
            )

        if action == "reset":
            self._active = None
            self._run_log.clear()
            return EngineResponse(True, self.name, action, {"reset": True})

        return EngineResponse(False, self.name, action, error="Unhandled")