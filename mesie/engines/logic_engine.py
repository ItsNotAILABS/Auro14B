"""Logic engine — rule evaluation driving control and workflow arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope


@dataclass
class LogicRule:
    name: str
    condition: str
    action: str
    target_engine: str
    target_action: str
    payload: Dict[str, Any]


class LogicEngine(Engine):
    name = "logic"
    capabilities = ["add_rule", "evaluate", "list_rules", "clear"]

    def __init__(self) -> None:
        self._rules: List[LogicRule] = []

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        p = message.payload

        if action == "add_rule":
            rule = LogicRule(
                name=p["name"],
                condition=p["condition"],
                action=p.get("then_action", p.get("action", "notify")),
                target_engine=p.get("target_engine", "control"),
                target_action=p.get("target_action", "evaluate"),
                payload=p.get("payload", {}),
            )
            self._rules.append(rule)
            return EngineResponse(True, self.name, action, {"rule_count": len(self._rules)})

        if action == "list_rules":
            return EngineResponse(
                True,
                self.name,
                action,
                {"rules": [{"name": r.name, "condition": r.condition} for r in self._rules]},
            )

        if action == "clear":
            self._rules.clear()
            return EngineResponse(True, self.name, action, {"cleared": True})

        if action == "evaluate":
            ctx = p.get("context", {})
            fired: List[Dict[str, Any]] = []
            for rule in self._rules:
                if self._check(rule.condition, ctx):
                    fired.append(
                        {
                            "rule": rule.name,
                            "target_engine": rule.target_engine,
                            "target_action": rule.target_action,
                            "then_action": rule.action,
                            "payload": {**rule.payload, **ctx},
                        }
                    )
            return EngineResponse(True, self.name, action, {"fired": fired, "count": len(fired)})

        return EngineResponse(False, self.name, action, error="Unhandled")

    @staticmethod
    def _check(condition: str, ctx: Dict[str, Any]) -> bool:
        if condition == "anomaly_high":
            return float(ctx.get("anomaly", 0)) > float(ctx.get("anomaly_threshold", 2.5))
        if condition == "similarity_low":
            return float(ctx.get("similarity", 1)) < float(ctx.get("similarity_threshold", 0.65))
        if condition == "workflow_incomplete":
            return not ctx.get("workflow_complete", True)
        if condition == "always":
            return True
        return False