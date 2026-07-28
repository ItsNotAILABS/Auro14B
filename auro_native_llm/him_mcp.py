"""MCP-enabled HIM adapter without coupling the core organism to transports."""
from __future__ import annotations

from typing import Any, Dict

from auro_native_llm.him.being import HIM
from auro_native_llm.mcp_bridge import AuroMCPBridge, ExecutionContext


class MCPEnabledHIM(HIM):
    def __init__(self, *args: Any, mcp_bridge: AuroMCPBridge, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mcp_bridge = mcp_bridge

    def whoami(self) -> Dict[str, Any]:
        identity = super().whoami()
        identity["tools"] = list(identity.get("tools", [])) + ["mcp_catalog", "mcp_call"]
        identity["mcp"] = {
            "uri": "mcp://localhost:8080",
            "tool_count": len(self.mcp_bridge.tools),
            "receipt_tip": self.mcp_bridge.receipts.tip,
            "governed": True,
        }
        return identity

    def plan(self, goal: str, sense: Dict[str, Any]) -> Dict[str, Any]:
        plan = super().plan(goal, sense)
        low = goal.lower()
        if any(
            token in low
            for token in (
                "mcp", "tool", "simulate", "service", "api", "hardware",
                "iot", "execute", "experiment", "bridge",
            )
        ):
            plan["actions"].append(
                {"tool": "mcp_catalog", "why": "discover governed external capabilities"}
            )
            plan["mcp_candidates"] = self.mcp_bridge.context_catalog(goal, limit=6)
            plan["max_steps"] = min(8, len(plan["actions"]) + 1)
        return plan

    def act(self, tool: str, goal: str) -> Dict[str, Any]:
        if tool == "mcp_catalog":
            catalog = self.mcp_bridge.context_catalog(goal, limit=12)
            return {
                "ok": True,
                "tool": tool,
                "text": catalog,
                "meta": {"tool_count": len(self.mcp_bridge.tools)},
            }
        if tool.startswith("mcp:"):
            name = tool[4:]
            result = self.mcp_bridge.call_tool(
                name,
                {"goal": goal},
                context=ExecutionContext(actor=self.name),
            )
            text = "\n".join(
                item.get("text", "") for item in result.get("content", [])
            )
            return {
                "ok": not result.get("isError"),
                "tool": tool,
                "text": text,
                "meta": result.get("meta", {}),
            }
        return super().act(tool, goal)
