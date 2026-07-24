"""Governed AURO capability runtime for MCP and REST/JS service bridges."""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    adapter: str
    source: str
    risk_tier: str = "C1"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str
    risk_tier: str


class BridgeAdapter(Protocol):
    name: str

    def list_tools(self) -> List[ToolSpec]: ...
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...
    def health(self) -> Dict[str, Any]: ...


class PolicyEngine:
    """Small, inspectable policy gate shared by every bridge adapter."""

    _dangerous = re.compile(
        r"(^|[._-])(shell|exec|command|delete|remove|wallet|transfer|trade|hardware|iot|deploy|write)([._-]|$)",
        re.IGNORECASE,
    )

    def classify(self, tool: ToolSpec) -> str:
        if tool.risk_tier in {"C0", "C1", "C2", "C3", "C4", "C5"}:
            return tool.risk_tier
        return "C3" if self._dangerous.search(tool.name) else "C1"

    def decide(self, tool: ToolSpec, *, confirmed: bool = False) -> PolicyDecision:
        tier = self.classify(tool)
        if tier in {"C4", "C5"}:
            return PolicyDecision(False, True, "high-risk capability is disabled by default", tier)
        if tier in {"C2", "C3"} and not confirmed:
            return PolicyDecision(False, True, "explicit user confirmation required", tier)
        return PolicyDecision(True, False, "policy gate passed", tier)


class ReceiptChain:
    """Append-only hash-linked receipts for discovery, denial, and execution."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self.tip = "0" * 64
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.tip = json.loads(line)["hash"]

    def append(self, event: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        receipt = {
            "schema": "auro.bridge.receipt.v1",
            "event": event,
            "ts": time.time(),
            "previous_hash": self.tip,
            "payload": dict(payload),
        }
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), default=str).encode()
        receipt["hash"] = hashlib.sha256(encoded).hexdigest()
        self.tip = receipt["hash"]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, sort_keys=True, default=str) + "\n")
        return receipt


class MCPHTTPAdapter:
    """Minimal MCP JSON-RPC client for HTTP-accessible servers or bridges."""

    def __init__(self, name: str, endpoint: str, *, headers: Optional[Mapping[str, str]] = None, timeout: float = 15.0):
        self.name = name
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._request_id = 0

    def _rpc(self, method: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        self._request_id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": dict(params or {})}).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"content-type": "application/json", "accept": "application/json, text/event-stream", **self.headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("error"):
            raise RuntimeError(f"MCP {method} failed: {payload['error']}")
        return payload.get("result")

    def initialize(self) -> Any:
        return self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "AURO", "version": "1"},
            },
        )

    def list_tools(self) -> List[ToolSpec]:
        result = self._rpc("tools/list") or {}
        tools = []
        for item in result.get("tools", []):
            annotations = item.get("annotations") or {}
            tools.append(
                ToolSpec(
                    name=item["name"],
                    description=item.get("description", ""),
                    input_schema=item.get("inputSchema") or {"type": "object"},
                    adapter=self.name,
                    source=self.endpoint,
                    risk_tier=annotations.get("riskTier", "C1"),
                    metadata={"mcp_annotations": annotations},
                )
            )
        return tools

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        return self._rpc("tools/call", {"name": name, "arguments": dict(arguments)})

    def health(self) -> Dict[str, Any]:
        try:
            result = self.initialize()
            return {"ok": True, "adapter": self.name, "endpoint": self.endpoint, "server": result}
        except Exception as exc:
            return {"ok": False, "adapter": self.name, "endpoint": self.endpoint, "error": str(exc)[:300]}


class JSServiceAdapter:
    """REST/JS service bridge generated from a compact OpenAPI document."""

    def __init__(self, name: str, base_url: str, schema: Mapping[str, Any], *, headers: Optional[Mapping[str, str]] = None, timeout: float = 15.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.schema = dict(schema)
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._operations: Dict[str, Dict[str, Any]] = {}

    def list_tools(self) -> List[ToolSpec]:
        tools: List[ToolSpec] = []
        self._operations.clear()
        for path, methods in (self.schema.get("paths") or {}).items():
            for method, operation in methods.items():
                if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                    continue
                name = operation.get("operationId") or re.sub(r"[^a-zA-Z0-9_]+", "_", f"{method}_{path}").strip("_")
                body_schema = (((operation.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema")
                params = operation.get("parameters") or []
                properties, required = {}, []
                for item in params:
                    properties[item["name"]] = item.get("schema") or {"type": "string"}
                    if item.get("required"):
                        required.append(item["name"])
                if body_schema:
                    properties["body"] = body_schema
                input_schema: Dict[str, Any] = {"type": "object", "properties": properties}
                if required:
                    input_schema["required"] = required
                risk = operation.get("x-auro-risk-tier") or ("C1" if method.lower() == "get" else "C2")
                self._operations[name] = {"path": path, "method": method.upper(), "parameters": params}
                tools.append(ToolSpec(name, operation.get("summary") or operation.get("description") or name, input_schema, self.name, self.base_url, risk))
        return tools

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name not in self._operations:
            self.list_tools()
        operation = self._operations[name]
        arguments = dict(arguments)
        path = operation["path"]
        query = []
        for item in operation["parameters"]:
            value = arguments.get(item["name"])
            if value is None:
                continue
            if item.get("in") == "path":
                path = path.replace("{" + item["name"] + "}", urllib.parse.quote(str(value)))
            elif item.get("in") == "query":
                query.append((item["name"], str(value)))
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        body = None if "body" not in arguments else json.dumps(arguments["body"]).encode()
        request = urllib.request.Request(
            url,
            data=body,
            headers={"accept": "application/json", "content-type": "application/json", **self.headers},
            method=operation["method"],
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {"ok": True, "status": response.status}

    def health(self) -> Dict[str, Any]:
        return {"ok": True, "adapter": self.name, "base_url": self.base_url, "generated_tools": len(self.list_tools())}


class AuroCapabilityRegistry:
    """Unified registry shared by AURO, HIM, browser UI, and mobile clients."""

    def __init__(self, *, policy: Optional[PolicyEngine] = None, receipts: Optional[ReceiptChain] = None):
        self.policy = policy or PolicyEngine()
        self.receipts = receipts or ReceiptChain()
        self.adapters: Dict[str, BridgeAdapter] = {}
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, adapter: BridgeAdapter) -> None:
        self.adapters[adapter.name] = adapter

    def refresh(self) -> Dict[str, Any]:
        self.tools.clear()
        errors = {}
        for adapter in self.adapters.values():
            try:
                for tool in adapter.list_tools():
                    self.tools[f"{adapter.name}.{tool.name}"] = tool
            except Exception as exc:
                errors[adapter.name] = str(exc)[:300]
        receipt = self.receipts.append(
            "registry.refresh",
            {"tool_count": len(self.tools), "adapters": sorted(self.adapters), "errors": errors},
        )
        return {"ok": not errors, "tool_count": len(self.tools), "errors": errors, "receipt": receipt}

    def compact_catalog(self, query: str = "", *, max_tools: int = 12) -> List[Dict[str, Any]]:
        terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
        ranked = []
        risk_order = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
        for canonical, tool in self.tools.items():
            haystack = f"{canonical} {tool.description}".lower()
            score = sum(term in haystack for term in terms)
            risk_rank = risk_order.get(self.policy.classify(tool), 6)
            ranked.append((score, risk_rank, canonical, tool))
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [
            {
                "name": canonical,
                "description": tool.description[:240],
                "input_schema": tool.input_schema,
                "risk_tier": self.policy.classify(tool),
                "source": tool.source,
            }
            for _, _, canonical, tool in ranked[:max_tools]
        ]

    def invoke(self, canonical_name: str, arguments: Optional[Mapping[str, Any]] = None, *, confirmed: bool = False) -> Dict[str, Any]:
        if canonical_name not in self.tools:
            raise KeyError(f"unknown capability: {canonical_name}")
        tool = self.tools[canonical_name]
        decision = self.policy.decide(tool, confirmed=confirmed)
        if not decision.allowed:
            receipt = self.receipts.append(
                "tool.denied",
                {"tool": canonical_name, "arguments": dict(arguments or {}), "decision": asdict(decision)},
            )
            return {"ok": False, "denied": True, "decision": asdict(decision), "receipt": receipt}
        started = time.perf_counter()
        try:
            result = self.adapters[tool.adapter].call_tool(tool.name, dict(arguments or {}))
            receipt = self.receipts.append(
                "tool.executed",
                {
                    "tool": canonical_name,
                    "ok": True,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "result_sha256": hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest(),
                },
            )
            return {"ok": True, "result": result, "decision": asdict(decision), "receipt": receipt}
        except Exception as exc:
            receipt = self.receipts.append("tool.failed", {"tool": canonical_name, "error": str(exc)[:300]})
            return {"ok": False, "error": str(exc)[:300], "decision": asdict(decision), "receipt": receipt}

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "schema": "auro.capability_registry.v1",
            "adapters": {name: adapter.health() for name, adapter in self.adapters.items()},
            "tool_count": len(self.tools),
            "receipt_tip": self.receipts.tip,
        }


class AuroBridgeRuntime:
    """Bind a capability registry to an AURO/HIM instance without replacing it."""

    def __init__(self, auro: Any, registry: AuroCapabilityRegistry):
        self.auro = auro
        self.registry = registry

    def context_for(self, goal: str, *, max_tools: int = 8) -> Dict[str, Any]:
        return {
            "schema": "auro.bridge.context.v1",
            "goal": goal,
            "tools": self.registry.compact_catalog(goal, max_tools=max_tools),
            "registry_health": self.registry.health(),
        }

    def execute(self, goal: str, *, tool: Optional[str] = None, arguments: Optional[Mapping[str, Any]] = None, confirmed: bool = False) -> Dict[str, Any]:
        context = self.context_for(goal)
        if hasattr(self.auro, "colony") and hasattr(self.auro.colony, "context"):
            self.auro.colony.context.ingest(
                json.dumps(context, sort_keys=True)[:4000],
                kind="system",
                meta={"bridge": "capability_registry"},
            )
        tool_result = self.registry.invoke(tool, arguments, confirmed=confirmed) if tool else None
        if hasattr(self.auro, "run"):
            auro_result = self.auro.run(goal)
        elif hasattr(self.auro, "chat"):
            auro_result = self.auro.chat(goal)
        else:
            auro_result = {"ok": False, "error": "AURO object has no run/chat method"}
        return {
            "schema": "auro.bridge.run.v1",
            "ok": bool((auro_result or {}).get("ok", True)),
            "goal": goal,
            "context": context,
            "tool_result": tool_result,
            "auro_result": auro_result,
        }
