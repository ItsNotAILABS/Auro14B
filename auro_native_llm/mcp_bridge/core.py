"""Governed MCP and service bridge for AURO/HIM.

Transport adapters stay separate from policy, execution, context compression,
and hash-linked receipts. The bridge implements compact JSON-RPC MCP methods
(`initialize`, `ping`, `tools/list`, `tools/call`) and a native Python API.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional


class RiskTier(IntEnum):
    READ_ONLY = 0
    STANDARD = 1
    CONFIRM = 2
    ISOLATED = 3
    DENY = 4


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    adapter: str
    operation: str
    risk_tier: RiskTier = RiskTier.STANDARD
    tags: tuple[str, ...] = ()
    timeout_seconds: float = 20.0

    def mcp_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "adapter": self.adapter,
                "riskTier": int(self.risk_tier),
                "tags": list(self.tags),
            },
        }


@dataclass
class ExecutionContext:
    actor: str = "auro"
    session_id: str = field(default_factory=lambda: f"mcp-{uuid.uuid4().hex[:12]}")
    confirmed: bool = False
    allow_isolated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReceipt:
    receipt_id: str
    tool: str
    adapter: str
    operation: str
    risk_tier: int
    allowed: bool
    ok: bool
    started_at: float
    duration_ms: float
    input_sha256: str
    output_sha256: str
    previous_sha256: str
    receipt_sha256: str
    error: Optional[str] = None


class PolicyDenied(RuntimeError):
    pass


class BridgePolicy:
    """Central policy envelope shared by every adapter."""

    def __init__(
        self,
        *,
        maximum_tier: RiskTier = RiskTier.ISOLATED,
        denied_tools: Iterable[str] = (),
        require_confirmation_from: RiskTier = RiskTier.CONFIRM,
    ) -> None:
        self.maximum_tier = maximum_tier
        self.denied_tools = set(denied_tools)
        self.require_confirmation_from = require_confirmation_from

    def authorize(self, tool: ToolDefinition, context: ExecutionContext) -> None:
        if tool.name in self.denied_tools or tool.risk_tier >= RiskTier.DENY:
            raise PolicyDenied(f"tool denied by policy: {tool.name}")
        if tool.risk_tier > self.maximum_tier:
            raise PolicyDenied(f"risk tier {int(tool.risk_tier)} exceeds policy maximum")
        if tool.risk_tier >= self.require_confirmation_from and not context.confirmed:
            raise PolicyDenied(f"confirmation required for {tool.name}")
        if tool.risk_tier >= RiskTier.ISOLATED and not context.allow_isolated:
            raise PolicyDenied(f"isolated execution not enabled for {tool.name}")


class ReceiptChain:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.tip = "0" * 64
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.tip = json.loads(line)["receipt_sha256"]
                except (KeyError, json.JSONDecodeError):
                    continue

    @staticmethod
    def hash_value(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def append(self, payload: MutableMapping[str, Any]) -> ExecutionReceipt:
        payload["previous_sha256"] = self.tip
        unsigned = dict(payload)
        unsigned.pop("receipt_sha256", None)
        payload["receipt_sha256"] = self.hash_value(unsigned)
        receipt = ExecutionReceipt(**payload)
        self.tip = receipt.receipt_sha256
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(receipt), sort_keys=True) + "\n")
        return receipt


class Adapter:
    name = "adapter"

    def list_tools(self) -> Iterable[ToolDefinition]:
        raise NotImplementedError

    def invoke(self, operation: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        raise NotImplementedError


class LocalAdapter(Adapter):
    name = "local"

    def __init__(self) -> None:
        self._tools: Dict[str, tuple[ToolDefinition, Callable[[Mapping[str, Any]], Any]]] = {}

    def register(self, definition: ToolDefinition, handler: Callable[[Mapping[str, Any]], Any]) -> None:
        if definition.adapter != self.name:
            raise ValueError("local definition must use adapter='local'")
        self._tools[definition.operation] = (definition, handler)

    def list_tools(self) -> Iterable[ToolDefinition]:
        return (item[0] for item in self._tools.values())

    def invoke(self, operation: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        try:
            return self._tools[operation][1](arguments)
        except KeyError as exc:
            raise KeyError(f"unknown local operation: {operation}") from exc


class HTTPServiceAdapter(Adapter):
    """Separate JS/legacy-service adapter driven by a compact manifest."""

    name = "service"

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        allowed_hosts: Iterable[str] = ("127.0.0.1", "localhost"),
    ) -> None:
        self.base_url = str(manifest.get("base_url", "")).rstrip("/")
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.hostname not in set(allowed_hosts):
            raise ValueError(f"service host not allowlisted: {parsed.hostname}")
        self._definitions: list[ToolDefinition] = []
        for raw in manifest.get("tools", []):
            method = str(raw.get("method", "GET")).upper()
            self._definitions.append(
                ToolDefinition(
                    name=str(raw["name"]),
                    description=str(raw.get("description", "Service operation")),
                    input_schema=dict(raw.get("input_schema", {"type": "object", "properties": {}})),
                    adapter=self.name,
                    operation=f"{method} {raw['path']}",
                    risk_tier=RiskTier(int(raw.get("risk_tier", RiskTier.STANDARD))),
                    tags=tuple(raw.get("tags", ("service",))),
                    timeout_seconds=float(raw.get("timeout_seconds", 20.0)),
                )
            )

    def list_tools(self) -> Iterable[ToolDefinition]:
        return iter(self._definitions)

    def invoke(self, operation: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        method, path = operation.split(" ", 1)
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if method == "GET":
            query = urllib.parse.urlencode(arguments, doseq=True)
            if query:
                url += "?" + query
        else:
            data = json.dumps(dict(arguments)).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"text": body, "status": response.status}


class OpenAPIAdapter(HTTPServiceAdapter):
    """Generate tools from a safe subset of OpenAPI 3.x."""

    name = "openapi"

    def __init__(
        self,
        specification: Mapping[str, Any],
        *,
        allowed_hosts: Iterable[str] = ("127.0.0.1", "localhost"),
    ) -> None:
        servers = specification.get("servers") or []
        base_url = servers[0].get("url") if servers else ""
        tools = []
        for path, path_item in specification.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                    continue
                properties: Dict[str, Any] = {}
                required: list[str] = []
                for parameter in operation.get("parameters", []):
                    name = parameter["name"]
                    properties[name] = dict(parameter.get("schema", {"type": "string"}))
                    if parameter.get("required"):
                        required.append(name)
                body_schema = (
                    (((operation.get("requestBody") or {}).get("content") or {}).get("application/json") or {})
                    .get("schema")
                )
                if isinstance(body_schema, Mapping):
                    properties["body"] = dict(body_schema)
                risk = operation.get(
                    "x-auro-risk-tier",
                    RiskTier.READ_ONLY if method.lower() == "get" else RiskTier.CONFIRM,
                )
                tools.append(
                    {
                        "name": operation.get("operationId")
                        or f"{method}_{path}".replace("/", "_").strip("_"),
                        "description": operation.get("summary")
                        or operation.get("description")
                        or "OpenAPI operation",
                        "method": method.upper(),
                        "path": path,
                        "input_schema": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                        "risk_tier": int(risk),
                        "tags": tuple(operation.get("tags", ("openapi",))),
                    }
                )
        super().__init__({"base_url": base_url, "tools": tools}, allowed_hosts=allowed_hosts)
        self.name = "openapi"
        self._definitions = [
            ToolDefinition(
                name=item.name,
                description=item.description,
                input_schema=item.input_schema,
                adapter=self.name,
                operation=item.operation,
                risk_tier=item.risk_tier,
                tags=item.tags,
                timeout_seconds=item.timeout_seconds,
            )
            for item in self._definitions
        ]


class MCPHTTPAdapter(Adapter):
    """Upstream MCP JSON-RPC adapter over HTTP POST."""

    name = "mcp"

    def __init__(
        self,
        endpoint: str,
        *,
        namespace: str,
        allowed_hosts: Iterable[str] = ("127.0.0.1", "localhost"),
    ) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.hostname not in set(allowed_hosts):
            raise ValueError(f"MCP host not allowlisted: {parsed.hostname}")
        self.endpoint = endpoint
        self.namespace = namespace
        self._tools: list[ToolDefinition] = []

    def _rpc(self, method: str, params: Mapping[str, Any] | None = None, timeout: float = 20.0) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": dict(params or {}),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
        if "error" in body:
            raise RuntimeError(str(body["error"]))
        return body.get("result")

    def refresh(self) -> None:
        result = self._rpc("tools/list") or {}
        definitions = []
        for raw in result.get("tools", []):
            annotations = raw.get("annotations") or {}
            definitions.append(
                ToolDefinition(
                    name=f"{self.namespace}.{raw['name']}",
                    description=raw.get("description", "Upstream MCP tool"),
                    input_schema=raw.get("inputSchema", {"type": "object", "properties": {}}),
                    adapter=self.name,
                    operation=raw["name"],
                    risk_tier=RiskTier(int(annotations.get("riskTier", RiskTier.CONFIRM))),
                    tags=("mcp", self.namespace),
                )
            )
        self._tools = definitions

    def list_tools(self) -> Iterable[ToolDefinition]:
        if not self._tools:
            self.refresh()
        return iter(self._tools)

    def invoke(self, operation: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        return self._rpc(
            "tools/call",
            {"name": operation, "arguments": dict(arguments)},
            timeout=timeout,
        )


class AuroMCPBridge:
    protocol_version = "2025-03-26"

    def __init__(
        self,
        *,
        policy: BridgePolicy | None = None,
        receipt_path: str | Path | None = "artifacts/mcp/receipts.jsonl",
        max_output_chars: int = 12_000,
    ) -> None:
        self.policy = policy or BridgePolicy()
        self.receipts = ReceiptChain(receipt_path)
        self.max_output_chars = max_output_chars
        self.adapters: Dict[str, Adapter] = {}
        self.tools: Dict[str, ToolDefinition] = {}

    def add_adapter(self, adapter: Adapter) -> None:
        if adapter.name in self.adapters:
            raise ValueError(f"adapter already registered: {adapter.name}")
        self.adapters[adapter.name] = adapter
        for tool in adapter.list_tools():
            if tool.name in self.tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self.tools[tool.name] = tool

    def list_tools(self, *, compact: bool = False) -> list[Dict[str, Any]]:
        values = sorted(self.tools.values(), key=lambda item: item.name)
        if not compact:
            return [item.mcp_schema() for item in values]
        return [
            {
                "name": item.name,
                "description": item.description[:160],
                "riskTier": int(item.risk_tier),
                "adapter": item.adapter,
            }
            for item in values
        ]

    def context_catalog(self, query: str = "", limit: int = 12) -> str:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        ranked = []
        for tool in self.tools.values():
            haystack = f"{tool.name} {tool.description} {' '.join(tool.tags)}".lower()
            score = sum(term in haystack for term in terms)
            ranked.append((score, tool.name, tool))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return "\n".join(
            f"{item.name} | risk={int(item.risk_tier)} | {item.description[:180]}"
            for _, _, item in ranked[:limit]
        )

    def _compact_output(self, value: Any) -> Dict[str, Any]:
        encoded = json.dumps(value, sort_keys=True, default=str)
        truncated = len(encoded) > self.max_output_chars
        return {
            "content": [{"type": "text", "text": encoded[: self.max_output_chars]}],
            "structuredContent": value if not truncated else None,
            "meta": {
                "chars": len(encoded),
                "truncated": truncated,
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            },
            "isError": False,
        }

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        context: ExecutionContext | None = None,
    ) -> Dict[str, Any]:
        arguments = dict(arguments or {})
        context = context or ExecutionContext()
        if name not in self.tools:
            raise KeyError(f"unknown tool: {name}")
        tool = self.tools[name]
        started = time.time()
        input_hash = ReceiptChain.hash_value(arguments)
        allowed = ok = False
        error: Optional[str] = None
        output: Any = None
        try:
            self.policy.authorize(tool, context)
            allowed = True
            output = self.adapters[tool.adapter].invoke(
                tool.operation,
                arguments,
                tool.timeout_seconds,
            )
            ok = True
        except Exception as exc:
            error = str(exc)[:500]
        result = (
            self._compact_output(output)
            if ok
            else {
                "content": [{"type": "text", "text": error or "execution failed"}],
                "structuredContent": None,
                "meta": {},
                "isError": True,
            }
        )
        receipt = self.receipts.append(
            {
                "receipt_id": f"rcpt-{uuid.uuid4().hex[:16]}",
                "tool": tool.name,
                "adapter": tool.adapter,
                "operation": tool.operation,
                "risk_tier": int(tool.risk_tier),
                "allowed": allowed,
                "ok": ok,
                "started_at": started,
                "duration_ms": (time.time() - started) * 1000.0,
                "input_sha256": input_hash,
                "output_sha256": result.get("meta", {}).get("sha256")
                or ReceiptChain.hash_value(error or ""),
                "previous_sha256": "",
                "receipt_sha256": "",
                "error": error,
            }
        )
        result["meta"] = {**result.get("meta", {}), "receipt": asdict(receipt)}
        return result

    def handle_jsonrpc(
        self,
        message: Mapping[str, Any],
        *,
        context: ExecutionContext | None = None,
    ) -> Dict[str, Any]:
        request_id = message.get("id")
        try:
            method = message["method"]
            params = message.get("params") or {}
            if method == "initialize":
                result = {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "auro-mcp-bridge", "version": "1.0.0"},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                result = self.call_tool(
                    params["name"],
                    params.get("arguments") or {},
                    context=context,
                )
            else:
                raise KeyError(f"unsupported method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(exc)[:500]},
            }
