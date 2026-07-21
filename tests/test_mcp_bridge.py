import json
from pathlib import Path

from auro_native_llm.mcp_bridge import (
    AuroMCPBridge,
    ExecutionContext,
    LocalAdapter,
    OpenAPIAdapter,
    RiskTier,
    ToolDefinition,
)


def build(tmp_path: Path):
    bridge = AuroMCPBridge(receipt_path=tmp_path / "receipts.jsonl", max_output_chars=80)
    local = LocalAdapter()
    local.register(
        ToolDefinition(
            name="lab.echo",
            description="Echo lab input",
            input_schema={"type": "object"},
            adapter="local",
            operation="echo",
            risk_tier=RiskTier.READ_ONLY,
        ),
        lambda args: {"echo": args},
    )
    local.register(
        ToolDefinition(
            name="lab.execute",
            description="Execute a controlled experiment",
            input_schema={"type": "object"},
            adapter="local",
            operation="execute",
            risk_tier=RiskTier.CONFIRM,
        ),
        lambda args: {"executed": args},
    )
    bridge.add_adapter(local)
    return bridge


def test_mcp_initialize_list_and_call(tmp_path):
    bridge = build(tmp_path)
    initialized = bridge.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert initialized["result"]["serverInfo"]["name"] == "auro-mcp-bridge"
    listed = bridge.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {tool["name"] for tool in listed["result"]["tools"]} == {
        "lab.echo",
        "lab.execute",
    }
    called = bridge.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "lab.echo", "arguments": {"x": 4}},
        }
    )
    assert called["result"]["isError"] is False
    assert called["result"]["structuredContent"] == {"echo": {"x": 4}}


def test_confirmation_policy_and_receipt_chain(tmp_path):
    bridge = build(tmp_path)
    denied = bridge.call_tool("lab.execute", {"x": 1})
    assert denied["isError"] is True
    assert denied["meta"]["receipt"]["allowed"] is False
    allowed = bridge.call_tool(
        "lab.execute", {"x": 2}, context=ExecutionContext(confirmed=True)
    )
    assert allowed["isError"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "receipts.jsonl").read_text().splitlines()
    ]
    assert rows[1]["previous_sha256"] == rows[0]["receipt_sha256"]


def test_output_compaction(tmp_path):
    bridge = build(tmp_path)
    result = bridge.call_tool("lab.echo", {"blob": "x" * 200})
    assert result["meta"]["truncated"] is True
    assert result["structuredContent"] is None


def test_openapi_generation_is_risk_aware():
    spec = {
        "openapi": "3.1.0",
        "servers": [{"url": "http://127.0.0.1:9999"}],
        "paths": {
            "/health": {
                "get": {"operationId": "health", "summary": "Health"}
            },
            "/run": {
                "post": {
                    "operationId": "run_sim",
                    "summary": "Run simulation",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        }
                    },
                }
            },
        },
    }
    adapter = OpenAPIAdapter(spec)
    tools = {tool.name: tool for tool in adapter.list_tools()}
    assert tools["health"].risk_tier == RiskTier.READ_ONLY
    assert tools["run_sim"].risk_tier == RiskTier.CONFIRM
    assert tools["run_sim"].adapter == "openapi"


def test_compact_catalog_ranks_matching_tools(tmp_path):
    bridge = build(tmp_path)
    catalog = bridge.context_catalog("execute experiment", limit=1)
    assert catalog.startswith("lab.execute")
