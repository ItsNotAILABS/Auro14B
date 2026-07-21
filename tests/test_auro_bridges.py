import json
from pathlib import Path

from auro_native_llm.bridges import (
    AuroBridgeRuntime,
    AuroCapabilityRegistry,
    JSServiceAdapter,
    ReceiptChain,
    ToolSpec,
)


class FakeAdapter:
    name = "fake"

    def list_tools(self):
        return [
            ToolSpec("read_status", "Read status", {"type": "object"}, self.name, "fake://local", "C1"),
            ToolSpec("deploy_service", "Deploy service", {"type": "object"}, self.name, "fake://local", "C3"),
        ]

    def call_tool(self, name, arguments):
        return {"name": name, "arguments": dict(arguments), "value": 7}

    def health(self):
        return {"ok": True}


class FakeContext:
    def __init__(self):
        self.items = []

    def ingest(self, text, **kwargs):
        self.items.append((text, kwargs))


class FakeAuro:
    def __init__(self):
        self.colony = type("Colony", (), {"context": FakeContext()})()

    def run(self, goal):
        return {"ok": True, "answer": goal}


def test_registry_discovery_execution_denial_and_receipt_chain(tmp_path: Path):
    chain = ReceiptChain(tmp_path / "receipts.jsonl")
    registry = AuroCapabilityRegistry(receipts=chain)
    registry.register(FakeAdapter())
    refresh = registry.refresh()
    assert refresh["tool_count"] == 2
    assert registry.invoke("fake.read_status", {"x": 1})["ok"]
    denied = registry.invoke("fake.deploy_service", {})
    assert denied["denied"] and denied["decision"]["requires_confirmation"]
    assert registry.invoke("fake.deploy_service", {}, confirmed=True)["ok"]
    receipts = [
        json.loads(line)
        for line in (tmp_path / "receipts.jsonl").read_text().splitlines()
    ]
    assert len(receipts) == 4
    assert all(
        receipts[index]["previous_hash"] == receipts[index - 1]["hash"]
        for index in range(1, len(receipts))
    )


def test_openapi_tool_generation_keeps_js_bridge_separate():
    schema = {
        "paths": {
            "/lab/{experiment}": {
                "get": {
                    "operationId": "lab_read",
                    "summary": "Read experiment",
                    "parameters": [
                        {
                            "name": "experiment",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                },
                "post": {
                    "operationId": "lab_run",
                    "summary": "Run experiment",
                    "x-auro-risk-tier": "C3",
                    "requestBody": {
                        "content": {
                            "application/json": {"schema": {"type": "object"}}
                        }
                    },
                },
            }
        }
    }
    adapter = JSServiceAdapter("js_lab", "http://127.0.0.1:9000", schema)
    tools = {tool.name: tool for tool in adapter.list_tools()}
    assert tools["lab_read"].risk_tier == "C1"
    assert tools["lab_run"].risk_tier == "C3"
    assert tools["lab_read"].adapter == "js_lab"


def test_compact_catalog_and_auro_context_injection():
    registry = AuroCapabilityRegistry()
    registry.register(FakeAdapter())
    registry.refresh()
    auro = FakeAuro()
    runtime = AuroBridgeRuntime(auro, registry)
    report = runtime.execute("check service status", tool="fake.read_status")
    assert report["ok"]
    assert report["tool_result"]["ok"]
    assert auro.colony.context.items
    assert report["context"]["tools"][0]["name"] == "fake.read_status"
