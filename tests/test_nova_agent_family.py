import json
from pathlib import Path

from auro_native_llm.nova import NOVA_AGENT_FAMILY, NovaAgentFamily, model_agent_taxonomy
from auro_native_llm.production_fleet.browser_gateway import BrowserTaskBroker
from auro_native_llm.production_fleet.office import NativeOffice


class Caps:
    def __init__(self, root):
        self.browser = BrowserTaskBroker(root / "tasks.json")
        self.office = NativeOffice()

    def call(self, name, arguments, approved=False):
        if name == "browser.task.enqueue":
            return {"ok": True, "output": self.browser.enqueue(arguments["kind"], arguments["payload"]), "receipt": "browser"}
        if name == "office.create_bundle":
            if not approved:
                return {"ok": False, "denied": True}
            return {"ok": True, "output": self.office.create_bundle(arguments["out_dir"], arguments["title"], arguments["sections"], arguments.get("table") or []), "receipt": "office"}
        if name == "compute.engines":
            return {"ok": True, "output": {"embedded": ["MESIE"]}, "receipt": "engine"}
        return {"ok": False, "denied": True}


def generator(messages, options):
    system = messages[0]["content"]
    if "NOVA Hermes" in system:
        action = {"name": "browser.task.enqueue", "arguments": {"kind": "research", "payload": {"objective": "verify"}}, "reason": "test"}
    elif "NOVA Publisher" in system:
        action = {"name": "office.create_bundle", "arguments": {"out_dir": "unused", "title": "Trial", "sections": [{"heading": "Result", "body": "passed"}], "table": [["status"], ["passed"]]}, "reason": "test"}
    else:
        action = {"name": "compute.engines", "arguments": {}, "reason": "test"}
    return {"text": json.dumps({"answer": "task complete", "actions": [action]})}


def test_taxonomy_separates_models_agents_tools_and_engines():
    taxonomy = model_agent_taxonomy([{"id": "m1", "parameter_count": 10}])
    assert taxonomy["models"][0]["parameter_count"] == 10
    assert len(taxonomy["nova_agents"]) == len(NOVA_AGENT_FAMILY)
    assert "Agent count never contributes" in taxonomy["accounting_laws"][0]


def test_hermes_executes_browser_task_with_model_identity(tmp_path):
    family = NovaAgentFamily(generator, {"id": "m1", "model": "AURO", "provider": "local"}, Caps(tmp_path))
    result = family.run("nova.hermes", "research", execute=True)
    assert result.ok and len(result.receipt_hash) == 64
    assert result.model_lane["id"] == "m1"
    assert result.executions[0]["ok"]


def test_publisher_requires_approval_and_creates_all_formats(tmp_path):
    caps = Caps(tmp_path)
    def publish(messages, options):
        return {"text": json.dumps({"answer": "publish", "actions": [{"name": "office.create_bundle", "arguments": {"out_dir": str(tmp_path / "out"), "title": "NEXUS", "sections": [{"heading": "Status", "body": "ready"}], "table": [["lane", "status"], ["NOVA", "passed"]]}, "reason": "deliver"}]})}
    family = NovaAgentFamily(publish, {"id": "m1", "model": "AURO", "provider": "local"}, caps)
    denied = family.run("nova.publisher", "publish", execute=True)
    assert denied.executions[0]["denied"]
    allowed = family.run("nova.publisher", "publish", execute=True, approved_capabilities={"office.create_bundle"})
    files = allowed.executions[0]["output"]["files"]
    assert {Path(x["path"]).suffix for x in files} == {".md", ".csv", ".docx", ".xlsx", ".pdf"}
