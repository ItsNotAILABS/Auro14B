"""Run model-alone and NOVA-agent capability trials with sealed receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from auro_native_llm.nova import NovaAgentFamily, model_agent_taxonomy
from auro_native_llm.production_fleet.browser_gateway import BrowserTaskBroker
from auro_native_llm.production_fleet.office import NativeOffice
from auro_native_llm.production_fleet.runtime import ModelEndpoint, NativeOpenWeightGenerator, OpenAICompatibleGenerator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "nexus-nova-trials"


class TrialCapabilities:
    """Exercise real repository subsystems with temporary state and no unsafe shell lane."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.browser = BrowserTaskBroker(root / "browser-tasks.json")
        self.office = NativeOffice()

    def call(self, name: str, arguments: dict[str, Any], approved: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        if name == "browser.task.enqueue":
            output = self.browser.enqueue(arguments["kind"], arguments["payload"])
        elif name == "browser.task.status":
            output = self.browser.get(arguments["task_id"])
        elif name == "browser.tasks.list":
            output = {"tasks": self.browser.list(int(arguments.get("limit", 50)))}
        elif name == "office.create_bundle":
            if not approved:
                output = {"ok": False, "denied": True, "reason": "explicit approval required"}
                return self._seal(name, False, output, started)
            output = self.office.create_bundle(
                arguments["out_dir"], arguments["title"], arguments["sections"], arguments.get("table") or []
            )
        elif name == "compute.engines":
            output = {"embedded": ["MESIE", "MatDaemon", "HIMBrain"], "mode": "repository-contract"}
        elif name == "brain.cycle":
            output = {"observation": arguments["observation"], "importance": arguments.get("importance", 0.5), "execute_requested": arguments.get("execute_requested", False)}
        elif name.startswith("skill."):
            output = {"objective": arguments.get("objective", ""), "playbook_loaded": True}
        else:
            output = {"ok": False, "denied": True, "reason": "trial capability not enabled"}
            return self._seal(name, False, output, started)
        return self._seal(name, True, output, started)

    @staticmethod
    def _seal(name: str, ok: bool, output: Any, started: float) -> dict[str, Any]:
        body = {"ok": ok, "capability": name, "output": output, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        body["receipt_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        return body


def deterministic_generator(messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
    objective = messages[-1]["content"]
    system = messages[0]["content"]
    if "NOVA Hermes" in system:
        actions = [{"name": "browser.task.enqueue", "arguments": {"kind": "research", "payload": {"objective": objective}}, "reason": "queue governed browser research"}]
    elif "NOVA Publisher" in system:
        actions = [{"name": "office.create_bundle", "arguments": {"out_dir": str(OUT / "deliverables"), "title": "NEXUS NOVA Trial Report", "sections": [{"heading": "Objective", "body": objective}, {"heading": "Result", "body": "NOVA Publisher created a multi-format deliverable bundle through the governed office capability."}], "table": [["lane", "status"], ["publisher", "passed"]]}, "reason": "create the required deliverable bundle"}]
    elif "NOVA Engineer" in system:
        actions = [{"name": "compute.engines", "arguments": {}, "reason": "inspect engine plane"}]
    else:
        actions = []
    return {"text": json.dumps({"answer": f"Completed bounded analysis for: {objective}", "actions": actions})}


def load_generators() -> list[tuple[str, Any, dict[str, Any]]]:
    lanes: list[tuple[str, Any, dict[str, Any]]] = []
    checkpoint = os.getenv("AURO_NATIVE_CHECKPOINT", str(ROOT / "checkpoints" / "open" / "HIM-native-v0"))
    if Path(checkpoint).exists():
        generator = NativeOpenWeightGenerator(checkpoint)
        lanes.append(("auro-native", generator, {"id": "auro-native", "model": "HIM-native-v0", "provider": "repository-native-open-weights", "checkpoint": checkpoint}))
    external_url = os.getenv("NOVA_COMPARE_BASE_URL", "").strip()
    external_model = os.getenv("NOVA_COMPARE_MODEL", "").strip()
    if external_url and external_model:
        endpoint = ModelEndpoint("comparison-local", external_url, external_model, None, "comparison", os.getenv("NOVA_COMPARE_API_KEY_ENV") or None)
        lanes.append(("comparison-local", OpenAICompatibleGenerator(endpoint), {"id": endpoint.id, "model": endpoint.model, "provider": "external-local-endpoint", "base_url": endpoint.base_url}))
    lanes.append(("contract-control", deterministic_generator, {"id": "contract-control", "model": "deterministic-contract-control", "provider": "test-control"}))
    return lanes


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(prefix="nexus-nova-") as tmp:
        capabilities = TrialCapabilities(Path(tmp))
        for lane_id, generator, model_lane in load_generators():
            model_prompt = "Produce a concise operational plan for a browser research task, an engine inspection, and a publication deliverable."
            started = time.perf_counter()
            try:
                raw = generator([{"role": "user", "content": model_prompt}], {"temperature": 0.2, "max_tokens": 256})
                model_alone = {"ok": bool(str(raw.get("text", "")).strip()), "text": str(raw.get("text", "")), "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
            except Exception as exc:
                model_alone = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
            family = NovaAgentFamily(generator, model_lane, capabilities)
            agent_runs = []
            for agent_id, objective, approvals in (
                ("nova.hermes", "Research the configured NEXUS capability surface and preserve a browser-task receipt.", set()),
                ("nova.engineer", "Inspect the available engine plane and report MESIE and compute readiness.", set()),
                ("nova.publisher", "Create a publishable NEXUS trial bundle in all supported formats.", {"office.create_bundle"}),
            ):
                try:
                    agent_runs.append(family.run(agent_id, objective, execute=True, approved_capabilities=approvals).public())
                except Exception as exc:
                    agent_runs.append({"agent_id": agent_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            trials.append({"lane": model_lane, "model_alone": model_alone, "nova_agents": agent_runs, "family": family.manifest()})
    report = {
        "schema": "nexus.nova.capability-trials.v1",
        "generated_at": time.time(),
        "taxonomy": model_agent_taxonomy([x[2] for x in load_generators()]),
        "trials": trials,
        "comparison_contract": {
            "lanes": ["model-alone", "NOVA-over-AURO", "NOVA-over-external-local-model"],
            "external_lane_status": "configured" if os.getenv("NOVA_COMPARE_BASE_URL") else "not configured; no result claimed",
            "tasks": ["browser", "web-worker", "engine", "deliverable"],
            "rule": "A pass requires a usable answer plus capability and artifact receipts where execution is requested.",
        },
    }
    report["receipt_hash"] = hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()
    (OUT / "trial-results.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT / "MODEL_AGENT_TAXONOMY.json").write_text(json.dumps(report["taxonomy"], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    control = next(x for x in trials if x["lane"]["id"] == "contract-control")
    return report if control["model_alone"]["ok"] and all(x.get("ok") for x in control["nova_agents"]) else {}


if __name__ == "__main__":
    raise SystemExit(0 if run() else 2)
