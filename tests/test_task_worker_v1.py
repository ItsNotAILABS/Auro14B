import json
from pathlib import Path

from auro_native_llm.production_fleet.task_runtime import DurableTaskRuntime
from auro_native_llm.production_fleet.task_worker import (
    CouncilStepExecutor,
    CouncilTaskWorker,
)


class FakeCouncil:
    configured = True

    def __init__(self):
        self.prompts = []

    def respond(self, message):
        prompt = json.loads(message)
        self.prompts.append(prompt)
        step = prompt["step"]
        dependencies = prompt.get("dependency_context") or []
        return {
            "turn_id": f"turn-{len(self.prompts)}",
            "text": f"Completed {step['step_id']} with {len(dependencies)} dependency records.",
            "structured_answer": {
                "answer": f"Completed {step['step_id']} with verified bounded context.",
                "key_points": ["Used only explicit dependency outputs and artifact hashes"],
                "recommendations": ["Preserve validation evidence"],
                "caveats": ["No external tool execution is claimed"],
                "confidence": 0.9,
                "citations": [f"step:{step['step_id']}"],
            },
            "evidence_class": "E3-validated-output",
            "release_evidence_ready": False,
            "blockers": ["test fixture is not a promoted model"],
            "runtime_receipt": {"receipt_sha256": "a" * 64},
        }


def test_council_worker_completes_deep_reasoning_graph_and_artifacts(tmp_path: Path):
    runtime = DurableTaskRuntime(
        tmp_path / "tasks.sqlite3",
        tmp_path / "artifacts",
        signing_key="s" * 32,
    )
    runtime.create_run(
        {
            "run_id": "council-worker-run",
            "objective": "Analyze a system and deliver a reviewed report.",
            "quality_mode": "deep",
            "tasks": [
                {
                    "step_id": "analysis",
                    "title": "System analysis",
                    "objective": "Analyze architecture, risks, and evidence.",
                    "kind": "analysis",
                    "required_capabilities": ["analysis"],
                    "artifacts": [
                        {"name": "ANALYSIS.md", "media_type": "text/markdown"}
                    ],
                }
            ],
        },
        principal_id="user-1",
        organization_id="org-1",
    )
    council = FakeCouncil()
    worker = CouncilTaskWorker(
        runtime,
        CouncilStepExecutor(council),
        run_id="council-worker-run",
        principal_id="user-1",
        organization_id="org-1",
        worker_id="council-worker",
    )
    result = worker.run_until_idle(max_steps=20, idle_rounds=1)
    assert result["final_status"] == "succeeded"

    run = runtime.get_run(
        "council-worker-run",
        principal_id="user-1",
        organization_id="org-1",
    )
    assert run["progress"]["fraction"] == 1.0
    assert {item["step_id"] for item in run["steps"]} == {
        "analysis",
        "review:analysis",
        "final-synthesis",
    }
    names = {item["name"] for item in run["artifacts"]}
    assert "ANALYSIS.md" in names
    assert "analysis-review.json" in names
    assert "FINAL_REPORT.md" in names
    assert "DELIVERY_INDEX.json" in names
    assert all(len(item["sha256"]) == 64 for item in run["artifacts"])
    assert run["result"]["evidence_class"] == "E4-signed-receipt"

    review_prompts = [
        item for item in council.prompts if item["step"]["step_id"] == "review:analysis"
    ]
    assert len(review_prompts) == 1
    assert review_prompts[0]["dependency_context"][0]["step_id"] == "analysis"
    assert review_prompts[0]["dependency_context"][0]["output"]["summary"]
    assert review_prompts[0]["instruction"].startswith("Complete one bounded task")


def test_council_worker_does_not_claim_code_or_execution_tasks(tmp_path: Path):
    runtime = DurableTaskRuntime(tmp_path / "tasks.sqlite3", tmp_path / "artifacts")
    runtime.create_run(
        {
            "run_id": "code-run",
            "objective": "Execute code",
            "quality_mode": "fast",
            "tasks": [
                {
                    "step_id": "code",
                    "objective": "Run a build and tests.",
                    "kind": "code",
                    "required_capabilities": ["code", "test"],
                }
            ],
        },
        principal_id="user-1",
    )
    worker = CouncilTaskWorker(
        runtime,
        CouncilStepExecutor(FakeCouncil()),
        run_id="code-run",
        principal_id="user-1",
    )
    assert worker.run_once() is None
    state = runtime.get_run("code-run", principal_id="user-1")
    assert state["status"] == "queued"
    assert state["steps"][0]["status"] == "ready"
