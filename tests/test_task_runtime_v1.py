import json
from pathlib import Path
import time
import zipfile

import pytest

from auro_native_llm.production_fleet.task_runtime import (
    DurableTaskRuntime,
    TaskPlanningUnavailable,
    compile_plan,
)


ALL_CAPABILITIES = (
    "research",
    "code",
    "build",
    "test",
    "write",
    "review",
    "artifact-validation",
    "synthesis",
)


def runtime(tmp_path: Path, *, signing: bool = True) -> DurableTaskRuntime:
    return DurableTaskRuntime(
        tmp_path / "tasks.sqlite3",
        tmp_path / "artifacts",
        signing_key="s" * 32 if signing else None,
        signer_id="test-task-runtime",
    )


def artifacts_for(step: dict) -> list[dict]:
    output = []
    for requirement in step["artifact_contract"]:
        name = requirement["name"]
        media_type = requirement["media_type"]
        if media_type == "application/json":
            output.append(
                {
                    "name": name,
                    "media_type": media_type,
                    "json": {"ok": True, "step_id": step["step_id"]},
                }
            )
        else:
            output.append(
                {
                    "name": name,
                    "media_type": media_type,
                    "content": f"# Artifact for {step['step_id']}\n\nValidated output.\n",
                }
            )
    return output


def drain(runtime: DurableTaskRuntime, run_id: str) -> dict:
    for _ in range(100):
        current = runtime.get_run(run_id, principal_id="user-1", organization_id="org-1")
        if current["status"] in {"succeeded", "partial", "failed", "cancelled"}:
            return current
        step = runtime.claim_step(
            run_id,
            worker_id="worker-1",
            capabilities=ALL_CAPABILITIES,
            lease_seconds=300,
        )
        if step is None:
            raise AssertionError(f"run stalled: {json.dumps(current, indent=2)}")
        runtime.complete_step(
            run_id,
            step["step_id"],
            worker_id="worker-1",
            lease_token=step["lease_token"],
            output={"summary": f"completed {step['step_id']}"},
            artifacts=artifacts_for(step),
            validation={"passed": True},
        )
    raise AssertionError("run did not terminate")


def test_compile_multiple_tasks_into_deep_review_and_synthesis_graph():
    plan = compile_plan(
        {
            "objective": "Research, implement, test, and publish a feature.",
            "quality_mode": "deep",
            "tasks": [
                {
                    "step_id": "research",
                    "title": "Research",
                    "objective": "Collect evidence and requirements.",
                    "kind": "research",
                    "required_capabilities": ["research"],
                },
                {
                    "step_id": "implement",
                    "title": "Implementation",
                    "objective": "Implement the approved design.",
                    "kind": "code",
                    "dependencies": ["research"],
                    "required_capabilities": ["code"],
                },
                {
                    "step_id": "test",
                    "title": "Tests",
                    "objective": "Run tests and produce evidence.",
                    "kind": "test",
                    "dependencies": ["implement"],
                    "required_capabilities": ["test"],
                },
            ],
        }
    )
    ids = {step.step_id for step in plan.steps}
    assert {"research", "implement", "test"}.issubset(ids)
    assert {"review:research", "review:implement", "review:test"}.issubset(ids)
    assert "final-synthesis" in ids
    assert len(plan.plan_sha256) == 64
    assert plan.to_dict()["private_chain_of_thought_exported"] is False


def test_cycle_is_rejected_before_run_creation():
    with pytest.raises(ValueError, match="cycle"):
        compile_plan(
            {
                "objective": "Invalid cyclic plan",
                "quality_mode": "fast",
                "tasks": [
                    {"step_id": "a", "objective": "A", "dependencies": ["b"]},
                    {"step_id": "b", "objective": "B", "dependencies": ["a"]},
                ],
            }
        )


def test_multiple_dependent_tasks_execute_and_deliver_hashed_artifacts(tmp_path):
    rt = runtime(tmp_path)
    created = rt.create_run(
        {
            "run_id": "multi-run",
            "objective": "Research and implement a verified module.",
            "quality_mode": "fast",
            "tasks": [
                {
                    "step_id": "research",
                    "objective": "Produce a research note.",
                    "kind": "research",
                    "required_capabilities": ["research"],
                    "artifacts": [
                        {"name": "research.md", "media_type": "text/markdown"}
                    ],
                },
                {
                    "step_id": "code",
                    "objective": "Implement the researched design.",
                    "kind": "code",
                    "dependencies": ["research"],
                    "required_capabilities": ["code"],
                    "artifacts": [
                        {"name": "module.py", "media_type": "text/x-python"}
                    ],
                },
            ],
        },
        principal_id="user-1",
        organization_id="org-1",
    )
    assert created["progress"]["total_steps"] == 2

    first = rt.claim_step(
        "multi-run",
        worker_id="worker-1",
        capabilities=["research"],
    )
    assert first["step_id"] == "research"
    assert rt.claim_step("multi-run", worker_id="worker-2", capabilities=["code"]) is None

    finished_first = rt.complete_step(
        "multi-run",
        "research",
        worker_id="worker-1",
        lease_token=first["lease_token"],
        output={"reasoning_summary": ["Evidence collected and bounded."]},
        artifacts=[
            {
                "name": "research.md",
                "media_type": "text/markdown",
                "content": "# Research\n\nEvidence-backed requirements.\n",
            }
        ],
        validation={"passed": True},
    )
    assert finished_first["status"] == "succeeded"

    second = rt.claim_step(
        "multi-run",
        worker_id="worker-2",
        capabilities=["code"],
    )
    assert second["step_id"] == "code"
    rt.complete_step(
        "multi-run",
        "code",
        worker_id="worker-2",
        lease_token=second["lease_token"],
        output={"summary": "module implemented"},
        artifacts=[
            {
                "name": "module.py",
                "media_type": "text/x-python",
                "content": "def ready():\n    return True\n",
            }
        ],
        validation={"passed": True},
    )

    completed = rt.get_run("multi-run", principal_id="user-1", organization_id="org-1")
    assert completed["status"] == "succeeded"
    assert completed["progress"]["fraction"] == 1.0
    assert len(completed["artifacts"]) == 2
    assert all(len(item["sha256"]) == 64 for item in completed["artifacts"])
    assert completed["result"]["evidence_class"] == "E4-signed-receipt"
    assert rt.verify_event_chain("multi-run")["valid"] is True


def test_exhaustive_run_adds_reviews_synthesis_quality_gate_and_bundle(tmp_path):
    rt = runtime(tmp_path)
    rt.create_run(
        {
            "run_id": "exhaustive-run",
            "objective": "Deliver a research note and working implementation.",
            "quality_mode": "exhaustive",
            "tasks": [
                {
                    "step_id": "research",
                    "title": "Research",
                    "objective": "Research the problem.",
                    "kind": "research",
                    "required_capabilities": ["research"],
                    "artifacts": [{"name": "research.md", "media_type": "text/markdown"}],
                },
                {
                    "step_id": "implementation",
                    "title": "Implementation",
                    "objective": "Implement it.",
                    "kind": "code",
                    "dependencies": ["research"],
                    "required_capabilities": ["code"],
                    "artifacts": [{"name": "implementation.py", "media_type": "text/x-python"}],
                },
            ],
        },
        principal_id="user-1",
        organization_id="org-1",
    )
    completed = drain(rt, "exhaustive-run")
    ids = {step["step_id"] for step in completed["steps"]}
    assert "review:research" in ids
    assert "review:implementation" in ids
    assert "final-synthesis" in ids
    assert "delivery-quality-gate" in ids
    bundle = rt.build_bundle(
        "exhaustive-run",
        principal_id="user-1",
        organization_id="org-1",
    )
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "RUN_RECEIPT.json" in names
    assert "ARTIFACT_MANIFEST.json" in names
    assert any(name.endswith("FINAL_REPORT.md") for name in names)
    assert any(name.endswith("QUALITY_GATE.json") for name in names)


def test_lease_expiry_recovers_after_restart_and_reissues_step(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    root = tmp_path / "artifacts"
    first_runtime = DurableTaskRuntime(db, root)
    first_runtime.create_run(
        {
            "run_id": "lease-run",
            "objective": "Long running task",
            "quality_mode": "fast",
            "tasks": [{"step_id": "long", "objective": "Run for a long time."}],
        },
        principal_id="user-1",
    )
    first = first_runtime.claim_step("lease-run", worker_id="worker-a")
    assert first["step_id"] == "long"
    first_runtime.db.execute(
        "UPDATE task_steps SET lease_expires_at=? WHERE run_id=? AND step_id=?",
        (int(time.time()) - 1, "lease-run", "long"),
    )
    first_runtime.db.commit()
    first_runtime.close()

    second_runtime = DurableTaskRuntime(db, root)
    second = second_runtime.claim_step("lease-run", worker_id="worker-b")
    assert second["step_id"] == "long"
    assert second["attempts"] == 2
    with pytest.raises(PermissionError):
        second_runtime.complete_step(
            "lease-run",
            "long",
            worker_id="worker-a",
            lease_token=first["lease_token"],
        )


def test_retry_then_terminal_failure_is_visible(tmp_path):
    rt = runtime(tmp_path, signing=False)
    rt.create_run(
        {
            "run_id": "retry-run",
            "objective": "Fail visibly",
            "quality_mode": "fast",
            "tasks": [
                {
                    "step_id": "fragile",
                    "objective": "Run a fragile operation.",
                    "max_attempts": 2,
                }
            ],
        },
        principal_id="user-1",
    )
    first = rt.claim_step("retry-run", worker_id="worker")
    retried = rt.fail_step(
        "retry-run",
        "fragile",
        worker_id="worker",
        lease_token=first["lease_token"],
        error="first failure",
        retry_delay_seconds=0,
    )
    assert retried["status"] == "ready"
    second = rt.claim_step("retry-run", worker_id="worker")
    terminal = rt.fail_step(
        "retry-run",
        "fragile",
        worker_id="worker",
        lease_token=second["lease_token"],
        error="second failure",
        retry_delay_seconds=0,
    )
    assert terminal["status"] == "failed"
    run = rt.get_run("retry-run", principal_id="user-1")
    assert run["status"] == "failed"
    assert run["result"]["evidence_class"] == "E3-validated-output"


def test_pause_resume_cancel_and_principal_isolation(tmp_path):
    rt = runtime(tmp_path)
    rt.create_run(
        {
            "run_id": "controlled-run",
            "objective": "Controlled task",
            "quality_mode": "fast",
            "tasks": [{"step_id": "task", "objective": "Do the task."}],
        },
        principal_id="owner",
        organization_id="org-1",
    )
    paused = rt.pause("controlled-run", principal_id="owner", organization_id="org-1")
    assert paused["status"] == "paused"
    assert rt.claim_step("controlled-run", worker_id="worker") is None
    resumed = rt.resume("controlled-run", principal_id="owner", organization_id="org-1")
    assert resumed["status"] == "queued"
    with pytest.raises(PermissionError):
        rt.get_run("controlled-run", principal_id="stranger", organization_id="org-2")
    cancelled = rt.cancel("controlled-run", principal_id="owner", organization_id="org-1")
    assert cancelled["status"] == "cancelled"
    assert all(step["status"] == "cancelled" for step in cancelled["steps"])


def test_high_risk_step_waits_for_trusted_approval_verifier(tmp_path):
    rt = runtime(tmp_path)
    created = rt.create_run(
        {
            "run_id": "approval-run",
            "objective": "Production mutation",
            "quality_mode": "fast",
            "tasks": [
                {
                    "step_id": "deploy",
                    "objective": "Deploy to production.",
                    "kind": "tool",
                    "risk_class": 4,
                    "approval_required": True,
                }
            ],
        },
        principal_id="owner",
    )
    assert created["status"] == "awaiting_approval"
    assert rt.claim_step("approval-run", worker_id="worker") is None
    with pytest.raises(PermissionError, match="approval verifier"):
        rt.record_verified_approval(
            "approval-run",
            "deploy",
            {"approval_id": "approval-1", "receipt_sha256": "a" * 64},
        )


def test_workspace_artifact_path_cannot_escape_step_root(tmp_path):
    rt = runtime(tmp_path)
    rt.create_run(
        {
            "run_id": "artifact-run",
            "objective": "Artifact safety",
            "quality_mode": "fast",
            "tasks": [{"step_id": "write", "objective": "Write file."}],
        },
        principal_id="owner",
    )
    claimed = rt.claim_step("artifact-run", worker_id="worker")
    with pytest.raises((ValueError, FileNotFoundError)):
        rt.complete_step(
            "artifact-run",
            "write",
            worker_id="worker",
            lease_token=claimed["lease_token"],
            artifacts=[{"name": "stolen.txt", "workspace_path": "../../outside.txt"}],
        )


def test_council_planning_request_fails_closed_without_planner():
    with pytest.raises(TaskPlanningUnavailable):
        compile_plan(
            {
                "objective": "Decompose this with the council",
                "plan_with_council": True,
            }
        )
