import hashlib

from auro_native_llm.production_fleet.task_service import MissionService
from auro_native_llm.tasks import (
    ArtifactStore,
    CouncilTaskExecutor,
    MissionOrchestrator,
    MissionPlanner,
    MissionStore,
    PackageTaskExecutor,
)


class FakeCouncil:
    configured = True

    def __init__(self):
        self.calls = []

    def respond(self, message, full_parent_context=None):
        self.calls.append((message, full_parent_context))
        digest = hashlib.sha256(message.encode()).hexdigest()
        return {
            "schema": "auro.2b-council.turn.v1",
            "text": "A bounded result was produced. Architecture is not trained capability.",
            "structured_answer": {
                "answer": "A bounded result was produced.",
                "key_points": ["Architecture is not trained capability"],
                "confidence": 0.9,
            },
            "consensus_votes": [
                {"consensus": "Use the bounded result.", "confidence": 0.9},
                {"consensus": "Preserve evidence boundaries.", "confidence": 0.8},
            ],
            "runtime_receipt": {"receipt_sha256": digest},
            "mesie_receipts": [{"receipt_sha256": digest[::-1]}],
            "atomic_agent_count": 9,
            "model_backed_atomic_count": 9,
            "evidence_class": "E4-signed-receipt",
            "release_evidence_ready": True,
            "blockers": [],
        }


def build(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    council = FakeCouncil()
    council_exec = CouncilTaskExecutor(council, artifacts)
    package_exec = PackageTaskExecutor(artifacts)
    orchestrator = MissionOrchestrator(
        store,
        artifacts,
        {
            "reasoning": council_exec,
            "research": council_exec,
            "implementation": council_exec,
            "artifact": council_exec,
            "review": council_exec,
            "synthesis": council_exec,
            "package": package_exec,
            "default": council_exec,
        },
        planner=MissionPlanner(),
    )
    return store, artifacts, council, orchestrator


def test_default_mission_runs_dependency_graph_and_delivers_bundle(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    mission = orchestrator.create(
        objective="Research, implement, review, and package a product proposal.",
        title="Product proposal",
        operator_id="alfredo",
        organization_id="itsnotai",
        max_parallel=3,
    )
    assert len(mission["tasks"]) == 7
    assert mission["status"] == "queued"

    result = orchestrator.run_burst(
        mission["mission_id"],
        worker_id="test-worker",
        max_tasks=20,
        time_budget_seconds=30,
    )
    snapshot = result["mission"]
    assert snapshot["status"] == "completed"
    assert snapshot["progress"]["completed"] == 7
    assert snapshot["artifact_manifest_sha256"]
    paths = {item["relative_path"] for item in snapshot["artifacts"]}
    assert "FINAL_RESULT.md" in paths
    assert "ARTIFACT_MANIFEST.json" in paths
    assert "mission-artifacts.zip" in paths
    assert len(snapshot["decisions"]) == 7
    assert all(item["confidence"] >= 0 for item in snapshot["decisions"])
    assert len(council.calls) >= 6
    assert any(event["event_type"] == "task_heartbeat" for event in snapshot["events"])


def test_parallel_tasks_do_not_run_before_dependency(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    mission = orchestrator.create(
        objective="Do a multi-step task.",
        tasks=[
            {"task_id": "a", "kind": "reasoning", "objective": "first"},
            {"task_id": "b", "kind": "reasoning", "objective": "second", "depends_on": ["a"]},
        ],
        operator_id="alfredo",
        organization_id="itsnotai",
    )
    leased = store.lease_ready_task("worker", mission_id=mission["mission_id"])
    assert leased["task_id"] == "a"
    assert store.lease_ready_task("other", mission_id=mission["mission_id"]) is None


def test_cycle_is_rejected(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    try:
        orchestrator.create(
            objective="bad",
            tasks=[
                {"task_id": "a", "depends_on": ["b"]},
                {"task_id": "b", "depends_on": ["a"]},
            ],
            operator_id="alfredo",
            organization_id="itsnotai",
        )
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("cycle should fail")


def test_tenant_service_rejects_cross_org_access(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    service = MissionService(store=store, artifacts=artifacts, orchestrator=orchestrator)
    mission = service.create(
        {"objective": "tenant work"},
        operator_id="alfredo",
        organization_id="itsnotai",
    )
    assert service.get(
        mission["mission_id"], operator_id="alfredo", organization_id="itsnotai"
    )["mission_id"] == mission["mission_id"]
    try:
        service.get(
            mission["mission_id"], operator_id="alfredo", organization_id="other"
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("cross-org read should fail")


def test_pause_resume_and_cancel(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    mission = orchestrator.create(
        objective="pause me",
        operator_id="alfredo",
        organization_id="itsnotai",
    )
    paused = store.pause(mission["mission_id"])
    assert paused["status"] == "paused"
    resumed = store.resume(mission["mission_id"])
    assert resumed["status"] == "queued"
    cancelled = store.cancel(mission["mission_id"])
    assert cancelled["status"] == "cancelled"


def test_idempotency_returns_existing_mission(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    first = orchestrator.create(
        objective="idempotent",
        operator_id="alfredo",
        organization_id="itsnotai",
        idempotency_key="stable-request-1",
    )
    second = orchestrator.create(
        objective="idempotent",
        operator_id="alfredo",
        organization_id="itsnotai",
        idempotency_key="stable-request-1",
    )
    assert first["mission_id"] == second["mission_id"]


def test_expired_lease_is_recovered_and_requeued(tmp_path):
    store, artifacts, council, orchestrator = build(tmp_path)
    mission = orchestrator.create(
        objective="recover",
        tasks=[{"task_id": "a", "kind": "reasoning", "objective": "recover"}],
        operator_id="alfredo",
        organization_id="itsnotai",
    )
    leased = store.lease_ready_task("worker-a", mission_id=mission["mission_id"], lease_seconds=30)
    assert leased["status"] == "running"
    recovered = store.recover_expired(int(leased["lease_expires_at_unix"]) + 1)
    assert recovered == 1
    assert store.get_task("a")["status"] == "queued"


def test_artifact_path_traversal_is_rejected(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    try:
        artifacts.write_text("mission", "../escape.txt", "no")
    except ValueError as exc:
        assert "traverse" in str(exc)
    else:
        raise AssertionError("path traversal should fail")
