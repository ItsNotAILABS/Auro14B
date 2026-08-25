from pathlib import Path

from auro_native_llm.work.harness import HarnessStore, IndependentHarnessFabric
from auro_native_llm.work.harness_orchestrator import FanoutPlan, HarnessOrchestrator
from auro_native_llm.work.skill_forge import HarnessSkillForge


def _executor(state, task):
    return {"ok": True, "summary": f"done:{task.objective}", "harness": state.id}


def test_harness_state_survives_store_reload(tmp_path: Path):
    fabric = IndependentHarnessFabric(HarnessStore(tmp_path), executor=_executor)
    state = fabric.create_harness("multi day objective")
    result = fabric.run_once(state.id, worker_id="w1")
    assert result["ok"] is True
    loaded = HarnessStore(tmp_path).load(state.id)
    assert loaded.state == "completed"
    assert loaded.completed_tasks == 1


def test_fanout_creates_independent_child_directories(tmp_path: Path):
    fabric = IndependentHarnessFabric(HarnessStore(tmp_path), executor=_executor)
    parent = fabric.create_harness("parent")
    children = fabric.fan_out(parent.id, ["research", "implement", "review"])
    assert len(children) == 3
    assert len({child.id for child in children}) == 3
    assert all((tmp_path / child.id / "state.json").exists() for child in children)
    assert all(child.parent_id == parent.id for child in children)


def test_orchestrator_rejoins_children_and_distills_skill(tmp_path: Path):
    fabric = IndependentHarnessFabric(HarnessStore(tmp_path), executor=_executor)
    orchestrator = HarnessOrchestrator(fabric)
    parent = fabric.create_harness("build a durable subsystem")
    live = fabric.store.load(parent.id)
    live.tasks.clear()
    fabric.store.save(live)
    plan = FanoutPlan(
        objective=parent.objective,
        subproblems=[
            {"objective": "implement core", "role": "coder", "completion_criteria": "works"},
            {"objective": "review core", "role": "reviewer", "completion_criteria": "reviewed"},
        ],
        planner_text="test",
    )
    children = orchestrator.fan_out_plan(parent.id, plan)
    join = fabric.add_task(parent.id, "join")
    live = fabric.store.load(parent.id)
    live.tasks[join.id].state = "waiting_children"
    fabric.store.save(live)
    result = orchestrator.advance_tree(parent.id, worker_id="test", cycles_per_child=4)
    assert result["parent"]["state"] == "completed"
    assert all(fabric.store.load(child.id).state == "completed" for child in children)
    forge = HarnessSkillForge(fabric.store)
    assert forge.select(parent.objective, limit=5)


def test_pause_resume_and_lease_exclusion(tmp_path: Path):
    fabric = IndependentHarnessFabric(HarnessStore(tmp_path), executor=_executor)
    state = fabric.create_harness("objective")
    assert fabric.pause(state.id).state == "paused"
    assert fabric.resume(state.id).state == "active"
    lease = fabric.store.acquire_lease(state.id, "worker-a", 120)
    try:
        try:
            fabric.store.acquire_lease(state.id, "worker-b", 120)
            assert False, "second worker should not acquire active lease"
        except RuntimeError:
            pass
    finally:
        fabric.store.release_lease(lease)
