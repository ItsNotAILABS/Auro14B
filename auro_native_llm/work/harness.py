"""Durable independent work harness fabric for AURO/NOVA.

A harness is a resumable work organism, not a transient sub-agent. It owns its
objective, task graph, state directory, event journal, memory notes, child
harnesses, leases, and execution receipts. Harnesses may recursively fan out
independent child harnesses and rejoin their results.

The implementation is process-safe through atomic JSON replacement and explicit
leases. It intentionally does not promise background execution by itself: a
worker must call ``run_once`` / ``run_until_blocked`` or the HTTP API. The state
survives process restarts so work can continue over hours or days.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import json
import os
import tempfile
import time
import uuid


TERMINAL_TASK_STATES = {"completed", "failed", "cancelled"}
HARNESS_TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _now() -> float:
    return time.time()


def _sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class HarnessTask:
    id: str
    objective: str
    state: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    assigned_agent: str | None = None
    child_harness_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def ready(self, tasks: dict[str, "HarnessTask"]) -> bool:
        if self.state not in {"pending", "retry"}:
            return False
        return all(tasks.get(dep) is not None and tasks[dep].state == "completed" for dep in self.depends_on)


@dataclass
class HarnessState:
    id: str
    objective: str
    state: str = "active"
    parent_id: str | None = None
    root_id: str | None = None
    depth: int = 0
    model_id: str = "Auro-2B"
    agent_roster: list[str] = field(default_factory=lambda: ["planner", "researcher", "coder", "reviewer"])
    memory_notes: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    tasks: dict[str, HarnessTask] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    last_heartbeat: float = 0.0
    cycles: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    final_summary: str = ""
    version: str = "3.0"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tasks"] = {task_id: asdict(task) for task_id, task in self.tasks.items()}
        value["schema"] = "auro.independent-harness.state.v3"
        return value

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HarnessState":
        tasks = {k: HarnessTask(**v) for k, v in dict(raw.get("tasks") or {}).items()}
        allowed = {name for name in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in allowed and k != "tasks"}
        return cls(**kwargs, tasks=tasks)


@dataclass(frozen=True)
class HarnessLease:
    harness_id: str
    worker_id: str
    acquired_at: float
    expires_at: float
    nonce: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schema": "auro.independent-harness.lease.v1"}


class HarnessStore:
    """Filesystem persistence for resumable harness instances."""

    def __init__(self, root: str | Path = "state/harnesses") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, harness_id: str) -> Path:
        return self.root / harness_id

    def state_path(self, harness_id: str) -> Path:
        return self.directory(harness_id) / "state.json"

    def lease_path(self, harness_id: str) -> Path:
        return self.directory(harness_id) / "lease.json"

    def journal_path(self, harness_id: str) -> Path:
        return self.directory(harness_id) / "events.jsonl"

    def create(self, state: HarnessState) -> HarnessState:
        path = self.state_path(state.id)
        if path.exists():
            raise FileExistsError(state.id)
        self.save(state)
        self.append_event(state.id, "harness_created", {"objective": state.objective, "parent_id": state.parent_id})
        return state

    def save(self, state: HarnessState) -> None:
        state.updated_at = _now()
        payload = state.to_dict()
        payload["state_sha256"] = _sha(payload)
        _atomic_json(self.state_path(state.id), payload)

    def load(self, harness_id: str) -> HarnessState:
        raw = json.loads(self.state_path(harness_id).read_text(encoding="utf-8"))
        return HarnessState.from_dict(raw)

    def exists(self, harness_id: str) -> bool:
        return self.state_path(harness_id).exists()

    def list(self) -> list[HarnessState]:
        states = []
        for path in sorted(self.root.glob("*/state.json")):
            try:
                states.append(HarnessState.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return sorted(states, key=lambda x: x.updated_at, reverse=True)

    def append_event(self, harness_id: str, event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self.journal_path(harness_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous_hash = "0" * 64
        if path.exists():
            try:
                last = path.read_text(encoding="utf-8").splitlines()[-1]
                previous_hash = json.loads(last).get("event_hash", previous_hash)
            except Exception:
                pass
        record = {
            "schema": "auro.independent-harness.event.v1",
            "at": _now(),
            "event": event,
            "payload": payload or {},
            "previous_hash": previous_hash,
        }
        record["event_hash"] = _sha(record)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def acquire_lease(self, harness_id: str, worker_id: str, ttl_seconds: int = 300) -> HarnessLease:
        now = _now()
        path = self.lease_path(harness_id)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if float(raw.get("expires_at", 0)) > now and raw.get("worker_id") != worker_id:
                    raise RuntimeError(f"harness {harness_id} leased by {raw.get('worker_id')}")
            except json.JSONDecodeError:
                pass
        lease = HarnessLease(harness_id, worker_id, now, now + max(30, int(ttl_seconds)), uuid.uuid4().hex)
        _atomic_json(path, lease.to_dict())
        self.append_event(harness_id, "lease_acquired", {"worker_id": worker_id, "expires_at": lease.expires_at})
        return lease

    def release_lease(self, lease: HarnessLease) -> None:
        path = self.lease_path(lease.harness_id)
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("worker_id") == lease.worker_id and raw.get("nonce") == lease.nonce:
            path.unlink(missing_ok=True)
            self.append_event(lease.harness_id, "lease_released", {"worker_id": lease.worker_id})


TaskExecutor = Callable[[HarnessState, HarnessTask], dict[str, Any]]


class IndependentHarnessFabric:
    """Create, resume, fan out, and rejoin durable independent harnesses."""

    def __init__(self, store: HarnessStore | None = None, executor: TaskExecutor | None = None) -> None:
        self.store = store or HarnessStore(os.getenv("AURO_HARNESS_ROOT", "state/harnesses"))
        self.executor = executor or self._default_executor
        self.max_depth = max(1, int(os.getenv("AURO_HARNESS_MAX_DEPTH", "4")))
        self.max_children = max(1, int(os.getenv("AURO_HARNESS_MAX_CHILDREN", "32")))

    def create_harness(
        self,
        objective: str,
        *,
        parent_id: str | None = None,
        model_id: str = "Auro-2B",
        agent_roster: Iterable[str] | None = None,
        tasks: Iterable[dict[str, Any] | str] | None = None,
    ) -> HarnessState:
        if not objective.strip():
            raise ValueError("objective is required")
        parent = self.store.load(parent_id) if parent_id else None
        depth = (parent.depth + 1) if parent else 0
        if depth > self.max_depth:
            raise RuntimeError("harness max depth exceeded")
        harness_id = "h_" + uuid.uuid4().hex
        root_id = parent.root_id or parent.id if parent else harness_id
        state = HarnessState(
            id=harness_id,
            objective=objective.strip(),
            parent_id=parent_id,
            root_id=root_id,
            depth=depth,
            model_id=model_id,
            agent_roster=list(agent_roster or ["planner", "researcher", "coder", "reviewer"]),
        )
        if tasks:
            for item in tasks:
                if isinstance(item, str):
                    self._add_task_to_state(state, item)
                else:
                    self._add_task_to_state(state, str(item["objective"]), depends_on=list(item.get("depends_on") or []), max_attempts=int(item.get("max_attempts", 3)))
        else:
            self._add_task_to_state(state, objective.strip())
        self.store.create(state)
        if parent:
            parent.child_ids.append(state.id)
            self.store.save(parent)
            self.store.append_event(parent.id, "child_created", {"child_id": state.id, "objective": state.objective})
        return state

    def _add_task_to_state(self, state: HarnessState, objective: str, *, depends_on: list[str] | None = None, max_attempts: int = 3) -> HarnessTask:
        task = HarnessTask("t_" + uuid.uuid4().hex, objective.strip(), depends_on=list(depends_on or []), max_attempts=max(1, int(max_attempts)))
        state.tasks[task.id] = task
        return task

    def add_task(self, harness_id: str, objective: str, *, depends_on: list[str] | None = None, max_attempts: int = 3) -> HarnessTask:
        state = self.store.load(harness_id)
        task = self._add_task_to_state(state, objective, depends_on=depends_on, max_attempts=max_attempts)
        self.store.save(state)
        self.store.append_event(harness_id, "task_added", {"task_id": task.id, "objective": task.objective, "depends_on": task.depends_on})
        return task

    def fan_out(self, harness_id: str, subproblems: Iterable[str], *, model_id: str | None = None) -> list[HarnessState]:
        parent = self.store.load(harness_id)
        objectives = [x.strip() for x in subproblems if x and x.strip()]
        if len(parent.child_ids) + len(objectives) > self.max_children:
            raise RuntimeError("harness child limit exceeded")
        children = []
        for objective in objectives:
            child = self.create_harness(objective, parent_id=harness_id, model_id=model_id or parent.model_id, agent_roster=parent.agent_roster)
            children.append(child)
        return children

    def pause(self, harness_id: str) -> HarnessState:
        state = self.store.load(harness_id)
        if state.state not in HARNESS_TERMINAL_STATES:
            state.state = "paused"
            self.store.save(state)
            self.store.append_event(harness_id, "harness_paused")
        return state

    def resume(self, harness_id: str) -> HarnessState:
        state = self.store.load(harness_id)
        if state.state == "paused":
            state.state = "active"
            self.store.save(state)
            self.store.append_event(harness_id, "harness_resumed")
        return state

    def cancel(self, harness_id: str) -> HarnessState:
        state = self.store.load(harness_id)
        state.state = "cancelled"
        self.store.save(state)
        self.store.append_event(harness_id, "harness_cancelled")
        return state

    def _ready_tasks(self, state: HarnessState) -> list[HarnessTask]:
        return [task for task in state.tasks.values() if task.ready(state.tasks)]

    def _refresh_terminal_state(self, state: HarnessState) -> None:
        states = [task.state for task in state.tasks.values()]
        state.completed_tasks = sum(s == "completed" for s in states)
        state.failed_tasks = sum(s == "failed" for s in states)
        if states and all(s in TERMINAL_TASK_STATES for s in states):
            state.state = "failed" if any(s == "failed" for s in states) else "completed"
            state.final_summary = self.aggregate(state.id)["summary"]

    def run_once(self, harness_id: str, *, worker_id: str = "local", lease_seconds: int = 300) -> dict[str, Any]:
        lease = self.store.acquire_lease(harness_id, worker_id, lease_seconds)
        try:
            state = self.store.load(harness_id)
            if state.state != "active":
                return {"ok": True, "harness": state.to_dict(), "executed": None, "reason": f"state={state.state}"}
            ready = self._ready_tasks(state)
            if not ready:
                self._refresh_terminal_state(state)
                self.store.save(state)
                return {"ok": True, "harness": state.to_dict(), "executed": None, "reason": "no_ready_task"}
            task = ready[0]
            task.state = "running"
            task.attempts += 1
            task.updated_at = _now()
            state.cycles += 1
            state.last_heartbeat = _now()
            self.store.save(state)
            self.store.append_event(harness_id, "task_started", {"task_id": task.id, "attempt": task.attempts, "worker_id": worker_id})
            try:
                result = self.executor(state, task)
                task.result = result
                task.state = "completed" if result.get("ok", False) else ("retry" if task.attempts < task.max_attempts else "failed")
                task.error = None if result.get("ok", False) else str(result.get("error") or "task_failed")[:1000]
            except Exception as exc:
                task.error = f"{type(exc).__name__}: {exc}"[:1000]
                task.state = "retry" if task.attempts < task.max_attempts else "failed"
            task.updated_at = _now()
            self._refresh_terminal_state(state)
            self.store.save(state)
            self.store.append_event(harness_id, "task_finished", {"task_id": task.id, "state": task.state, "attempts": task.attempts, "result_sha256": _sha(task.result or {"error": task.error})})
            return {"ok": task.state == "completed", "harness": state.to_dict(), "executed": asdict(task)}
        finally:
            self.store.release_lease(lease)

    def run_until_blocked(self, harness_id: str, *, worker_id: str = "local", max_cycles: int = 32) -> dict[str, Any]:
        cycles = []
        for _ in range(max(1, int(max_cycles))):
            result = self.run_once(harness_id, worker_id=worker_id)
            cycles.append(result)
            state = self.store.load(harness_id)
            if state.state != "active" or result.get("executed") is None:
                break
        return {"schema": "auro.independent-harness.run.v3", "harness_id": harness_id, "cycles": cycles, "state": self.store.load(harness_id).to_dict()}

    def aggregate(self, harness_id: str) -> dict[str, Any]:
        state = self.store.load(harness_id)
        child_results = []
        for child_id in state.child_ids:
            if self.store.exists(child_id):
                child = self.store.load(child_id)
                child_results.append({"id": child.id, "state": child.state, "summary": child.final_summary, "completed_tasks": child.completed_tasks, "failed_tasks": child.failed_tasks})
        local = [
            {"task_id": task.id, "objective": task.objective, "state": task.state, "result": task.result, "error": task.error}
            for task in state.tasks.values()
        ]
        summary = f"{state.completed_tasks}/{len(state.tasks)} local tasks completed; {sum(c['state'] == 'completed' for c in child_results)}/{len(child_results)} child harnesses completed."
        return {"schema": "auro.independent-harness.aggregate.v3", "harness_id": state.id, "summary": summary, "local_tasks": local, "children": child_results}

    def _default_executor(self, state: HarnessState, task: HarnessTask) -> dict[str, Any]:
        from auro_native_llm.work.agent import WorkAgent
        agent = WorkAgent(model_id=state.model_id, lite=os.getenv("AURO_HARNESS_FULL_MODEL", "0") != "1", use_scripture=True, max_tool_steps=max(1, int(os.getenv("AURO_HARNESS_TOOL_STEPS", "8"))))
        result = agent.run(task.objective)
        return result.to_dict()

    def manifest(self) -> dict[str, Any]:
        states = self.store.list()
        return {
            "schema": "auro.independent-harness.fabric.v3",
            "persistent": True,
            "recursive_fanout": True,
            "lease_protected": True,
            "resume_across_restart": True,
            "background_execution_claim": False,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
            "harness_count": len(states),
            "active": sum(s.state == "active" for s in states),
            "paused": sum(s.state == "paused" for s in states),
            "completed": sum(s.state == "completed" for s in states),
        }


__all__ = ["HarnessTask", "HarnessState", "HarnessLease", "HarnessStore", "IndependentHarnessFabric"]
