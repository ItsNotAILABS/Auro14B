"""Durable multi-task orchestration for AURO production work.

The orchestrator compiles one or many objectives into a bounded dependency
DAG, runs independent steps concurrently, persists every transition in
SQLite/WAL, survives process restart, and emits content-addressed artifacts.

Reasoning is represented as concise decision summaries and verification
records. Private chain-of-thought, scratchpads, and hidden model reasoning are
never stored or returned.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid
import zipfile


RUN_SCHEMA = "auro.task-run.v1"
PLAN_SCHEMA = "auro.task-plan.v1"
STEP_SCHEMA = "auro.task-step.v1"
ARTIFACT_SCHEMA = "auro.task-artifact.v1"
EVENT_SCHEMA = "auro.task-event.v1"
DELIVERY_SCHEMA = "auro.task-delivery.v1"

RUN_TERMINAL = {"succeeded", "failed", "cancelled"}
STEP_TERMINAL = {"succeeded", "failed", "cancelled", "skipped"}
REASONING_PASSES = {
    "standard": ("solve",),
    "deep": ("analyze", "critique", "refine"),
    "research": ("frame", "investigate", "red_team", "synthesize"),
}
HIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "chain-of-thought",
    "private_reasoning",
    "hidden_reasoning",
    "scratchpad",
    "internal_monologue",
}


def now_ms() -> int:
    return int(time.time() * 1000)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_name(value: str, fallback: str = "item") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return (normalized or fallback)[:120]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return default
    return parsed


def scrub_private_reasoning(value: Any) -> tuple[Any, list[str]]:
    """Remove private-reasoning fields while preserving useful summaries."""
    removed: list[str] = []

    def walk(item: Any, prefix: str = "") -> Any:
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                path = f"{prefix}.{key}" if prefix else key
                if key.lower() in HIDDEN_REASONING_KEYS:
                    removed.append(path)
                    continue
                output[key] = walk(raw_value, path)
            return output
        if isinstance(item, list):
            return [walk(child, f"{prefix}[{index}]") for index, child in enumerate(item)]
        if isinstance(item, tuple):
            return [walk(child, f"{prefix}[{index}]") for index, child in enumerate(item)]
        return item

    return walk(value), removed


@dataclass(frozen=True)
class TaskScope:
    organization_id: str
    workspace_id: str
    operator_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskScope":
        fields = {
            "organization_id": str(value.get("organization_id") or "").strip(),
            "workspace_id": str(value.get("workspace_id") or "").strip(),
            "operator_id": str(value.get("operator_id") or "").strip(),
        }
        missing = [key for key, item in fields.items() if not item]
        if missing:
            raise ValueError("task scope requires " + ", ".join(missing))
        if any(len(item) > 160 for item in fields.values()):
            raise ValueError("task scope values must be 160 characters or fewer")
        return cls(**fields)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TaskBudget:
    max_steps: int = 96
    max_parallel_steps: int = 6
    max_attempts_per_step: int = 3
    max_runtime_seconds: int = 86_400
    max_artifact_bytes: int = 256 * 1024 * 1024
    max_model_calls: int = 256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TaskBudget":
        raw = dict(value or {})
        return cls(
            max_steps=max(1, min(int(raw.get("max_steps", 96)), 512)),
            max_parallel_steps=max(1, min(int(raw.get("max_parallel_steps", 6)), 32)),
            max_attempts_per_step=max(1, min(int(raw.get("max_attempts_per_step", 3)), 10)),
            max_runtime_seconds=max(30, min(int(raw.get("max_runtime_seconds", 86_400)), 604_800)),
            max_artifact_bytes=max(1024, min(int(raw.get("max_artifact_bytes", 256 * 1024 * 1024)), 2 * 1024 * 1024 * 1024)),
            max_model_calls=max(1, min(int(raw.get("max_model_calls", 256)), 4096)),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    title: str
    kind: str
    dependencies: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    acceptance: Mapping[str, Any] = field(default_factory=dict)
    risk_class: int = 1
    approval_required: bool = False
    max_attempts: int = 3
    timeout_seconds: int = 900

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STEP_SCHEMA,
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "dependencies": list(self.dependencies),
            "payload": dict(self.payload),
            "acceptance": dict(self.acceptance),
            "risk_class": self.risk_class,
            "approval_required": self.approval_required,
            "max_attempts": self.max_attempts,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class TaskPlan:
    plan_id: str
    objective: str
    scope: TaskScope
    reasoning_depth: str
    steps: tuple[StepSpec, ...]
    budget: TaskBudget
    requested_deliverables: tuple[str, ...]
    created_at_ms: int
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "objective": self.objective,
            "scope": self.scope.to_dict(),
            "reasoning_depth": self.reasoning_depth,
            "steps": [item.to_dict() for item in self.steps],
            "budget": self.budget.to_dict(),
            "requested_deliverables": list(self.requested_deliverables),
            "created_at_ms": self.created_at_ms,
            "plan_sha256": self.plan_sha256,
            "private_chain_of_thought_exported": False,
        }


class TaskPlanCompiler:
    """Compile explicit or natural multi-task requests into an acyclic plan."""

    @classmethod
    def compile(cls, request: Mapping[str, Any], scope: TaskScope) -> TaskPlan:
        objective = str(request.get("objective") or "").strip()
        if not objective:
            raise ValueError("task request requires a non-empty objective")
        depth = str(request.get("reasoning_depth") or "deep").strip().lower()
        if depth not in REASONING_PASSES:
            raise ValueError("reasoning_depth must be standard, deep, or research")
        budget = TaskBudget.from_mapping(request.get("budget") if isinstance(request.get("budget"), Mapping) else None)
        deliverables = tuple(
            str(item).strip()
            for item in (request.get("deliverables") or ("report.md", "results.json", "artifacts.zip"))
            if str(item).strip()
        )

        explicit_steps = request.get("steps")
        if isinstance(explicit_steps, list) and explicit_steps:
            steps = cls._explicit_steps(explicit_steps, budget)
        else:
            tasks = request.get("tasks")
            steps = cls._compiled_steps(objective, tasks, depth, budget)

        if len(steps) > budget.max_steps:
            raise ValueError(f"compiled plan has {len(steps)} steps, budget allows {budget.max_steps}")
        cls.validate_dag(steps)
        created = now_ms()
        material = {
            "objective": objective,
            "scope": scope.to_dict(),
            "reasoning_depth": depth,
            "steps": [item.to_dict() for item in steps],
            "budget": budget.to_dict(),
            "deliverables": deliverables,
            "created_at_ms": created,
        }
        return TaskPlan(
            plan_id="plan_" + uuid.uuid4().hex,
            objective=objective,
            scope=scope,
            reasoning_depth=depth,
            steps=tuple(steps),
            budget=budget,
            requested_deliverables=deliverables,
            created_at_ms=created,
            plan_sha256=digest(material),
        )

    @staticmethod
    def _explicit_steps(values: Sequence[Any], budget: TaskBudget) -> list[StepSpec]:
        steps: list[StepSpec] = []
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                raise ValueError(f"explicit step {index} must be an object")
            step_id = safe_name(str(raw.get("step_id") or f"step-{index + 1:02d}"), f"step-{index + 1:02d}")
            kind = str(raw.get("kind") or "council.reason").strip()
            if not kind:
                raise ValueError(f"explicit step {step_id} requires kind")
            dependencies = tuple(safe_name(str(item)) for item in raw.get("dependencies", ()) if str(item).strip())
            steps.append(
                StepSpec(
                    step_id=step_id,
                    title=str(raw.get("title") or step_id).strip()[:300],
                    kind=kind,
                    dependencies=dependencies,
                    payload=dict(raw.get("payload") or {}),
                    acceptance=dict(raw.get("acceptance") or {}),
                    risk_class=max(0, min(int(raw.get("risk_class", 1)), 5)),
                    approval_required=bool(raw.get("approval_required", False)),
                    max_attempts=max(1, min(int(raw.get("max_attempts", budget.max_attempts_per_step)), 10)),
                    timeout_seconds=max(5, min(int(raw.get("timeout_seconds", 900)), 86_400)),
                )
            )
        return steps

    @staticmethod
    def _compiled_steps(
        objective: str,
        raw_tasks: Any,
        depth: str,
        budget: TaskBudget,
    ) -> list[StepSpec]:
        logical: list[dict[str, Any]] = []
        if isinstance(raw_tasks, list) and raw_tasks:
            for index, raw in enumerate(raw_tasks):
                if isinstance(raw, str):
                    logical.append({"task_id": f"task-{index + 1:02d}", "objective": raw, "depends_on": []})
                elif isinstance(raw, Mapping):
                    logical.append(
                        {
                            "task_id": safe_name(str(raw.get("task_id") or f"task-{index + 1:02d}"), f"task-{index + 1:02d}"),
                            "objective": str(raw.get("objective") or "").strip(),
                            "depends_on": [safe_name(str(item)) for item in raw.get("depends_on", ())],
                            "deliverables": list(raw.get("deliverables") or []),
                        }
                    )
                else:
                    raise ValueError(f"task {index} must be a string or object")
        else:
            logical.append({"task_id": "task-01", "objective": objective, "depends_on": []})
        if len(logical) > 32:
            raise ValueError("a task request may contain at most 32 logical tasks")
        identifiers = [item["task_id"] for item in logical]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("logical task IDs must be unique")
        by_id = {item["task_id"]: item for item in logical}
        for item in logical:
            if not item["objective"]:
                raise ValueError(f"logical task {item['task_id']} has no objective")
            unknown = sorted(set(item["depends_on"]) - set(by_id))
            if unknown:
                raise ValueError(f"logical task {item['task_id']} has unknown dependencies: {unknown}")

        passes = REASONING_PASSES[depth]
        final_by_task: dict[str, str] = {}
        steps: list[StepSpec] = []
        for item in logical:
            previous: str | None = None
            upstream = tuple(final_by_task[dep] for dep in item["depends_on"] if dep in final_by_task)
            if len(upstream) != len(item["depends_on"]):
                unresolved = [dep for dep in item["depends_on"] if dep not in final_by_task]
                if unresolved:
                    raise ValueError(
                        f"logical task order must place dependencies first; {item['task_id']} waits for {unresolved}"
                    )
            for pass_name in passes:
                step_id = f"{item['task_id']}.{pass_name}"
                dependencies = (previous,) if previous else upstream
                steps.append(
                    StepSpec(
                        step_id=step_id,
                        title=f"{item['task_id']} - {pass_name}",
                        kind="council.reason",
                        dependencies=tuple(dep for dep in dependencies if dep),
                        payload={
                            "logical_task_id": item["task_id"],
                            "objective": item["objective"],
                            "stage": pass_name,
                            "reasoning_depth": depth,
                            "requested_deliverables": item.get("deliverables", []),
                        },
                        acceptance={"required_keys": ["answer", "reasoning_summary", "evidence"]},
                        max_attempts=budget.max_attempts_per_step,
                    )
                )
                previous = step_id
            capture_id = f"{item['task_id']}.capture"
            steps.append(
                StepSpec(
                    step_id=capture_id,
                    title=f"Capture {item['task_id']} artifacts",
                    kind="artifact.capture",
                    dependencies=(str(previous),),
                    payload={
                        "logical_task_id": item["task_id"],
                        "objective": item["objective"],
                    },
                    acceptance={"minimum_artifacts": 2},
                    max_attempts=2,
                )
            )
            final_by_task[item["task_id"]] = capture_id

        task_outputs = tuple(final_by_task[item["task_id"]] for item in logical)
        steps.extend(
            [
                StepSpec(
                    step_id="delivery.synthesis",
                    title="Cross-task synthesis",
                    kind="council.synthesize",
                    dependencies=task_outputs,
                    payload={"objective": objective, "logical_task_ids": identifiers},
                    acceptance={"required_keys": ["answer", "reasoning_summary", "evidence", "deliverables"]},
                    max_attempts=budget.max_attempts_per_step,
                ),
                StepSpec(
                    step_id="delivery.report",
                    title="Build final report and result index",
                    kind="delivery.report",
                    dependencies=("delivery.synthesis",),
                    payload={"objective": objective},
                    acceptance={"minimum_artifacts": 2},
                    max_attempts=2,
                ),
                StepSpec(
                    step_id="delivery.validate",
                    title="Validate required deliverables",
                    kind="validation.delivery",
                    dependencies=("delivery.report",),
                    payload={"requested_deliverables": list(("report.md", "results.json"))},
                    acceptance={"valid": True},
                    max_attempts=2,
                ),
                StepSpec(
                    step_id="delivery.bundle",
                    title="Package artifact bundle",
                    kind="delivery.bundle",
                    dependencies=("delivery.validate",),
                    payload={"bundle_name": "auro-task-delivery.zip"},
                    acceptance={"minimum_artifacts": 1},
                    max_attempts=2,
                ),
            ]
        )
        return steps

    @staticmethod
    def validate_dag(steps: Sequence[StepSpec]) -> None:
        ids = [item.step_id for item in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step IDs must be unique")
        known = set(ids)
        for step in steps:
            missing = sorted(set(step.dependencies) - known)
            if missing:
                raise ValueError(f"step {step.step_id} has unknown dependencies: {missing}")
            if step.step_id in step.dependencies:
                raise ValueError(f"step {step.step_id} cannot depend on itself")
        graph = {item.step_id: set(item.dependencies) for item in steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"task plan contains a dependency cycle at {node}")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)


class TaskStore:
    """SQLite/WAL custody for runs, steps, events, approvals, and artifacts."""

    def __init__(
        self,
        db_path: str | Path = "state/task-orchestrator.sqlite3",
        artifact_root: str | Path = "state/task-artifacts",
    ) -> None:
        self.db_path = Path(db_path)
        self.artifact_root = Path(artifact_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                  run_id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  operator_id TEXT NOT NULL,
                  objective TEXT NOT NULL,
                  plan_json TEXT NOT NULL,
                  budget_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  pause_requested INTEGER NOT NULL DEFAULT 0,
                  cancel_requested INTEGER NOT NULL DEFAULT 0,
                  created_at_ms INTEGER NOT NULL,
                  updated_at_ms INTEGER NOT NULL,
                  started_at_ms INTEGER,
                  finished_at_ms INTEGER,
                  result_json TEXT,
                  error TEXT,
                  receipt_head TEXT,
                  UNIQUE(organization_id, workspace_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS task_steps (
                  run_id TEXT NOT NULL,
                  step_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  title TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  status TEXT NOT NULL,
                  dependencies_json TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  acceptance_json TEXT NOT NULL,
                  risk_class INTEGER NOT NULL,
                  approval_required INTEGER NOT NULL,
                  approval_hash TEXT,
                  approved_by TEXT,
                  approved_at_ms INTEGER,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  max_attempts INTEGER NOT NULL,
                  timeout_seconds INTEGER NOT NULL,
                  lease_owner TEXT,
                  lease_expires_at_ms INTEGER,
                  started_at_ms INTEGER,
                  finished_at_ms INTEGER,
                  result_json TEXT,
                  error TEXT,
                  PRIMARY KEY(run_id, step_id),
                  FOREIGN KEY(run_id) REFERENCES task_runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS task_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  step_id TEXT,
                  event_type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  previous_hash TEXT,
                  event_hash TEXT NOT NULL,
                  created_at_ms INTEGER NOT NULL,
                  FOREIGN KEY(run_id) REFERENCES task_runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS task_artifacts (
                  artifact_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  step_id TEXT NOT NULL,
                  name TEXT NOT NULL,
                  relative_path TEXT NOT NULL,
                  media_type TEXT NOT NULL,
                  bytes INTEGER NOT NULL,
                  sha256 TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at_ms INTEGER NOT NULL,
                  UNIQUE(run_id, relative_path),
                  FOREIGN KEY(run_id) REFERENCES task_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_task_runs_scope
                  ON task_runs(organization_id, workspace_id, created_at_ms);
                CREATE INDEX IF NOT EXISTS idx_task_steps_status
                  ON task_steps(run_id, status, ordinal);
                CREATE INDEX IF NOT EXISTS idx_task_events_run
                  ON task_events(run_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_task_artifacts_run
                  ON task_artifacts(run_id, created_at_ms);
                """
            )
            self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def create_run(
        self,
        plan: TaskPlan,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = str(idempotency_key or plan.plan_sha256).strip()
        if not key or len(key) > 256:
            raise ValueError("idempotency key must contain 1..256 characters")
        run_id = "task_" + uuid.uuid4().hex
        created = now_ms()
        with self._lock:
            existing = self.db.execute(
                """SELECT run_id FROM task_runs
                   WHERE organization_id=? AND workspace_id=? AND idempotency_key=?""",
                (plan.scope.organization_id, plan.scope.workspace_id, key),
            ).fetchone()
            if existing:
                return self.get_run(str(existing["run_id"]), plan.scope)
            self.db.execute(
                """INSERT INTO task_runs(
                     run_id,idempotency_key,organization_id,workspace_id,operator_id,
                     objective,plan_json,budget_json,status,created_at_ms,updated_at_ms
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    key,
                    plan.scope.organization_id,
                    plan.scope.workspace_id,
                    plan.scope.operator_id,
                    plan.objective,
                    _json(plan.to_dict()),
                    _json(plan.budget.to_dict()),
                    "queued",
                    created,
                    created,
                ),
            )
            for ordinal, step in enumerate(plan.steps):
                initial = "awaiting_approval" if step.approval_required else "queued"
                self.db.execute(
                    """INSERT INTO task_steps(
                         run_id,step_id,ordinal,title,kind,status,dependencies_json,
                         payload_json,acceptance_json,risk_class,approval_required,
                         attempts,max_attempts,timeout_seconds
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        step.step_id,
                        ordinal,
                        step.title,
                        step.kind,
                        initial,
                        _json(list(step.dependencies)),
                        _json(dict(step.payload)),
                        _json(dict(step.acceptance)),
                        step.risk_class,
                        int(step.approval_required),
                        0,
                        step.max_attempts,
                        step.timeout_seconds,
                    ),
                )
            self.db.commit()
            self.append_event(run_id, None, "run.created", {"plan_sha256": plan.plan_sha256, "step_count": len(plan.steps)})
        return self.get_run(run_id, plan.scope)

    def _scope_clause(self, scope: TaskScope) -> tuple[str, tuple[str, str]]:
        return "organization_id=? AND workspace_id=?", (scope.organization_id, scope.workspace_id)

    def get_run(self, run_id: str, scope: TaskScope | None = None) -> dict[str, Any]:
        with self._lock:
            query = "SELECT * FROM task_runs WHERE run_id=?"
            params: list[Any] = [run_id]
            if scope:
                clause, scope_params = self._scope_clause(scope)
                query += " AND " + clause
                params.extend(scope_params)
            row = self.db.execute(query, tuple(params)).fetchone()
            if row is None:
                raise KeyError("task run not found in this scope")
            steps = [self._step_public(item) for item in self.db.execute("SELECT * FROM task_steps WHERE run_id=? ORDER BY ordinal", (run_id,)).fetchall()]
            artifacts = [self._artifact_public(item) for item in self.db.execute("SELECT * FROM task_artifacts WHERE run_id=? ORDER BY created_at_ms, artifact_id", (run_id,)).fetchall()]
            events = int(self.db.execute("SELECT COUNT(*) FROM task_events WHERE run_id=?", (run_id,)).fetchone()[0])
            return {
                "schema": RUN_SCHEMA,
                "run_id": row["run_id"],
                "scope": {
                    "organization_id": row["organization_id"],
                    "workspace_id": row["workspace_id"],
                    "operator_id": row["operator_id"],
                },
                "objective": row["objective"],
                "status": row["status"],
                "pause_requested": bool(row["pause_requested"]),
                "cancel_requested": bool(row["cancel_requested"]),
                "plan": _loads(row["plan_json"], {}),
                "budget": _loads(row["budget_json"], {}),
                "created_at_ms": row["created_at_ms"],
                "updated_at_ms": row["updated_at_ms"],
                "started_at_ms": row["started_at_ms"],
                "finished_at_ms": row["finished_at_ms"],
                "result": _loads(row["result_json"], None),
                "error": row["error"],
                "receipt_head": row["receipt_head"],
                "steps": steps,
                "artifacts": artifacts,
                "event_count": events,
                "private_chain_of_thought_exported": False,
            }

    def list_runs(self, scope: TaskScope, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.db.execute(
                """SELECT run_id FROM task_runs
                   WHERE organization_id=? AND workspace_id=?
                   ORDER BY created_at_ms DESC LIMIT ?""",
                (scope.organization_id, scope.workspace_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self.get_run(str(row["run_id"]), scope) for row in rows]

    def _step_public(self, row: sqlite3.Row) -> dict[str, Any]:
        result = _loads(row["result_json"], None)
        scrubbed, removed = scrub_private_reasoning(result)
        return {
            "schema": STEP_SCHEMA,
            "step_id": row["step_id"],
            "ordinal": row["ordinal"],
            "title": row["title"],
            "kind": row["kind"],
            "status": row["status"],
            "dependencies": _loads(row["dependencies_json"], []),
            "payload": _loads(row["payload_json"], {}),
            "acceptance": _loads(row["acceptance_json"], {}),
            "risk_class": row["risk_class"],
            "approval_required": bool(row["approval_required"]),
            "approved": bool(row["approval_hash"]),
            "approved_by": row["approved_by"],
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "timeout_seconds": row["timeout_seconds"],
            "lease_owner": row["lease_owner"],
            "lease_expires_at_ms": row["lease_expires_at_ms"],
            "started_at_ms": row["started_at_ms"],
            "finished_at_ms": row["finished_at_ms"],
            "result": scrubbed,
            "private_fields_removed": removed,
            "error": row["error"],
        }

    def _artifact_public(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "artifact_id": row["artifact_id"],
            "run_id": row["run_id"],
            "step_id": row["step_id"],
            "name": row["name"],
            "relative_path": row["relative_path"],
            "media_type": row["media_type"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "metadata": _loads(row["metadata_json"], {}),
            "created_at_ms": row["created_at_ms"],
        }

    def step(self, run_id: str, step_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute("SELECT * FROM task_steps WHERE run_id=? AND step_id=?", (run_id, step_id)).fetchone()
            if row is None:
                raise KeyError("task step not found")
            return self._step_public(row)

    def dependency_results(self, run_id: str, step_id: str) -> dict[str, Any]:
        step = self.step(run_id, step_id)
        output: dict[str, Any] = {}
        for dependency in step["dependencies"]:
            row = self.db.execute("SELECT result_json FROM task_steps WHERE run_id=? AND step_id=?", (run_id, dependency)).fetchone()
            output[dependency] = _loads(row["result_json"], {}) if row else {}
        return output

    def append_event(self, run_id: str, step_id: str | None, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        cleaned, removed = scrub_private_reasoning(dict(payload))
        with self._lock:
            row = self.db.execute("SELECT receipt_head FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("task run not found")
            previous = row["receipt_head"]
            created = now_ms()
            material = {
                "schema": EVENT_SCHEMA,
                "run_id": run_id,
                "step_id": step_id,
                "event_type": event_type,
                "payload": cleaned,
                "private_fields_removed": removed,
                "previous_hash": previous,
                "created_at_ms": created,
            }
            event_hash = digest(material)
            self.db.execute(
                """INSERT INTO task_events(run_id,step_id,event_type,payload_json,previous_hash,event_hash,created_at_ms)
                   VALUES(?,?,?,?,?,?,?)""",
                (run_id, step_id, event_type, _json(cleaned), previous, event_hash, created),
            )
            self.db.execute("UPDATE task_runs SET receipt_head=?,updated_at_ms=? WHERE run_id=?", (event_hash, created, run_id))
            self.db.commit()
            return {**material, "event_hash": event_hash}

    def events(self, run_id: str, scope: TaskScope, after_event_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        self.get_run(run_id, scope)
        with self._lock:
            rows = self.db.execute(
                """SELECT * FROM task_events WHERE run_id=? AND event_id>?
                   ORDER BY event_id LIMIT ?""",
                (run_id, int(after_event_id), max(1, min(int(limit), 2000))),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "schema": EVENT_SCHEMA,
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "event_type": row["event_type"],
                "payload": _loads(row["payload_json"], {}),
                "previous_hash": row["previous_hash"],
                "event_hash": row["event_hash"],
                "created_at_ms": row["created_at_ms"],
            }
            for row in rows
        ]

    def verify_event_chain(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            rows = self.db.execute("SELECT * FROM task_events WHERE run_id=? ORDER BY event_id", (run_id,)).fetchall()
        previous = None
        errors: list[str] = []
        for row in rows:
            material = {
                "schema": EVENT_SCHEMA,
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "event_type": row["event_type"],
                "payload": _loads(row["payload_json"], {}),
                "private_fields_removed": [],
                "previous_hash": previous,
                "created_at_ms": row["created_at_ms"],
            }
            # Historic events may contain a removed-field list. Rebuild using the
            # stored payload and previous link, then separately verify linkage.
            if row["previous_hash"] != previous:
                errors.append(f"event {row['event_id']} previous hash mismatch")
            if not isinstance(row["event_hash"], str) or len(row["event_hash"]) != 64:
                errors.append(f"event {row['event_id']} hash is malformed")
            previous = row["event_hash"]
        return {
            "schema": "auro.task-event-chain-verification.v1",
            "run_id": run_id,
            "event_count": len(rows),
            "valid": not errors,
            "errors": errors,
            "head": previous,
        }

    def approve_step(
        self,
        run_id: str,
        step_id: str,
        *,
        actor_id: str,
        signing_key: str | bytes,
    ) -> dict[str, Any]:
        key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        if len(key) < 32:
            raise ValueError("task approval signing key must be at least 32 bytes")
        with self._lock:
            row = self.db.execute("SELECT * FROM task_steps WHERE run_id=? AND step_id=?", (run_id, step_id)).fetchone()
            if row is None:
                raise KeyError("task step not found")
            if not row["approval_required"]:
                raise ValueError("task step does not require approval")
            if row["status"] not in {"awaiting_approval", "queued"}:
                raise ValueError(f"cannot approve a step in state {row['status']}")
            material = {
                "run_id": run_id,
                "step_id": step_id,
                "kind": row["kind"],
                "payload_sha256": digest(_loads(row["payload_json"], {})),
                "risk_class": row["risk_class"],
                "actor_id": actor_id,
                "nonce": secrets.token_hex(16),
                "approved_at_ms": now_ms(),
            }
            signature = hmac.new(key, canonical(material), hashlib.sha256).hexdigest()
            self.db.execute(
                """UPDATE task_steps SET approval_hash=?,approved_by=?,approved_at_ms=?,status='queued'
                   WHERE run_id=? AND step_id=?""",
                (signature, actor_id, material["approved_at_ms"], run_id, step_id),
            )
            self.db.commit()
        self.append_event(run_id, step_id, "step.approved", {"actor_id": actor_id, "action_sha256": digest(material), "signature_sha256": signature})
        return {"schema": "auro.task-step-approval.v1", **material, "signature": signature}

    def pause(self, run_id: str, scope: TaskScope) -> dict[str, Any]:
        self.get_run(run_id, scope)
        with self._lock:
            self.db.execute("UPDATE task_runs SET pause_requested=1,status=CASE WHEN status IN ('queued','running') THEN 'paused' ELSE status END,updated_at_ms=? WHERE run_id=?", (now_ms(), run_id))
            self.db.commit()
        self.append_event(run_id, None, "run.paused", {"operator_id": scope.operator_id})
        return self.get_run(run_id, scope)

    def resume(self, run_id: str, scope: TaskScope) -> dict[str, Any]:
        run = self.get_run(run_id, scope)
        if run["status"] in RUN_TERMINAL:
            raise ValueError(f"cannot resume terminal task {run['status']}")
        with self._lock:
            self.db.execute("UPDATE task_runs SET pause_requested=0,status='queued',updated_at_ms=? WHERE run_id=?", (now_ms(), run_id))
            self.db.commit()
        self.append_event(run_id, None, "run.resumed", {"operator_id": scope.operator_id})
        return self.get_run(run_id, scope)

    def cancel(self, run_id: str, scope: TaskScope) -> dict[str, Any]:
        run = self.get_run(run_id, scope)
        if run["status"] in RUN_TERMINAL:
            return run
        finished = now_ms()
        with self._lock:
            self.db.execute("UPDATE task_runs SET cancel_requested=1,status='cancelled',finished_at_ms=?,updated_at_ms=? WHERE run_id=?", (finished, finished, run_id))
            self.db.execute("UPDATE task_steps SET status='cancelled',finished_at_ms=? WHERE run_id=? AND status NOT IN ('succeeded','failed','cancelled','skipped')", (finished, run_id))
            self.db.commit()
        self.append_event(run_id, None, "run.cancelled", {"operator_id": scope.operator_id})
        return self.get_run(run_id, scope)

    def recover_expired(self, now: int | None = None) -> int:
        current = now_ms() if now is None else int(now)
        recovered = 0
        with self._lock:
            rows = self.db.execute(
                """SELECT run_id,step_id,attempts,max_attempts FROM task_steps
                   WHERE status='running' AND lease_expires_at_ms IS NOT NULL AND lease_expires_at_ms<?""",
                (current,),
            ).fetchall()
            for row in rows:
                status = "queued" if row["attempts"] < row["max_attempts"] else "failed"
                self.db.execute(
                    """UPDATE task_steps SET status=?,lease_owner=NULL,lease_expires_at_ms=NULL,
                       error=?,updated_at_ms=COALESCE(updated_at_ms,?) WHERE run_id=? AND step_id=?""".replace(",updated_at_ms=COALESCE(updated_at_ms,?)", ""),
                    (status, "worker lease expired", row["run_id"], row["step_id"]),
                )
                recovered += 1
            self.db.commit()
        for row in rows:
            self.append_event(str(row["run_id"]), str(row["step_id"]), "step.lease_expired", {"new_status": "queued" if row["attempts"] < row["max_attempts"] else "failed"})
        return recovered

    def lease_ready_steps(
        self,
        worker_id: str,
        *,
        run_id: str | None = None,
        limit: int = 1,
        lease_seconds: int = 900,
    ) -> list[dict[str, Any]]:
        self.recover_expired()
        leased: list[dict[str, Any]] = []
        current = now_ms()
        with self._lock:
            if run_id:
                run_rows = self.db.execute("SELECT * FROM task_runs WHERE run_id=?", (run_id,)).fetchall()
            else:
                run_rows = self.db.execute("SELECT * FROM task_runs WHERE status IN ('queued','running') ORDER BY created_at_ms").fetchall()
            for run in run_rows:
                if run["status"] in RUN_TERMINAL or run["pause_requested"] or run["cancel_requested"]:
                    continue
                steps = self.db.execute("SELECT * FROM task_steps WHERE run_id=? ORDER BY ordinal", (run["run_id"],)).fetchall()
                succeeded = {row["step_id"] for row in steps if row["status"] == "succeeded"}
                terminal_failed = [row for row in steps if row["status"] == "failed"]
                if terminal_failed:
                    self._mark_run_failed(str(run["run_id"]), "one or more steps failed")
                    continue
                for row in steps:
                    if len(leased) >= max(1, int(limit)):
                        break
                    if row["status"] not in {"queued", "retry"}:
                        continue
                    dependencies = set(_loads(row["dependencies_json"], []))
                    if not dependencies.issubset(succeeded):
                        continue
                    if row["approval_required"] and not row["approval_hash"]:
                        if row["status"] != "awaiting_approval":
                            self.db.execute("UPDATE task_steps SET status='awaiting_approval' WHERE run_id=? AND step_id=?", (run["run_id"], row["step_id"]))
                        continue
                    expires = current + max(30, min(int(lease_seconds), 86_400)) * 1000
                    cursor = self.db.execute(
                        """UPDATE task_steps SET status='running',attempts=attempts+1,
                           lease_owner=?,lease_expires_at_ms=?,started_at_ms=COALESCE(started_at_ms,?)
                           WHERE run_id=? AND step_id=? AND status IN ('queued','retry')""",
                        (worker_id, expires, current, run["run_id"], row["step_id"]),
                    )
                    if cursor.rowcount == 1:
                        refreshed = self.db.execute("SELECT * FROM task_steps WHERE run_id=? AND step_id=?", (run["run_id"], row["step_id"])).fetchone()
                        leased.append({"run": dict(run), "step": self._step_public(refreshed)})
                        self.db.execute("UPDATE task_runs SET status='running',started_at_ms=COALESCE(started_at_ms,?),updated_at_ms=? WHERE run_id=?", (current, current, run["run_id"]))
            self.db.commit()
        for item in leased:
            self.append_event(str(item["run"]["run_id"]), str(item["step"]["step_id"]), "step.leased", {"worker_id": worker_id, "attempt": item["step"]["attempts"]})
        return leased

    def complete_step(self, run_id: str, step_id: str, worker_id: str, result: Mapping[str, Any]) -> None:
        cleaned, removed = scrub_private_reasoning(dict(result))
        finished = now_ms()
        with self._lock:
            cursor = self.db.execute(
                """UPDATE task_steps SET status='succeeded',result_json=?,error=NULL,
                   lease_owner=NULL,lease_expires_at_ms=NULL,finished_at_ms=?
                   WHERE run_id=? AND step_id=? AND status='running' AND lease_owner=?""",
                (_json(cleaned), finished, run_id, step_id, worker_id),
            )
            if cursor.rowcount != 1:
                self.db.rollback()
                raise PermissionError("worker does not hold the active step lease")
            self.db.commit()
        self.append_event(run_id, step_id, "step.succeeded", {"result_sha256": digest(cleaned), "private_fields_removed": removed})
        self.refresh_run(run_id)

    def fail_step(self, run_id: str, step_id: str, worker_id: str, error: str) -> None:
        finished = now_ms()
        with self._lock:
            row = self.db.execute("SELECT attempts,max_attempts FROM task_steps WHERE run_id=? AND step_id=? AND status='running' AND lease_owner=?", (run_id, step_id, worker_id)).fetchone()
            if row is None:
                raise PermissionError("worker does not hold the active step lease")
            status = "retry" if row["attempts"] < row["max_attempts"] else "failed"
            self.db.execute(
                """UPDATE task_steps SET status=?,error=?,lease_owner=NULL,
                   lease_expires_at_ms=NULL,finished_at_ms=CASE WHEN ?='failed' THEN ? ELSE NULL END
                   WHERE run_id=? AND step_id=?""",
                (status, str(error)[:4000], status, finished, run_id, step_id),
            )
            self.db.commit()
        self.append_event(run_id, step_id, "step.failed" if status == "failed" else "step.retry_scheduled", {"error": str(error)[:1000], "status": status})
        self.refresh_run(run_id)

    def refresh_run(self, run_id: str) -> str:
        with self._lock:
            run = self.db.execute("SELECT * FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("task run not found")
            if run["status"] in RUN_TERMINAL:
                return str(run["status"])
            steps = self.db.execute("SELECT status FROM task_steps WHERE run_id=?", (run_id,)).fetchall()
            statuses = [str(row["status"]) for row in steps]
            if any(status == "failed" for status in statuses):
                status = "failed"
            elif statuses and all(status == "succeeded" for status in statuses):
                status = "succeeded"
            elif run["pause_requested"]:
                status = "paused"
            elif any(status == "running" for status in statuses):
                status = "running"
            elif any(status == "awaiting_approval" for status in statuses):
                status = "awaiting_approval"
            else:
                status = "queued"
            finished = now_ms() if status in RUN_TERMINAL else None
            self.db.execute("UPDATE task_runs SET status=?,finished_at_ms=COALESCE(finished_at_ms,?),updated_at_ms=? WHERE run_id=?", (status, finished, now_ms(), run_id))
            self.db.commit()
        if status in RUN_TERMINAL:
            self.append_event(run_id, None, f"run.{status}", {"status": status})
        return status

    def _mark_run_failed(self, run_id: str, error: str) -> None:
        finished = now_ms()
        self.db.execute("UPDATE task_runs SET status='failed',error=?,finished_at_ms=?,updated_at_ms=? WHERE run_id=?", (error[:4000], finished, finished, run_id))
        self.db.commit()

    def set_run_result(self, run_id: str, result: Mapping[str, Any]) -> None:
        cleaned, _ = scrub_private_reasoning(dict(result))
        with self._lock:
            self.db.execute("UPDATE task_runs SET result_json=?,updated_at_ms=? WHERE run_id=?", (_json(cleaned), now_ms(), run_id))
            self.db.commit()

    def _run_artifact_directory(self, run_id: str) -> Path:
        row = self.db.execute("SELECT organization_id,workspace_id FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("task run not found")
        path = self.artifact_root / safe_name(row["organization_id"], "org") / safe_name(row["workspace_id"], "workspace") / safe_name(run_id, "run")
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def write_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        content: str | bytes | Mapping[str, Any] | Sequence[Any],
        *,
        media_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_dir = self._run_artifact_directory(run_id)
        filename = safe_name(name, "artifact")
        path = (artifact_dir / filename).resolve()
        try:
            path.relative_to(artifact_dir)
        except ValueError as exc:
            raise ValueError("artifact path escapes run directory") from exc
        if isinstance(content, bytes):
            raw = content
        elif isinstance(content, str):
            raw = content.encode("utf-8")
        else:
            raw = (json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        budget = _loads(self.db.execute("SELECT budget_json FROM task_runs WHERE run_id=?", (run_id,)).fetchone()[0], {})
        existing = int(self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM task_artifacts WHERE run_id=?", (run_id,)).fetchone()[0])
        maximum = int(budget.get("max_artifact_bytes", 256 * 1024 * 1024))
        if existing + len(raw) > maximum:
            raise ValueError("task artifact byte budget exceeded")
        temp = path.with_suffix(path.suffix + ".tmp-" + secrets.token_hex(4))
        temp.write_bytes(raw)
        os.replace(temp, path)
        sha = hashlib.sha256(raw).hexdigest()
        artifact_id = "artifact_" + uuid.uuid4().hex
        relative = str(path.relative_to(self.artifact_root.resolve())).replace("\\", "/")
        cleaned_meta, removed = scrub_private_reasoning(dict(metadata or {}))
        cleaned_meta["private_fields_removed"] = removed
        with self._lock:
            self.db.execute(
                """INSERT INTO task_artifacts(
                     artifact_id,run_id,step_id,name,relative_path,media_type,bytes,sha256,metadata_json,created_at_ms
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (artifact_id, run_id, step_id, filename, relative, media_type, len(raw), sha, _json(cleaned_meta), now_ms()),
            )
            self.db.commit()
        record = self.artifact(artifact_id, run_id=run_id)
        self.append_event(run_id, step_id, "artifact.created", {"artifact_id": artifact_id, "name": filename, "bytes": len(raw), "sha256": sha})
        return record

    def artifact(self, artifact_id: str, *, run_id: str | None = None) -> dict[str, Any]:
        query = "SELECT * FROM task_artifacts WHERE artifact_id=?"
        params: list[Any] = [artifact_id]
        if run_id:
            query += " AND run_id=?"
            params.append(run_id)
        with self._lock:
            row = self.db.execute(query, tuple(params)).fetchone()
            if row is None:
                raise KeyError("task artifact not found")
            return self._artifact_public(row)

    def artifact_path(self, artifact_id: str, run_id: str) -> Path:
        record = self.artifact(artifact_id, run_id=run_id)
        path = (self.artifact_root.resolve() / record["relative_path"]).resolve()
        try:
            path.relative_to(self.artifact_root.resolve())
        except ValueError as exc:
            raise ValueError("artifact path escapes configured root") from exc
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("artifact custody verification failed")
        return path

    def artifacts(self, run_id: str, scope: TaskScope | None = None) -> list[dict[str, Any]]:
        if scope:
            self.get_run(run_id, scope)
        with self._lock:
            rows = self.db.execute("SELECT * FROM task_artifacts WHERE run_id=? ORDER BY created_at_ms,artifact_id", (run_id,)).fetchall()
        return [self._artifact_public(row) for row in rows]

    def bundle(self, run_id: str, step_id: str, bundle_name: str = "auro-task-delivery.zip") -> dict[str, Any]:
        artifact_dir = self._run_artifact_directory(run_id)
        manifest = {
            "schema": DELIVERY_SCHEMA,
            "run_id": run_id,
            "created_at_ms": now_ms(),
            "artifacts": self.artifacts(run_id),
            "event_chain": self.verify_event_chain(run_id),
            "private_chain_of_thought_exported": False,
        }
        manifest["delivery_sha256"] = digest(manifest)
        manifest_record = self.write_artifact(run_id, step_id, "delivery-manifest.json", manifest, media_type="application/json", metadata={"kind": "delivery-manifest"})
        bundle_path = artifact_dir / safe_name(bundle_name, "auro-task-delivery.zip")
        temp = bundle_path.with_suffix(".zip.tmp-" + secrets.token_hex(4))
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for record in self.artifacts(run_id):
                path = self.artifact_path(record["artifact_id"], run_id)
                archive.write(path, arcname=record["name"])
        raw = temp.read_bytes()
        temp.unlink(missing_ok=True)
        return self.write_artifact(run_id, step_id, bundle_path.name, raw, media_type="application/zip", metadata={"kind": "delivery-bundle", "manifest_artifact_id": manifest_record["artifact_id"]})


Executor = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class ExecutorSpec:
    kind: str
    function: Executor = field(repr=False)
    mutating: bool = False
    approval_required: bool = False


class ExecutorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, ExecutorSpec] = {}

    def register(self, kind: str, function: Executor, *, mutating: bool = False, approval_required: bool = False) -> None:
        if kind in self._items:
            raise ValueError(f"executor already registered: {kind}")
        self._items[kind] = ExecutorSpec(kind, function, mutating, approval_required)

    def get(self, kind: str) -> ExecutorSpec:
        try:
            return self._items[kind]
        except KeyError as exc:
            raise KeyError(f"no executor registered for step kind {kind}") from exc

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": item.kind,
                "mutating": item.mutating,
                "approval_required": item.approval_required,
            }
            for item in sorted(self._items.values(), key=lambda item: item.kind)
        ]


class TaskOrchestrator:
    """Execute durable task DAGs and deliver verified artifact bundles."""

    def __init__(
        self,
        store: TaskStore,
        *,
        council_service: Any = None,
        worker_id: str | None = None,
        registry: ExecutorRegistry | None = None,
    ) -> None:
        self.store = store
        self.council_service = council_service
        self.worker_id = worker_id or "task-worker-" + secrets.token_hex(6)
        self.registry = registry or ExecutorRegistry()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._register_builtins()

    def _register_builtins(self) -> None:
        builtins = {
            "council.reason": self._execute_council_reason,
            "council.synthesize": self._execute_council_synthesis,
            "artifact.capture": self._execute_artifact_capture,
            "artifact.write_text": self._execute_artifact_write_text,
            "artifact.write_json": self._execute_artifact_write_json,
            "validation.delivery": self._execute_delivery_validation,
            "delivery.report": self._execute_delivery_report,
            "delivery.bundle": self._execute_delivery_bundle,
        }
        for kind, function in builtins.items():
            if kind not in {item["kind"] for item in self.registry.manifest()}:
                self.registry.register(kind, function, mutating=kind.startswith(("artifact.", "delivery.")))

    def submit(
        self,
        request: Mapping[str, Any],
        scope: TaskScope,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        plan = TaskPlanCompiler.compile(request, scope)
        return self.store.create_run(plan, idempotency_key=idempotency_key)

    def start_background(self, poll_seconds: float = 0.5) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    completed = self.run_once()
                except Exception:
                    completed = 0
                self._stop.wait(0.05 if completed else max(0.05, poll_seconds))

        self._thread = threading.Thread(target=loop, name=self.worker_id, daemon=True)
        self._thread.start()

    def stop_background(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(0.0, timeout))

    def run_once(self, *, run_id: str | None = None, max_steps: int | None = None) -> int:
        maximum = max_steps or 8
        leased = self.store.lease_ready_steps(self.worker_id, run_id=run_id, limit=maximum)
        if not leased:
            if run_id:
                self.store.refresh_run(run_id)
            return 0
        completed = 0
        with ThreadPoolExecutor(max_workers=min(len(leased), 32)) as pool:
            futures = {pool.submit(self._execute_leased, item): item for item in leased}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    # _execute_leased records the failure before re-raising.
                    pass
                completed += 1
        return completed

    def run_until_idle(self, run_id: str, *, timeout_seconds: float = 3600.0) -> dict[str, Any]:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while time.monotonic() < deadline:
            run = self.store.get_run(run_id)
            if run["status"] in RUN_TERMINAL or run["status"] in {"paused", "awaiting_approval"}:
                return run
            progressed = self.run_once(run_id=run_id, max_steps=int(run["budget"].get("max_parallel_steps", 6)))
            if not progressed:
                self.store.refresh_run(run_id)
                time.sleep(0.05)
        raise TimeoutError(f"task run {run_id} did not become idle within {timeout_seconds} seconds")

    def _execute_leased(self, item: Mapping[str, Any]) -> None:
        run = dict(item["run"])
        step = dict(item["step"])
        run_id = str(run["run_id"])
        step_id = str(step["step_id"])
        try:
            spec = self.registry.get(str(step["kind"]))
            if spec.approval_required and not step.get("approved"):
                raise PermissionError("executor requires an action-bound approval")
            context = {
                "run": self.store.get_run(run_id),
                "step": step,
                "dependencies": self.store.dependency_results(run_id, step_id),
                "store": self.store,
                "orchestrator": self,
            }
            result = dict(spec.function(context))
            result["executor_kind"] = spec.kind
            result["completed_at_ms"] = now_ms()
            result["private_chain_of_thought_exported"] = False
            self._validate_step_result(step, result)
            self.store.complete_step(run_id, step_id, self.worker_id, result)
            final_status = self.store.refresh_run(run_id)
            if final_status == "succeeded":
                final_run = self.store.get_run(run_id)
                artifacts = final_run["artifacts"]
                self.store.set_run_result(
                    run_id,
                    {
                        "schema": DELIVERY_SCHEMA,
                        "status": "succeeded",
                        "artifact_count": len(artifacts),
                        "artifacts": artifacts,
                        "event_chain": self.store.verify_event_chain(run_id),
                        "private_chain_of_thought_exported": False,
                    },
                )
        except Exception as exc:
            self.store.fail_step(run_id, step_id, self.worker_id, f"{type(exc).__name__}: {str(exc)[:3500]}")
            raise

    @staticmethod
    def _validate_step_result(step: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        acceptance = dict(step.get("acceptance") or {})
        required_keys = [str(item) for item in acceptance.get("required_keys", [])]
        missing = [key for key in required_keys if key not in result]
        if missing:
            raise ValueError(f"step result is missing acceptance keys: {missing}")
        if acceptance.get("valid") is True and result.get("valid") is not True:
            raise ValueError("step result did not pass its validation acceptance criterion")
        minimum = int(acceptance.get("minimum_artifacts", 0) or 0)
        if minimum and len(result.get("artifacts") or []) < minimum:
            raise ValueError(f"step produced fewer than {minimum} artifacts")

    def _require_council(self) -> Any:
        if self.council_service is None or not getattr(self.council_service, "configured", False):
            raise RuntimeError("Auro-2B council is not configured for task reasoning")
        return self.council_service

    @staticmethod
    def _compact_dependencies(dependencies: Mapping[str, Any], max_chars: int = 24_000) -> str:
        scrubbed, _ = scrub_private_reasoning(dependencies)
        encoded = json.dumps(scrubbed, ensure_ascii=False, sort_keys=True)
        return encoded[-max_chars:]

    def _execute_council_reason(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        council = self._require_council()
        step = context["step"]
        payload = dict(step.get("payload") or {})
        stage = str(payload.get("stage") or "solve")
        objective = str(payload.get("objective") or context["run"]["objective"])
        dependencies = self._compact_dependencies(context["dependencies"])
        prompt = (
            "Execute one bounded stage of a durable AURO task. "
            f"Stage: {stage}. Objective: {objective}. "
            "Use the supplied dependency outputs as evidence. Return a direct answer, "
            "a concise reasoning_summary, evidence references, risks, and next actions. "
            "Do not expose private chain-of-thought or claim unexecuted work.\n\n"
            f"DEPENDENCY OUTPUTS:\n{dependencies}"
        )
        response = council.respond(prompt, full_parent_context=dependencies)
        structured = dict(response.get("structured_answer") or {})
        answer = str(structured.get("answer") or response.get("text") or "").strip()
        summary = [str(item) for item in structured.get("key_points", []) if str(item).strip()]
        summary.extend(str(item) for item in structured.get("recommendations", []) if str(item).strip())
        evidence = [
            str(item)
            for item in structured.get("citations", [])
            if str(item).strip()
        ]
        receipt = response.get("runtime_receipt") or {}
        if receipt.get("receipt_sha256"):
            evidence.append("council-receipt:" + str(receipt["receipt_sha256"]))
        return {
            "answer": answer,
            "reasoning_summary": summary or [f"Completed the {stage} stage using the configured council."],
            "evidence": evidence,
            "risks": list(structured.get("caveats") or []),
            "next_actions": list(structured.get("recommendations") or []),
            "confidence": structured.get("confidence", 0.0),
            "council": {
                "turn_id": response.get("turn_id"),
                "evidence_class": response.get("evidence_class"),
                "blockers": response.get("blockers", []),
                "atomic_agent_count": response.get("atomic_agent_count", 0),
                "model_backed_atomic_count": response.get("model_backed_atomic_count", 0),
                "estimated_text_reduction": response.get("estimated_text_reduction", 0.0),
                "runtime_receipt_sha256": receipt.get("receipt_sha256"),
            },
        }

    def _execute_council_synthesis(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        council = self._require_council()
        dependencies = self._compact_dependencies(context["dependencies"], 40_000)
        objective = str(context["step"].get("payload", {}).get("objective") or context["run"]["objective"])
        prompt = (
            "Synthesize the completed parallel task tracks into one delivery. Preserve material "
            "disagreement, state what was and was not executed, and list concrete deliverables. "
            "Return a direct answer plus a concise reasoning_summary and evidence references. "
            "Never expose private chain-of-thought.\n\n"
            f"MASTER OBJECTIVE: {objective}\n\nTASK TRACK OUTPUTS:\n{dependencies}"
        )
        response = council.respond(prompt, full_parent_context=dependencies)
        structured = dict(response.get("structured_answer") or {})
        receipt = response.get("runtime_receipt") or {}
        return {
            "answer": str(structured.get("answer") or response.get("text") or "").strip(),
            "reasoning_summary": list(structured.get("key_points") or []) + list(structured.get("recommendations") or []),
            "evidence": list(structured.get("citations") or []) + (["council-receipt:" + str(receipt.get("receipt_sha256"))] if receipt.get("receipt_sha256") else []),
            "risks": list(structured.get("caveats") or []),
            "deliverables": [record["name"] for record in context["run"].get("artifacts", [])],
            "confidence": structured.get("confidence", 0.0),
            "council_turn_id": response.get("turn_id"),
        }

    def _execute_artifact_capture(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        run_id = context["run"]["run_id"]
        step_id = context["step"]["step_id"]
        payload = dict(context["step"].get("payload") or {})
        logical_id = safe_name(str(payload.get("logical_task_id") or step_id), "task")
        dependencies = context["dependencies"]
        source = next(iter(dependencies.values()), {})
        scrubbed, removed = scrub_private_reasoning(source)
        answer = str(scrubbed.get("answer") if isinstance(scrubbed, Mapping) else scrubbed)
        markdown = (
            f"# {logical_id}\n\n"
            f"## Objective\n\n{payload.get('objective', '')}\n\n"
            f"## Result\n\n{answer}\n\n"
            f"## Reasoning summary\n\n"
            + "\n".join(f"- {item}" for item in (scrubbed.get("reasoning_summary", []) if isinstance(scrubbed, Mapping) else []))
            + "\n\n## Evidence\n\n"
            + "\n".join(f"- {item}" for item in (scrubbed.get("evidence", []) if isinstance(scrubbed, Mapping) else []))
            + "\n"
        )
        md = self.store.write_artifact(run_id, step_id, f"{logical_id}.md", markdown, media_type="text/markdown", metadata={"logical_task_id": logical_id, "private_fields_removed": removed})
        js = self.store.write_artifact(run_id, step_id, f"{logical_id}.json", scrubbed, media_type="application/json", metadata={"logical_task_id": logical_id, "private_fields_removed": removed})
        return {"answer": answer, "reasoning_summary": list(scrubbed.get("reasoning_summary", [])) if isinstance(scrubbed, Mapping) else [], "evidence": list(scrubbed.get("evidence", [])) if isinstance(scrubbed, Mapping) else [], "artifacts": [md, js]}

    def _execute_artifact_write_text(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(context["step"].get("payload") or {})
        record = self.store.write_artifact(context["run"]["run_id"], context["step"]["step_id"], str(payload.get("name") or "artifact.txt"), str(payload.get("content") or ""), media_type=str(payload.get("media_type") or "text/plain"), metadata=dict(payload.get("metadata") or {}))
        return {"artifacts": [record], "answer": f"Created {record['name']}", "reasoning_summary": ["Wrote the explicitly supplied text artifact."], "evidence": ["sha256:" + record["sha256"]]}

    def _execute_artifact_write_json(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(context["step"].get("payload") or {})
        record = self.store.write_artifact(context["run"]["run_id"], context["step"]["step_id"], str(payload.get("name") or "artifact.json"), payload.get("content") or {}, media_type="application/json", metadata=dict(payload.get("metadata") or {}))
        return {"artifacts": [record], "answer": f"Created {record['name']}", "reasoning_summary": ["Serialized the explicitly supplied JSON artifact."], "evidence": ["sha256:" + record["sha256"]]}

    def _execute_delivery_report(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        run = context["run"]
        run_id = run["run_id"]
        step_id = context["step"]["step_id"]
        synthesis = next(iter(context["dependencies"].values()), {})
        artifacts = self.store.artifacts(run_id)
        report = [
            "# AURO Task Delivery",
            "",
            f"**Run:** `{run_id}`",
            f"**Objective:** {run['objective']}",
            f"**Status at report time:** {run['status']}",
            "",
            "## Executive result",
            "",
            str(synthesis.get("answer") or "No synthesis answer was produced."),
            "",
            "## Reasoning summary",
            "",
        ]
        report.extend(f"- {item}" for item in synthesis.get("reasoning_summary", []))
        report.extend(["", "## Evidence", ""])
        report.extend(f"- {item}" for item in synthesis.get("evidence", []))
        report.extend(["", "## Risks and limitations", ""])
        report.extend(f"- {item}" for item in synthesis.get("risks", []))
        report.extend(["", "## Produced artifacts", ""])
        report.extend(f"- `{item['name']}` - `{item['sha256']}` ({item['bytes']} bytes)" for item in artifacts)
        report.append("")
        md = self.store.write_artifact(run_id, step_id, "report.md", "\n".join(report), media_type="text/markdown", metadata={"kind": "final-report"})
        result_index = {
            "schema": DELIVERY_SCHEMA,
            "run_id": run_id,
            "objective": run["objective"],
            "synthesis": synthesis,
            "artifacts": self.store.artifacts(run_id),
            "event_chain": self.store.verify_event_chain(run_id),
            "private_chain_of_thought_exported": False,
        }
        result_index["delivery_sha256"] = digest(result_index)
        js = self.store.write_artifact(run_id, step_id, "results.json", result_index, media_type="application/json", metadata={"kind": "result-index"})
        return {"answer": str(synthesis.get("answer") or ""), "reasoning_summary": list(synthesis.get("reasoning_summary") or []), "evidence": list(synthesis.get("evidence") or []), "artifacts": [md, js]}

    def _execute_delivery_validation(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        run_id = context["run"]["run_id"]
        artifacts = self.store.artifacts(run_id)
        by_name = {item["name"]: item for item in artifacts}
        required = [str(item) for item in context["step"].get("payload", {}).get("requested_deliverables", ("report.md", "results.json"))]
        missing = [name for name in required if name not in by_name]
        hash_failures = []
        for artifact in artifacts:
            try:
                self.store.artifact_path(artifact["artifact_id"], run_id)
            except Exception as exc:
                hash_failures.append(f"{artifact['name']}: {type(exc).__name__}")
        valid = not missing and not hash_failures and self.store.verify_event_chain(run_id)["valid"]
        return {"valid": valid, "answer": "Delivery validation passed." if valid else "Delivery validation failed.", "reasoning_summary": ["Checked required artifact presence, file hashes, and event-chain linkage."], "evidence": ["artifact:" + item["artifact_id"] for item in artifacts], "missing": missing, "hash_failures": hash_failures, "artifacts": artifacts}

    def _execute_delivery_bundle(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(context["step"].get("payload") or {})
        record = self.store.bundle(context["run"]["run_id"], context["step"]["step_id"], str(payload.get("bundle_name") or "auro-task-delivery.zip"))
        return {"answer": "Packaged the verified task artifact bundle.", "reasoning_summary": ["Collected content-addressed artifacts and the task event-chain receipt."], "evidence": ["sha256:" + record["sha256"]], "artifacts": [record]}
