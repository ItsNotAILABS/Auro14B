"""Durable multi-task and long-running work orchestration for AURO.

The runtime turns one objective into an explicit task graph, executes only
through leased workers, preserves compact reasoning summaries rather than hidden
chain-of-thought, validates outputs, and delivers content-addressed artifacts.

It is intentionally execution-backend neutral. A worker may be an AURO council,
POCKET Agent, sandbox, app bottle, mini OS, browser node, or human operator. The
runtime coordinates and verifies work; it does not silently grant tool access.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
import uuid
import zipfile


RUN_SCHEMA = "auro.task-run.v1"
PLAN_SCHEMA = "auro.task-plan.v1"
STEP_SCHEMA = "auro.task-step.v1"
ARTIFACT_SCHEMA = "auro.task-artifact.v1"
RECEIPT_SCHEMA = "auro.task-run.receipt.v1"
EVENT_SCHEMA = "auro.task-event.v1"

RUN_TERMINAL = {"succeeded", "partial", "failed", "cancelled"}
STEP_TERMINAL = {"succeeded", "failed", "cancelled", "skipped", "blocked"}
STEP_ACTIVE = {"leased", "running"}
QUALITY_MODES = {"fast", "standard", "deep", "exhaustive"}
TASK_KINDS = {
    "analysis",
    "research",
    "code",
    "build",
    "test",
    "write",
    "package",
    "review",
    "synthesize",
    "tool",
    "other",
}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_FILE = re.compile(r"[^A-Za-z0-9._-]+")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _now() -> int:
    return int(time.time())


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    parsed = json.loads(value)
    return parsed


def _safe_file_name(name: str) -> str:
    value = _SAFE_FILE.sub("-", str(name).strip()).strip(".-")
    if not value or value in {".", ".."}:
        raise ValueError("artifact name is invalid")
    return value[:180]


def _require_identifier(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not _ID.fullmatch(text):
        raise ValueError(f"{field_name} must match {_ID.pattern}")
    return text


def _unique_strings(value: Any, *, limit: int = 64) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected an array of strings")
    output: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in output:
            output.append(text[:300])
        if len(output) >= limit:
            break
    return tuple(output)


@dataclass(frozen=True)
class TaskBudget:
    max_steps: int = 96
    max_runtime_seconds: int = 7 * 24 * 3600
    max_artifact_bytes: int = 512 * 1024 * 1024
    max_total_attempts: int = 256
    max_inline_artifact_bytes: int = 512 * 1024
    max_events: int = 50_000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TaskBudget":
        raw = dict(value or {})
        return cls(
            max_steps=max(1, min(int(raw.get("max_steps", cls.max_steps)), 512)),
            max_runtime_seconds=max(
                60,
                min(int(raw.get("max_runtime_seconds", cls.max_runtime_seconds)), 30 * 24 * 3600),
            ),
            max_artifact_bytes=max(
                1_048_576,
                min(int(raw.get("max_artifact_bytes", cls.max_artifact_bytes)), 20 * 1024**3),
            ),
            max_total_attempts=max(
                1,
                min(int(raw.get("max_total_attempts", cls.max_total_attempts)), 4096),
            ),
            max_inline_artifact_bytes=max(
                1024,
                min(
                    int(raw.get("max_inline_artifact_bytes", cls.max_inline_artifact_bytes)),
                    8 * 1024 * 1024,
                ),
            ),
            max_events=max(100, min(int(raw.get("max_events", cls.max_events)), 1_000_000)),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRequirement:
    name: str
    media_type: str = "application/octet-stream"
    required: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactRequirement":
        return cls(
            name=_safe_file_name(str(value.get("name") or "artifact")),
            media_type=str(value.get("media_type") or "application/octet-stream")[:160],
            required=bool(value.get("required", True)),
        )


@dataclass(frozen=True)
class TaskStepSpec:
    step_id: str
    title: str
    objective: str
    kind: str = "other"
    dependencies: tuple[str, ...] = ()
    priority: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 3600
    required_capabilities: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRequirement, ...] = ()
    validation: Mapping[str, Any] = field(default_factory=dict)
    reasoning_depth: str = "standard"
    risk_class: int = 1
    approval_required: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "TaskStepSpec":
        step_id = str(value.get("step_id") or value.get("id") or f"task-{index + 1}")
        step_id = _require_identifier(step_id, "step_id")
        title = str(value.get("title") or value.get("name") or step_id).strip()
        objective = str(value.get("objective") or value.get("description") or "").strip()
        if not objective:
            raise ValueError(f"step {step_id} requires an objective")
        kind = str(value.get("kind") or "other").strip().lower()
        if kind not in TASK_KINDS:
            raise ValueError(f"unsupported task kind for {step_id}: {kind}")
        risk = max(0, min(int(value.get("risk_class", 1)), 5))
        artifacts_raw = value.get("artifacts") or []
        if not isinstance(artifacts_raw, list):
            raise TypeError(f"step {step_id} artifacts must be an array")
        return cls(
            step_id=step_id,
            title=title[:240],
            objective=objective[:20_000],
            kind=kind,
            dependencies=_unique_strings(value.get("dependencies") or (), limit=128),
            priority=max(-1000, min(int(value.get("priority", 0)), 1000)),
            max_attempts=max(1, min(int(value.get("max_attempts", 3)), 20)),
            timeout_seconds=max(30, min(int(value.get("timeout_seconds", 3600)), 7 * 24 * 3600)),
            required_capabilities=_unique_strings(value.get("required_capabilities") or (), limit=64),
            artifacts=tuple(ArtifactRequirement.from_mapping(item) for item in artifacts_raw),
            validation=dict(value.get("validation") or {}),
            reasoning_depth=str(value.get("reasoning_depth") or "standard")[:40],
            risk_class=risk,
            approval_required=bool(value.get("approval_required", risk >= 3)),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [asdict(item) for item in self.artifacts]
        return value


@dataclass(frozen=True)
class CompiledPlan:
    objective: str
    quality_mode: str
    steps: tuple[TaskStepSpec, ...]
    reasoning_summary: tuple[str, ...]
    assumptions: tuple[str, ...]
    plan_sha256: str
    schema: str = PLAN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objective": self.objective,
            "quality_mode": self.quality_mode,
            "steps": [item.to_dict() for item in self.steps],
            "reasoning_summary": list(self.reasoning_summary),
            "assumptions": list(self.assumptions),
            "plan_sha256": self.plan_sha256,
            "private_chain_of_thought_exported": False,
        }


class TaskPlanningUnavailable(RuntimeError):
    pass


class TaskPlanner(Protocol):
    def plan(self, objective: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CouncilTaskPlanner:
    """Use the configured Auro-2B council only for structured task planning."""

    def __init__(self, council_service: Any):
        self.council_service = council_service

    def plan(self, objective: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not getattr(self.council_service, "configured", False):
            raise TaskPlanningUnavailable("Auro-2B council is not configured for task planning")
        prompt = {
            "instruction": (
                "Create an executable dependency graph. Return JSON only with keys tasks, "
                "reasoning_summary, assumptions. Each task needs step_id, title, objective, kind, "
                "dependencies, required_capabilities, artifacts, validation, max_attempts, "
                "timeout_seconds, risk_class, approval_required, reasoning_depth. Do not include "
                "hidden chain-of-thought or claim that any task executed."
            ),
            "objective": objective,
            "requested_deliverables": payload.get("deliverables") or [],
            "constraints": payload.get("constraints") or [],
            "quality_mode": payload.get("quality_mode", "deep"),
        }
        response = self.council_service.respond(json.dumps(prompt, ensure_ascii=False))
        candidates = [
            response.get("structured_answer"),
            response.get("text"),
        ]
        for candidate in candidates:
            if isinstance(candidate, Mapping) and isinstance(candidate.get("tasks"), list):
                return dict(candidate)
            if isinstance(candidate, str):
                raw = candidate.strip()
                if "{" in raw and "}" in raw:
                    raw = raw[raw.find("{") : raw.rfind("}") + 1]
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, Mapping) and isinstance(parsed.get("tasks"), list):
                    return dict(parsed)
        raise TaskPlanningUnavailable("council did not return a valid task-plan contract")


def _validate_dag(steps: Sequence[TaskStepSpec]) -> None:
    by_id = {item.step_id: item for item in steps}
    if len(by_id) != len(steps):
        raise ValueError("task step IDs must be unique")
    for item in steps:
        unknown = sorted(set(item.dependencies) - set(by_id))
        if unknown:
            raise ValueError(f"step {item.step_id} has unknown dependencies: {unknown}")
        if item.step_id in item.dependencies:
            raise ValueError(f"step {item.step_id} cannot depend on itself")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError(f"task dependency cycle detected at {step_id}")
        visiting.add(step_id)
        for dependency in by_id[step_id].dependencies:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in by_id:
        visit(step_id)


def _terminal_steps(steps: Sequence[TaskStepSpec]) -> list[str]:
    depended_on = {dependency for item in steps for dependency in item.dependencies}
    return [item.step_id for item in steps if item.step_id not in depended_on]


def _expand_quality(steps: Sequence[TaskStepSpec], mode: str) -> tuple[TaskStepSpec, ...]:
    output = list(steps)
    review_ids: list[str] = []
    if mode in {"deep", "exhaustive"}:
        for item in steps:
            if item.kind in {"review", "synthesize", "package"}:
                continue
            review_id = _require_identifier(f"review:{item.step_id}", "review step_id")
            review_ids.append(review_id)
            output.append(
                TaskStepSpec(
                    step_id=review_id,
                    title=f"Review: {item.title}",
                    objective=(
                        f"Independently review the output of {item.step_id}. Check correctness, "
                        "evidence, completeness, security, contradictions, and artifact validity."
                    ),
                    kind="review",
                    dependencies=(item.step_id,),
                    priority=item.priority - 1,
                    max_attempts=2,
                    timeout_seconds=min(item.timeout_seconds, 3600),
                    required_capabilities=("review",),
                    artifacts=(
                        ArtifactRequirement(f"{item.step_id}-review.json", "application/json", True),
                    ),
                    validation={"require_artifacts": True},
                    reasoning_depth="deep",
                    risk_class=1,
                    approval_required=False,
                    metadata={"reviews_step": item.step_id},
                )
            )

    dependencies = tuple(review_ids or _terminal_steps(steps))
    if mode in {"standard", "deep", "exhaustive"}:
        output.append(
            TaskStepSpec(
                step_id="final-synthesis",
                title="Final synthesis",
                objective=(
                    "Integrate all completed work into one coherent delivery. Preserve material "
                    "disagreement, state limitations, cite artifacts, and produce a concise "
                    "reasoning summary without hidden chain-of-thought."
                ),
                kind="synthesize",
                dependencies=dependencies,
                priority=-50,
                max_attempts=3,
                timeout_seconds=7200,
                required_capabilities=("synthesis",),
                artifacts=(
                    ArtifactRequirement("FINAL_REPORT.md", "text/markdown", True),
                    ArtifactRequirement("DELIVERY_INDEX.json", "application/json", True),
                ),
                validation={"require_artifacts": True},
                reasoning_depth="deep" if mode != "standard" else "standard",
                risk_class=1,
                approval_required=False,
                metadata={"system_generated": True},
            )
        )

    if mode == "exhaustive":
        output.append(
            TaskStepSpec(
                step_id="delivery-quality-gate",
                title="Delivery quality gate",
                objective=(
                    "Validate the complete delivery against the original objective, required "
                    "artifacts, evidence references, tests, security boundaries, and claim limits."
                ),
                kind="review",
                dependencies=("final-synthesis",),
                priority=-75,
                max_attempts=2,
                timeout_seconds=3600,
                required_capabilities=("review", "artifact-validation"),
                artifacts=(
                    ArtifactRequirement("QUALITY_GATE.json", "application/json", True),
                ),
                validation={"require_artifacts": True},
                reasoning_depth="exhaustive",
                risk_class=1,
                approval_required=False,
                metadata={"system_generated": True, "final_gate": True},
            )
        )
    return tuple(output)


def compile_plan(
    payload: Mapping[str, Any],
    *,
    planner: TaskPlanner | None = None,
) -> CompiledPlan:
    objective = str(payload.get("objective") or "").strip()
    if not objective:
        raise ValueError("task run requires an objective")
    quality_mode = str(payload.get("quality_mode") or "deep").strip().lower()
    if quality_mode not in QUALITY_MODES:
        raise ValueError(f"quality_mode must be one of {sorted(QUALITY_MODES)}")

    tasks = payload.get("tasks")
    planning_payload: Mapping[str, Any] = payload
    if tasks is None:
        deliverables = payload.get("deliverables")
        if isinstance(deliverables, list) and deliverables:
            tasks = [
                {
                    "step_id": f"deliverable-{index + 1}",
                    "title": str(item.get("title") if isinstance(item, Mapping) else item),
                    "objective": str(item.get("objective") if isinstance(item, Mapping) else item),
                    "kind": str(item.get("kind", "other")) if isinstance(item, Mapping) else "other",
                    "dependencies": list(item.get("dependencies", [])) if isinstance(item, Mapping) else [],
                    "artifacts": list(item.get("artifacts", [])) if isinstance(item, Mapping) else [],
                    "required_capabilities": list(item.get("required_capabilities", [])) if isinstance(item, Mapping) else [],
                    "risk_class": int(item.get("risk_class", 1)) if isinstance(item, Mapping) else 1,
                }
                for index, item in enumerate(deliverables)
            ]
        elif bool(payload.get("plan_with_council", False)):
            if planner is None:
                raise TaskPlanningUnavailable("plan_with_council requested but no planner is configured")
            planning_payload = planner.plan(objective, payload)
            tasks = planning_payload.get("tasks")
        else:
            tasks = [
                {
                    "step_id": "primary-task",
                    "title": "Primary task",
                    "objective": objective,
                    "kind": "other",
                }
            ]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task plan must contain at least one task")

    base = tuple(TaskStepSpec.from_mapping(dict(item), index) for index, item in enumerate(tasks))
    _validate_dag(base)
    expanded = _expand_quality(base, quality_mode)
    _validate_dag(expanded)

    budget = TaskBudget.from_mapping(payload.get("budget") if isinstance(payload.get("budget"), Mapping) else None)
    if len(expanded) > budget.max_steps:
        raise ValueError(f"compiled plan has {len(expanded)} steps; budget allows {budget.max_steps}")

    reasoning_summary = _unique_strings(
        planning_payload.get("reasoning_summary")
        or [
            f"Compiled {len(base)} requested tasks into {len(expanded)} dependency-aware steps.",
            f"Quality mode is {quality_mode}; review and synthesis stages are explicit tasks.",
            "Workers receive bounded step inputs and artifact contracts rather than unrestricted run state.",
        ],
        limit=32,
    )
    assumptions = _unique_strings(
        planning_payload.get("assumptions")
        or payload.get("assumptions")
        or [
            "External execution requires a separately authorized worker.",
            "Completion requires declared artifact and validation evidence.",
        ],
        limit=32,
    )
    material = {
        "schema": PLAN_SCHEMA,
        "objective": objective,
        "quality_mode": quality_mode,
        "steps": [item.to_dict() for item in expanded],
        "reasoning_summary": list(reasoning_summary),
        "assumptions": list(assumptions),
    }
    return CompiledPlan(
        objective=objective,
        quality_mode=quality_mode,
        steps=expanded,
        reasoning_summary=reasoning_summary,
        assumptions=assumptions,
        plan_sha256=_sha(material),
    )


class DurableTaskRuntime:
    """SQLite/WAL DAG scheduler with leases, retries, events, and artifacts."""

    def __init__(
        self,
        database: str | Path = "state/task-runs.sqlite3",
        artifact_root: str | Path = "state/task-artifacts",
        *,
        signing_key: str | bytes | None = None,
        signer_id: str = "auro-task-runtime",
        planner: TaskPlanner | None = None,
        approval_verifier: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        raw_key = signing_key if signing_key is not None else os.getenv("AURO_TASK_RECEIPT_HMAC_KEY", "")
        self.signing_key = raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key
        self.signer_id = signer_id
        self.planner = planner
        self.approval_verifier = approval_verifier
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.database, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self._schema()

    def _schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
              run_id TEXT PRIMARY KEY,
              principal_id TEXT NOT NULL,
              organization_id TEXT,
              objective TEXT NOT NULL,
              quality_mode TEXT NOT NULL,
              status TEXT NOT NULL,
              plan_json TEXT NOT NULL,
              plan_sha256 TEXT NOT NULL,
              budget_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              result_json TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              paused INTEGER NOT NULL DEFAULT 0,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              completed_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_task_runs_scope
              ON task_runs(organization_id, principal_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS task_steps (
              step_id TEXT NOT NULL,
              run_id TEXT NOT NULL REFERENCES task_runs(run_id) ON DELETE CASCADE,
              title TEXT NOT NULL,
              objective TEXT NOT NULL,
              kind TEXT NOT NULL,
              dependencies_json TEXT NOT NULL,
              priority INTEGER NOT NULL,
              status TEXT NOT NULL,
              max_attempts INTEGER NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              timeout_seconds INTEGER NOT NULL,
              required_capabilities_json TEXT NOT NULL,
              artifact_contract_json TEXT NOT NULL,
              validation_contract_json TEXT NOT NULL,
              reasoning_depth TEXT NOT NULL,
              risk_class INTEGER NOT NULL,
              approval_required INTEGER NOT NULL,
              approval_json TEXT,
              available_at INTEGER NOT NULL,
              lease_owner TEXT,
              lease_token_sha256 TEXT,
              lease_expires_at INTEGER,
              heartbeat_at INTEGER,
              output_json TEXT,
              validation_json TEXT,
              error TEXT,
              metadata_json TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              completed_at INTEGER,
              PRIMARY KEY(run_id, step_id)
            );
            CREATE INDEX IF NOT EXISTS idx_task_steps_sched
              ON task_steps(run_id, status, available_at, priority DESC, created_at);

            CREATE TABLE IF NOT EXISTS task_artifacts (
              artifact_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL REFERENCES task_runs(run_id) ON DELETE CASCADE,
              step_id TEXT NOT NULL,
              name TEXT NOT NULL,
              relative_path TEXT NOT NULL,
              media_type TEXT NOT NULL,
              bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              UNIQUE(run_id, step_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_task_artifacts_run
              ON task_artifacts(run_id, step_id, created_at);

            CREATE TABLE IF NOT EXISTS task_events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL REFERENCES task_runs(run_id) ON DELETE CASCADE,
              step_id TEXT,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              previous_hash TEXT,
              event_hash TEXT NOT NULL,
              created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_task_events_run
              ON task_events(run_id, sequence);
            """
        )
        self.db.commit()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self.db.rollback()
                raise
            else:
                self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def create_run(
        self,
        payload: Mapping[str, Any],
        *,
        principal_id: str,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        principal = _require_identifier(principal_id, "principal_id")
        organization = (
            _require_identifier(organization_id, "organization_id")
            if organization_id
            else None
        )
        plan = compile_plan(payload, planner=self.planner)
        budget = TaskBudget.from_mapping(
            payload.get("budget") if isinstance(payload.get("budget"), Mapping) else None
        )
        run_id = _require_identifier(
            str(payload.get("run_id") or f"run-{uuid.uuid4().hex}"),
            "run_id",
        )
        now = _now()
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "deliverables": payload.get("deliverables") or [],
                "constraints": payload.get("constraints") or [],
                "private_chain_of_thought_exported": False,
            }
        )
        with self._transaction():
            self.db.execute(
                """INSERT INTO task_runs
                   (run_id, principal_id, organization_id, objective, quality_mode,
                    status, plan_json, plan_sha256, budget_json, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    principal,
                    organization,
                    plan.objective,
                    plan.quality_mode,
                    _json(plan.to_dict()),
                    plan.plan_sha256,
                    _json(budget.to_dict()),
                    _json(metadata),
                    now,
                    now,
                ),
            )
            for step in plan.steps:
                initial = "awaiting_approval" if step.approval_required else "pending"
                self.db.execute(
                    """INSERT INTO task_steps
                       (step_id, run_id, title, objective, kind, dependencies_json,
                        priority, status, max_attempts, timeout_seconds,
                        required_capabilities_json, artifact_contract_json,
                        validation_contract_json, reasoning_depth, risk_class,
                        approval_required, available_at, metadata_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        step.step_id,
                        run_id,
                        step.title,
                        step.objective,
                        step.kind,
                        _json(list(step.dependencies)),
                        step.priority,
                        initial,
                        step.max_attempts,
                        step.timeout_seconds,
                        _json(list(step.required_capabilities)),
                        _json([asdict(item) for item in step.artifacts]),
                        _json(dict(step.validation)),
                        step.reasoning_depth,
                        step.risk_class,
                        int(step.approval_required),
                        now,
                        _json(dict(step.metadata)),
                        now,
                        now,
                    ),
                )
            self._event_locked(
                run_id,
                None,
                "run.created",
                {
                    "schema": RUN_SCHEMA,
                    "plan_sha256": plan.plan_sha256,
                    "step_count": len(plan.steps),
                    "quality_mode": plan.quality_mode,
                    "reasoning_summary": list(plan.reasoning_summary),
                },
            )
            self._refresh_locked(run_id, now)
        return self.get_run(run_id, principal_id=principal, organization_id=organization)

    def _authorize_scope(
        self,
        row: sqlite3.Row,
        principal_id: str,
        organization_id: str | None,
    ) -> None:
        principal = _require_identifier(principal_id, "principal_id")
        organization = organization_id or None
        if row["principal_id"] == principal:
            return
        if organization and row["organization_id"] == organization:
            return
        raise PermissionError("task run belongs to another principal or organization")

    def list_runs(
        self,
        *,
        principal_id: str,
        organization_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        principal = _require_identifier(principal_id, "principal_id")
        limit = max(1, min(int(limit), 200))
        with self._lock:
            if organization_id:
                organization = _require_identifier(organization_id, "organization_id")
                rows = self.db.execute(
                    """SELECT * FROM task_runs
                       WHERE principal_id=? OR organization_id=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (principal, organization, limit),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT * FROM task_runs WHERE principal_id=? ORDER BY created_at DESC LIMIT ?",
                    (principal, limit),
                ).fetchall()
        return [self._run_public(row, include_steps=False) for row in rows]

    def get_run(
        self,
        run_id: str,
        *,
        principal_id: str,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = _require_identifier(run_id, "run_id")
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM task_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError("unknown task run")
            self._authorize_scope(row, principal_id, organization_id)
            steps = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? ORDER BY priority DESC, created_at, step_id",
                (run_id,),
            ).fetchall()
            artifacts = self.db.execute(
                "SELECT * FROM task_artifacts WHERE run_id=? ORDER BY created_at, artifact_id",
                (run_id,),
            ).fetchall()
        value = self._run_public(row, include_steps=False)
        value["steps"] = [self._step_public(item, include_lease=False) for item in steps]
        value["artifacts"] = [self._artifact_public(item) for item in artifacts]
        value["progress"] = self._progress(steps)
        return value

    def events(
        self,
        run_id: str,
        *,
        principal_id: str,
        organization_id: str | None = None,
        after_sequence: int = 0,
        limit: int = 1000,
    ) -> dict[str, Any]:
        self.get_run(run_id, principal_id=principal_id, organization_id=organization_id)
        with self._lock:
            rows = self.db.execute(
                """SELECT * FROM task_events WHERE run_id=? AND sequence>?
                   ORDER BY sequence LIMIT ?""",
                (run_id, max(0, int(after_sequence)), max(1, min(int(limit), 5000))),
            ).fetchall()
        events = [
            {
                "schema": EVENT_SCHEMA,
                "sequence": row["sequence"],
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "event_type": row["event_type"],
                "payload": _load(row["payload_json"], {}),
                "previous_hash": row["previous_hash"],
                "event_hash": row["event_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {
            "schema": "auro.task-events.v1",
            "run_id": run_id,
            "events": events,
            "chain": self.verify_event_chain(run_id),
        }

    def claim_step(
        self,
        run_id: str,
        *,
        worker_id: str,
        capabilities: Sequence[str] = (),
        lease_seconds: int = 900,
    ) -> dict[str, Any] | None:
        run_id = _require_identifier(run_id, "run_id")
        worker_id = _require_identifier(worker_id, "worker_id")
        capability_set = {str(item) for item in capabilities}
        now = _now()
        lease_seconds = max(30, min(int(lease_seconds), 24 * 3600))
        token = secrets.token_urlsafe(32)
        token_sha = _sha_bytes(token.encode("utf-8"))
        with self._transaction():
            self._recover_expired_locked(run_id, now)
            self._refresh_locked(run_id, now)
            run = self.db.execute("SELECT * FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("unknown task run")
            if run["status"] in RUN_TERMINAL or run["paused"] or run["cancel_requested"]:
                return None
            budget = TaskBudget.from_mapping(_load(run["budget_json"], {}))
            total_attempts = int(
                self.db.execute(
                    "SELECT COALESCE(SUM(attempts),0) FROM task_steps WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            if total_attempts >= budget.max_total_attempts:
                self._fail_run_budget_locked(run_id, "maximum total attempts exceeded", now)
                return None
            if now - int(run["created_at"]) > budget.max_runtime_seconds:
                self._fail_run_budget_locked(run_id, "maximum runtime exceeded", now)
                return None

            candidates = self.db.execute(
                """SELECT * FROM task_steps
                   WHERE run_id=? AND status='ready' AND available_at<=?
                   ORDER BY priority DESC, created_at, step_id""",
                (run_id, now),
            ).fetchall()
            selected = None
            for item in candidates:
                required = set(_load(item["required_capabilities_json"], []))
                if required.issubset(capability_set):
                    selected = item
                    break
            if selected is None:
                return None
            expires = now + min(lease_seconds, int(selected["timeout_seconds"]))
            cursor = self.db.execute(
                """UPDATE task_steps SET status='leased', attempts=attempts+1,
                   lease_owner=?, lease_token_sha256=?, lease_expires_at=?, heartbeat_at=?,
                   updated_at=? WHERE run_id=? AND step_id=? AND status='ready'""",
                (
                    worker_id,
                    token_sha,
                    expires,
                    now,
                    now,
                    run_id,
                    selected["step_id"],
                ),
            )
            if cursor.rowcount != 1:
                return None
            self.db.execute(
                "UPDATE task_runs SET status='running', updated_at=? WHERE run_id=?",
                (now, run_id),
            )
            self._event_locked(
                run_id,
                selected["step_id"],
                "step.leased",
                {
                    "worker_id": worker_id,
                    "lease_expires_at": expires,
                    "attempt": int(selected["attempts"]) + 1,
                    "capabilities": sorted(capability_set),
                },
            )
            row = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
                (run_id, selected["step_id"]),
            ).fetchone()
        value = self._step_public(row, include_lease=True)
        value["lease_token"] = token
        value["workspace"] = str(self.workspace_path(run_id, row["step_id"]))
        return value

    def heartbeat(
        self,
        run_id: str,
        step_id: str,
        *,
        worker_id: str,
        lease_token: str,
        extend_seconds: int = 900,
    ) -> dict[str, Any]:
        now = _now()
        with self._transaction():
            row = self._leased_step_locked(run_id, step_id, worker_id, lease_token, now)
            expires = now + min(
                max(30, int(extend_seconds)),
                int(row["timeout_seconds"]),
                24 * 3600,
            )
            self.db.execute(
                """UPDATE task_steps SET status='running', heartbeat_at=?, lease_expires_at=?,
                   updated_at=? WHERE run_id=? AND step_id=?""",
                (now, expires, now, run_id, step_id),
            )
            self._event_locked(
                run_id,
                step_id,
                "step.heartbeat",
                {"worker_id": worker_id, "lease_expires_at": expires},
            )
            updated = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
                (run_id, step_id),
            ).fetchone()
        return self._step_public(updated, include_lease=True)

    def progress(
        self,
        run_id: str,
        step_id: str,
        *,
        worker_id: str,
        lease_token: str,
        progress: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        with self._transaction():
            self._leased_step_locked(run_id, step_id, worker_id, lease_token, now)
            bounded = dict(progress)
            encoded = _canonical(bounded)
            if len(encoded) > 128 * 1024:
                raise ValueError("progress event exceeds 128 KiB")
            self._event_locked(run_id, step_id, "step.progress", bounded)
            self.db.execute(
                "UPDATE task_steps SET status='running', heartbeat_at=?, updated_at=? WHERE run_id=? AND step_id=?",
                (now, now, run_id, step_id),
            )
        return {"ok": True, "run_id": run_id, "step_id": step_id, "observed_at": now}

    def complete_step(
        self,
        run_id: str,
        step_id: str,
        *,
        worker_id: str,
        lease_token: str,
        output: Mapping[str, Any] | None = None,
        artifacts: Sequence[Mapping[str, Any]] = (),
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self._transaction():
            row = self._leased_step_locked(run_id, step_id, worker_id, lease_token, now)
            artifact_rows = self._store_artifacts_locked(
                run_id,
                step_id,
                artifacts,
                run_budget=self._budget_for_run_locked(run_id),
            )
            errors = self._validate_completion(row, output or {}, artifact_rows, validation or {})
            if errors:
                retry = int(row["attempts"]) < int(row["max_attempts"])
                status = "retry_wait" if retry else "failed"
                available_at = now + min(300, 5 * 2 ** max(0, int(row["attempts"]) - 1))
                self.db.execute(
                    """UPDATE task_steps SET status=?, available_at=?, output_json=?, validation_json=?,
                       error=?, lease_owner=NULL, lease_token_sha256=NULL, lease_expires_at=NULL,
                       heartbeat_at=NULL, updated_at=?, completed_at=? WHERE run_id=? AND step_id=?""",
                    (
                        status,
                        available_at,
                        _json(dict(output or {})),
                        _json({"valid": False, "errors": errors, **dict(validation or {})}),
                        "; ".join(errors)[:4000],
                        now,
                        None if retry else now,
                        run_id,
                        step_id,
                    ),
                )
                self._event_locked(
                    run_id,
                    step_id,
                    "step.validation_failed",
                    {"errors": errors, "retry": retry, "available_at": available_at},
                )
            else:
                self.db.execute(
                    """UPDATE task_steps SET status='succeeded', output_json=?, validation_json=?,
                       error=NULL, lease_owner=NULL, lease_token_sha256=NULL,
                       lease_expires_at=NULL, heartbeat_at=NULL, updated_at=?, completed_at=?
                       WHERE run_id=? AND step_id=?""",
                    (
                        _json(dict(output or {})),
                        _json({"valid": True, **dict(validation or {})}),
                        now,
                        now,
                        run_id,
                        step_id,
                    ),
                )
                self._event_locked(
                    run_id,
                    step_id,
                    "step.succeeded",
                    {
                        "output_sha256": _sha(dict(output or {})),
                        "artifacts": [item["artifact_id"] for item in artifact_rows],
                    },
                )
            self._refresh_locked(run_id, now)
            updated = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
                (run_id, step_id),
            ).fetchone()
        return self._step_public(updated, include_lease=False)

    def fail_step(
        self,
        run_id: str,
        step_id: str,
        *,
        worker_id: str,
        lease_token: str,
        error: str,
        retry_delay_seconds: int = 30,
    ) -> dict[str, Any]:
        now = _now()
        with self._transaction():
            row = self._leased_step_locked(run_id, step_id, worker_id, lease_token, now)
            retry = int(row["attempts"]) < int(row["max_attempts"])
            status = "retry_wait" if retry else "failed"
            available_at = now + max(0, min(int(retry_delay_seconds), 3600))
            self.db.execute(
                """UPDATE task_steps SET status=?, available_at=?, error=?, lease_owner=NULL,
                   lease_token_sha256=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                   updated_at=?, completed_at=? WHERE run_id=? AND step_id=?""",
                (
                    status,
                    available_at,
                    str(error)[:4000],
                    now,
                    None if retry else now,
                    run_id,
                    step_id,
                ),
            )
            self._event_locked(
                run_id,
                step_id,
                "step.failed" if not retry else "step.retry_scheduled",
                {"error": str(error)[:1000], "retry": retry, "available_at": available_at},
            )
            self._refresh_locked(run_id, now)
            updated = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
                (run_id, step_id),
            ).fetchone()
        return self._step_public(updated, include_lease=False)

    def record_verified_approval(
        self,
        run_id: str,
        step_id: str,
        approval: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.approval_verifier is None:
            raise PermissionError("no trusted approval verifier is configured")
        now = _now()
        with self._transaction():
            row = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
                (run_id, step_id),
            ).fetchone()
            if row is None:
                raise KeyError("unknown task step")
            action = self._approval_action(row)
            if not self.approval_verifier(dict(approval), action):
                raise PermissionError("approval verification failed")
            self.db.execute(
                "UPDATE task_steps SET approval_json=?, status='pending', updated_at=? WHERE run_id=? AND step_id=?",
                (_json(dict(approval)), now, run_id, step_id),
            )
            self._event_locked(
                run_id,
                step_id,
                "step.approved",
                {
                    "approval_id": approval.get("approval_id"),
                    "approval_receipt_sha256": approval.get("receipt_sha256"),
                    "action_hash": _sha(action),
                },
            )
            self._refresh_locked(run_id, now)
            updated = self.db.execute(
                "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
                (run_id, step_id),
            ).fetchone()
        return self._step_public(updated, include_lease=False)

    def pause(self, run_id: str, *, principal_id: str, organization_id: str | None = None) -> dict[str, Any]:
        return self._set_run_flag(run_id, principal_id, organization_id, paused=True)

    def resume(self, run_id: str, *, principal_id: str, organization_id: str | None = None) -> dict[str, Any]:
        return self._set_run_flag(run_id, principal_id, organization_id, paused=False)

    def cancel(self, run_id: str, *, principal_id: str, organization_id: str | None = None) -> dict[str, Any]:
        self.get_run(run_id, principal_id=principal_id, organization_id=organization_id)
        now = _now()
        with self._transaction():
            self.db.execute(
                "UPDATE task_runs SET cancel_requested=1, status='cancelled', updated_at=?, completed_at=? WHERE run_id=?",
                (now, now, run_id),
            )
            self.db.execute(
                """UPDATE task_steps SET status='cancelled', updated_at=?, completed_at=?,
                   lease_owner=NULL, lease_token_sha256=NULL, lease_expires_at=NULL
                   WHERE run_id=? AND status NOT IN ('succeeded','failed','cancelled','skipped','blocked')""",
                (now, now, run_id),
            )
            self._event_locked(run_id, None, "run.cancelled", {})
            self._finalize_locked(run_id, now)
        return self.get_run(run_id, principal_id=principal_id, organization_id=organization_id)

    def _set_run_flag(
        self,
        run_id: str,
        principal_id: str,
        organization_id: str | None,
        *,
        paused: bool,
    ) -> dict[str, Any]:
        current = self.get_run(run_id, principal_id=principal_id, organization_id=organization_id)
        if current["status"] in RUN_TERMINAL:
            return current
        now = _now()
        with self._transaction():
            self.db.execute(
                "UPDATE task_runs SET paused=?, status=?, updated_at=? WHERE run_id=?",
                (int(paused), "paused" if paused else "queued", now, run_id),
            )
            self._event_locked(run_id, None, "run.paused" if paused else "run.resumed", {})
            if not paused:
                self._refresh_locked(run_id, now)
        return self.get_run(run_id, principal_id=principal_id, organization_id=organization_id)

    def workspace_path(self, run_id: str, step_id: str) -> Path:
        run_id = _require_identifier(run_id, "run_id")
        step_id = _require_identifier(step_id, "step_id")
        path = self.artifact_root / run_id / "workspace" / _safe_file_name(step_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_bundle(
        self,
        run_id: str,
        *,
        principal_id: str,
        organization_id: str | None = None,
    ) -> Path:
        state = self.get_run(run_id, principal_id=principal_id, organization_id=organization_id)
        events = self.events(
            run_id,
            principal_id=principal_id,
            organization_id=organization_id,
            limit=50_000,
        )
        root = self.artifact_root / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "RUN_STATE.json").write_text(
            json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (root / "EVENTS.json").write_text(
            json.dumps(events, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        bundle = root / f"{run_id}-delivery.zip"
        temporary = bundle.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                if path in {bundle, temporary}:
                    continue
                archive.write(path, path.relative_to(root))
        temporary.replace(bundle)
        return bundle

    def verify_event_chain(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM task_events WHERE run_id=? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        previous = None
        for row in rows:
            material = {
                "schema": EVENT_SCHEMA,
                "run_id": row["run_id"],
                "step_id": row["step_id"],
                "event_type": row["event_type"],
                "payload": _load(row["payload_json"], {}),
                "previous_hash": previous,
                "created_at": row["created_at"],
            }
            expected = _sha(material)
            if row["previous_hash"] != previous or row["event_hash"] != expected:
                return {
                    "valid": False,
                    "events": len(rows),
                    "failed_sequence": row["sequence"],
                    "expected": expected,
                    "actual": row["event_hash"],
                }
            previous = row["event_hash"]
        return {"valid": True, "events": len(rows), "root_hash": previous}

    def status(self) -> dict[str, Any]:
        with self._lock:
            run_counts = {
                row["status"]: int(row["count"])
                for row in self.db.execute(
                    "SELECT status, COUNT(*) AS count FROM task_runs GROUP BY status"
                ).fetchall()
            }
            step_counts = {
                row["status"]: int(row["count"])
                for row in self.db.execute(
                    "SELECT status, COUNT(*) AS count FROM task_steps GROUP BY status"
                ).fetchall()
            }
        return {
            "schema": "auro.task-runtime.status.v1",
            "database": str(self.database),
            "artifact_root": str(self.artifact_root),
            "runs": run_counts,
            "steps": step_counts,
            "quality_modes": sorted(QUALITY_MODES),
            "signed_receipts": bool(self.signing_key),
            "planner_configured": self.planner is not None,
            "approval_verifier_configured": self.approval_verifier is not None,
            "capabilities": [
                "multiple-dependent-tasks",
                "long-running-leases",
                "pause-resume-cancel",
                "bounded-retries",
                "deep-review-graphs",
                "artifact-delivery",
                "tamper-evident-events",
                "principal-and-organization-scope",
            ],
            "claim_boundary": (
                "runtime source and local tests do not prove an external worker executed, "
                "a model reasoned correctly, or an artifact is useful until validation evidence exists"
            ),
        }

    def _event_locked(
        self,
        run_id: str,
        step_id: str | None,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> str:
        run = self.db.execute("SELECT budget_json FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise KeyError("unknown task run")
        budget = TaskBudget.from_mapping(_load(run["budget_json"], {}))
        count = int(
            self.db.execute("SELECT COUNT(*) FROM task_events WHERE run_id=?", (run_id,)).fetchone()[0]
        )
        if count >= budget.max_events:
            raise RuntimeError("task event budget exhausted")
        previous_row = self.db.execute(
            "SELECT event_hash FROM task_events WHERE run_id=? ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        previous = previous_row["event_hash"] if previous_row else None
        created_at = _now()
        material = {
            "schema": EVENT_SCHEMA,
            "run_id": run_id,
            "step_id": step_id,
            "event_type": event_type,
            "payload": dict(payload),
            "previous_hash": previous,
            "created_at": created_at,
        }
        event_hash = _sha(material)
        self.db.execute(
            """INSERT INTO task_events
               (run_id, step_id, event_type, payload_json, previous_hash, event_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, step_id, event_type, _json(dict(payload)), previous, event_hash, created_at),
        )
        return event_hash

    def _recover_expired_locked(self, run_id: str, now: int) -> None:
        rows = self.db.execute(
            """SELECT * FROM task_steps WHERE run_id=? AND status IN ('leased','running')
               AND lease_expires_at IS NOT NULL AND lease_expires_at<?""",
            (run_id, now),
        ).fetchall()
        for row in rows:
            retry = int(row["attempts"]) < int(row["max_attempts"])
            status = "retry_wait" if retry else "failed"
            self.db.execute(
                """UPDATE task_steps SET status=?, available_at=?, error=?, lease_owner=NULL,
                   lease_token_sha256=NULL, lease_expires_at=NULL, heartbeat_at=NULL,
                   updated_at=?, completed_at=? WHERE run_id=? AND step_id=?""",
                (
                    status,
                    now + 5,
                    "worker lease expired",
                    now,
                    None if retry else now,
                    run_id,
                    row["step_id"],
                ),
            )
            self._event_locked(
                run_id,
                row["step_id"],
                "step.lease_expired",
                {"worker_id": row["lease_owner"], "retry": retry},
            )

    def _refresh_locked(self, run_id: str, now: int) -> None:
        run = self.db.execute("SELECT * FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise KeyError("unknown task run")
        if run["status"] in RUN_TERMINAL or run["cancel_requested"]:
            return
        if run["paused"]:
            self.db.execute(
                "UPDATE task_runs SET status='paused', updated_at=? WHERE run_id=?",
                (now, run_id),
            )
            return

        steps = self.db.execute(
            "SELECT * FROM task_steps WHERE run_id=? ORDER BY created_at, step_id",
            (run_id,),
        ).fetchall()
        status_by_id = {row["step_id"]: row["status"] for row in steps}
        for row in steps:
            status = row["status"]
            if status in STEP_TERMINAL or status in STEP_ACTIVE or status == "ready":
                continue
            dependencies = _load(row["dependencies_json"], [])
            dependency_statuses = [status_by_id.get(item, "missing") for item in dependencies]
            if any(item in {"failed", "cancelled", "blocked"} for item in dependency_statuses):
                self.db.execute(
                    "UPDATE task_steps SET status='blocked', error=?, updated_at=?, completed_at=? WHERE run_id=? AND step_id=?",
                    ("dependency failed, cancelled, or blocked", now, now, run_id, row["step_id"]),
                )
                status_by_id[row["step_id"]] = "blocked"
                self._event_locked(
                    run_id,
                    row["step_id"],
                    "step.blocked",
                    {"dependency_statuses": dict(zip(dependencies, dependency_statuses))},
                )
                continue
            if not all(item in {"succeeded", "skipped"} for item in dependency_statuses):
                continue
            if row["approval_required"] and not row["approval_json"]:
                if status != "awaiting_approval":
                    self.db.execute(
                        "UPDATE task_steps SET status='awaiting_approval', updated_at=? WHERE run_id=? AND step_id=?",
                        (now, run_id, row["step_id"]),
                    )
                status_by_id[row["step_id"]] = "awaiting_approval"
                continue
            if int(row["available_at"]) <= now:
                self.db.execute(
                    "UPDATE task_steps SET status='ready', updated_at=? WHERE run_id=? AND step_id=?",
                    (now, run_id, row["step_id"]),
                )
                status_by_id[row["step_id"]] = "ready"
                self._event_locked(run_id, row["step_id"], "step.ready", {})

        steps = self.db.execute(
            "SELECT * FROM task_steps WHERE run_id=?",
            (run_id,),
        ).fetchall()
        statuses = [row["status"] for row in steps]
        if statuses and all(item in {"succeeded", "skipped"} for item in statuses):
            new_status = "succeeded"
        elif any(item in STEP_ACTIVE for item in statuses):
            new_status = "running"
        elif any(item == "ready" for item in statuses):
            new_status = "queued"
        elif any(item == "awaiting_approval" for item in statuses):
            new_status = "awaiting_approval"
        elif statuses and all(item in STEP_TERMINAL for item in statuses):
            new_status = "partial" if any(item == "succeeded" for item in statuses) else "failed"
        else:
            new_status = "queued"
        completed_at = now if new_status in RUN_TERMINAL else None
        self.db.execute(
            "UPDATE task_runs SET status=?, updated_at=?, completed_at=? WHERE run_id=?",
            (new_status, now, completed_at, run_id),
        )
        if new_status in RUN_TERMINAL:
            self._event_locked(run_id, None, f"run.{new_status}", {})
            self._finalize_locked(run_id, now)

    def _finalize_locked(self, run_id: str, now: int) -> None:
        run = self.db.execute("SELECT * FROM task_runs WHERE run_id=?", (run_id,)).fetchone()
        steps = self.db.execute("SELECT * FROM task_steps WHERE run_id=? ORDER BY created_at, step_id", (run_id,)).fetchall()
        artifacts = self.db.execute("SELECT * FROM task_artifacts WHERE run_id=? ORDER BY created_at, artifact_id", (run_id,)).fetchall()
        chain = self.verify_event_chain(run_id)
        manifest = {
            "schema": "auro.task-artifact-manifest.v1",
            "run_id": run_id,
            "status": run["status"],
            "plan_sha256": run["plan_sha256"],
            "artifacts": [self._artifact_public(item) for item in artifacts],
            "event_chain": chain,
            "completed_steps": sum(item["status"] == "succeeded" for item in steps),
            "failed_steps": sum(item["status"] in {"failed", "blocked"} for item in steps),
            "created_at": now,
        }
        manifest["manifest_sha256"] = _sha(manifest)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "run_id": run_id,
            "status": run["status"],
            "principal_id": run["principal_id"],
            "organization_id": run["organization_id"],
            "objective_sha256": _sha_bytes(run["objective"].encode("utf-8")),
            "plan_sha256": run["plan_sha256"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "event_root_hash": chain.get("root_hash"),
            "completed_at": now,
            "private_chain_of_thought_exported": False,
        }
        receipt_hash = _sha(receipt)
        signature = (
            hmac.new(self.signing_key, receipt_hash.encode("ascii"), hashlib.sha256).hexdigest()
            if self.signing_key
            else None
        )
        receipt.update(
            {
                "receipt_sha256": receipt_hash,
                "signature": signature,
                "signer_id": self.signer_id if signature else None,
                "evidence_class": "E4-signed-receipt" if signature else "E3-validated-output",
                "external_custody_proven": False,
            }
        )
        root = self.artifact_root / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "ARTIFACT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (root / "RUN_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        result = {
            "artifact_manifest": str(root / "ARTIFACT_MANIFEST.json"),
            "run_receipt": str(root / "RUN_RECEIPT.json"),
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "receipt_sha256": receipt_hash,
            "evidence_class": receipt["evidence_class"],
        }
        self.db.execute(
            "UPDATE task_runs SET result_json=?, updated_at=? WHERE run_id=?",
            (_json(result), now, run_id),
        )

    def _store_artifacts_locked(
        self,
        run_id: str,
        step_id: str,
        artifacts: Sequence[Mapping[str, Any]],
        *,
        run_budget: TaskBudget,
    ) -> list[dict[str, Any]]:
        existing_bytes = int(
            self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM task_artifacts WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
        )
        output: list[dict[str, Any]] = []
        for raw in artifacts:
            item = dict(raw)
            name = _safe_file_name(str(item.get("name") or f"artifact-{len(output)+1}"))
            media_type = str(item.get("media_type") or "application/octet-stream")[:160]
            data: bytes
            if "content" in item:
                data = str(item["content"]).encode("utf-8")
            elif "json" in item:
                data = (json.dumps(item["json"], indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            elif "base64" in item:
                data = base64.b64decode(str(item["base64"]), validate=True)
            elif "workspace_path" in item:
                workspace = self.workspace_path(run_id, step_id).resolve()
                source = (workspace / str(item["workspace_path"])).resolve()
                try:
                    source.relative_to(workspace)
                except ValueError as exc:
                    raise ValueError("artifact workspace path escapes the step workspace") from exc
                if not source.is_file():
                    raise FileNotFoundError(f"artifact workspace file missing: {source}")
                data = source.read_bytes()
            else:
                raise ValueError(f"artifact {name} requires content, json, base64, or workspace_path")
            if len(data) > run_budget.max_inline_artifact_bytes and "workspace_path" not in item:
                raise ValueError(
                    f"inline artifact {name} exceeds {run_budget.max_inline_artifact_bytes} bytes; use workspace_path"
                )
            existing_bytes += len(data)
            if existing_bytes > run_budget.max_artifact_bytes:
                raise RuntimeError("run artifact byte budget exceeded")
            destination = self.artifact_root / run_id / "artifacts" / _safe_file_name(step_id) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            artifact_id = "artifact-" + uuid.uuid4().hex
            sha = _sha_bytes(data)
            metadata = dict(item.get("metadata") or {})
            self.db.execute(
                """INSERT OR REPLACE INTO task_artifacts
                   (artifact_id, run_id, step_id, name, relative_path, media_type,
                    bytes, sha256, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    run_id,
                    step_id,
                    name,
                    str(destination.relative_to(self.artifact_root / run_id)),
                    media_type,
                    len(data),
                    sha,
                    _json(metadata),
                    _now(),
                ),
            )
            output.append(
                {
                    "schema": ARTIFACT_SCHEMA,
                    "artifact_id": artifact_id,
                    "name": name,
                    "media_type": media_type,
                    "bytes": len(data),
                    "sha256": sha,
                    "relative_path": str(destination.relative_to(self.artifact_root / run_id)),
                    "metadata": metadata,
                }
            )
        return output

    @staticmethod
    def _validate_completion(
        row: sqlite3.Row,
        output: Mapping[str, Any],
        artifacts: Sequence[Mapping[str, Any]],
        validation: Mapping[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        contract = _load(row["artifact_contract_json"], [])
        names = {str(item.get("name")) for item in artifacts}
        for requirement in contract:
            if requirement.get("required", True) and str(requirement.get("name")) not in names:
                errors.append(f"required artifact missing: {requirement.get('name')}")
        validation_contract = _load(row["validation_contract_json"], {})
        if validation_contract.get("require_artifacts") and not artifacts:
            errors.append("validation contract requires at least one artifact")
        required_output_keys = validation_contract.get("required_output_keys") or []
        for key in required_output_keys:
            if key not in output:
                errors.append(f"required output key missing: {key}")
        if validation.get("passed") is False:
            errors.append("worker validation explicitly failed")
        return errors

    def _leased_step_locked(
        self,
        run_id: str,
        step_id: str,
        worker_id: str,
        lease_token: str,
        now: int,
    ) -> sqlite3.Row:
        row = self.db.execute(
            "SELECT * FROM task_steps WHERE run_id=? AND step_id=?",
            (run_id, step_id),
        ).fetchone()
        if row is None:
            raise KeyError("unknown task step")
        if row["status"] not in STEP_ACTIVE:
            raise PermissionError("task step is not actively leased")
        if row["lease_owner"] != worker_id:
            raise PermissionError("worker does not own the task lease")
        if not row["lease_token_sha256"] or not hmac.compare_digest(
            str(row["lease_token_sha256"]),
            _sha_bytes(str(lease_token).encode("utf-8")),
        ):
            raise PermissionError("task lease token is invalid")
        if int(row["lease_expires_at"] or 0) < now:
            raise PermissionError("task lease expired")
        return row

    def _budget_for_run_locked(self, run_id: str) -> TaskBudget:
        row = self.db.execute(
            "SELECT budget_json FROM task_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError("unknown task run")
        return TaskBudget.from_mapping(_load(row["budget_json"], {}))

    def _approval_action(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema": "auro.task-step.action.v1",
            "run_id": row["run_id"],
            "step_id": row["step_id"],
            "kind": row["kind"],
            "objective_sha256": _sha_bytes(row["objective"].encode("utf-8")),
            "risk_class": row["risk_class"],
            "required_capabilities": _load(row["required_capabilities_json"], []),
        }

    def _fail_run_budget_locked(self, run_id: str, reason: str, now: int) -> None:
        self.db.execute(
            "UPDATE task_runs SET status='failed', result_json=?, updated_at=?, completed_at=? WHERE run_id=?",
            (_json({"error": reason}), now, now, run_id),
        )
        self.db.execute(
            """UPDATE task_steps SET status='cancelled', error=?, updated_at=?, completed_at=?
               WHERE run_id=? AND status NOT IN ('succeeded','failed','cancelled','skipped','blocked')""",
            (reason, now, now, run_id),
        )
        self._event_locked(run_id, None, "run.budget_failed", {"reason": reason})
        self._finalize_locked(run_id, now)

    @staticmethod
    def _progress(steps: Sequence[sqlite3.Row]) -> dict[str, Any]:
        total = len(steps)
        counts: dict[str, int] = {}
        for row in steps:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        completed = sum(counts.get(item, 0) for item in {"succeeded", "skipped"})
        return {
            "total_steps": total,
            "completed_steps": completed,
            "fraction": round(completed / max(total, 1), 6),
            "status_counts": counts,
        }

    @staticmethod
    def _run_public(row: sqlite3.Row, *, include_steps: bool) -> dict[str, Any]:
        del include_steps
        return {
            "schema": RUN_SCHEMA,
            "run_id": row["run_id"],
            "principal_id": row["principal_id"],
            "organization_id": row["organization_id"],
            "objective": row["objective"],
            "quality_mode": row["quality_mode"],
            "status": row["status"],
            "plan": _load(row["plan_json"], {}),
            "plan_sha256": row["plan_sha256"],
            "budget": _load(row["budget_json"], {}),
            "metadata": _load(row["metadata_json"], {}),
            "result": _load(row["result_json"], None),
            "cancel_requested": bool(row["cancel_requested"]),
            "paused": bool(row["paused"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _step_public(row: sqlite3.Row, *, include_lease: bool) -> dict[str, Any]:
        value = {
            "schema": STEP_SCHEMA,
            "run_id": row["run_id"],
            "step_id": row["step_id"],
            "title": row["title"],
            "objective": row["objective"],
            "kind": row["kind"],
            "dependencies": _load(row["dependencies_json"], []),
            "priority": row["priority"],
            "status": row["status"],
            "max_attempts": row["max_attempts"],
            "attempts": row["attempts"],
            "timeout_seconds": row["timeout_seconds"],
            "required_capabilities": _load(row["required_capabilities_json"], []),
            "artifact_contract": _load(row["artifact_contract_json"], []),
            "validation_contract": _load(row["validation_contract_json"], {}),
            "reasoning_depth": row["reasoning_depth"],
            "risk_class": row["risk_class"],
            "approval_required": bool(row["approval_required"]),
            "approved": bool(row["approval_json"]),
            "output": _load(row["output_json"], None),
            "validation": _load(row["validation_json"], None),
            "error": row["error"],
            "metadata": _load(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }
        if include_lease:
            value.update(
                {
                    "lease_owner": row["lease_owner"],
                    "lease_expires_at": row["lease_expires_at"],
                    "heartbeat_at": row["heartbeat_at"],
                }
            )
        return value

    @staticmethod
    def _artifact_public(row: sqlite3.Row) -> dict[str, Any]:
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
            "metadata": _load(row["metadata_json"], {}),
            "created_at": row["created_at"],
        }


class TaskRuntimeService:
    """Environment-configured production facade around DurableTaskRuntime."""

    def __init__(self, runtime: DurableTaskRuntime):
        self.runtime = runtime

    @classmethod
    def from_env(cls, council_service: Any | None = None) -> "TaskRuntimeService":
        planner = CouncilTaskPlanner(council_service) if council_service is not None else None
        runtime = DurableTaskRuntime(
            database=os.getenv("AURO_TASK_DB", "state/task-runs.sqlite3"),
            artifact_root=os.getenv("AURO_TASK_ARTIFACT_ROOT", "state/task-artifacts"),
            signing_key=os.getenv("AURO_TASK_RECEIPT_HMAC_KEY", "") or None,
            signer_id=os.getenv("AURO_TASK_RECEIPT_SIGNER", "auro-task-runtime"),
            planner=planner,
        )
        return cls(runtime)

    def status(self) -> dict[str, Any]:
        return self.runtime.status()
