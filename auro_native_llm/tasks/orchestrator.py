"""Durable multi-task and long-running mission orchestration for AURO.

The orchestrator increases deliberation through explicit passes, dependencies,
reviews, and synthesis. It records bounded decision summaries rather than hidden
chain-of-thought. Every material deliverable is written as a hashed artifact.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import time
import uuid
from typing import Any, Callable, Mapping, Protocol, Sequence

from .artifacts import ArtifactRecord, ArtifactStore
from .store import MissionStore


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


def _bounded(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "..."


def _safe_task_id(value: str) -> str:
    clean = "".join(character if character.isalnum() or character in "_-" else "-" for character in value)
    clean = clean.strip("-_")
    if not clean:
        clean = uuid.uuid4().hex[:12]
    return clean[:120]


@dataclass(frozen=True)
class TaskExecutionResult:
    summary: str
    output: Mapping[str, Any]
    artifacts: tuple[ArtifactRecord, ...] = ()
    decision: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    accepted: bool = True
    schema: str = "auro.mission.task-result.v1"

    def public(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "summary": self.summary,
            "output": dict(self.output),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "decision": dict(self.decision),
            "metrics": dict(self.metrics),
            "evidence": list(self.evidence),
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }


class TaskExecutor(Protocol):
    def execute(
        self,
        mission: Mapping[str, Any],
        task: Mapping[str, Any],
        dependency_results: Sequence[Mapping[str, Any]],
        progress: Callable[[float, str], None] | None = None,
    ) -> TaskExecutionResult: ...


class MissionPlanner:
    """Normalize explicit tasks or build a deep default dependency graph."""

    KINDS = {"reasoning", "research", "implementation", "artifact", "review", "synthesis", "package"}

    def plan(
        self,
        objective: str,
        *,
        title: str = "AURO mission",
        tasks: Sequence[Mapping[str, Any]] | None = None,
        deliverables: Sequence[Mapping[str, Any] | str] = (),
        operator_id: str = "operator",
        organization_id: str = "default",
        max_parallel: int = 3,
        budget: Mapping[str, Any] | None = None,
        deadline_unix: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        objective = str(objective).strip()
        if not objective:
            raise ValueError("mission objective is required")
        normalized = (
            self._normalize_explicit(tasks, objective)
            if tasks
            else self._default_tasks(objective, deliverables)
        )
        mission_id = "mission_" + uuid.uuid4().hex
        return {
            "schema": "auro.mission.plan.v1",
            "mission_id": mission_id,
            "idempotency_key": idempotency_key,
            "operator_id": operator_id,
            "organization_id": organization_id,
            "title": str(title)[:300],
            "objective": objective,
            "max_parallel": max(1, min(int(max_parallel), 16)),
            "budget": dict(budget or {}),
            "deadline_unix": deadline_unix,
            "tasks": normalized,
            "reasoning_policy": {
                "private_chain_of_thought_exported": False,
                "decision_summaries_recorded": True,
                "independent_analysis_and_review": True,
                "artifact_delivery_required": True,
            },
        }

    def _normalize_explicit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        mission_objective: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(tasks):
            task_id = _safe_task_id(str(raw.get("task_id") or f"task-{index + 1}"))
            if task_id in seen:
                raise ValueError(f"duplicate task_id: {task_id}")
            seen.add(task_id)
            kind = str(raw.get("kind") or "reasoning")
            if kind not in self.KINDS:
                raise ValueError(f"unsupported task kind: {kind}")
            output.append(
                {
                    "task_id": task_id,
                    "title": str(raw.get("title") or task_id),
                    "objective": str(raw.get("objective") or mission_objective),
                    "kind": kind,
                    "depends_on": [str(item) for item in raw.get("depends_on", []) or []],
                    "priority": int(raw.get("priority", 0)),
                    "model_lane": str(raw.get("model_lane") or "auro-2b-council"),
                    "reasoning_rounds": max(1, min(int(raw.get("reasoning_rounds", 2)), 6)),
                    "max_attempts": max(1, min(int(raw.get("max_attempts", 3)), 10)),
                    "timeout_seconds": max(30, min(int(raw.get("timeout_seconds", 1800)), 86_400)),
                    "payload": dict(raw.get("payload") or {}),
                    "acceptance_criteria": list(raw.get("acceptance_criteria") or []),
                    "required_artifacts": list(raw.get("required_artifacts") or []),
                }
            )
        return output

    def _default_tasks(
        self,
        objective: str,
        deliverables: Sequence[Mapping[str, Any] | str],
    ) -> list[dict[str, Any]]:
        deliverable_paths: list[str] = []
        for index, raw in enumerate(deliverables):
            if isinstance(raw, Mapping):
                path = str(raw.get("path") or raw.get("name") or f"deliverable-{index + 1}.md")
            else:
                path = str(raw)
            if path.strip():
                deliverable_paths.append(path.strip())
        if not deliverable_paths:
            deliverable_paths = ["deliverables/mission-report.md", "deliverables/mission-report.json"]

        return [
            {
                "task_id": "interpret",
                "title": "Interpret objective and constraints",
                "objective": f"Resolve scope, assumptions, constraints, success criteria, and unknowns for: {objective}",
                "kind": "reasoning",
                "depends_on": [],
                "priority": 100,
                "model_lane": "auro-2b-council",
                "reasoning_rounds": 2,
                "max_attempts": 3,
                "timeout_seconds": 1800,
                "payload": {"phase": "discovery"},
                "acceptance_criteria": ["scope is explicit", "unknowns are listed", "success criteria are testable"],
                "required_artifacts": ["tasks/interpret/brief.md", "tasks/interpret/brief.json"],
            },
            {
                "task_id": "evidence",
                "title": "Collect and evaluate evidence",
                "objective": "Identify the strongest available evidence, contradictions, risks, and missing validation needed for the mission.",
                "kind": "research",
                "depends_on": ["interpret"],
                "priority": 80,
                "model_lane": "auro-2b-council",
                "reasoning_rounds": 3,
                "max_attempts": 3,
                "timeout_seconds": 3600,
                "payload": {"phase": "evidence"},
                "acceptance_criteria": ["claims are evidence-bound", "gaps are explicit"],
                "required_artifacts": ["tasks/evidence/evidence-review.md", "tasks/evidence/evidence-review.json"],
            },
            {
                "task_id": "solution",
                "title": "Develop the implementation or solution",
                "objective": "Produce the strongest practical solution, implementation design, or work product for the mission objective.",
                "kind": "implementation",
                "depends_on": ["interpret"],
                "priority": 80,
                "model_lane": "auro-2b-council",
                "reasoning_rounds": 3,
                "max_attempts": 3,
                "timeout_seconds": 7200,
                "payload": {"phase": "production"},
                "acceptance_criteria": ["solution is actionable", "failure behavior is specified"],
                "required_artifacts": ["tasks/solution/solution.md", "tasks/solution/solution.json"],
            },
            {
                "task_id": "deliverables",
                "title": "Create requested deliverables",
                "objective": "Convert the evidence and solution into polished user-facing deliverables without introducing unsupported claims.",
                "kind": "artifact",
                "depends_on": ["evidence", "solution"],
                "priority": 60,
                "model_lane": "auro-2b-council",
                "reasoning_rounds": 2,
                "max_attempts": 3,
                "timeout_seconds": 3600,
                "payload": {"phase": "artifact-production"},
                "acceptance_criteria": ["all requested deliverables exist", "artifact hashes are recorded"],
                "required_artifacts": deliverable_paths,
            },
            {
                "task_id": "red-team",
                "title": "Red-team and quality review",
                "objective": "Challenge the proposed work for logical gaps, unsafe assumptions, missing evidence, usability problems, and incomplete deliverables.",
                "kind": "review",
                "depends_on": ["deliverables"],
                "priority": 50,
                "model_lane": "auro-2b-council",
                "reasoning_rounds": 3,
                "max_attempts": 2,
                "timeout_seconds": 3600,
                "payload": {"phase": "quality-gate", "fail_on_blockers": False},
                "acceptance_criteria": ["defects are prioritized", "residual risks are visible"],
                "required_artifacts": ["tasks/red-team/review.md", "tasks/red-team/review.json"],
            },
            {
                "task_id": "synthesize",
                "title": "Synthesize final answer and handoff",
                "objective": "Reconcile the solution and red-team findings into a final, coherent, user-ready result with explicit next actions.",
                "kind": "synthesis",
                "depends_on": ["red-team"],
                "priority": 40,
                "model_lane": "auro-2b-council",
                "reasoning_rounds": 2,
                "max_attempts": 3,
                "timeout_seconds": 3600,
                "payload": {"phase": "final-synthesis"},
                "acceptance_criteria": ["result is complete", "remaining blockers are explicit"],
                "required_artifacts": ["FINAL_RESULT.md", "FINAL_RESULT.json"],
            },
            {
                "task_id": "package",
                "title": "Package all artifacts and receipts",
                "objective": "Create a complete artifact manifest and downloadable mission bundle.",
                "kind": "package",
                "depends_on": ["synthesize"],
                "priority": 10,
                "model_lane": "deterministic-packager",
                "reasoning_rounds": 1,
                "max_attempts": 2,
                "timeout_seconds": 900,
                "payload": {"phase": "package"},
                "acceptance_criteria": ["bundle exists", "manifest hash exists"],
                "required_artifacts": ["ARTIFACT_MANIFEST.json", "mission-artifacts.zip"],
            },
        ]


class CouncilTaskExecutor:
    """Use the Auro-2B council for bounded multi-pass task execution."""

    def __init__(self, council_service: Any, artifacts: ArtifactStore) -> None:
        self.council = council_service
        self.artifacts = artifacts

    def execute(
        self,
        mission: Mapping[str, Any],
        task: Mapping[str, Any],
        dependency_results: Sequence[Mapping[str, Any]],
        progress: Callable[[float, str], None] | None = None,
    ) -> TaskExecutionResult:
        if not getattr(self.council, "configured", False):
            raise RuntimeError("Auro-2B council is not configured for task execution")
        dependency_context = self._dependency_context(dependency_results)
        rounds = max(1, min(int(task.get("reasoning_rounds", 1)), 6))
        previous = ""
        round_results: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index in range(rounds):
            if progress:
                progress(index / max(rounds + 1, 1), f"reasoning pass {index + 1}/{rounds}")
            prompt = self._round_prompt(mission, task, dependency_context, previous, index, rounds)
            response = self.council.respond(prompt, full_parent_context=dependency_context or None)
            round_results.append(response)
            previous = _bounded(response.get("text", ""), 8_000)

        if progress:
            progress(rounds / max(rounds + 1, 1), "rendering task artifacts")
        final = round_results[-1]
        structured = dict(final.get("structured_answer") or {})
        text = str(final.get("text") or structured.get("answer") or "").strip()
        blockers = tuple(str(item) for item in final.get("blockers", []) if str(item).strip())
        artifacts = self._write_artifacts(mission, task, text, structured, round_results)
        evidence = self._evidence(round_results)
        confidence = self._confidence(structured, final)
        options = [
            str(item.get("consensus") or "")
            for item in final.get("consensus_votes", [])
            if str(item.get("consensus") or "").strip()
        ]
        decision = {
            "summary": _bounded(text, 2_000),
            "options": options[:12],
            "decision": str(structured.get("answer") or text),
            "evidence": list(evidence),
            "confidence": confidence,
            "blockers": list(blockers),
            "private_chain_of_thought_exported": False,
        }
        required = {str(item) for item in task.get("required_artifacts", [])}
        produced = {item.relative_path for item in artifacts}
        missing = sorted(required - produced)
        fail_on_blockers = bool(task.get("payload", {}).get("fail_on_blockers", False))
        accepted = bool(text) and not missing and not (fail_on_blockers and blockers)
        result_payload = {
            "text": text,
            "structured_answer": structured,
            "round_count": len(round_results),
            "round_receipts": [item.get("runtime_receipt") for item in round_results],
            "evidence_class": final.get("evidence_class"),
            "release_evidence_ready": final.get("release_evidence_ready", False),
            "blockers": list(blockers),
            "missing_required_artifacts": missing,
        }
        metrics = {
            "reasoning_rounds": len(round_results),
            "elapsed_ms": round((time.perf_counter() - started) * 1_000, 3),
            "atomic_agent_count": sum(int(item.get("atomic_agent_count", 0)) for item in round_results),
            "model_backed_atomic_count": sum(int(item.get("model_backed_atomic_count", 0)) for item in round_results),
            "artifact_count": len(artifacts),
            "confidence": confidence,
        }
        return TaskExecutionResult(
            summary=text,
            output=result_payload,
            artifacts=tuple(artifacts),
            decision=decision,
            metrics=metrics,
            evidence=evidence,
            blockers=tuple([*blockers, *[f"missing artifact: {item}" for item in missing]]),
            accepted=accepted,
        )

    @staticmethod
    def _dependency_context(results: Sequence[Mapping[str, Any]]) -> str:
        blocks: list[str] = []
        for item in results:
            task_id = str(item.get("task_id") or "dependency")
            result = item.get("result") or {}
            summary = result.get("summary") if isinstance(result, Mapping) else ""
            if not summary and isinstance(result, Mapping):
                summary = result.get("output", {}).get("text") if isinstance(result.get("output"), Mapping) else ""
            blocks.append(f"[{task_id}] {_bounded(summary, 6_000)}")
        return "\n".join(blocks)[:24_000]

    @staticmethod
    def _round_prompt(
        mission: Mapping[str, Any],
        task: Mapping[str, Any],
        dependencies: str,
        previous: str,
        index: int,
        rounds: int,
    ) -> str:
        if index == 0:
            mode = "Develop an independent, evidence-bound solution."
        elif index == rounds - 1:
            mode = "Produce the final decision after correcting weaknesses found in prior passes."
        else:
            mode = "Critique the previous pass, test alternatives, and improve the solution."
        return f"""AURO MISSION TASK\nMission: {mission.get('title')}\nMission objective: {mission.get('objective')}\nTask: {task.get('title')}\nTask objective: {task.get('objective')}\nTask kind: {task.get('kind')}\nAcceptance criteria: {json.dumps(task.get('acceptance_criteria', []), ensure_ascii=False)}\nPass: {index + 1} of {rounds}\nMode: {mode}\n\nDependency results:\n{dependencies or '[none]'}\n\nPrevious pass:\n{previous or '[none]'}\n\nReturn a direct result, bounded reasoning summary, evidence references, confidence, caveats, and next actions. Do not expose private chain-of-thought or claim unexecuted actions."""

    def _write_artifacts(
        self,
        mission: Mapping[str, Any],
        task: Mapping[str, Any],
        text: str,
        structured: Mapping[str, Any],
        rounds: Sequence[Mapping[str, Any]],
    ) -> list[ArtifactRecord]:
        mission_id = str(mission["mission_id"])
        task_id = str(task["task_id"])
        records: list[ArtifactRecord] = []
        report = {
            "schema": "auro.mission.task-report.v1",
            "mission_id": mission_id,
            "task_id": task_id,
            "kind": task.get("kind"),
            "answer": text,
            "structured_answer": dict(structured),
            "round_count": len(rounds),
            "round_receipt_hashes": [
                (item.get("runtime_receipt") or {}).get("receipt_sha256")
                for item in rounds
            ],
            "claim_boundary": "decision summary and outputs only; private chain-of-thought is not stored",
        }
        default_json = f"tasks/{task_id}/report.json"
        default_md = f"tasks/{task_id}/result.md"
        records.append(self.artifacts.write_json(mission_id, default_json, report, task_id=task_id, label="task report"))
        records.append(self.artifacts.write_text(mission_id, default_md, text + "\n", task_id=task_id, media_type="text/markdown; charset=utf-8", label="task result"))
        required = [str(item) for item in task.get("required_artifacts", [])]
        for relative in required:
            if relative in {default_json, default_md}:
                continue
            if relative.lower().endswith(".json"):
                record = self.artifacts.write_json(mission_id, relative, report, task_id=task_id, label="required deliverable")
            else:
                record = self.artifacts.write_text(
                    mission_id,
                    relative,
                    text + "\n",
                    task_id=task_id,
                    media_type=("text/markdown; charset=utf-8" if relative.lower().endswith(".md") else "text/plain; charset=utf-8"),
                    label="required deliverable",
                )
            records.append(record)
        return records

    @staticmethod
    def _evidence(rounds: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        values: list[str] = []
        for item in rounds:
            receipt = item.get("runtime_receipt") or {}
            digest = receipt.get("receipt_sha256") or receipt.get("digest")
            if digest:
                values.append("receipt:" + str(digest))
            for stage in item.get("mesie_receipts", []) or []:
                stage_digest = stage.get("receipt_sha256") if isinstance(stage, Mapping) else None
                if stage_digest:
                    values.append("mesie:" + str(stage_digest))
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _confidence(structured: Mapping[str, Any], final: Mapping[str, Any]) -> float:
        try:
            return max(0.0, min(float(structured.get("confidence", 0.0)), 1.0))
        except (TypeError, ValueError):
            votes = [
                float(item.get("confidence", 0.0))
                for item in final.get("consensus_votes", [])
                if isinstance(item, Mapping)
            ]
            return sum(votes) / max(len(votes), 1)


class PackageTaskExecutor:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def execute(
        self,
        mission: Mapping[str, Any],
        task: Mapping[str, Any],
        dependency_results: Sequence[Mapping[str, Any]],
        progress: Callable[[float, str], None] | None = None,
    ) -> TaskExecutionResult:
        mission_id = str(mission["mission_id"])
        if progress:
            progress(0.25, "building artifact manifest")
        manifest = self.artifacts.manifest(mission_id)
        manifest_record = self.artifacts.write_json(
            mission_id,
            "ARTIFACT_MANIFEST.json",
            manifest,
            task_id=str(task["task_id"]),
            label="artifact manifest",
        )
        if progress:
            progress(0.6, "building mission bundle")
        bundle = self.artifacts.build_bundle(mission_id)
        if progress:
            progress(0.95, "finalizing package receipt")
        decision = {
            "summary": "Packaged mission artifacts and their content hashes.",
            "options": [],
            "decision": "Use the generated bundle as the mission delivery package.",
            "evidence": ["manifest:" + manifest["manifest_sha256"], "bundle:" + bundle.sha256],
            "confidence": 1.0,
            "blockers": [],
            "private_chain_of_thought_exported": False,
        }
        return TaskExecutionResult(
            summary="Mission artifacts were packaged with a hash manifest.",
            output={"manifest": manifest, "bundle": bundle.to_dict()},
            artifacts=(manifest_record, bundle),
            decision=decision,
            metrics={"artifact_count": manifest["artifact_count"], "bundle_bytes": bundle.bytes},
            evidence=("manifest:" + manifest["manifest_sha256"], "bundle:" + bundle.sha256),
            accepted=True,
        )


class MissionOrchestrator:
    """Create, execute, resume, and package dependency-aware missions."""

    def __init__(
        self,
        store: MissionStore,
        artifacts: ArtifactStore,
        executors: Mapping[str, TaskExecutor],
        *,
        planner: MissionPlanner | None = None,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.executors = dict(executors)
        self.planner = planner or MissionPlanner()

    def create(self, **request: Any) -> dict[str, Any]:
        plan = self.planner.plan(**request)
        return self.store.create_mission(plan)

    def run_burst(
        self,
        mission_id: str,
        *,
        worker_id: str,
        max_tasks: int = 20,
        time_budget_seconds: int = 300,
        capabilities: Sequence[str] = (),
    ) -> dict[str, Any]:
        started = time.monotonic()
        executed = 0
        failures = 0
        while executed < max(1, min(int(max_tasks), 500)):
            snapshot = self.store.get_mission(mission_id, include_events=False)
            if snapshot["status"] in {"completed", "failed", "cancelled", "paused"}:
                break
            if time.monotonic() - started >= max(1, int(time_budget_seconds)):
                break
            concurrency = min(int(snapshot["max_parallel"]), max_tasks - executed)
            leased: list[dict[str, Any]] = []
            for slot in range(max(1, concurrency)):
                task = self.store.lease_ready_task(
                    f"{worker_id}:{slot}",
                    mission_id=mission_id,
                    lease_seconds=86_400,
                    capabilities=capabilities,
                )
                if task is None:
                    break
                leased.append(task)
            if not leased:
                break

            with ThreadPoolExecutor(max_workers=len(leased)) as pool:
                futures: dict[Future[TaskExecutionResult], tuple[dict[str, Any], str]] = {}
                for slot, task in enumerate(leased):
                    lease_worker = f"{worker_id}:{slot}"
                    dependencies = [self.store.get_task(item) for item in task.get("depends_on", [])]
                    mission = self.store.get_mission(mission_id, include_events=False)
                    executor = self.executors.get(str(task["kind"])) or self.executors.get("default")
                    if executor is None:
                        self.store.fail_task(task["task_id"], lease_worker, f"no executor for task kind {task['kind']}", terminal=True)
                        failures += 1
                        executed += 1
                        continue

                    def report_progress(value: float, note: str, *, task_id=task["task_id"], owner=lease_worker):
                        self.store.heartbeat(
                            task_id,
                            owner,
                            progress=value,
                            lease_seconds=max(int(task.get("timeout_seconds", 1800)) + 120, 900),
                            note=note,
                        )

                    futures[
                        pool.submit(
                            executor.execute,
                            mission,
                            task,
                            dependencies,
                            report_progress,
                        )
                    ] = (task, lease_worker)

                for future in as_completed(futures):
                    task, lease_worker = futures[future]
                    try:
                        result = future.result(timeout=int(task.get("timeout_seconds", 1800)))
                        public = result.public()
                        if result.accepted:
                            self.store.complete_task(
                                task["task_id"],
                                lease_worker,
                                public,
                                artifacts=[item.to_dict() for item in result.artifacts],
                                decision=result.decision,
                            )
                        else:
                            self.store.fail_task(
                                task["task_id"],
                                lease_worker,
                                "; ".join(result.blockers) or "task acceptance failed",
                                terminal=True,
                            )
                            failures += 1
                    except Exception as exc:
                        self.store.fail_task(
                            task["task_id"],
                            lease_worker,
                            f"{type(exc).__name__}: {str(exc)[:2000]}",
                            retry_delay_seconds=30,
                        )
                        failures += 1
                    executed += 1

        snapshot = self.store.get_mission(mission_id)
        if snapshot["status"] == "completed" and not snapshot.get("result_summary"):
            self._finalize(snapshot)
            snapshot = self.store.get_mission(mission_id)
        return {
            "schema": "auro.mission.run-burst.v1",
            "mission": snapshot,
            "tasks_executed": executed,
            "task_failures_or_retries": failures,
            "elapsed_ms": round((time.monotonic() - started) * 1_000, 3),
            "stopped_because": (
                "mission-terminal"
                if snapshot["status"] in {"completed", "failed", "cancelled", "paused"}
                else "idle-or-budget-exhausted"
            ),
        }

    def _finalize(self, mission: Mapping[str, Any]) -> None:
        mission_id = str(mission["mission_id"])
        synthesis = next(
            (
                task
                for task in mission.get("tasks", [])
                if task.get("task_id") == "synthesize" and task.get("result")
            ),
            None,
        )
        summary = "Mission completed."
        if synthesis:
            summary = str((synthesis.get("result") or {}).get("summary") or summary)
        manifest = self.artifacts.manifest(mission_id)
        self.store.set_result_summary(mission_id, summary, manifest["manifest_sha256"])

    def status(self) -> dict[str, Any]:
        return {
            "schema": "auro.mission-orchestrator.status.v1",
            "database": str(self.store.path),
            "artifact_root": str(self.artifacts.root),
            "executors": sorted(self.executors),
            "features": [
                "dependency-aware-DAG",
                "parallel-ready-tasks",
                "pause-resume-cancel",
                "lease-recovery",
                "bounded-retries",
                "multi-pass-council-reasoning",
                "decision-summaries-without-private-chain-of-thought",
                "content-addressed-artifacts",
                "artifact-bundles",
            ],
        }
