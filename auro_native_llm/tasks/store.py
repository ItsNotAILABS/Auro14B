"""Durable dependency-aware mission store for AURO long-running tasks."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence


TERMINAL_TASK_STATES = {"completed", "failed", "cancelled", "blocked"}
ACTIVE_MISSION_STATES = {"queued", "running"}


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


def _now() -> int:
    return int(time.time())


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class MissionStore:
    """SQLite/WAL state for missions, tasks, leases, artifacts, and decisions."""

    def __init__(self, path: str | Path = "state/mission-orchestrator.sqlite3") -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS missions (
              mission_id TEXT PRIMARY KEY,
              idempotency_key TEXT UNIQUE,
              operator_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              title TEXT NOT NULL,
              objective TEXT NOT NULL,
              status TEXT NOT NULL,
              max_parallel INTEGER NOT NULL,
              budget_json TEXT NOT NULL,
              deadline_unix INTEGER,
              plan_sha256 TEXT NOT NULL,
              result_summary TEXT,
              artifact_manifest_sha256 TEXT,
              error TEXT,
              created_at_unix INTEGER NOT NULL,
              updated_at_unix INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
              task_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE CASCADE,
              ordinal INTEGER NOT NULL,
              title TEXT NOT NULL,
              objective TEXT NOT NULL,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              priority INTEGER NOT NULL,
              model_lane TEXT,
              reasoning_rounds INTEGER NOT NULL,
              max_attempts INTEGER NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              timeout_seconds INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              acceptance_json TEXT NOT NULL,
              required_artifacts_json TEXT NOT NULL,
              result_json TEXT,
              error TEXT,
              progress REAL NOT NULL DEFAULT 0,
              available_at_unix INTEGER NOT NULL,
              lease_owner TEXT,
              lease_expires_at_unix INTEGER,
              created_at_unix INTEGER NOT NULL,
              updated_at_unix INTEGER NOT NULL,
              UNIQUE(mission_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS dependencies (
              task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
              depends_on_task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
              PRIMARY KEY(task_id, depends_on_task_id)
            );
            CREATE TABLE IF NOT EXISTS mission_events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE CASCADE,
              task_id TEXT,
              event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              observed_at_unix INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              artifact_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE CASCADE,
              task_id TEXT,
              relative_path TEXT NOT NULL,
              media_type TEXT NOT NULL,
              bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              label TEXT NOT NULL,
              created_at_unix INTEGER NOT NULL,
              UNIQUE(mission_id, relative_path, sha256)
            );
            CREATE TABLE IF NOT EXISTS decisions (
              decision_id TEXT PRIMARY KEY,
              mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE CASCADE,
              task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
              summary TEXT NOT NULL,
              options_json TEXT NOT NULL,
              decision TEXT NOT NULL,
              evidence_json TEXT NOT NULL,
              confidence REAL NOT NULL,
              blockers_json TEXT NOT NULL,
              decision_sha256 TEXT NOT NULL,
              created_at_unix INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_ready
              ON tasks(status, available_at_unix, priority, ordinal);
            CREATE INDEX IF NOT EXISTS idx_tasks_mission ON tasks(mission_id, status);
            CREATE INDEX IF NOT EXISTS idx_events_mission ON mission_events(mission_id, sequence);
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

    def create_mission(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        tasks = [dict(item) for item in spec.get("tasks", [])]
        if not tasks:
            raise ValueError("mission requires at least one task")
        self._validate_graph(tasks)
        mission_id = str(spec.get("mission_id") or "mission_" + uuid.uuid4().hex)
        idempotency_key = str(spec.get("idempotency_key") or "").strip() or None
        now = _now()
        plan_material = {
            "objective": spec.get("objective"),
            "tasks": tasks,
            "budget": spec.get("budget", {}),
            "max_parallel": spec.get("max_parallel", 3),
        }
        plan_sha = _sha(plan_material)
        with self._transaction():
            if idempotency_key:
                existing = self.db.execute(
                    "SELECT mission_id FROM missions WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return self.get_mission(str(existing["mission_id"]))
            self.db.execute(
                """INSERT INTO missions (
                    mission_id,idempotency_key,operator_id,organization_id,title,objective,
                    status,max_parallel,budget_json,deadline_unix,plan_sha256,
                    created_at_unix,updated_at_unix
                ) VALUES (?,?,?,?,?,?, 'queued',?,?,?,?,?,?)""",
                (
                    mission_id,
                    idempotency_key,
                    str(spec.get("operator_id") or "operator"),
                    str(spec.get("organization_id") or "default"),
                    str(spec.get("title") or "AURO mission")[:300],
                    str(spec.get("objective") or "").strip(),
                    max(1, min(int(spec.get("max_parallel", 3)), 32)),
                    _json(dict(spec.get("budget") or {})),
                    int(spec["deadline_unix"]) if spec.get("deadline_unix") else None,
                    plan_sha,
                    now,
                    now,
                ),
            )
            task_ids = {str(item["task_id"]) for item in tasks}
            for ordinal, task in enumerate(tasks):
                task_id = str(task["task_id"])
                self.db.execute(
                    """INSERT INTO tasks (
                        task_id,mission_id,ordinal,title,objective,kind,status,priority,
                        model_lane,reasoning_rounds,max_attempts,timeout_seconds,payload_json,
                        acceptance_json,required_artifacts_json,available_at_unix,
                        created_at_unix,updated_at_unix
                    ) VALUES (?,?,?,?,?,?,'queued',?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        task_id,
                        mission_id,
                        ordinal,
                        str(task.get("title") or task_id)[:300],
                        str(task.get("objective") or "").strip(),
                        str(task.get("kind") or "reasoning"),
                        max(-1000, min(int(task.get("priority", 0)), 1000)),
                        str(task.get("model_lane") or "auro-2b-council"),
                        max(1, min(int(task.get("reasoning_rounds", 1)), 12)),
                        max(1, min(int(task.get("max_attempts", 3)), 20)),
                        max(10, min(int(task.get("timeout_seconds", 900)), 86_400)),
                        _json(dict(task.get("payload") or {})),
                        _json(list(task.get("acceptance_criteria") or [])),
                        _json(list(task.get("required_artifacts") or [])),
                        now,
                        now,
                        now,
                    ),
                )
                for dependency in task.get("depends_on", []) or []:
                    dependency_id = str(dependency)
                    if dependency_id not in task_ids:
                        raise ValueError(f"unknown task dependency: {dependency_id}")
                    self.db.execute(
                        "INSERT INTO dependencies(task_id,depends_on_task_id) VALUES (?,?)",
                        (task_id, dependency_id),
                    )
            self._event_locked(
                mission_id,
                None,
                "mission_created",
                {"plan_sha256": plan_sha, "task_count": len(tasks)},
                now,
            )
        return self.get_mission(mission_id)

    @staticmethod
    def _validate_graph(tasks: Sequence[Mapping[str, Any]]) -> None:
        ids = [str(item.get("task_id") or "") for item in tasks]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("task IDs must be non-empty and unique")
        dependencies = {
            str(task["task_id"]): {str(item) for item in task.get("depends_on", []) or []}
            for task in tasks
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError("task dependency graph contains a cycle")
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                if dependency not in dependencies:
                    raise ValueError(f"unknown task dependency: {dependency}")
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in ids:
            visit(task_id)

    def _event_locked(
        self,
        mission_id: str,
        task_id: str | None,
        event_type: str,
        payload: Mapping[str, Any],
        observed_at: int | None = None,
    ) -> None:
        self.db.execute(
            """INSERT INTO mission_events(
                mission_id,task_id,event_type,payload_json,observed_at_unix
            ) VALUES (?,?,?,?,?)""",
            (
                mission_id,
                task_id,
                event_type,
                _json(dict(payload)),
                observed_at or _now(),
            ),
        )

    def event(
        self,
        mission_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        task_id: str | None = None,
    ) -> None:
        with self._transaction():
            self._event_locked(mission_id, task_id, event_type, payload)

    def recover_expired(self, now: int | None = None) -> int:
        current = _now() if now is None else int(now)
        with self._transaction():
            rows = self.db.execute(
                """SELECT task_id,mission_id,attempts,max_attempts FROM tasks
                   WHERE status='running' AND lease_expires_at_unix < ?""",
                (current,),
            ).fetchall()
            recovered = 0
            for row in rows:
                terminal = int(row["attempts"]) >= int(row["max_attempts"])
                new_status = "failed" if terminal else "queued"
                delay = min(300, 2 ** max(0, int(row["attempts"]) - 1))
                self.db.execute(
                    """UPDATE tasks SET status=?,available_at_unix=?,lease_owner=NULL,
                       lease_expires_at_unix=NULL,error=?,updated_at_unix=? WHERE task_id=?""",
                    (
                        new_status,
                        current + delay,
                        "worker lease expired",
                        current,
                        row["task_id"],
                    ),
                )
                self._event_locked(
                    row["mission_id"],
                    row["task_id"],
                    "task_lease_expired",
                    {"terminal": terminal, "retry_delay_seconds": delay},
                    current,
                )
                recovered += 1
            self._refresh_all_missions_locked(current)
            return recovered

    def lease_ready_task(
        self,
        worker_id: str,
        *,
        mission_id: str | None = None,
        lease_seconds: int = 900,
        capabilities: Iterable[str] = (),
    ) -> dict[str, Any] | None:
        current = _now()
        capability_set = {str(item) for item in capabilities}
        self.recover_expired(current)
        with self._transaction():
            query = """
                SELECT t.* FROM tasks t
                JOIN missions m ON m.mission_id=t.mission_id
                WHERE t.status='queued'
                  AND t.available_at_unix <= ?
                  AND m.status IN ('queued','running')
                  AND (? IS NULL OR t.mission_id = ?)
                  AND NOT EXISTS (
                    SELECT 1 FROM dependencies d
                    JOIN tasks parent ON parent.task_id=d.depends_on_task_id
                    WHERE d.task_id=t.task_id AND parent.status!='completed'
                  )
                ORDER BY t.priority DESC, t.ordinal, t.created_at_unix
            """
            rows = self.db.execute(query, (current, mission_id, mission_id)).fetchall()
            selected = None
            for row in rows:
                payload = _load(row["payload_json"], {})
                required = {str(item) for item in payload.get("required_capabilities", [])}
                if required and not required.issubset(capability_set):
                    continue
                selected = row
                break
            if selected is None:
                self._mark_dependency_blockers_locked(current)
                self._refresh_all_missions_locked(current)
                return None
            expires = current + max(30, min(int(lease_seconds), 86_400))
            cursor = self.db.execute(
                """UPDATE tasks SET status='running',attempts=attempts+1,
                   lease_owner=?,lease_expires_at_unix=?,updated_at_unix=?
                   WHERE task_id=? AND status='queued'""",
                (worker_id, expires, current, selected["task_id"]),
            )
            if cursor.rowcount != 1:
                return None
            self.db.execute(
                "UPDATE missions SET status='running',updated_at_unix=? WHERE mission_id=?",
                (current, selected["mission_id"]),
            )
            self._event_locked(
                selected["mission_id"],
                selected["task_id"],
                "task_leased",
                {"worker_id": worker_id, "lease_expires_at_unix": expires},
                current,
            )
            row = self.db.execute(
                "SELECT * FROM tasks WHERE task_id=?",
                (selected["task_id"],),
            ).fetchone()
            return self._task_public(row)

    def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        progress: float | None = None,
        lease_seconds: int = 900,
        note: str = "",
    ) -> dict[str, Any]:
        current = _now()
        expires = current + max(30, min(int(lease_seconds), 86_400))
        with self._transaction():
            row = self.db.execute(
                "SELECT mission_id FROM tasks WHERE task_id=? AND status='running' AND lease_owner=?",
                (task_id, worker_id),
            ).fetchone()
            if row is None:
                raise PermissionError("worker does not hold the active task lease")
            normalized = None if progress is None else max(0.0, min(float(progress), 0.99))
            self.db.execute(
                """UPDATE tasks SET lease_expires_at_unix=?,progress=COALESCE(?,progress),
                   updated_at_unix=? WHERE task_id=?""",
                (expires, normalized, current, task_id),
            )
            self._event_locked(
                row["mission_id"],
                task_id,
                "task_heartbeat",
                {"progress": normalized, "note": str(note)[:500]},
                current,
            )
        return self.get_task(task_id)

    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        result: Mapping[str, Any],
        *,
        artifacts: Sequence[Mapping[str, Any]] = (),
        decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = _now()
        with self._transaction():
            row = self.db.execute(
                "SELECT mission_id FROM tasks WHERE task_id=? AND status='running' AND lease_owner=?",
                (task_id, worker_id),
            ).fetchone()
            if row is None:
                raise PermissionError("worker does not hold the active task lease")
            mission_id = str(row["mission_id"])
            self.db.execute(
                """UPDATE tasks SET status='completed',result_json=?,error=NULL,progress=1,
                   lease_owner=NULL,lease_expires_at_unix=NULL,updated_at_unix=?
                   WHERE task_id=?""",
                (_json(dict(result)), current, task_id),
            )
            for artifact in artifacts:
                self._insert_artifact_locked(mission_id, task_id, artifact)
            if decision:
                self._insert_decision_locked(mission_id, task_id, decision)
            self._event_locked(
                mission_id,
                task_id,
                "task_completed",
                {
                    "result_sha256": _sha(dict(result)),
                    "artifact_count": len(artifacts),
                },
                current,
            )
            self._mark_dependency_blockers_locked(current)
            self._refresh_mission_locked(mission_id, current)
        return self.get_task(task_id)

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: str,
        *,
        retry_delay_seconds: int = 30,
        terminal: bool = False,
    ) -> dict[str, Any]:
        current = _now()
        with self._transaction():
            row = self.db.execute(
                "SELECT mission_id,attempts,max_attempts FROM tasks WHERE task_id=? AND status='running' AND lease_owner=?",
                (task_id, worker_id),
            ).fetchone()
            if row is None:
                raise PermissionError("worker does not hold the active task lease")
            exhausted = int(row["attempts"]) >= int(row["max_attempts"])
            final = bool(terminal or exhausted)
            new_status = "failed" if final else "queued"
            delay = max(0, min(int(retry_delay_seconds), 86_400))
            self.db.execute(
                """UPDATE tasks SET status=?,error=?,available_at_unix=?,lease_owner=NULL,
                   lease_expires_at_unix=NULL,updated_at_unix=? WHERE task_id=?""",
                (new_status, str(error)[:4000], current + delay, current, task_id),
            )
            self._event_locked(
                row["mission_id"],
                task_id,
                "task_failed" if final else "task_retry_scheduled",
                {"error": str(error)[:1000], "retry_delay_seconds": delay},
                current,
            )
            self._mark_dependency_blockers_locked(current)
            self._refresh_mission_locked(str(row["mission_id"]), current)
        return self.get_task(task_id)

    def pause(self, mission_id: str) -> dict[str, Any]:
        return self._set_mission_state(mission_id, "paused")

    def resume(self, mission_id: str) -> dict[str, Any]:
        with self._transaction():
            row = self.db.execute(
                "SELECT status FROM missions WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if row["status"] not in {"paused", "failed"}:
                raise ValueError("only paused or failed missions can be resumed")
            now = _now()
            self.db.execute(
                "UPDATE missions SET status='queued',error=NULL,updated_at_unix=? WHERE mission_id=?",
                (now, mission_id),
            )
            self.db.execute(
                """UPDATE tasks SET status='queued',error=NULL,available_at_unix=?,updated_at_unix=?
                   WHERE mission_id=? AND status IN ('blocked','failed') AND attempts < max_attempts""",
                (now, now, mission_id),
            )
            self._event_locked(mission_id, None, "mission_resumed", {}, now)
        return self.get_mission(mission_id)

    def cancel(self, mission_id: str) -> dict[str, Any]:
        with self._transaction():
            now = _now()
            row = self.db.execute(
                "SELECT status FROM missions WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if row["status"] in {"completed", "cancelled"}:
                return self.get_mission(mission_id)
            self.db.execute(
                "UPDATE missions SET status='cancelled',updated_at_unix=? WHERE mission_id=?",
                (now, mission_id),
            )
            self.db.execute(
                """UPDATE tasks SET status='cancelled',lease_owner=NULL,
                   lease_expires_at_unix=NULL,updated_at_unix=?
                   WHERE mission_id=? AND status NOT IN ('completed','failed','cancelled')""",
                (now, mission_id),
            )
            self._event_locked(mission_id, None, "mission_cancelled", {}, now)
        return self.get_mission(mission_id)

    def _set_mission_state(self, mission_id: str, status: str) -> dict[str, Any]:
        with self._transaction():
            row = self.db.execute(
                "SELECT status FROM missions WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(mission_id)
            if row["status"] in {"completed", "cancelled"}:
                raise ValueError("terminal mission cannot change state")
            now = _now()
            self.db.execute(
                "UPDATE missions SET status=?,updated_at_unix=? WHERE mission_id=?",
                (status, now, mission_id),
            )
            self._event_locked(mission_id, None, f"mission_{status}", {}, now)
        return self.get_mission(mission_id)

    def set_result_summary(
        self,
        mission_id: str,
        summary: str,
        artifact_manifest_sha256: str | None = None,
    ) -> None:
        with self._transaction():
            self.db.execute(
                """UPDATE missions SET result_summary=?,artifact_manifest_sha256=?,
                   updated_at_unix=? WHERE mission_id=?""",
                (str(summary), artifact_manifest_sha256, _now(), mission_id),
            )

    def _insert_artifact_locked(
        self,
        mission_id: str,
        task_id: str | None,
        artifact: Mapping[str, Any],
    ) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO artifacts(
                artifact_id,mission_id,task_id,relative_path,media_type,bytes,sha256,label,created_at_unix
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                str(artifact["artifact_id"]),
                mission_id,
                task_id,
                str(artifact["relative_path"]),
                str(artifact.get("media_type") or "application/octet-stream"),
                int(artifact.get("bytes") or 0),
                str(artifact["sha256"]),
                str(artifact.get("label") or ""),
                int(artifact.get("created_at_unix") or _now()),
            ),
        )

    def add_artifact(
        self,
        mission_id: str,
        artifact: Mapping[str, Any],
        *,
        task_id: str | None = None,
    ) -> None:
        with self._transaction():
            self._insert_artifact_locked(mission_id, task_id, artifact)
            self._event_locked(
                mission_id,
                task_id,
                "artifact_registered",
                {
                    "artifact_id": artifact.get("artifact_id"),
                    "sha256": artifact.get("sha256"),
                    "relative_path": artifact.get("relative_path"),
                },
            )

    def _insert_decision_locked(
        self,
        mission_id: str,
        task_id: str,
        decision: Mapping[str, Any],
    ) -> None:
        material = {
            "mission_id": mission_id,
            "task_id": task_id,
            "summary": str(decision.get("summary") or ""),
            "options": list(decision.get("options") or []),
            "decision": str(decision.get("decision") or ""),
            "evidence": list(decision.get("evidence") or []),
            "confidence": max(0.0, min(float(decision.get("confidence", 0.0)), 1.0)),
            "blockers": list(decision.get("blockers") or []),
        }
        digest = _sha(material)
        self.db.execute(
            """INSERT INTO decisions(
                decision_id,mission_id,task_id,summary,options_json,decision,
                evidence_json,confidence,blockers_json,decision_sha256,created_at_unix
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "decision_" + digest[:24],
                mission_id,
                task_id,
                material["summary"],
                _json(material["options"]),
                material["decision"],
                _json(material["evidence"]),
                material["confidence"],
                _json(material["blockers"]),
                digest,
                _now(),
            ),
        )

    def _mark_dependency_blockers_locked(self, current: int) -> None:
        rows = self.db.execute(
            """SELECT DISTINCT child.task_id,child.mission_id
               FROM tasks child
               JOIN dependencies d ON d.task_id=child.task_id
               JOIN tasks parent ON parent.task_id=d.depends_on_task_id
               WHERE child.status='queued' AND parent.status IN ('failed','cancelled','blocked')"""
        ).fetchall()
        for row in rows:
            self.db.execute(
                "UPDATE tasks SET status='blocked',error=?,updated_at_unix=? WHERE task_id=?",
                ("dependency did not complete", current, row["task_id"]),
            )
            self._event_locked(
                row["mission_id"],
                row["task_id"],
                "task_blocked",
                {"reason": "dependency did not complete"},
                current,
            )

    def _refresh_all_missions_locked(self, current: int) -> None:
        mission_ids = [
            str(row["mission_id"])
            for row in self.db.execute("SELECT mission_id FROM missions").fetchall()
        ]
        for mission_id in mission_ids:
            self._refresh_mission_locked(mission_id, current)

    def _refresh_mission_locked(self, mission_id: str, current: int) -> None:
        mission = self.db.execute(
            "SELECT status,deadline_unix FROM missions WHERE mission_id=?",
            (mission_id,),
        ).fetchone()
        if mission is None or mission["status"] in {"paused", "cancelled"}:
            return
        if mission["deadline_unix"] and int(mission["deadline_unix"]) < current:
            self.db.execute(
                "UPDATE missions SET status='failed',error=?,updated_at_unix=? WHERE mission_id=?",
                ("mission deadline exceeded", current, mission_id),
            )
            return
        states = [
            str(row["status"])
            for row in self.db.execute(
                "SELECT status FROM tasks WHERE mission_id=?",
                (mission_id,),
            ).fetchall()
        ]
        if states and all(state == "completed" for state in states):
            status = "completed"
        elif states and all(state in TERMINAL_TASK_STATES for state in states):
            status = "failed" if any(state in {"failed", "blocked"} for state in states) else "cancelled"
        elif any(state == "running" for state in states):
            status = "running"
        else:
            status = "queued"
        self.db.execute(
            "UPDATE missions SET status=?,updated_at_unix=? WHERE mission_id=?",
            (status, current, mission_id),
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            return self._task_public(row)

    def _task_public(self, row: sqlite3.Row) -> dict[str, Any]:
        task_id = str(row["task_id"])
        dependencies = [
            str(item["depends_on_task_id"])
            for item in self.db.execute(
                "SELECT depends_on_task_id FROM dependencies WHERE task_id=? ORDER BY depends_on_task_id",
                (task_id,),
            ).fetchall()
        ]
        return {
            "task_id": task_id,
            "mission_id": row["mission_id"],
            "ordinal": row["ordinal"],
            "title": row["title"],
            "objective": row["objective"],
            "kind": row["kind"],
            "status": row["status"],
            "priority": row["priority"],
            "model_lane": row["model_lane"],
            "reasoning_rounds": row["reasoning_rounds"],
            "max_attempts": row["max_attempts"],
            "attempts": row["attempts"],
            "timeout_seconds": row["timeout_seconds"],
            "payload": _load(row["payload_json"], {}),
            "acceptance_criteria": _load(row["acceptance_json"], []),
            "required_artifacts": _load(row["required_artifacts_json"], []),
            "result": _load(row["result_json"], None),
            "error": row["error"],
            "progress": row["progress"],
            "available_at_unix": row["available_at_unix"],
            "lease_owner": row["lease_owner"],
            "lease_expires_at_unix": row["lease_expires_at_unix"],
            "depends_on": dependencies,
            "created_at_unix": row["created_at_unix"],
            "updated_at_unix": row["updated_at_unix"],
        }

    def get_mission(self, mission_id: str, *, include_events: bool = True) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM missions WHERE mission_id=?",
                (mission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(mission_id)
            tasks = [
                self._task_public(item)
                for item in self.db.execute(
                    "SELECT * FROM tasks WHERE mission_id=? ORDER BY ordinal",
                    (mission_id,),
                ).fetchall()
            ]
            artifacts = [
                dict(item)
                for item in self.db.execute(
                    "SELECT * FROM artifacts WHERE mission_id=? ORDER BY created_at_unix,relative_path",
                    (mission_id,),
                ).fetchall()
            ]
            decisions = []
            for item in self.db.execute(
                "SELECT * FROM decisions WHERE mission_id=? ORDER BY created_at_unix,decision_id",
                (mission_id,),
            ).fetchall():
                value = dict(item)
                value["options"] = _load(value.pop("options_json"), [])
                value["evidence"] = _load(value.pop("evidence_json"), [])
                value["blockers"] = _load(value.pop("blockers_json"), [])
                decisions.append(value)
            events = []
            if include_events:
                for item in self.db.execute(
                    "SELECT * FROM mission_events WHERE mission_id=? ORDER BY sequence",
                    (mission_id,),
                ).fetchall():
                    value = dict(item)
                    value["payload"] = _load(value.pop("payload_json"), {})
                    events.append(value)
            total = len(tasks)
            completed = sum(task["status"] == "completed" for task in tasks)
            running = sum(task["status"] == "running" for task in tasks)
            failed = sum(task["status"] in {"failed", "blocked"} for task in tasks)
            return {
                "schema": "auro.mission.snapshot.v1",
                "mission_id": row["mission_id"],
                "idempotency_key": row["idempotency_key"],
                "operator_id": row["operator_id"],
                "organization_id": row["organization_id"],
                "title": row["title"],
                "objective": row["objective"],
                "status": row["status"],
                "max_parallel": row["max_parallel"],
                "budget": _load(row["budget_json"], {}),
                "deadline_unix": row["deadline_unix"],
                "plan_sha256": row["plan_sha256"],
                "result_summary": row["result_summary"],
                "artifact_manifest_sha256": row["artifact_manifest_sha256"],
                "error": row["error"],
                "created_at_unix": row["created_at_unix"],
                "updated_at_unix": row["updated_at_unix"],
                "progress": {
                    "total": total,
                    "completed": completed,
                    "running": running,
                    "failed_or_blocked": failed,
                    "fraction": round(completed / max(total, 1), 6),
                },
                "tasks": tasks,
                "artifacts": artifacts,
                "decisions": decisions,
                "events": events,
            }

    def list_missions(
        self,
        *,
        organization_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._lock:
            bounded = max(1, min(int(limit), 500))
            if organization_id is None:
                rows = self.db.execute(
                    "SELECT mission_id FROM missions ORDER BY created_at_unix DESC LIMIT ?",
                    (bounded,),
                ).fetchall()
            else:
                rows = self.db.execute(
                    """SELECT mission_id FROM missions WHERE organization_id=?
                       ORDER BY created_at_unix DESC LIMIT ?""",
                    (str(organization_id), bounded),
                ).fetchall()
            return [
                self.get_mission(str(row["mission_id"]), include_events=False)
                for row in rows
            ]
