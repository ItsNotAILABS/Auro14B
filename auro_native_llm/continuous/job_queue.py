"""Durable SQLite job queue for bounded AURO continuous-training workers."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

ALLOWED_ENTRYPOINTS = {"scripts/train_him_sft.py"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DurableJobQueue:
    def __init__(self, path: str | Path = "state/continuous-jobs.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              job_sha256 TEXT NOT NULL UNIQUE,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              max_attempts INTEGER NOT NULL,
              available_at INTEGER NOT NULL,
              lease_owner TEXT,
              lease_expires_at INTEGER,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              result_json TEXT,
              error TEXT
            );
            """
        )
        self.db.commit()

    def enqueue(self, kind: str, payload: Mapping[str, Any], max_attempts: int = 3) -> dict[str, Any]:
        document = {"kind": kind, "payload": dict(payload)}
        digest = hashlib.sha256(canonical(document)).hexdigest()
        existing = self.db.execute("SELECT * FROM jobs WHERE job_sha256 = ?", (digest,)).fetchone()
        if existing:
            return dict(existing)
        now = int(time.time())
        job_id = f"job-{uuid.uuid4().hex}"
        self.db.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, NULL, NULL, ?, ?, NULL, NULL)",
            (job_id, digest, kind, json.dumps(dict(payload), sort_keys=True), max(1, max_attempts), now, now, now),
        )
        self.db.commit()
        return dict(self.db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def recover_expired(self, now: int | None = None) -> int:
        current = int(time.time()) if now is None else int(now)
        cursor = self.db.execute(
            """UPDATE jobs SET status='queued', lease_owner=NULL, lease_expires_at=NULL, updated_at=?
               WHERE status='leased' AND lease_expires_at < ? AND attempts < max_attempts""",
            (current, current),
        )
        self.db.commit()
        return cursor.rowcount

    def lease(self, worker_id: str, lease_seconds: int = 300, now: int | None = None) -> dict[str, Any] | None:
        current = int(time.time()) if now is None else int(now)
        self.recover_expired(current)
        self.db.execute("BEGIN IMMEDIATE")
        row = self.db.execute(
            """SELECT * FROM jobs WHERE status='queued' AND available_at <= ? AND attempts < max_attempts
               ORDER BY created_at, job_id LIMIT 1""",
            (current,),
        ).fetchone()
        if row is None:
            self.db.rollback()
            return None
        expires = current + max(30, min(int(lease_seconds), 3600))
        self.db.execute(
            """UPDATE jobs SET status='leased', attempts=attempts+1, lease_owner=?, lease_expires_at=?, updated_at=?
               WHERE job_id=? AND status='queued'""",
            (worker_id, expires, current, row["job_id"]),
        )
        self.db.commit()
        leased = dict(self.db.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone())
        leased["payload"] = json.loads(leased["payload_json"])
        return leased

    def complete(self, job_id: str, worker_id: str, result: Mapping[str, Any]) -> None:
        cursor = self.db.execute(
            """UPDATE jobs SET status='completed', result_json=?, lease_owner=NULL, lease_expires_at=NULL, updated_at=?
               WHERE job_id=? AND status='leased' AND lease_owner=?""",
            (json.dumps(dict(result), sort_keys=True), int(time.time()), job_id, worker_id),
        )
        if cursor.rowcount != 1:
            self.db.rollback()
            raise PermissionError("worker does not hold the active lease")
        self.db.commit()

    def fail(self, job_id: str, worker_id: str, error: str, retry_delay: int = 60) -> None:
        row = self.db.execute("SELECT attempts, max_attempts FROM jobs WHERE job_id=? AND lease_owner=?", (job_id, worker_id)).fetchone()
        if row is None:
            raise PermissionError("worker does not hold the active lease")
        terminal = int(row["attempts"]) >= int(row["max_attempts"])
        self.db.execute(
            """UPDATE jobs SET status=?, error=?, available_at=?, lease_owner=NULL, lease_expires_at=NULL, updated_at=?
               WHERE job_id=? AND lease_owner=?""",
            ("failed" if terminal else "queued", error[:4000], int(time.time()) + max(0, retry_delay), int(time.time()), job_id, worker_id),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def execute_training_job(job: Mapping[str, Any], repository_root: str | Path = ".") -> dict[str, Any]:
    """Execute one allowlisted training job and verify its declared output."""
    root = Path(repository_root).resolve()
    entrypoint = str(job.get("entrypoint") or "")
    if entrypoint not in ALLOWED_ENTRYPOINTS:
        raise PermissionError(f"training entrypoint is not allowlisted: {entrypoint}")
    resume = Path(str(job.get("resume_checkpoint") or ""))
    if not resume.is_absolute():
        resume = root / resume
    if not resume.is_dir():
        raise FileNotFoundError(f"resume checkpoint missing: {resume}")
    output = Path(str(job.get("output_checkpoint") or ""))
    if not output.is_absolute():
        output = root / output
    command = [
        "python", str(root / entrypoint),
        "--resume", str(resume),
        "--output", str(output),
        "--epochs", str(max(1, min(int(job.get("epochs", 1)), 10))),
        "--seq-len", str(max(32, min(int(job.get("seq_len", 256)), 8192))),
        "--lr", str(float(job.get("learning_rate", 0.001))),
    ]
    steps = int(job.get("steps_per_epoch", 0))
    if steps:
        command += ["--steps-per-epoch", str(max(1, min(steps, 10000)))]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=int(job.get("timeout_seconds", 7200)))
    if completed.returncode != 0:
        raise RuntimeError(f"training failed ({completed.returncode}): {completed.stderr[-2000:]}")
    report_path = output / "HIM_SFT_REPORT.json"
    if not report_path.is_file():
        raise RuntimeError("training exited successfully but HIM_SFT_REPORT.json is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ok") or Path(str(report.get("checkpoint") or "")).name != output.name:
        raise RuntimeError("training report does not prove the declared output checkpoint")
    artifacts = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        artifacts.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if len(artifacts) < 2:
        raise RuntimeError("candidate checkpoint output is incomplete")
    manifest = {
        "schema": "auro.training.execution-receipt.v1",
        "entrypoint": entrypoint,
        "resume_checkpoint": str(resume),
        "output_checkpoint": str(output),
        "command": command,
        "returncode": completed.returncode,
        "report_sha256": sha256_file(report_path),
        "artifacts": artifacts,
        "stdout_tail": completed.stdout[-4000:],
    }
    manifest["receipt_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    (output / "TRAINING_EXECUTION_RECEIPT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
