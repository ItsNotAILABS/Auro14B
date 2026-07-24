from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .models import JobState, TrainingJob
from .policy import ExecutionPolicy
from .receipts import write_receipt


@dataclass(frozen=True)
class ExecutionResult:
    job_id: str
    return_code: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    receipt_path: str
    state: str


class GovernedRunner:
    """Execute approved training commands without invoking a shell."""

    def __init__(self, policy: ExecutionPolicy, receipt_dir: str | Path) -> None:
        self.policy = policy
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        job: TrainingJob,
        cwd: str | Path,
        *,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        self.policy.validate(job.command, cwd)
        workdir = Path(cwd).expanduser().resolve()
        job_dir = self.receipt_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        receipt_path = job_dir / "execution-receipt.json"

        started = time.monotonic()
        job.state = JobState.RUNNING
        timed_out = False
        return_code = -1

        try:
            completed = subprocess.run(
                list(job.command),
                cwd=workdir,
                env=dict(env) if env is not None else None,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
            return_code = completed.returncode
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            job.state = JobState.SUCCEEDED if return_code == 0 else JobState.FAILED
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout_path.write_text(_decode_output(exc.stdout), encoding="utf-8")
            stderr_path.write_text(_decode_output(exc.stderr), encoding="utf-8")
            job.state = JobState.FAILED

        duration = time.monotonic() - started
        receipt = write_receipt(
            receipt_path,
            {
                "job_id": job.job_id,
                "model_id": job.model_id,
                "dataset_id": job.dataset.dataset_id,
                "command": list(job.command),
                "cwd": str(workdir),
                "assigned_nodes": list(job.assigned_nodes),
                "return_code": return_code,
                "duration_seconds": duration,
                "timed_out": timed_out,
                "state": job.state.value,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            },
        )
        return ExecutionResult(
            job_id=job.job_id,
            return_code=return_code,
            duration_seconds=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            receipt_path=str(receipt_path),
            state=receipt["state"],
        )


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
