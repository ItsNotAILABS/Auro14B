"""Background Python scripts under the JS service shell (Pythonista pattern).

Inspired by:
  - Pythonista background script lifetime
  - potential-succotash windows-runtime-sdk BackgroundTaskScheduler
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

PHI_HEARTBEAT_MS = 873


@dataclass
class BackgroundJob:
    job_id: str
    name: str
    state: str = "registered"  # registered|running|done|error|cancelled
    interval_s: float = 0.0  # 0 = one-shot
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_error: Optional[str] = None
    result: Any = None
    logs: List[str] = field(default_factory=list)
    runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "state": self.state,
            "interval_s": self.interval_s,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "runs": self.runs,
            "logs_tail": self.logs[-20:],
            "result_preview": str(self.result)[:500] if self.result is not None else None,
        }


class BackgroundScheduler:
    """JS service keeps this alive; Python jobs run in worker threads."""

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pyista")
        self._jobs: Dict[str, BackgroundJob] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"bg_{int(time.time())}_{self._seq}"

    def submit(
        self,
        fn: Callable[[], Any],
        *,
        name: str = "script",
        interval_s: float = 0.0,
        job_id: Optional[str] = None,
    ) -> BackgroundJob:
        jid = job_id or self._next_id()
        job = BackgroundJob(job_id=jid, name=name, interval_s=interval_s, state="running")
        with self._lock:
            self._jobs[jid] = job

        def runner() -> Any:
            job.started_at = time.time()
            job.state = "running"
            try:
                if interval_s and interval_s > 0:
                    # periodic until cancelled
                    while job.state != "cancelled":
                        job.runs += 1
                        job.logs.append(f"run#{job.runs} @ {time.time():.0f}")
                        job.result = fn()
                        if job.state == "cancelled":
                            break
                        time.sleep(interval_s)
                    job.state = "done" if job.state != "cancelled" else "cancelled"
                else:
                    job.runs = 1
                    job.result = fn()
                    job.state = "done"
            except Exception as exc:
                job.state = "error"
                job.last_error = f"{exc}\n{traceback.format_exc()}"
                job.logs.append(job.last_error)
            finally:
                job.finished_at = time.time()
            return job.result

        fut = self._pool.submit(runner)
        with self._lock:
            self._futures[jid] = fut
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.state = "cancelled"
        return True

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [j.to_dict() for j in self._jobs.values()]

    def status(self) -> Dict[str, Any]:
        return {
            "jobs": len(self._jobs),
            "running": sum(1 for j in self._jobs.values() if j.state == "running"),
            "heartbeat_ms": PHI_HEARTBEAT_MS,
            "jobs_list": self.list_jobs()[-20:],
        }
