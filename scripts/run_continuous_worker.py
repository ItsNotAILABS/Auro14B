#!/usr/bin/env python3
"""Run a durable AURO continuous-training worker.

The worker executes one job by default. ``--forever`` turns it into a service;
leases and retry state remain in SQLite across restarts.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT))
    from auro_native_llm.continuous.job_queue import DurableJobQueue, execute_training_job

    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="state/continuous-jobs.sqlite3")
    parser.add_argument("--worker-id", default=f"{socket.gethostname()}-{int(time.time())}")
    parser.add_argument("--lease-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--forever", action="store_true")
    args = parser.parse_args()

    queue = DurableJobQueue(ROOT / args.queue)
    processed = 0
    try:
        while True:
            job = queue.lease(args.worker_id, lease_seconds=args.lease_seconds)
            if job is None:
                if not args.forever:
                    break
                time.sleep(max(0.5, args.poll_seconds))
                continue
            try:
                if job["kind"] != "training":
                    raise PermissionError(f"unsupported durable job kind: {job['kind']}")
                result = execute_training_job(job["payload"], ROOT)
                queue.complete(job["job_id"], args.worker_id, result)
                print(json.dumps({"job_id": job["job_id"], "status": "completed", "receipt_sha256": result["receipt_sha256"]}), flush=True)
            except Exception as exc:
                queue.fail(job["job_id"], args.worker_id, f"{type(exc).__name__}: {exc}")
                print(json.dumps({"job_id": job["job_id"], "status": "failed", "error": str(exc)}), flush=True)
            processed += 1
            if not args.forever:
                break
    finally:
        queue.close()
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())
