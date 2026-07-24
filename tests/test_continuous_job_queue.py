import json
import time

import pytest

from auro_native_llm.continuous.job_queue import DurableJobQueue, execute_training_job


def test_queue_persists_leases_and_recovers_expired_work(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    queue = DurableJobQueue(path)
    queued = queue.enqueue("training", {"entrypoint": "scripts/train_him_sft.py", "resume_checkpoint": "missing"})
    base = int(queued["available_at"])
    leased = queue.lease("worker-a", lease_seconds=30, now=base)
    assert leased["job_id"] == queued["job_id"]
    queue.close()

    reopened = DurableJobQueue(path)
    assert reopened.lease("worker-b", lease_seconds=30, now=base + 10) is None
    assert reopened.recover_expired(now=base + 31) == 1
    recovered = reopened.lease("worker-b", lease_seconds=30, now=base + 31)
    assert recovered["job_id"] == queued["job_id"]
    assert recovered["attempts"] == 2
    reopened.close()


def test_completion_requires_lease_owner_and_is_durable(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    queue = DurableJobQueue(path)
    queued = queue.enqueue("training", {"entrypoint": "scripts/train_him_sft.py"})
    queue.lease("worker-a", now=int(queued["available_at"]))
    with pytest.raises(PermissionError):
        queue.complete(queued["job_id"], "worker-b", {"ok": True})
    queue.complete(queued["job_id"], "worker-a", {"ok": True})
    row = queue.db.execute("SELECT status, result_json FROM jobs WHERE job_id=?", (queued["job_id"],)).fetchone()
    assert row["status"] == "completed"
    assert json.loads(row["result_json"])["ok"] is True
    queue.close()


def test_training_execution_rejects_unapproved_entrypoints(tmp_path):
    with pytest.raises(PermissionError, match="not allowlisted"):
        execute_training_job({"entrypoint": "scripts/arbitrary.py"}, tmp_path)


def test_training_execution_requires_real_resume_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="resume checkpoint missing"):
        execute_training_job(
            {
                "entrypoint": "scripts/train_him_sft.py",
                "resume_checkpoint": "checkpoints/missing",
                "output_checkpoint": "checkpoints/output",
            },
            tmp_path,
        )
