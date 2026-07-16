from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mesie.training_fabric.models import DatasetManifest, JobState, TrainingJob
from mesie.training_fabric.policy import ExecutionPolicy, PolicyViolation
from mesie.training_fabric.runner import GovernedRunner


def dataset() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="python-owned-v1",
        version="1.0.0",
        root_uri="livevault://datasets/python-owned-v1",
        sha256="0" * 64,
        token_count=1000,
        ownership_basis="owned",
        tokenizer_id="medina-tokenizer-v1",
    )


def test_policy_denies_unapproved_executable(tmp_path: Path) -> None:
    policy = ExecutionPolicy.from_roots([tmp_path])
    with pytest.raises(PolicyViolation):
        policy.validate(["bash", "-lc", "echo unsafe"], tmp_path)


def test_policy_denies_workdir_outside_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    policy = ExecutionPolicy.from_roots([allowed])
    with pytest.raises(PolicyViolation):
        policy.validate([sys.executable, "-c", "print('x')"], tmp_path)


def test_runner_executes_and_writes_receipt(tmp_path: Path) -> None:
    policy = ExecutionPolicy.from_roots(
        [tmp_path], allowed_executables=[Path(sys.executable).name]
    )
    job = TrainingJob(
        job_id="job-success",
        model_id="auro-python-4b",
        dataset=dataset(),
        required_gpus=0,
        min_gpu_memory_gb=0,
        command=[sys.executable, "-c", "print('medina-runner-ok')"],
    )
    result = GovernedRunner(policy, tmp_path / "receipts").run(job, tmp_path)

    assert result.return_code == 0
    assert result.state == JobState.SUCCEEDED.value
    assert job.state is JobState.SUCCEEDED
    assert Path(result.stdout_path).read_text(encoding="utf-8").strip() == "medina-runner-ok"
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["job_id"] == job.job_id
    assert receipt["receipt_sha256"]
    assert receipt["timed_out"] is False


def test_runner_records_failure(tmp_path: Path) -> None:
    policy = ExecutionPolicy.from_roots(
        [tmp_path], allowed_executables=[Path(sys.executable).name]
    )
    job = TrainingJob(
        job_id="job-failure",
        model_id="auro-python-4b",
        dataset=dataset(),
        required_gpus=0,
        min_gpu_memory_gb=0,
        command=[sys.executable, "-c", "raise SystemExit(7)"],
    )
    result = GovernedRunner(policy, tmp_path / "receipts").run(job, tmp_path)

    assert result.return_code == 7
    assert result.state == JobState.FAILED.value
    assert job.state is JobState.FAILED


def test_runner_records_timeout(tmp_path: Path) -> None:
    policy = ExecutionPolicy.from_roots(
        [tmp_path], allowed_executables=[Path(sys.executable).name]
    )
    job = TrainingJob(
        job_id="job-timeout",
        model_id="auro-python-4b",
        dataset=dataset(),
        required_gpus=0,
        min_gpu_memory_gb=0,
        command=[sys.executable, "-c", "import time; time.sleep(2)"],
    )
    result = GovernedRunner(policy, tmp_path / "receipts").run(
        job, tmp_path, timeout_seconds=0.01
    )

    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert result.return_code == -1
    assert result.state == JobState.FAILED.value
    assert receipt["timed_out"] is True
