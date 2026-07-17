from __future__ import annotations

import json
import sys
from pathlib import Path

from mesie.training_fabric import (
    DatasetManifest,
    ExecutionPolicy,
    GovernedRunner,
    TrainingJob,
)


def run_training_governed(
    config_path: str | Path,
    *,
    workdir: str | Path = ".",
    receipt_dir: str | Path = "artifacts/auro-foundry/mesie-receipts",
    timeout_seconds: float | None = None,
):
    """Launch Auro training through MESIE's shell-free governed runner."""
    config_path = Path(config_path).expanduser().resolve()
    root = Path(workdir).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_dir = Path(config["dataset_dir"]).expanduser().resolve()
    manifest_path = dataset_dir / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = DatasetManifest(
        dataset_id=manifest.get("schema", "auro-foundry-dataset"),
        version="1",
        root_uri=str(dataset_dir),
        sha256=manifest["manifest_sha256"],
        token_count=int(manifest["train"]["tokens"] + manifest["validation"]["tokens"]),
        ownership_basis="operator-authorized Medina repositories and local roots",
        tokenizer_id=manifest["tokenizer_sha256"],
    )
    command = [sys.executable, "-m", "auro_foundry.cli", "train", "--config", str(config_path)]
    job = TrainingJob(
        job_id=str(config.get("run_name", "auro-foundry-run")),
        model_id=str(config["model"]["model_id"]),
        dataset=dataset,
        required_gpus=0,
        min_gpu_memory_gb=0,
        command=command,
    )
    policy = ExecutionPolicy.from_roots([root], allowed_executables=[Path(sys.executable).name])
    return GovernedRunner(policy, receipt_dir).run(
        job,
        root,
        timeout_seconds=timeout_seconds,
    )
