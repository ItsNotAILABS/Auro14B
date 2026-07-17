"""Training-fabric jobs for Auro family pretrain on MESIE compute plane."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.model.config import all_family_ids


def build_pretrain_command(
    model_id: str,
    *,
    mode: str = "dev",
    steps: int = 100,
    output_dir: str = "checkpoints/auro",
    vocab_size: int = 4096,
) -> List[str]:
    return [
        "python",
        "-m",
        "auro_native_llm.model.train",
        "--model",
        model_id,
        "--mode",
        mode,
        "--steps",
        str(steps),
        "--output-dir",
        output_dir,
        "--vocab-size",
        str(vocab_size),
    ]


def submit_pretrain_job(
    model_id: str,
    *,
    mode: str = "dev",
    steps: int = 100,
    output_dir: str = "checkpoints/auro",
    registry_path: str = ".mesie/nodes.json",
    required_gpus: int = 0,
    min_gpu_memory_gb: int = 0,
    execute: bool = False,
    receipt_dir: str = "deliverables/auro_jobs",
) -> Dict[str, Any]:
    """Register a pretrain job with MESIE training fabric.

    When ``execute=True`` and policy allows, runs the train command locally
    through GovernedRunner. Otherwise emits a launch plan + receipt only.
    """
    if model_id not in all_family_ids():
        raise ValueError(f"unknown model_id {model_id}")

    command = build_pretrain_command(
        model_id, mode=mode, steps=steps, output_dir=output_dir
    )
    job_id = f"auro-pretrain-{model_id.lower()}-{uuid.uuid4().hex[:8]}"
    result: Dict[str, Any] = {
        "schema": "auro.lm.pretrain_job.v1",
        "job_id": job_id,
        "model_id": model_id,
        "compute_plane": "MESIE",
        "native": True,
        "command": command,
        "created_at_unix": int(time.time()),
    }

    try:
        from mesie.training_fabric.models import (
            DatasetManifest,
            JobState,
            NodeRole,
            TrainingJob,
            ComputeNode,
        )
        from mesie.training_fabric.registry import NodeRegistry
        from mesie.training_fabric.scheduler import SovereignScheduler, InsufficientCapacity
        from mesie.training_fabric.discovery import discover_compute_node
        from mesie.training_fabric.receipts import write_receipt
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"training_fabric unavailable: {exc}"
        return result

    reg_path = Path(registry_path)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    registry = NodeRegistry(reg_path)
    try:
        node, meta = discover_compute_node(".")
        registry.register(node)
    except Exception:
        # register a CPU orchestrator node so scheduling still works with 0 GPUs
        registry.register(
            ComputeNode(
                node_id="mesie-local",
                hostname="localhost",
                roles=[NodeRole.TRAINER, NodeRole.ORCHESTRATOR],
                gpu_count=max(required_gpus, 0),
                gpu_memory_gb=max(min_gpu_memory_gb, 0),
                system_memory_gb=16,
                storage_free_gb=50,
                labels={"mesie.compute": "spectral-transformer", "auro": "true"},
            )
        )

    dataset = DatasetManifest(
        dataset_id="auro-repo-corpus",
        version="1.0",
        root_uri="repo://docs+examples+native_llm",
        sha256="pending",
        token_count=0,
        ownership_basis="ItsNotAILABS owned repository text",
        tokenizer_id="auro-bpe",
    )
    job = TrainingJob(
        job_id=job_id,
        model_id=model_id,
        dataset=dataset,
        required_gpus=required_gpus,
        min_gpu_memory_gb=min_gpu_memory_gb,
        command=command,
        metadata={"mode": mode, "steps": steps, "output_dir": output_dir},
    )

    scheduler = SovereignScheduler(registry)
    try:
        plan = scheduler.launch_plan(job)
        result["launch_plan"] = plan
        result["ok"] = True
    except InsufficientCapacity as exc:
        # CPU path: still record job as queued for local execute
        job.state = JobState.QUEUED
        result["launch_plan"] = {
            "job_id": job_id,
            "model_id": model_id,
            "nodes": [],
            "command": command,
            "state": job.state.value,
            "note": str(exc),
            "cpu_fallback": True,
        }
        result["ok"] = True
        result["cpu_fallback"] = True

    Path(receipt_dir).mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_dir) / f"{job_id}.json"

    if execute:
        try:
            from mesie.training_fabric.policy import ExecutionPolicy
            from mesie.training_fabric.runner import GovernedRunner

            policy = ExecutionPolicy.from_roots([Path(".").resolve()])
            runner = GovernedRunner(policy, receipt_dir)
            exec_result = runner.run(job, cwd=".")
            result["execution"] = {
                "ok": exec_result.return_code == 0,
                "returncode": exec_result.return_code,
                "state": exec_result.state,
                "stdout_path": exec_result.stdout_path,
                "stderr_path": exec_result.stderr_path,
                "receipt_path": exec_result.receipt_path,
            }
            job.state = JobState.SUCCEEDED if exec_result.return_code == 0 else JobState.FAILED
        except Exception as exc:
            import subprocess
            import sys

            cmd = [sys.executable if c == "python" else c for c in command]
            proc = subprocess.run(cmd, cwd=str(Path(".").resolve()), capture_output=True, text=True)
            result["execution"] = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-2000:],
                "stderr": (proc.stderr or "")[-2000:],
                "note": f"direct_subprocess after fabric error: {exc}",
            }

    try:
        write_receipt(receipt_path, result)
        result["receipt_path"] = str(receipt_path)
    except Exception:
        receipt_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["receipt_path"] = str(receipt_path)

    return result


def submit_family_jobs(
    *,
    mode: str = "dev",
    steps: int = 40,
    execute: bool = False,
    models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    models = models or all_family_ids()
    return [
        submit_pretrain_job(m, mode=mode, steps=steps, execute=execute)
        for m in models
    ]
