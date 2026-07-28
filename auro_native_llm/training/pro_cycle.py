"""Bounded professional training-cycle planner for AURO model candidates.

This module builds a provenance-bearing curriculum and executable job manifest.
It never claims training occurred; completion requires checkpoint, logs, hashes,
and evaluation receipts produced by the invoked trainer and harness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class CurriculumRecord:
    record_id: str
    suite: str
    prompt: str
    answer: str
    source: str
    source_sha256: str
    license: str = "repository-authored"
    admitted: bool = True


def load_jsonl_records(paths: Sequence[str | Path]) -> list[CurriculumRecord]:
    records: list[CurriculumRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = list(row.get("messages") or [])
            prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user").strip()
            answer = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "assistant").strip()
            if not prompt or not answer:
                continue
            suite = str(row.get("suite") or row.get("category") or "general")
            record_id = str(row.get("id") or f"{path.stem}-{index}")
            records.append(CurriculumRecord(record_id, suite, prompt, answer, str(path), file_sha))
    return records


def deduplicate(records: Iterable[CurriculumRecord]) -> tuple[list[CurriculumRecord], int]:
    unique: dict[str, CurriculumRecord] = {}
    total = 0
    for record in records:
        total += 1
        key = _sha({"prompt": record.prompt.strip(), "answer": record.answer.strip()})
        unique.setdefault(key, record)
    return list(unique.values()), total - len(unique)


def build_curriculum_manifest(records: Iterable[CurriculumRecord], required_suites: Sequence[str]) -> dict[str, Any]:
    admitted, duplicate_count = deduplicate(record for record in records if record.admitted)
    counts: dict[str, int] = {}
    for record in admitted:
        counts[record.suite] = counts.get(record.suite, 0) + 1
    missing = [suite for suite in required_suites if counts.get(suite, 0) == 0]
    payload: dict[str, Any] = {
        "schema": "auro.pro-curriculum.v1",
        "records": [asdict(record) for record in admitted],
        "record_count": len(admitted),
        "duplicate_count": duplicate_count,
        "suite_counts": counts,
        "required_suites": list(required_suites),
        "missing_suites": missing,
        "ready": bool(admitted) and not missing,
        "provenance_required": True,
    }
    payload["curriculum_sha256"] = _sha(payload)
    return payload


def build_training_job(
    curriculum: Mapping[str, Any],
    *,
    model_id: str,
    resume_checkpoint: str,
    output_checkpoint: str,
    epochs: int = 3,
    seq_len: int = 256,
    learning_rate: float = 0.001,
) -> dict[str, Any]:
    blockers = []
    if not curriculum.get("ready"):
        blockers.append("curriculum is missing one or more required suites")
    if not resume_checkpoint:
        blockers.append("resume checkpoint is required")
    job: dict[str, Any] = {
        "schema": "auro.pro-training-job.v1",
        "model_id": model_id,
        "entrypoint": "scripts/train_him_sft.py",
        "resume_checkpoint": resume_checkpoint,
        "output_checkpoint": output_checkpoint,
        "curriculum_sha256": curriculum.get("curriculum_sha256"),
        "training_records": int(curriculum.get("record_count", 0)),
        "epochs": max(1, int(epochs)),
        "seq_len": max(32, int(seq_len)),
        "learning_rate": float(learning_rate),
        "required_outputs": [
            "HIM_SFT_REPORT.json",
            "checkpoint hash manifest",
            "loss history",
            "resume proof",
            "pro evaluation report",
            "drift comparison",
        ],
        "promotion_policy": {
            "minimum_harness_score": 0.85,
            "all_critical_suites_pass": True,
            "maximum_regression": 0.02,
            "human_authorization_required": True,
        },
        "blockers": blockers,
        "runnable": not blockers,
    }
    job["job_sha256"] = _sha(job)
    return job


def write_cycle_bundle(root: str | Path, curriculum: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    curriculum_path = root / "curriculum.json"
    job_path = root / "training-job.json"
    receipt_path = root / "cycle-receipt.json"
    curriculum_path.write_text(json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": "auro.pro-cycle-receipt.v1",
        "curriculum_sha256": curriculum.get("curriculum_sha256"),
        "job_sha256": job.get("job_sha256"),
        "status": "scheduled" if job.get("runnable") else "blocked",
        "training_completed": False,
    }
    receipt["receipt_sha256"] = _sha(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"curriculum": str(curriculum_path), "job": str(job_path), "receipt": str(receipt_path)}
