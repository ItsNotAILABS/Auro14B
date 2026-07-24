#!/usr/bin/env python3
"""Build a bounded AURO professional training cycle from repository SFT data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", default=["data/him_sft.jsonl"])
    parser.add_argument("--model-id", default="Auro-2B")
    parser.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_him_sft")
    parser.add_argument("--output-checkpoint", default="checkpoints/candidates/Auro-2B-pro")
    parser.add_argument("--bundle", default="artifacts/pro-training-cycle")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    from auro_native_llm.training.pro_cycle import (
        build_curriculum_manifest,
        build_training_job,
        load_jsonl_records,
        write_cycle_bundle,
    )

    paths = [ROOT / path for path in args.data]
    records = load_jsonl_records(paths)
    required = ("general",)
    curriculum = build_curriculum_manifest(records, required)
    job = build_training_job(
        curriculum,
        model_id=args.model_id,
        resume_checkpoint=args.resume,
        output_checkpoint=args.output_checkpoint,
        epochs=args.epochs,
        seq_len=args.seq_len,
        learning_rate=args.lr,
    )
    written = write_cycle_bundle(ROOT / args.bundle, curriculum, job)
    print(json.dumps({
        "curriculum_ready": curriculum["ready"],
        "records": curriculum["record_count"],
        "job_runnable": job["runnable"],
        "job_sha256": job["job_sha256"],
        "files": written,
    }, indent=2))
    return 0 if job["runnable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
