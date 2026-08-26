from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DistributedTrainContract:
    schema: str = "auro.st14b.distributed_train_contract.v1"
    model_id: str = "AURO-ST-14B"
    parameter_count: int = 14_339_691_520
    precision: str = "bf16"
    strategy: str = "fsdp2_or_zero3"
    sequence_length: int = 8192
    global_batch_tokens: int = 4_194_304
    checkpoint_format: str = "safetensors_sharded"
    activation_checkpointing: bool = True
    native_gqa_required: bool = True
    tokenizer_digest_required: bool = True
    corpus_digest_required: bool = True
    resume_verification_required: bool = True
    maturity: str = "PROTOTYPE"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path, allowed: set[str]) -> dict[str, Any]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in allowed:
            continue
        if any(part in {".git", "node_modules", "artifacts", "checkpoints", "dist"} for part in p.parts):
            continue
        files.append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": files, "sha256": digest}


def write_launch_receipt(root: Path, out: Path, tokenizer: Path | None = None) -> dict[str, Any]:
    contract = DistributedTrainContract()
    corpus = hash_tree(root, {".md", ".txt", ".json", ".py", ".js", ".ts", ".tsx", ".mo"})
    tok = {"present": False, "sha256": None, "path": None}
    if tokenizer is not None and tokenizer.exists():
        tok = {"present": True, "sha256": sha256_file(tokenizer), "path": str(tokenizer)}
    receipt = {
        "schema": "auro.st14b.train_launch_receipt.v1",
        "created_at": int(time.time()),
        "host": socket.gethostname(),
        "contract": asdict(contract),
        "corpus_manifest": corpus,
        "tokenizer": tok,
        "environment": {
            "world_size": int(os.getenv("WORLD_SIZE", "1")),
            "rank": int(os.getenv("RANK", "0")),
            "local_rank": int(os.getenv("LOCAL_RANK", "0")),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        "promotion_boundary": {
            "full_parameter_training_started": False,
            "full_parameter_checkpoint_produced": False,
            "h100_runtime_verified": False,
            "production_promoted": False,
        },
    }
    receipt["evidence_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def launch(command: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(json.dumps({"dry_run": True, "command": command}, indent=2))
        return 0
    if not command:
        raise SystemExit("training command required unless --dry-run")
    return subprocess.call(command)


def main() -> None:
    p = argparse.ArgumentParser(description="AURO ST-14B distributed training launch boundary")
    p.add_argument("--root", default=".")
    p.add_argument("--tokenizer")
    p.add_argument("--receipt", default="artifacts/st14b/train-launch-receipt.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    receipt = write_launch_receipt(Path(args.root).resolve(), Path(args.receipt), Path(args.tokenizer) if args.tokenizer else None)
    if not receipt["corpus_manifest"]["files"]:
        raise SystemExit("refusing launch: corpus manifest is empty")
    if args.tokenizer and not receipt["tokenizer"]["present"]:
        raise SystemExit("refusing launch: tokenizer file missing")
    sys.exit(launch(args.command, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
