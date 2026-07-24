from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import TrainConfig
from .dataset import TokenBlockDataset
from .model import AuroForCausalLM


@dataclass(frozen=True)
class TrainingResult:
    run_dir: str
    final_checkpoint: str
    steps: int
    train_loss: float
    validation_loss: float | None
    tokens_seen: int
    duration_seconds: float
    receipt_path: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, world_size, rank, local_rank


def _resolve_device(requested: str, local_rank: int) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _precision(config: TrainConfig, device: torch.device) -> tuple[torch.dtype, bool]:
    if config.precision == "fp32" or device.type == "cpu":
        return torch.float32, False
    if config.precision == "bf16" or (config.precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, device.type == "cuda"


def _schedule(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    config: TrainConfig,
    step: int,
    tokens_seen: int,
    best_validation_loss: float | None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "schema": "auro.foundry.checkpoint.v1",
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config.to_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "best_validation_loss": best_validation_loss,
            "torch_version": torch.__version__,
        },
        temporary,
    )
    temporary.replace(target)
    target.with_suffix(target.suffix + ".sha256").write_text(_sha256_file(target) + "\n", encoding="utf-8")
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != _sha256_file(checkpoint_path):
        raise ValueError("checkpoint hash mismatch")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return payload


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, batches: int, dtype: torch.dtype) -> float:
    model.eval()
    losses: list[float] = []
    autocast = torch.autocast(device_type=device.type, dtype=dtype) if device.type in {"cuda", "cpu"} and dtype != torch.float32 else contextlib.nullcontext()
    for index, (inputs, targets) in enumerate(loader):
        if index >= batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast:
            output = model(inputs, targets)
        if output.loss is not None:
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train(config: TrainConfig) -> TrainingResult:
    config.validate()
    distributed, world_size, rank, local_rank = _distributed()
    if distributed and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    device = _resolve_device(config.device, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype, use_scaler = _precision(config, device)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "train-config.json")

    train_dataset = TokenBlockDataset(Path(config.dataset_dir) / "train.bin", config.sequence_length)
    validation_dataset = TokenBlockDataset(Path(config.dataset_dir) / "validation.bin", config.sequence_length)
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=config.seed) if distributed else None
    loader = DataLoader(train_dataset, batch_size=config.micro_batch_size, sampler=sampler, shuffle=sampler is None, drop_last=True, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=config.micro_batch_size, shuffle=False, drop_last=False)

    model = AuroForCausalLM(config.model).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    start_step = 0
    tokens_seen = 0
    best_validation_loss: float | None = None
    if config.resume_from:
        payload = load_checkpoint(config.resume_from, model=model, optimizer=optimizer, scaler=scaler, map_location=device)
        start_step = int(payload.get("step", 0))
        tokens_seen = int(payload.get("tokens_seen", 0))
        best_validation_loss = payload.get("best_validation_loss")

    iterator = iter(loader)
    running_loss = 0.0
    final_loss = float("nan")
    final_validation: float | None = None
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    autocast_factory = lambda: torch.autocast(device_type=device.type, dtype=dtype) if device.type in {"cuda", "cpu"} and dtype != torch.float32 else contextlib.nullcontext()

    for step in range(start_step, config.max_steps):
        if sampler is not None and step % max(len(loader), 1) == 0:
            sampler.set_epoch(step // max(len(loader), 1))
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_factory():
            output = model(inputs, targets)
            if output.loss is None:
                raise RuntimeError("model did not produce a training loss")
            loss = output.loss / config.gradient_accumulation_steps
        scaler.scale(loss).backward()
        running_loss += float(loss.detach().cpu()) * config.gradient_accumulation_steps
        tokens_seen += inputs.numel() * world_size

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            for group in optimizer.param_groups:
                group["lr"] = _schedule(step, config)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        final_loss = running_loss / max((step - start_step + 1), 1)
        if rank == 0 and (step + 1) % config.log_interval == 0:
            print(json.dumps({"step": step + 1, "loss": round(final_loss, 6), "tokens_seen": tokens_seen, "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        if (step + 1) % config.eval_interval == 0 or step + 1 == config.max_steps:
            final_validation = evaluate(model, validation_loader, device, batches=config.eval_batches, dtype=dtype)
            if best_validation_loss is None or final_validation < best_validation_loss:
                best_validation_loss = final_validation
                if rank == 0:
                    save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer, scaler=scaler, config=config, step=step + 1, tokens_seen=tokens_seen, best_validation_loss=best_validation_loss)
        if rank == 0 and ((step + 1) % config.checkpoint_interval == 0 or step + 1 == config.max_steps):
            save_checkpoint(run_dir / f"step-{step + 1:08d}.pt", model=model, optimizer=optimizer, scaler=scaler, config=config, step=step + 1, tokens_seen=tokens_seen, best_validation_loss=best_validation_loss)

    final_checkpoint = run_dir / "final.pt"
    if rank == 0:
        save_checkpoint(final_checkpoint, model=model, optimizer=optimizer, scaler=scaler, config=config, step=config.max_steps, tokens_seen=tokens_seen, best_validation_loss=best_validation_loss)
        receipt = {
            "schema": "auro.foundry.training_receipt.v1",
            "run_name": config.run_name,
            "model": config.model.to_dict(),
            "steps": config.max_steps,
            "tokens_seen": tokens_seen,
            "train_loss": final_loss,
            "validation_loss": final_validation,
            "duration_seconds": round(time.time() - started, 3),
            "checkpoint": str(final_checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(final_checkpoint),
            "world_size": world_size,
            "device": str(device),
            "dtype": str(dtype),
        }
        receipt_path = run_dir / "training-receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    else:
        receipt_path = run_dir / "training-receipt.json"

    if distributed:
        torch.distributed.barrier()
    return TrainingResult(str(run_dir), str(final_checkpoint), config.max_steps, final_loss, final_validation, tokens_seen, time.time() - started, str(receipt_path))
