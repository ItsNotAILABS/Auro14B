#!/usr/bin/env python3
"""Matched dense-vs-structured toy benchmark with a machine-readable receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch import nn

from auro_native_llm.experiments.structured_transformer import StructuredTransformerBlock, parameter_count


class DenseBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, values):
        normalized = self.norm1(values)
        mask = torch.ones(values.shape[1], values.shape[1], device=values.device, dtype=torch.bool).triu(1)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        values = values + attended
        return values + self.ff(self.norm2(values))


class ToyLM(nn.Module):
    def __init__(self, block: nn.Module, vocab: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.block = block
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens):
        return self.head(self.norm(self.block(self.embedding(tokens))))


@dataclass
class Result:
    name: str
    parameter_count: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    tokens_per_second: float


def run(name, model, batches, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    start = time.perf_counter()
    tokens = 0
    for inputs, targets in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        tokens += inputs.numel()
    elapsed = time.perf_counter() - start
    return Result(name, parameter_count(model), losses[0], losses[-1], elapsed, tokens / elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("evidence/structured-transformer-benchmark.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    vocab, d_model, heads, d_ff, batch, sequence = 97, 32, 4, 64, 8, 24
    batches = []
    generator = torch.Generator().manual_seed(args.seed + 1)
    for _ in range(args.steps):
        inputs = torch.randint(vocab, (batch, sequence), generator=generator)
        targets = torch.roll(inputs, shifts=-1, dims=1)
        batches.append((inputs, targets))
    torch.manual_seed(args.seed)
    dense = ToyLM(DenseBlock(d_model, heads, d_ff), vocab, d_model)
    torch.manual_seed(args.seed)
    structured = ToyLM(StructuredTransformerBlock(d_model, heads, d_ff=d_ff, seed=args.seed), vocab, d_model)
    results = [run("dense", dense, batches, 3e-3), run("structured", structured, batches, 3e-3)]
    payload = {
        "schema": "auro.structured_transformer_benchmark.v1",
        "config": {"steps": args.steps, "seed": args.seed, "vocab": vocab, "d_model": d_model, "heads": heads, "d_ff": d_ff, "batch": batch, "sequence": sequence},
        "results": [asdict(item) for item in results],
        "claims": {"speed_superiority": False, "quality_superiority": False, "hallucination_reduction": False},
        "promotion_status": "experiment_only",
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
