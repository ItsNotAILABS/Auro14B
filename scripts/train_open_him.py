#!/usr/bin/env python3
"""Train and package the local HIM fixture with immutable provenance."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auro_native_llm.open_weights import ByteTokenizer, OpenHIM, OpenHIMConfig, corpus_digest, verify_checkpoint


def examples(text: str, tokenizer: ByteTokenizer, context: int):
    ids = tokenizer.encode(text, bos=True, eos=True)
    padded = [tokenizer.pad_id] * context + ids
    x, y = [], []
    for index in range(context, len(padded) - 1):
        x.append(padded[index - context:index])
        y.append(padded[index])
    return np.asarray(x, np.int64), np.asarray(y, np.int64), len(ids)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/him_open_corpus.txt")
    parser.add_argument("--output", default="checkpoints/open/HIM-native-v0")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--source-commit", default=os.getenv("GITHUB_SHA") or os.getenv("AURO_SOURCE_COMMIT") or "")
    parser.add_argument("--runner-identity", default=(f"github-actions:{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else os.getenv("AURO_RUNNER_IDENTITY") or ""))
    parser.add_argument("--signing-key-env", default="AURO_TRAINING_RECEIPT_HMAC_KEY")
    args = parser.parse_args(argv)
    if not args.source_commit:
        parser.error("source commit is required")
    if not args.runner_identity:
        parser.error("runner identity is required")
    signing_key = os.getenv(args.signing_key_env, "")
    if not signing_key:
        parser.error(f"{args.signing_key_env} is required")

    started = time.time()
    data_path = Path(args.data)
    text = data_path.read_text(encoding="utf-8")
    model = OpenHIM(OpenHIMConfig())
    x, y, n_tokens = examples(text, model.tokenizer, model.config.context_length)
    split = max(1, int(len(x) * 0.9))
    train_x, eval_x = x[:split], x[split:]
    train_y, eval_y = y[:split], y[split:]
    if not len(eval_x):
        raise RuntimeError("held-out split is empty")

    rng = np.random.default_rng(model.config.seed)
    m = {name: np.zeros_like(value) for name, value in model.weights.items()}
    v = {name: np.zeros_like(value) for name, value in model.weights.items()}
    losses = []
    tokens_seen = 0
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train_x), size=min(args.batch_size, len(train_x)))
        xb, yb = train_x[idx], train_y[idx]
        logits, hidden, flat = model.logits(xb)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean())
        losses.append(loss)
        delta = probs
        delta[np.arange(len(yb)), yb] -= 1
        delta /= len(yb)
        grads = {"w2": hidden.T @ delta, "b2": delta.sum(0)}
        dh = (delta @ model.weights["w2"].T) * (1 - hidden * hidden)
        grads["w1"] = flat.T @ dh
        grads["b1"] = dh.sum(0)
        de = (dh @ model.weights["w1"].T).reshape(len(xb), model.config.context_length, model.config.embedding_dim)
        grads["embedding"] = np.zeros_like(model.weights["embedding"])
        np.add.at(grads["embedding"], xb, de)
        for name in model.weights:
            gradient = np.clip(grads[name], -1, 1)
            m[name] = 0.9 * m[name] + 0.1 * gradient
            v[name] = 0.999 * v[name] + 0.001 * gradient * gradient
            mh = m[name] / (1 - 0.9 ** step)
            vh = v[name] / (1 - 0.999 ** step)
            model.weights[name] -= args.lr * mh / (np.sqrt(vh) + 1e-8)
        tokens_seen += len(yb)

    eval_logits, _, _ = model.logits(eval_x)
    eval_logits -= eval_logits.max(1, keepdims=True)
    eval_probs = np.exp(eval_logits)
    eval_probs /= eval_probs.sum(1, keepdims=True)
    eval_loss = float(-np.log(eval_probs[np.arange(len(eval_y)), eval_y] + 1e-12).mean())
    roundtrip = "Auro → NOVA\ncode:\tφ"
    assert model.tokenizer.decode(model.tokenizer.encode(roundtrip)) == roundtrip
    report = {
        "schema": "auro.open_weight_training.v2",
        "model": "HIM-native-v0",
        "architecture": "context_mlp_causal_lm",
        "third_party_base_model": False,
        "distilled_from_api": False,
        "open_weights": True,
        "dtype": "float32",
        "num_parameters": model.num_parameters,
        "corpus_unique_tokens": n_tokens,
        "optimizer_tokens_seen": tokens_seen,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "context_length": model.config.context_length,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "eval_loss": eval_loss,
        "eval_perplexity": float(np.exp(min(eval_loss, 20))),
        "tokenizer_zero_unknown": True,
        "tokenizer_byte_round_trip": True,
        "elapsed_seconds": time.time() - started,
        "sample": model.generate("User: What is HIM?\nAssistant:", max_new_tokens=100, temperature=0.55),
        "claim_boundary": {"pipeline_proven": True, "assistant_quality_proven": False, "st14b_capability_proven": False, "mature_him_intelligence_proven": False},
    }
    package = model.save(args.output, report, corpus_sha256=corpus_digest(data_path), source_commit=args.source_commit, runner_identity=args.runner_identity, signing_key=signing_key)
    verification = verify_checkpoint(args.output, runner_signing_key=signing_key)
    print(json.dumps({**report, **package, "checkpoint_verification": verification}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
