"""Supervised fine-tune HIM-style Q→A pairs on the live Auro mind core.

Trains real next-token CE on prompt+answer sequences so dense generation
improves beyond pure knowledge routing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_sft(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def pair_text(row: Dict[str, Any]) -> Tuple[str, str]:
    msgs = row.get("messages") or []
    user = ""
    asst = ""
    for m in msgs:
        if m.get("role") == "user":
            user = str(m.get("content") or "")
        elif m.get("role") == "assistant":
            asst = str(m.get("content") or "")
    return user, asst


def main(argv: list[str] | None = None) -> int:
    import sys

    sys.path.insert(0, str(ROOT))

    p = argparse.ArgumentParser(description="Train HIM SFT on Auro live core")
    p.add_argument("--data", default="data/him_sft.jsonl")
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_physics")
    p.add_argument("--output", default="checkpoints/auro_minds/Auro-2B_him_sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--steps-per-epoch", type=int, default=0, help="0 = one step per sample")
    args = p.parse_args(argv)

    from auro_native_llm.organism.checkpoint import load_mind, save_mind
    from auro_native_llm.physics import get_physics_engine

    data_path = ROOT / args.data
    rows = load_sft(data_path)
    if not rows:
        print("no SFT rows", flush=True)
        return 2

    resume = Path(args.resume)
    print(f"[him-sft] load {resume}", flush=True)
    mind = load_mind(resume, chrome_mock=True, full_runtime=False)
    mind.language.physics = get_physics_engine()
    mind.train_every_act = False
    tok = mind.language.tokenizer
    seq_len = min(args.seq_len, mind.language.config.max_seq_len)

    # Build training sequences: User: …\nAssistant: …
    seqs: List[Tuple[List[int], str]] = []
    for row in rows:
        u, a = pair_text(row)
        if not u or not a:
            continue
        text = f"User: {u}\nAssistant: {a}"
        ids = tok.encode(text, add_bos=True, add_eos=True, max_length=None)
        for i in range(0, max(1, len(ids) - 1), seq_len):
            chunk = ids[i : i + seq_len]
            if len(chunk) < 8:
                continue
            if len(chunk) < seq_len:
                chunk = chunk + [tok.pad_id] * (seq_len - len(chunk))
            seqs.append((chunk[:seq_len], u[:200]))

    print(f"[him-sft] samples={len(rows)} sequences={len(seqs)} epochs={args.epochs}", flush=True)
    if not seqs:
        return 2

    steps0 = mind.language.train_steps
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    rng = np.random.default_rng(42)

    for ep in range(1, args.epochs + 1):
        order = list(range(len(seqs)))
        rng.shuffle(order)
        if args.steps_per_epoch > 0:
            order = order[: args.steps_per_epoch]
        losses = []
        for j, idx in enumerate(order):
            chunk, meaning = seqs[idx]
            arr = np.array([chunk], dtype=np.int64)
            lr = args.lr * (0.97 ** (ep - 1))
            m = mind.language.train_step(arr, arr, lr=lr, text_for_meaning=meaning)
            ce = float(m.get("ce", m.get("loss", 0)))
            losses.append(ce)
            if j == 0 or j == len(order) - 1 or (j + 1) % 10 == 0:
                print(
                    f"  [ep{ep} {j+1}/{len(order)}] ce={ce:.4f} "
                    f"L={m.get('loss', 0):.4f} phys={m.get('physics')}",
                    flush=True,
                )
        history.append(
            {
                "epoch": ep,
                "mean_ce": float(np.mean(losses)),
                "min_ce": float(np.min(losses)),
                "last_ce": losses[-1],
                "n_steps": len(losses),
            }
        )
        print(f"[him-sft] epoch {ep} mean_ce={history[-1]['mean_ce']:.4f}", flush=True)

    out = Path(args.output)
    save_mind(mind, out)
    # also train colony germs lightly on SFT text
    colony_info = {}
    try:
        col = mind.colony(n_extra_germs=24, context_tokens=500_000)
        texts = []
        for row in rows:
            u, a = pair_text(row)
            texts.append(f"{u}\n{a}")
            col.context.ingest(a, kind="doc", meta={"sft": row.get("id")})
        colony_info = col.train_germs(steps=2, texts=texts)
        colony_info["num_params_live"] = col.num_params_live
        colony_info["n_germs"] = col.n_germs
    except Exception as exc:
        colony_info = {"ok": False, "error": str(exc)[:200]}

    report = {
        "schema": "auro.him.sft.v1",
        "ok": True,
        "data": str(data_path),
        "n_pairs": len(rows),
        "n_sequences": len(seqs),
        "epochs": args.epochs,
        "num_params_live": mind.language.num_params,
        "train_steps_delta": mind.language.train_steps - steps0,
        "train_steps": mind.language.train_steps,
        "history": history,
        "mean_ce_first": history[0]["mean_ce"],
        "mean_ce_last": history[-1]["mean_ce"],
        "checkpoint": str(out),
        "colony": colony_info,
        "elapsed_s": time.time() - t0,
        "scaffold": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "HIM_SFT_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
