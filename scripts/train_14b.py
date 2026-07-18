"""Train Auro-14B family LLM (live SpectralGPT core + physics).

Claim boundary
--------------
``parameter_target`` for Auro-14B is 14B (architecture label).
Live params are the countable MESIE SpectralGPT stack built in ``dev``
mode (~1.5B on this ladder). This script trains that live core for real.

Usage
-----
  python scripts/train_14b.py
  python scripts/train_14b.py --steps 8 --rounds 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train Auro-14B live LLM core")
    p.add_argument("--steps", type=int, default=8, help="CE steps per round")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--output", default="checkpoints/auro_minds/Auro-14B")
    args = p.parse_args(argv)

    import sys

    sys.path.insert(0, str(ROOT))

    from auro_native_llm.model.auro_lm import AuroLanguageModel
    from auro_native_llm.model.tokenizer import AuroTokenizer
    from auro_native_llm.organism.mind import AuroMind
    from auro_native_llm.organism.checkpoint import save_mind
    from auro_native_llm.organism.self_train import Experience
    from auro_native_llm.physics import get_physics_engine

    t0 = time.time()
    print("[14B] building Auro-14B live core (family dev ladder)…", flush=True)
    # Family Auro-14B → MESIE spectral_gpt_base geometry (live ~1.5B)
    language = AuroLanguageModel.build("Auro-14B", mode="dev")
    language.physics = get_physics_engine()
    mind = AuroMind(language, chrome_mock=True, absorb_every_act=False, train_every_act=False)
    print(
        f"[14B] live_params={language.num_params:,} "
        f"H={language.config.hidden_dim} L={language.config.num_layers} "
        f"target={language.config.parameter_target:,}",
        flush=True,
    )

    # corpus blocks
    blocks: List[str] = [
        "Auro-14B orchestrator council MESIE SpectralGPT MoE rotary SwiGLU",
        "Physics regularized CE: dispersion coherence Kuramoto Landau Fisher",
        "GHOST hybrid MESIE deterministic path LLM escalate only when justified",
        "Power stack Hamiltonian Klein-Gordon RG Ising Burgers Walras Kelly",
        "Python AI Julia BRAIN virtual physics cores distributed think",
        "Spectral market free energy F=U-TS multi-embed hybrid compression",
        "NOVA promotion receipts coding harness reasoning gate M2 expansion",
        "ItsNotAILABS sovereign local-first agent memory receipts hash chain",
    ]
    try:
        from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

        gdb = GitHubKnowledgeDB()
        more = gdb.training_blocks(
            "MESIE SpectralGPT Auro orchestrator training",
            max_blocks=40,
            max_chars=120_000,
            top_k_retrieve=12,
        )
        blocks.extend(more[:40])
        print(f"[14B] github blocks +{len(more[:40])} docs={gdb.count()}", flush=True)
    except Exception as exc:
        print(f"[14B] github optional: {exc}", flush=True)

    # tokenizer: expand on corpus if small
    tok = language.tokenizer
    try:
        tok.train(blocks[:30], vocab_size=min(language.config.vocab_size, tok.vocab_size))
    except Exception:
        pass

    if mind.organs.trainer:
        for b in blocks[:60]:
            mind.organs.trainer.absorb(
                Experience(
                    text=b[:2000],
                    kind="train_14b",
                    model_id="Auro-14B",
                    reward=0.9,
                )
            )

    # sequences
    seq_len = min(args.seq_len, language.config.max_seq_len)
    seqs: List[List[int]] = []
    for b in blocks:
        ids = tok.encode(b, add_bos=True, add_eos=True, max_length=None)
        for i in range(0, max(1, len(ids) - 1), seq_len):
            chunk = ids[i : i + seq_len]
            if len(chunk) < 8:
                continue
            if len(chunk) < seq_len:
                chunk = chunk + [tok.pad_id] * (seq_len - len(chunk))
            seqs.append(chunk[:seq_len])
    if not seqs:
        ids = tok.encode("Auro-14B MESIE train", max_length=seq_len)
        while len(ids) < seq_len:
            ids.append(tok.pad_id)
        seqs = [ids[:seq_len]]

    print(f"[14B] sequences={len(seqs)} rounds={args.rounds}x{args.steps}", flush=True)
    steps0 = language.train_steps
    history: List[Dict[str, Any]] = []
    rng = np.random.default_rng(14)

    for rnd in range(1, args.rounds + 1):
        losses = []
        print(f"[14B] round {rnd}/{args.rounds}", flush=True)
        for step in range(1, args.steps + 1):
            batch = [
                seqs[int(rng.integers(0, len(seqs)))]
                for _ in range(min(args.batch_size, len(seqs)))
            ]
            arr = np.array(batch, dtype=np.int64)
            meaning = blocks[(rnd * step) % len(blocks)][:400]
            lr = args.lr * (0.995 ** ((rnd - 1) * args.steps + step))
            t1 = time.time()
            m = language.train_step(arr, arr, lr=lr, text_for_meaning=meaning)
            dt = time.time() - t1
            losses.append(float(m.get("ce", m.get("loss", 0))))
            print(
                f"  [r{rnd} {step}/{args.steps}] ce={m.get('ce', 0):.4f} "
                f"L={m.get('loss', 0):.4f} ppl={m.get('ppl', 0):.2f} "
                f"phys={m.get('physics')} R={m.get('phys_resonance', 0):.3f} "
                f"step_s={dt:.1f}",
                flush=True,
            )
        history.append(
            {
                "round": rnd,
                "mean_ce": float(np.mean(losses)),
                "last_ce": losses[-1],
                "min_ce": float(np.min(losses)),
                "train_steps": language.train_steps,
            }
        )

    out_dir = Path(args.output)
    meta = save_mind(mind, out_dir)
    report = {
        "schema": "auro.train.14b.v1",
        "ok": True,
        "model_id": "Auro-14B",
        "parameter_target": language.config.parameter_target,
        "num_params_live": language.num_params,
        "architecture": {
            "hidden_dim": language.config.hidden_dim,
            "num_layers": language.config.num_layers,
            "num_heads": language.config.num_heads,
            "ffn_dim": language.config.ffn_dim,
            "vocab_size": language.config.vocab_size,
            "use_moe": language.config.use_moe,
            "mesie_preset": language.config.mesie_preset,
        },
        "train_steps_before": steps0,
        "train_steps_after": language.train_steps,
        "train_steps_delta": language.train_steps - steps0,
        "history": history,
        "checkpoint": str(out_dir),
        "checkpoint_meta": meta,
        "elapsed_s": time.time() - t0,
        "claim_boundary": (
            "Live params are the trained runnable core. "
            "Family label Auro-14B is the architecture target (14B). "
            "Value is CE drop + physics metrics + durable checkpoint."
        ),
        "scaffold": False,
        "fake": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TRAIN_14B_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    md = (
        f"# Auro-14B Train Report\n\n"
        f"- Live params: **{language.num_params:,}**\n"
        f"- Target label: **{language.config.parameter_target:,}**\n"
        f"- Geometry: H={language.config.hidden_dim} L={language.config.num_layers}\n"
        f"- Steps: {steps0} → **{language.train_steps}**\n"
        f"- CE first→last: "
        f"**{history[0]['mean_ce']:.4f} → {history[-1]['mean_ce']:.4f}**\n"
        f"- Checkpoint: `{out_dir}`\n"
        f"- Claim: live core trained; 14B is family architecture target.\n"
    )
    (out_dir / "TRAIN_14B_REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "history"}, indent=2), flush=True)
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
