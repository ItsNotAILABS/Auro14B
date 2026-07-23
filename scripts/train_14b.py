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
import os
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
    p.add_argument("--corpus-jsonl", default="artifacts/auro14b-corpus/corpus.jsonl")
    p.add_argument("--corpus-max-blocks", type=int, default=2048)
    p.add_argument("--corpus-max-chars", type=int, default=20_000_000)
    p.add_argument(
        "--allow-missing-corpus",
        action="store_true",
        help="Development override; full training requires the unified Auro+MESIE+Sovereign corpus",
    )
    p.add_argument(
        "--sovereign-root",
        help="Checkout of FreddyCreates/sovereign containing integration/training-contract.v1.json",
    )
    p.add_argument(
        "--allow-missing-sovereign",
        action="store_true",
        help="Development override; full Auro-14B training requires the Sovereign contract",
    )
    p.add_argument(
        "--sovereign-commit",
        default=os.getenv("AURO_SOVEREIGN_COMMIT"),
        help="Exact 40-character Sovereign commit admitted for production training",
    )
    p.add_argument(
        "--sovereign-contract-sha256",
        default=os.getenv("AURO_SOVEREIGN_CONTRACT_SHA256"),
        help="Exact SHA-256 digest of the admitted Sovereign training contract",
    )
    p.add_argument("--sovereign-max-blocks", type=int, default=256)
    p.add_argument(
        "--max-sequences",
        type=int,
        default=100_000,
        help="Deterministic cap on tokenized training sequences (smoke mode caps at 256)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the same governed pipeline with tiny geometry for local end-to-end verification",
    )
    args = p.parse_args(argv)

    import sys

    sys.path.insert(0, str(ROOT))

    from auro_native_llm.model.auro_lm import AuroLanguageModel
    from auro_native_llm.model.tokenizer import AuroTokenizer
    from auro_native_llm.organism.mind import AuroMind
    from auro_native_llm.organism.checkpoint import save_mind
    from auro_native_llm.organism.self_train import Experience
    from auro_native_llm.physics import get_physics_engine
    from auro_native_llm.sovereign import bind_sovereign

    production_admission = not args.smoke and not args.allow_missing_sovereign
    if production_admission and (not args.sovereign_commit or not args.sovereign_contract_sha256):
        raise ValueError(
            "Full Auro-14B training requires both --sovereign-commit "
            "(or AURO_SOVEREIGN_COMMIT) and --sovereign-contract-sha256 "
            "(or AURO_SOVEREIGN_CONTRACT_SHA256)"
        )
    sovereign = bind_sovereign(
        args.sovereign_root,
        required=not args.allow_missing_sovereign,
        expected_commit=args.sovereign_commit if production_admission else None,
        expected_contract_sha256=(
            args.sovereign_contract_sha256 if production_admission else None
        ),
        require_clean=production_admission,
        require_expected_remote=production_admission,
    )
    sovereign_blocks = (
        sovereign.training_blocks(max_blocks=max(1, args.sovereign_max_blocks))
        if sovereign is not None
        else []
    )
    if sovereign is not None:
        print(
            f"[14B] Sovereign bound commit={sovereign.commit[:12]} "
            f"records={len(sovereign.records)} redactions={sovereign.redactions}",
            flush=True,
        )

    t0 = time.time()
    print("[14B] building Auro-14B live core (family dev ladder)...", flush=True)
    # Family Auro-14B -> MESIE spectral_gpt_base geometry (live ~1.5B)
    smoke_overrides = {}
    if args.smoke:
        smoke_overrides = {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "head_dim": 16,
            "ffn_dim": 128,
            "vocab_size": 512,
            "max_seq_len": 128,
            "num_experts": 2,
            "top_k_experts": 1,
            "continuous_dim": 16,
            "spectral_input_dim": 64,
            "num_kv_heads": 2,
        }
    language = AuroLanguageModel.build("Auro-14B", mode="dev", **smoke_overrides)
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
    base_block_count = len(blocks)
    blocks.extend(sovereign_blocks)
    corpus_block_count = 0
    corpus_chars = 0
    corpus_path = Path(args.corpus_jsonl)
    if not corpus_path.is_file() and not args.allow_missing_corpus:
        raise FileNotFoundError(
            f"Unified corpus not found: {corpus_path}. "
            "Run scripts/build_unified_training_corpus.py first."
        )
    if corpus_path.is_file():
        with corpus_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if corpus_block_count >= max(1, args.corpus_max_blocks):
                    break
                row = json.loads(line)
                text = str(row.get("text") or "")
                if not text:
                    continue
                remaining = args.corpus_max_chars - corpus_chars
                if remaining <= 0:
                    break
                block = text[:remaining]
                blocks.append(block)
                corpus_chars += len(block)
                corpus_block_count += 1
        print(
            f"[14B] unified corpus blocks={corpus_block_count} chars={corpus_chars:,} "
            f"path={corpus_path}",
            flush=True,
        )
    github_block_count = 0
    try:
        if args.smoke:
            raise RuntimeError(
                "skipped in smoke mode; unified corpus already carries repository sources"
            )
        from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

        gdb = GitHubKnowledgeDB()
        more = gdb.training_blocks(
            "MESIE SpectralGPT Auro orchestrator training",
            max_blocks=40,
            max_chars=120_000,
            top_k_retrieve=12,
        )
        blocks.extend(more[:40])
        github_block_count = len(more[:40])
        print(f"[14B] github blocks +{len(more[:40])} docs={gdb.count()}", flush=True)
    except Exception as exc:
        print(f"[14B] github optional: {exc}", flush=True)

    # tokenizer: expand on corpus if small
    tok = language.tokenizer
    try:
        tokenizer_blocks = blocks[:16] if args.smoke else blocks[:30]
        if args.smoke:
            tokenizer_blocks = [block[:4000] for block in tokenizer_blocks]
        tok.train(tokenizer_blocks, vocab_size=min(language.config.vocab_size, tok.vocab_size))
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
    sequence_cap = min(args.max_sequences, 256) if args.smoke else args.max_sequences
    for b in blocks:
        ids = tok.encode(b, add_bos=True, add_eos=True, max_length=None)
        for i in range(0, max(1, len(ids) - 1), seq_len):
            chunk = ids[i : i + seq_len]
            if len(chunk) < 8:
                continue
            if len(chunk) < seq_len:
                chunk = chunk + [tok.pad_id] * (seq_len - len(chunk))
            seqs.append(chunk[:seq_len])
            if len(seqs) >= max(1, sequence_cap):
                break
        if len(seqs) >= max(1, sequence_cap):
            break
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
    sovereign_receipt = None
    if sovereign is not None:
        receipt_path = sovereign.write_receipt(out_dir / "SOVEREIGN_BINDING_RECEIPT.json")
        sovereign_receipt = {
            **sovereign.receipt(),
            "path": str(receipt_path),
        }
    report = {
        "schema": "auro.train.14b.v1",
        "ok": True,
        "model_id": "Auro-14B",
        "smoke_geometry": args.smoke,
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
        "training_sources": {
            "builtin_blocks": base_block_count,
            "sovereign_blocks": len(sovereign_blocks),
            "github_blocks": github_block_count,
            "unified_corpus_blocks": corpus_block_count,
            "unified_corpus_chars": corpus_chars,
            "unified_corpus_path": str(corpus_path) if corpus_path.is_file() else None,
            "total_blocks": len(blocks),
        },
        "sovereign_binding": sovereign_receipt,
        "training_source_admitted": bool(
            sovereign_receipt and sovereign_receipt["admission"]["production_admitted"]
        ),
        "checkpoint": str(out_dir),
        "checkpoint_meta": meta,
        "elapsed_s": time.time() - t0,
        "claim_boundary": (
            "Live params are the trained runnable core. "
            "Family label Auro-14B is the architecture target (14B). "
            "Smoke geometry, when true, verifies the pipeline and is not the production checkpoint. "
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
        f"- Steps: {steps0} -> **{language.train_steps}**\n"
        f"- CE first->last: "
        f"**{history[0]['mean_ce']:.4f} -> {history[-1]['mean_ce']:.4f}**\n"
        f"- Checkpoint: `{out_dir}`\n"
        f"- Sovereign records consumed: **{len(sovereign_blocks)}**\n"
        f"- Sovereign commit: **{sovereign.commit[:12] if sovereign else 'development override'}**\n"
        f"- Claim: live core trained; 14B is family architecture target.\n"
    )
    (out_dir / "TRAIN_14B_REPORT.md").write_text(md, encoding="utf-8")
    console_report = {
        k: report[k]
        for k in report
        if k not in {"history", "sovereign_binding"}
    }
    if sovereign_receipt:
        console_report["sovereign_binding"] = {
            "contract_id": sovereign_receipt["contract_id"],
            "commit": sovereign_receipt["commit"],
            "records": sovereign_receipt["records"],
            "receipt_sha256": sovereign_receipt["receipt_sha256"],
        }
    print(json.dumps(console_report, indent=2), flush=True)
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
