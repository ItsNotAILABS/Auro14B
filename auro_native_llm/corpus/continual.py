"""Continuous training on GitHub knowledge DB with max embeddings.

Loop:
  harvest GitHubs → embed ALL docs at max dim → retrieve → train mind →
  re-index experience → pulse forever (or N rounds)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.corpus.github_db import GitHubKnowledgeDB
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.organism.checkpoint import save_mind
from auro_native_llm.organism.mind import AuroMind
from auro_native_llm.organism.self_train import Experience


@dataclass
class ContinualConfig:
    model_id: str = "Auro-2B"
    rounds: int = 5
    steps_per_round: int = 20
    batch_size: int = 2
    seq_len: int = 96
    vocab_size: int = 2048
    lr: float = 3e-3
    lite: bool = True
    include_github: bool = True
    include_succotash: bool = True
    max_files: int = 3000
    max_chars: int = 8_000_000
    reembed_each_round: bool = False  # full reembed only first time unless True
    top_k_retrieve: int = 24
    train_queries: Optional[List[str]] = None
    output_dir: str = "checkpoints/auro_minds"
    embed_target_dim: int = 0  # 0 = max concat
    # Resume / keep-going
    resume_checkpoint: Optional[str] = None  # e.g. checkpoints/auro_minds/Auro-2B_continual
    skip_harvest_if_docs: int = 500  # reuse DB if already populated
    expand_harvest: bool = True  # still pull more docs into existing DB


DEFAULT_QUERIES = [
    "MESIE spectral intelligence SpectralGPT MoE",
    "sovereign organism memory temple phi golden ratio",
    "Chrome CDP work agent Monaco Jupyter MCP",
    "Auro language model training corpus doctrine",
    "Pythonista UI bridge background scripts database",
    "defense organism sentry governance protocols",
    "FreddyCreates potential-succotash engines models agents",
    "receipts NOVA LOOM CAPSULA NeuroCore Solus",
]


def run_continual_training(cfg: Optional[ContinualConfig] = None) -> Dict[str, Any]:
    """Harvest → max-embed GitHub DB → train mind multi-round."""
    cfg = cfg or ContinualConfig()
    t0 = time.time()
    report: Dict[str, Any] = {
        "schema": "auro.continual.github.v1",
        "loop": "harvest → max_embed → retrieve → train → save",
        "config": asdict(cfg),
    }

    print("[continual] init GitHub knowledge DB + max embedder", flush=True)
    gdb = GitHubKnowledgeDB(max_dim=cfg.embed_target_dim)
    print(f"[continual] embedder dim={gdb.embedder.dim} info={gdb.embedder.info()}", flush=True)

    # 1) Harvest + max embed (reuse dense DB when already full)
    existing = gdb.count()
    if existing >= cfg.skip_harvest_if_docs and not cfg.expand_harvest:
        print(f"[continual] reusing GitHub DB docs={existing} (skip harvest)", flush=True)
        harvest = {"ok": True, "reused": True, "total": existing}
        if gdb.stats().get("embedding_n", 0) < existing:
            harvest["embeddings"] = gdb.rebuild_embeddings()
    else:
        print(
            f"[continual] harvesting GitHubs + succotash "
            f"(existing={existing}, expand={cfg.expand_harvest})…",
            flush=True,
        )
        harvest = gdb.harvest_and_ingest(
            include_github=cfg.include_github,
            include_succotash=cfg.include_succotash,
            max_files=cfg.max_files,
            max_chars=cfg.max_chars,
            reembed=True,
        )
    report["harvest"] = harvest
    report["db_stats"] = gdb.stats()
    print(
        f"[continual] DB docs={gdb.count()} repos={len(gdb.repo_counts())} "
        f"emb_dim={gdb.stats().get('embedding_dim')} n={gdb.stats().get('embedding_n')}",
        flush=True,
    )

    # 2) Build or resume mind
    from auro_native_llm.model.auro_lm import AuroLanguageModel
    from auro_native_llm.model.config import mesie_preset_dims
    from auro_native_llm.organism.checkpoint import load_mind

    mind: AuroMind
    resumed = False
    resume_path = cfg.resume_checkpoint
    if resume_path and Path(resume_path).exists():
        print(f"[continual] resuming mind from {resume_path}", flush=True)
        mind = load_mind(resume_path, chrome_mock=True)
        resumed = True
        # keep tokenizer from checkpoint language
        tokenizer = mind.language.tokenizer
    else:
        texts_for_tok = gdb.training_blocks(max_blocks=80, max_chars=200_000)
        tokenizer = AuroTokenizer(vocab_size=cfg.vocab_size)
        tokenizer.train(
            texts_for_tok[:60] if texts_for_tok else ["MESIE Auro spectral"],
            vocab_size=cfg.vocab_size,
        )

        overrides: Dict[str, Any] = {
            "vocab_size": max(cfg.vocab_size, tokenizer.vocab_size),
            "max_seq_len": max(cfg.seq_len, 128),
            "use_moe": True,
            "use_cross_modal": True,
            "use_spectral_encoder": True,
            "use_spectral_fusion": True,
            "learning_rate": cfg.lr,
        }
        if cfg.lite:
            tiny = mesie_preset_dims("spectral_gpt_tiny")
            overrides.update(tiny)
            overrides["mesie_preset"] = "spectral_gpt_tiny"
            overrides["vocab_size"] = max(cfg.vocab_size, tokenizer.vocab_size)
            overrides["max_seq_len"] = max(cfg.seq_len, 128)

        language = AuroLanguageModel.build(
            cfg.model_id, mode="dev", tokenizer=tokenizer, **overrides
        )
        mind = AuroMind(language, chrome_mock=True, absorb_every_act=True, train_every_act=False)

    report["resumed"] = resumed
    report["resume_checkpoint"] = resume_path
    if mind.organs.trainer:
        mind.organs.trainer.lr = cfg.lr
        mind.organs.trainer.batch_size = cfg.batch_size
        mind.organs.trainer.seq_len = cfg.seq_len

    # Seed buffer with embedding-tagged experiences from GitHub DB
    queries = cfg.train_queries or DEFAULT_QUERIES
    for q in queries:
        for hit in gdb.search(q, top_k=min(8, cfg.top_k_retrieve)):
            emb = gdb.embedder.embed_text(hit.text).tolist()
            mind.organs.trainer.absorb(
                Experience(
                    text=hit.text[:2000],
                    kind="github_db",
                    model_id=cfg.model_id,
                    reward=0.75 + 0.2 * max(0.0, min(1.0, hit.score)),
                    embedding=emb,
                    meta={"repo": hit.repo, "path": hit.path, "score": hit.score, "query": q},
                )
            )

    # 3) Multi-round train with retrieval refresh
    history: List[Dict[str, Any]] = []
    for rnd in range(1, cfg.rounds + 1):
        print(f"[continual] round {rnd}/{cfg.rounds}", flush=True)
        if cfg.reembed_each_round and rnd > 1:
            gdb.rebuild_embeddings()

        # gather sequences from retrieval-augmented blocks
        q = queries[(rnd - 1) % len(queries)]
        blocks = gdb.training_blocks(q, max_blocks=120, max_chars=400_000, top_k_retrieve=cfg.top_k_retrieve)
        if not blocks:
            blocks = gdb.training_blocks(max_blocks=80, max_chars=300_000)

        # structured CE steps
        seqs = []
        for b in blocks:
            ids = tokenizer.encode(b, add_bos=True, add_eos=True, max_length=None)
            for i in range(0, max(1, len(ids) - 1), cfg.seq_len):
                chunk = ids[i : i + cfg.seq_len]
                if len(chunk) < 8:
                    continue
                if len(chunk) < cfg.seq_len:
                    chunk = chunk + [tokenizer.pad_id] * (cfg.seq_len - len(chunk))
                seqs.append(chunk[: cfg.seq_len])
        if not seqs:
            seqs = [tokenizer.encode("MESIE Auro continuous train", max_length=cfg.seq_len)]
            while len(seqs[0]) < cfg.seq_len:
                seqs[0].append(tokenizer.pad_id)

        rng = np.random.default_rng(42 + rnd)
        losses = []
        for step in range(1, cfg.steps_per_round + 1):
            pick = [
                seqs[int(i)]
                for i in rng.integers(0, len(seqs), size=min(cfg.batch_size, len(seqs)))
            ]
            arr = np.array(pick, dtype=np.int64)
            meaning = blocks[step % len(blocks)][:400]
            m = mind.language.train_step(arr, arr, lr=cfg.lr * (0.997 ** step), text_for_meaning=meaning)
            if mind.organs.trainer:
                messy = mind.organs.trainer.train_on_model(mind.language, steps=1)
                m["messy"] = messy.get("last_loss")
            losses.append(float(m.get("ce", m.get("loss", 0))))
            if step == 1 or step == cfg.steps_per_round or step % 5 == 0:
                print(
                    f"  [r{rnd} step {step}/{cfg.steps_per_round}] ce={m.get('ce', 0):.4f} "
                    f"ppl={m.get('ppl', 0):.2f}",
                    flush=True,
                )

        # retrieval smoke
        probe_q = "MESIE SpectralGPT Auro native"
        hits = gdb.search(probe_q, top_k=5)
        gen = mind.generate(
            f"From GitHub knowledge about {q[:60]}:",
            max_new_tokens=24,
        )
        # pulse keep learning
        pulse = mind.pulse() if hasattr(mind, "pulse") else {"ok": True}

        row = {
            "round": rnd,
            "query": q,
            "mean_ce": float(np.mean(losses)) if losses else None,
            "last_ce": losses[-1] if losses else None,
            "blocks": len(blocks),
            "sequences": len(seqs),
            "retrieve_top": [h.to_dict() for h in hits],
            "generate_ok": getattr(gen, "ok", True),
            "pulse_ok": pulse.get("ok") if isinstance(pulse, dict) else True,
            "train_steps": mind.language.train_steps,
            "params_live": mind.language.num_params,
        }
        history.append(row)
        print(
            f"  [r{rnd}] mean_ce={row['mean_ce']:.4f} retrieve_top1="
            f"{hits[0].repo if hits else '-'}::{hits[0].path if hits else '-'}",
            flush=True,
        )

    # 4) Save mind + report
    out_dir = Path(cfg.output_dir) / f"{cfg.model_id.replace('/', '_')}_continual"
    meta = save_mind(mind, out_dir)
    report.update(
        {
            "ok": True,
            "history": history,
            "checkpoint": str(out_dir),
            "checkpoint_meta": meta,
            "num_params_live": mind.language.num_params,
            "train_steps": mind.language.train_steps,
            "db_stats_final": gdb.stats(),
            "elapsed_s": time.time() - t0,
            "claim_boundary": (
                "Live params = running model. GitHub DB + max embeddings are the info plane. "
                "Family labels remain architecture targets."
            ),
        }
    )
    (out_dir / "CONTINUAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = (
        f"# Continual GitHub Training\n\n"
        f"- Docs: **{gdb.count()}** across **{len(gdb.repo_counts())}** repos\n"
        f"- Embedding dim: **{gdb.stats().get('embedding_dim')}** (n={gdb.stats().get('embedding_n')})\n"
        f"- Rounds: **{cfg.rounds}** × {cfg.steps_per_round} steps\n"
        f"- Live params: **{mind.language.num_params:,}**\n"
        f"- Final mean CE (last round): **{history[-1].get('mean_ce') if history else 'n/a'}**\n"
        f"- Checkpoint: `{out_dir}`\n"
    )
    (out_dir / "CONTINUAL_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Continual train on GitHub knowledge DB + max embeddings")
    p.add_argument("--model", default="Auro-2B")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-files", type=int, default=2500)
    p.add_argument("--no-github", action="store_true")
    p.add_argument("--full-core", action="store_true")
    p.add_argument("--output-dir", default="checkpoints/auro_minds")
    p.add_argument("--resume", default=None, help="Mind checkpoint dir to continue")
    p.add_argument("--no-expand", action="store_true", help="Do not re-harvest if DB full")
    args = p.parse_args()
    report = run_continual_training(
        ContinualConfig(
            model_id=args.model,
            rounds=args.rounds,
            steps_per_round=args.steps,
            batch_size=args.batch_size,
            max_files=args.max_files,
            include_github=not args.no_github,
            lite=not args.full_core,
            output_dir=args.output_dir,
            resume_checkpoint=args.resume,
            expand_harvest=not args.no_expand,
        )
    )
    slim = {
        k: report[k]
        for k in (
            "ok",
            "num_params_live",
            "train_steps",
            "db_stats",
            "elapsed_s",
            "checkpoint",
        )
        if k in report
    }
    slim["history_tail"] = (report.get("history") or [])[-2:]
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
