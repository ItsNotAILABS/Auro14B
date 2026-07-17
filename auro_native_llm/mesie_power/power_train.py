"""Powered training — bigger MESIE core + multi-embed weighted batches."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from auro_native_llm.mesie_power.compress import MesieCompressor
from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder


@dataclass
class PowerProfile:
    """Live core geometry for powered training (still laptop-real)."""

    name: str = "mesie_power_mid"
    # Between tiny (256/4) and small (512/12) — real param jump
    hidden_dim: int = 384
    num_layers: int = 8
    num_heads: int = 8
    head_dim: int = 48
    ffn_dim: int = 1536
    num_experts: int = 8
    top_k_experts: int = 2
    max_seq_len: int = 256
    vocab_size: int = 2048
    use_moe: bool = True
    use_cross_modal: bool = True
    use_spectral_encoder: bool = True
    use_spectral_fusion: bool = True
    mesie_preset: str = "spectral_gpt_tiny"  # label only; dims override


# ladder
PROFILES = {
    "tiny": PowerProfile(name="tiny", hidden_dim=256, num_layers=4, num_heads=4, head_dim=64, ffn_dim=1024),
    "power": PowerProfile(),  # mid default
    "small": PowerProfile(
        name="mesie_small",
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        head_dim=64,
        ffn_dim=2048,
        num_experts=8,
        max_seq_len=384,
        vocab_size=4096,
        mesie_preset="spectral_gpt_small",
    ),
    # ~1.09B live params (measured SpectralGPT causal+MoE+xmodal+spec_enc)
    "billion": PowerProfile(
        name="mesie_1b",
        hidden_dim=1024,
        num_layers=16,
        num_heads=16,
        head_dim=64,
        ffn_dim=4096,
        num_experts=8,
        top_k_experts=2,
        max_seq_len=512,  # train-friendly; architecture can go higher
        vocab_size=16384,
        mesie_preset="spectral_gpt_base",
    ),
}


def power_family_overrides(profile: str | PowerProfile = "power") -> Dict[str, Any]:
    p = PROFILES.get(profile, profile) if isinstance(profile, str) else profile
    assert isinstance(p, PowerProfile)
    return {
        "hidden_dim": p.hidden_dim,
        "num_layers": p.num_layers,
        "num_heads": p.num_heads,
        "head_dim": p.head_dim,
        "ffn_dim": p.ffn_dim,
        "num_experts": p.num_experts,
        "top_k_experts": p.top_k_experts,
        "max_seq_len": p.max_seq_len,
        "vocab_size": p.vocab_size,
        "use_moe": p.use_moe,
        "use_cross_modal": p.use_cross_modal,
        "use_spectral_encoder": p.use_spectral_encoder,
        "use_spectral_fusion": p.use_spectral_fusion,
        "mesie_preset": p.mesie_preset,
        "positional_encoding": "rotary",
        "normalization": "rms_norm",
        "activation": "swiglu",
        "qk_norm": True,
        "num_modalities": 8,
    }


@dataclass
class PowerTrainConfig:
    model_id: str = "Auro-2B"
    profile: str = "power"
    steps: int = 40
    batch_size: int = 2
    seq_len: int = 128
    lr: float = 2e-3
    rounds: int = 3
    compress_method: str = "hybrid"
    compress_dim: int = 256
    multi_embed: bool = True
    resume_checkpoint: Optional[str] = "checkpoints/auro_minds/Auro-2B_continual"
    output_dir: str = "checkpoints/auro_minds"
    use_github: bool = True
    show: bool = True


def run_power_train(cfg: Optional[PowerTrainConfig] = None) -> Dict[str, Any]:
    """Train with multi-embed retrieval weighting + powered MESIE core."""
    from auro_native_llm.model.auro_lm import AuroLanguageModel
    from auro_native_llm.model.tokenizer import AuroTokenizer
    from auro_native_llm.organism.checkpoint import load_mind, save_mind
    from auro_native_llm.organism.mind import AuroMind
    from auro_native_llm.organism.self_train import Experience

    cfg = cfg or PowerTrainConfig()
    t0 = time.time()
    embedder = MultiMesieEmbedder()
    compressor = MesieCompressor(method=cfg.compress_method, target_dim=cfg.compress_dim)

    if cfg.show:
        print(f"[power] multi-embed dim={embedder.dim} views={embedder.info()['views']}", flush=True)
        print(f"[power] profile={cfg.profile} overrides={power_family_overrides(cfg.profile)}", flush=True)

    # GitHub blocks + multi-embed scores
    blocks: List[str] = []
    emb_matrix = None
    ids: List[str] = []
    if cfg.use_github:
        try:
            from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

            gdb = GitHubKnowledgeDB()
            blocks = gdb.training_blocks(max_blocks=200, max_chars=600_000)
            if cfg.show:
                print(f"[power] github blocks={len(blocks)} docs={gdb.count()}", flush=True)
        except Exception as exc:
            if cfg.show:
                print(f"[power] github fallback: {exc}", flush=True)
    if not blocks:
        blocks = [
            "MESIE SpectralGPT MoE Auro native continuous training",
            "Python organ doctrine PL-001 PL-009 autocycle",
            "phi golden ratio spectral embeddings compression",
        ]

    # multi-embed all blocks + compress bank
    if cfg.multi_embed:
        if cfg.show:
            print(f"[power] embedding {len(blocks)} blocks multi-MESIE…", flush=True)
        emb_matrix = embedder.embed_batch([b[:4000] for b in blocks])
        ids = [f"b{i}" for i in range(len(blocks))]
        bank = compressor.compress_bank(emb_matrix, ids, fit=True)
        if cfg.show:
            print(f"[power] compress {bank.info()}", flush=True)
    else:
        bank = None

    # Build or upgrade mind to power profile
    overrides = power_family_overrides(cfg.profile)
    mind: AuroMind
    upgraded = False
    if cfg.resume_checkpoint and Path(cfg.resume_checkpoint).exists():
        # resume weights only if same geometry; else fresh power core + absorb doctrine
        try:
            old = load_mind(cfg.resume_checkpoint, chrome_mock=True)
            same = (
                old.config.hidden_dim == overrides["hidden_dim"]
                and old.config.num_layers == overrides["num_layers"]
            )
            if same:
                mind = old
                if cfg.show:
                    print(f"[power] resumed same geometry params={mind.language.num_params}", flush=True)
            else:
                upgraded = True
                if cfg.show:
                    print(
                        f"[power] upgrade core {old.config.hidden_dim}x{old.config.num_layers} "
                        f"→ {overrides['hidden_dim']}x{overrides['num_layers']}",
                        flush=True,
                    )
                tok = old.language.tokenizer
                language = AuroLanguageModel.build(
                    cfg.model_id, mode="dev", tokenizer=tok, **overrides
                )
                mind = AuroMind(language, chrome_mock=True)
                # transfer experience buffer texts
                if old.organs.trainer and mind.organs.trainer:
                    for exp in list(old.organs.trainer.buffer)[-200:]:
                        mind.organs.trainer.absorb(exp)
            # drop heavy ref
            del old
        except Exception as exc:
            if cfg.show:
                print(f"[power] resume failed ({exc}); fresh power core", flush=True)
            tok = AuroTokenizer(vocab_size=overrides["vocab_size"])
            tok.train(blocks[:40], vocab_size=overrides["vocab_size"])
            language = AuroLanguageModel.build(
                cfg.model_id, mode="dev", tokenizer=tok, **overrides
            )
            mind = AuroMind(language, chrome_mock=True)
    else:
        tok = AuroTokenizer(vocab_size=overrides["vocab_size"])
        tok.train(blocks[:40], vocab_size=overrides["vocab_size"])
        language = AuroLanguageModel.build(
            cfg.model_id, mode="dev", tokenizer=tok, **overrides
        )
        mind = AuroMind(language, chrome_mock=True)

    # seed multi-embed experiences
    if emb_matrix is not None and mind.organs.trainer:
        for i, b in enumerate(blocks[:80]):
            vec = emb_matrix[i]
            code = compressor.transform(vec) if bank else vec
            mind.organs.trainer.absorb(
                Experience(
                    text=b[:2000],
                    kind="multi_embed_github",
                    model_id=cfg.model_id,
                    reward=0.85,
                    embedding=code.tolist(),
                    meta={"view": "multi_mesie", "compress": cfg.compress_method},
                )
            )

    tokenizer = mind.language.tokenizer
    history: List[Dict[str, Any]] = []
    steps0 = mind.language.train_steps

    for rnd in range(1, cfg.rounds + 1):
        # sample blocks weighted by multi-embed self-similarity to round query
        q = blocks[(rnd * 7) % len(blocks)]
        if emb_matrix is not None:
            qv = embedder.embed_text(q[:2000])
            scores = emb_matrix @ qv
            # temperature sample
            s = scores - scores.max()
            w = np.exp(s / 0.15)
            w = w / w.sum()
            pick_idx = np.random.choice(len(blocks), size=min(40, len(blocks)), replace=False, p=w)
        else:
            pick_idx = np.random.choice(len(blocks), size=min(40, len(blocks)), replace=False)

        seqs = []
        for i in pick_idx:
            ids_tok = tokenizer.encode(blocks[int(i)], add_bos=True, add_eos=True, max_length=None)
            sl = min(cfg.seq_len, mind.config.max_seq_len)
            for j in range(0, max(1, len(ids_tok) - 1), sl):
                chunk = ids_tok[j : j + sl]
                if len(chunk) < 8:
                    continue
                if len(chunk) < sl:
                    chunk = chunk + [tokenizer.pad_id] * (sl - len(chunk))
                seqs.append(chunk[:sl])
        if not seqs:
            seqs = [tokenizer.encode("MESIE power train", max_length=cfg.seq_len)]
            while len(seqs[0]) < cfg.seq_len:
                seqs[0].append(tokenizer.pad_id)

        losses = []
        for step in range(1, cfg.steps + 1):
            batch = [
                seqs[int(np.random.randint(0, len(seqs)))]
                for _ in range(min(cfg.batch_size, len(seqs)))
            ]
            arr = np.array(batch, dtype=np.int64)
            m = mind.language.train_step(
                arr,
                arr,
                lr=cfg.lr * (0.997 ** step),
                text_for_meaning=q[:300],
            )
            if mind.organs.trainer:
                mind.organs.trainer.train_on_model(mind.language, steps=1)
            losses.append(float(m.get("ce", m.get("loss", 0))))
            if cfg.show and (step == 1 or step == cfg.steps or step % 10 == 0):
                print(
                    f"  [power r{rnd} {step}/{cfg.steps}] ce={m.get('ce',0):.4f} ppl={m.get('ppl',0):.2f}",
                    flush=True,
                )
        history.append(
            {
                "round": rnd,
                "mean_ce": float(np.mean(losses)),
                "last_ce": losses[-1] if losses else None,
                "min_ce": float(np.min(losses)) if losses else None,
                "train_steps": mind.language.train_steps,
                "params": mind.language.num_params,
            }
        )
        if cfg.show:
            print(
                f"[power] round {rnd} mean_ce={history[-1]['mean_ce']:.4f} "
                f"min_ce={history[-1]['min_ce']:.4f} params={mind.language.num_params:,}",
                flush=True,
            )

    out_dir = Path(cfg.output_dir) / f"{cfg.model_id.replace('/', '_')}_power"
    meta = save_mind(mind, out_dir)
    # also write bank
    bank_path = None
    if bank is not None:
        bank_path = str(out_dir / "multi_embed_bank.npz")
        bank.save(bank_path)

    report = {
        "schema": "auro.mesie.power_train.v1",
        "ok": True,
        "profile": cfg.profile,
        "upgraded_core": upgraded,
        "num_params_live": mind.language.num_params,
        "architecture": {
            "hidden_dim": mind.config.hidden_dim,
            "num_layers": mind.config.num_layers,
            "num_experts": mind.config.num_experts,
            "use_moe": mind.config.use_moe,
            "use_cross_modal": mind.config.use_cross_modal,
            "use_spectral_encoder": mind.config.use_spectral_encoder,
        },
        "multi_embed": embedder.info(),
        "compress": compressor.info(),
        "bank": bank.info() if bank else None,
        "bank_path": bank_path,
        "train_steps_before": steps0,
        "train_steps_after": mind.language.train_steps,
        "train_steps_delta": mind.language.train_steps - steps0,
        "history": history,
        "checkpoint": str(out_dir),
        "checkpoint_meta": meta,
        "elapsed_s": time.time() - t0,
        "mesie_arsenal_used": [
            "SpectralVectorizer hi/mid",
            "SpectralFeatureEncoder",
            "Helix",
            "LSH",
            "MultiMeaningField",
            "SpectralGPT MoE+cross-modal+spectral-encoder",
            "φ multi-FFT",
            "SVD/hybrid compression",
            "GitHub knowledge DB",
        ],
    }
    (out_dir / "POWER_TRAIN_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = (
        f"# MESIE Power Train\n\n"
        f"- Profile: **{cfg.profile}**\n"
        f"- Live params: **{mind.language.num_params:,}**\n"
        f"- Multi-embed dim: **{embedder.dim}** → compress **{cfg.compress_dim}** ({cfg.compress_method})\n"
        f"- Steps: {steps0} → **{mind.language.train_steps}**\n"
        f"- Best min CE: **{min(h['min_ce'] for h in history):.4f}**\n"
        f"- Checkpoint: `{out_dir}`\n"
    )
    (out_dir / "POWER_TRAIN_REPORT.md").write_text(md, encoding="utf-8")
    if cfg.show:
        print(md, flush=True)
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="MESIE powered training")
    p.add_argument("--profile", default="power", choices=list(PROFILES.keys()))
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_continual")
    args = p.parse_args()
    report = run_power_train(
        PowerTrainConfig(
            profile=args.profile,
            steps=args.steps,
            rounds=args.rounds,
            resume_checkpoint=args.resume if Path(args.resume).exists() else None,
        )
    )
    print(json.dumps({k: report[k] for k in ("ok", "num_params_live", "train_steps_delta", "checkpoint", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
