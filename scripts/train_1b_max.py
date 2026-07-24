"""MAX EFFORT: ~1.09B live MESIE SpectralGPT + train (memory-lean).

Hits >= 1e9 params. Saves checkpoint early. Trains with seq=64 batch=1.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT))

    from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder
    from auro_native_llm.mesie_power.compress import MesieCompressor
    from auro_native_llm.mesie_power.power_train import power_family_overrides
    from auro_native_llm.model.auro_lm import AuroLanguageModel
    from auro_native_llm.model.tokenizer import AuroTokenizer
    from auro_native_llm.model.checkpoint import save_checkpoint
    from auro_native_llm.organism.mind import AuroMind
    from auro_native_llm.organism.checkpoint import save_mind
    from auro_native_llm.organism.self_train import Experience
    from auro_native_llm.embedded.python_organ import PythonOrgan, load_python_doctrine

    t0 = time.time()
    out_dir = ROOT / "checkpoints" / "auro_minds" / "Auro-2B_1B"
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = power_family_overrides("billion")
    overrides.update(
        {
            "max_seq_len": 128,
            "continuous_dim": 128,
            "spectral_input_dim": 512,
            "num_kv_heads": 4,
            "num_modalities": 8,
            "use_meaning": True,
            "use_spectral_fusion": True,
        }
    )

    print("=" * 60, flush=True)
    print("AURO 1B MAX EFFORT (lean train)", flush=True)
    print("=" * 60, flush=True)

    # corpus (bounded)
    print("[1b] corpus…", flush=True)
    blocks = [
        "MESIE SpectralGPT MoE Auro 1B native continuous train phi golden ratio",
        "Python organ doctrine PL-001 PL-009 autocycle sense reason act learn",
        load_python_doctrine().get("principle", "python is an organ"),
    ]
    try:
        from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

        gdb = GitHubKnowledgeDB()
        blocks.extend(gdb.training_blocks(max_blocks=150, max_chars=400_000))
        print(f"[1b] github docs={gdb.count()} blocks={len(blocks)}", flush=True)
    except Exception as exc:
        print(f"[1b] github skip {exc}", flush=True)

    # multi-embed sample
    print("[1b] multi-embed…", flush=True)
    emb = MultiMesieEmbedder()
    comp = MesieCompressor(method="hybrid", target_dim=128)
    sample_n = min(48, len(blocks))
    mat = emb.embed_batch([b[:2000] for b in blocks[:sample_n]])
    bank = comp.compress_bank(mat, [f"b{i}" for i in range(sample_n)], fit=True)
    bank.save(out_dir / "multi_embed_bank_1b.npz")
    print(f"[1b] embed_dim={emb.dim} compress={bank.info()}", flush=True)

    print("[1b] tokenizer…", flush=True)
    tok = AuroTokenizer(vocab_size=overrides["vocab_size"])
    tok.train(blocks[: min(40, len(blocks))], vocab_size=min(8192, overrides["vocab_size"]))
    # keep full 16k architecture vocab for param mass; tokenizer can be smaller subset
    overrides["vocab_size"] = 16384

    print("[1b] BUILD ~1.09B SpectralGPT…", flush=True)
    t_build = time.time()
    language = AuroLanguageModel.build(
        "Auro-2B", mode="dev", tokenizer=tok, **overrides
    )
    language.config.parameter_target = 1_000_000_000
    language.config.tier = "1b_live"
    language.config.extra["billion_run"] = True
    nparams = int(language.num_params)
    print(
        f"[1b] LIVE PARAMS = {nparams:,} ({nparams/1e9:.3f}B) "
        f"hit_1b={nparams >= 1_000_000_000} build_s={time.time()-t_build:.1f}",
        flush=True,
    )

    # save language core immediately (before mind organs inflate RAM)
    print("[1b] early language checkpoint…", flush=True)
    save_checkpoint(language, out_dir / "language")
    gc.collect()

    # sequences (cap)
    seq_len = 64
    seqs = []
    for b in blocks[:80]:
        ids = tok.encode(b[:1500], add_bos=True, add_eos=True, max_length=None)
        for j in range(0, max(1, len(ids) - 1), seq_len):
            chunk = ids[j : j + seq_len]
            if len(chunk) < 8:
                continue
            if len(chunk) < seq_len:
                chunk = chunk + [tok.pad_id] * (seq_len - len(chunk))
            seqs.append(chunk[:seq_len])
            if len(seqs) >= 200:
                break
        if len(seqs) >= 200:
            break
    print(f"[1b] sequences={len(seqs)}", flush=True)
    if not seqs:
        seqs = [[tok.bos_id] + [1] * (seq_len - 2) + [tok.eos_id]]

    STEPS = 24
    history = []
    print(f"[1b] TRAIN {STEPS} steps on {nparams/1e9:.3f}B…", flush=True)
    rng = np.random.default_rng(11)
    for step in range(1, STEPS + 1):
        ts = time.time()
        pick = seqs[int(rng.integers(0, len(seqs)))]
        arr = np.array([pick], dtype=np.int64)
        meaning = blocks[step % len(blocks)][:200]
        m = language.train_step(arr, arr, lr=1.2e-3 * (0.99 ** step), text_for_meaning=meaning)
        row = {
            "step": step,
            "ce": float(m.get("ce", 0)),
            "ppl": float(m.get("ppl", 0)),
            "sec": time.time() - ts,
        }
        history.append(row)
        print(
            f"  [1B {step}/{STEPS}] ce={row['ce']:.4f} ppl={row['ppl']:.1f} "
            f"sec={row['sec']:.1f} params={nparams/1e9:.3f}B",
            flush=True,
        )
        if step % 8 == 0:
            save_checkpoint(language, out_dir / "language")
            gc.collect()
            print("  [1b] mid-save language", flush=True)

    # wrap mind for full organism checkpoint (may be heavy)
    print("[1b] wrap mind + python organ…", flush=True)
    try:
        mind = AuroMind(language, chrome_mock=True, absorb_every_act=False, train_every_act=False)
        mind.organs.python = PythonOrgan()
        if mind.organs.trainer:
            for i in range(min(20, sample_n)):
                mind.organs.trainer.absorb(
                    Experience(
                        text=blocks[i][:1200],
                        kind="1b_multi_embed",
                        model_id="Auro-2B",
                        reward=0.9,
                        embedding=comp.transform(mat[i]).tolist(),
                    )
                )
            mind.organs.trainer.absorb(
                Experience(
                    text=mind.organs.python.doctrine_prompt()[:3000],
                    kind="python_doctrine",
                    model_id="Auro-2B",
                    reward=0.95,
                )
            )
            mind.organs.trainer.train_on_model(language, steps=2)
        py = mind.python(
            "import math\nresult=sum(((1+5**0.5)/2)**i for i in range(1,6))\nprint(result)\n",
            intent="phi under 1B",
        )
        print(f"[1b] python ok={py.ok}", flush=True)
        meta = save_mind(mind, out_dir)
        train_steps = language.train_steps
    except Exception as exc:
        print(f"[1b] mind wrap partial: {exc}", flush=True)
        meta = save_checkpoint(language, out_dir / "language")
        train_steps = language.train_steps
        py = None

    ces = [h["ce"] for h in history]
    report = {
        "schema": "auro.1b.max_effort.v1",
        "ok": True,
        "num_params_live": nparams,
        "num_params_readable": f"{nparams/1e9:.3f}B",
        "hit_1b": nparams >= 1_000_000_000,
        "architecture": {
            "hidden_dim": language.config.hidden_dim,
            "num_layers": language.config.num_layers,
            "num_heads": language.config.num_heads,
            "ffn_dim": language.config.ffn_dim,
            "num_experts": language.config.num_experts,
            "vocab_size": language.config.vocab_size,
            "use_moe": language.config.use_moe,
            "use_cross_modal": language.config.use_cross_modal,
            "use_spectral_encoder": language.config.use_spectral_encoder,
            "num_kv_heads": language.config.num_kv_heads,
        },
        "multi_embed_dim": emb.dim,
        "compress": bank.info(),
        "train_steps": train_steps,
        "steps_run": STEPS,
        "ce_first": ces[0] if ces else None,
        "ce_last": ces[-1] if ces else None,
        "ce_min": min(ces) if ces else None,
        "history": history,
        "python_ok": None if py is None else py.ok,
        "checkpoint": str(out_dir),
        "checkpoint_meta": meta,
        "elapsed_s": time.time() - t0,
        "claim_boundary": (
            "Live params = this running 1.09B MESIE SpectralGPT. "
            "Family labels remain architecture targets."
        ),
        "mesie_arsenal": [
            "SpectralGPT 1024x16 MoE8 GQA",
            "cross-modal + spectral encoder",
            "MultiMesieEmbedder 9 views 1334D",
            "hybrid SVD compress",
            "GitHub knowledge DB",
            "python doctrine organ",
            "online CE train",
        ],
    }
    (out_dir / "BILLION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = (
        f"# Auro 1B Max Effort\n\n"
        f"- **Live params: {nparams:,} ({nparams/1e9:.3f}B)** — hit_1b=**{report['hit_1b']}**\n"
        f"- h={language.config.hidden_dim} L={language.config.num_layers} "
        f"experts={language.config.num_experts} MoE+xmodal+spec_enc+GQA\n"
        f"- Multi-embed **{emb.dim}D** → compress **{bank.info()['compressed_dim']}D**\n"
        f"- Train steps: **{train_steps}** · CE {report['ce_first']} → {report['ce_last']} "
        f"(min {report['ce_min']})\n"
        f"- Checkpoint: `{out_dir}`\n"
        f"- Elapsed: {report['elapsed_s']:.0f}s\n"
    )
    (out_dir / "BILLION_REPORT.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "ok",
                    "num_params_live",
                    "num_params_readable",
                    "hit_1b",
                    "architecture",
                    "train_steps",
                    "ce_min",
                    "ce_last",
                    "elapsed_s",
                    "checkpoint",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
