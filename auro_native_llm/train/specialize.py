"""Specialize Auro mind: generative + embedding + self-configure + all repo skills.

Resumes continual checkpoint, absorbs:
  - every SKILL.md under user/.grok/skills and repo .grok/skills
  - generative / embed / self-config doctrine curriculum
  - GitHub knowledge DB retrieval (114 repos)
  - MESIE runtime APIs (helix, transformers, match, validate)

Then trains multi-round CE + messy self-train and saves
``checkpoints/auro_minds/Auro-2B_specialized``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HOME = Path.home()
_REPO = Path(__file__).resolve().parents[2]


@dataclass
class SpecializeConfig:
    model_id: str = "Auro-2B"
    resume_checkpoint: str = "checkpoints/auro_minds/Auro-2B_continual"
    output_name: str = "Auro-2B_specialized"
    output_dir: str = "checkpoints/auro_minds"
    rounds: int = 8
    steps_per_round: int = 40
    batch_size: int = 2
    seq_len: int = 128
    lr: float = 2.5e-3
    max_skill_chars: int = 120_000
    max_skills: int = 80
    github_top_k: int = 16
    messy_steps: int = 1
    embed_contrast_steps: int = 4
    show: bool = True


# ---------------------------------------------------------------------------
# Curriculum: generative · embedding · self-config · MESIE skills
# ---------------------------------------------------------------------------

GENERATIVE_DOCTRINE = [
    "GENERATIVE: Always produce usable structured output — code, JSON, plans, spectral reports.",
    "GENERATIVE: For coding tasks emit def solution(...): with real logic that passes asserts.",
    "GENERATIVE: SpectralGPT causal MoE generates text and spectral latents jointly.",
    "GENERATIVE: PSD/FAS/RotDnn generation via mesie.generation — generate_psd generate_fas generate_rotdnn.",
    "GENERATIVE: Masked Spectral Modeling reconstructs masked bands; InfoNCE aligns positive pairs.",
    "GENERATIVE: Temporal Prediction forecasts next spectral state for streaming protocols.",
    "GENERATIVE: Multi-component records keep lineage; never drop metadata when synthesizing.",
    "GENERATIVE: When asked to write code use Python first; include tests when possible.",
    "GENERATIVE: Agent replies must be complete sentences with concrete next actions.",
    "GENERATIVE: Foundation pretraining objectives: MSM + InfoNCE + TemporalPrediction.",
]

EMBEDDING_DOCTRINE = [
    "EMBEDDING: Spectra are structured objects not bare arrays — embed with SpectralVectorizer.",
    "EMBEDDING: HelixEncoder maps vectors to phase/radius/elevation + flat_embedding.",
    "EMBEDDING: Max multi-embed concatenates char/word/spectral/hash views then compress hybrid SVD.",
    "EMBEDDING: ANN + LSH fingerprint pipeline for approximate nearest neighbors on spectral banks.",
    "EMBEDDING: AI-native embeddings feed retrieval before generate — retrieve then reason then act.",
    "EMBEDDING: Resonance and coherence scores rank candidates; composite_score for match_records.",
    "EMBEDDING: Hierarchical HelixRetriever searches multi-scale helix projections.",
    "EMBEDDING: Cross-domain transfer keeps domain-invariant encoders across engineering verticals.",
    "EMBEDDING: Every mind experience stores optional embedding for priority training.",
    "EMBEDDING: Embed library folders into ANNIndex for my-library spectral search.",
]

GHOST_FOCUS = [
    "GHOST specialization: grounded claims, hardened policy, open receipts, "
    "local-first, traceable custody — hybrid MESIE Ghost Node + LLM escalate only.",
]

SELF_CONFIG_DOCTRINE = [
    "SELF_CONFIG: Probe installed mesie with prefer_user_mesie_install then bind MesieRuntimeStack.",
    "SELF_CONFIG: Capability map includes spectral transformers helix intelligence connectome miniverse.",
    "SELF_CONFIG: Auro-14B intelligence_level=autonomous; smaller lanes adaptive/predictive.",
    "SELF_CONFIG: Runtime injects multi-repo SDKs without shadowing live auro_foundry.",
    "SELF_CONFIG: After every act absorb Experience then train_on_model — always self-train.",
    "SELF_CONFIG: Promotion gate M0/M1/M2 requires coding pass_rate + reasoning accuracy + usable gen.",
    "SELF_CONFIG: compute_plane=MESIE only — no cloud LLM; ChaosCUDA/Julia when torch missing.",
    "SELF_CONFIG: Family labels 2B/4B/8B/14B are architecture targets; live_params are running core.",
    "SELF_CONFIG: Autocycle SENSE→REASON→ACT→OBSERVE→ABSORB→TRAIN→GOVERN→LOOP.",
    "SELF_CONFIG: Connectome 44 regions 68 connections; miniverse RecursiveMemoryContainer ScaleBridge.",
    "SELF_CONFIG: When gaps appear extend synthesizer helpers and re-run long harness.",
    "SELF_CONFIG: save_mind after specialization rounds; resume_checkpoint for continual growth.",
]

SKILL_QUERIES = [
    "mesie generate PSD FAS RotDnn spectral",
    "mesie embed SpectralVectorizer helix ANN LSH",
    "mesie match validate multi-level schema",
    "mesie neuroaix connectome brain regions",
    "mesie octopus polyglot arms AIS vector",
    "mesie enterprise monte carlo verticals",
    "mesie solus mini brain mini heart organism",
    "mesie foundation SpectralGPT MoE pretraining",
    "NOVA LOOM CAPSULA governance receipts promotion",
    "potential-succotash engines models agents",
    "self configure runtime capabilities inject SDK",
    "generative coding harness assert solution",
    "embedding multi-view compression hybrid SVD",
    "career mcp triple protocol loom",
    "chrome portal multi-site agents MCP",
]


def _skill_roots() -> List[Path]:
    return [
        _REPO / ".grok" / "skills",
        _HOME / ".grok" / "skills",
        _HOME / ".grok" / "bundled" / "skills",
        _HOME / "Documents" / "GitHub" / "Auro14B" / ".grok" / "skills",
    ]


def harvest_skills(max_skills: int = 80, max_chars: int = 120_000) -> List[Dict[str, str]]:
    """Load SKILL.md from all known skill trees (repo + user)."""
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    total_chars = 0
    for root in _skill_roots():
        if not root.exists():
            continue
        for skill_md in sorted(root.rglob("SKILL.md")):
            try:
                name = skill_md.parent.name
                key = f"{root.name}:{name}"
                if key in seen:
                    continue
                text = skill_md.read_text(encoding="utf-8", errors="ignore")
                if len(text) < 40:
                    continue
                # keep frontmatter + first body for trainability
                clip = text[:3500]
                if total_chars + len(clip) > max_chars:
                    continue
                seen.add(key)
                total_chars += len(clip)
                out.append(
                    {
                        "name": name,
                        "path": str(skill_md),
                        "text": f"SKILL {name}\n{clip}",
                    }
                )
                if len(out) >= max_skills:
                    return out
            except Exception:
                continue
    return out


def specialization_blocks(skills: Sequence[Dict[str, str]]) -> List[str]:
    blocks: List[str] = []
    blocks.extend(GENERATIVE_DOCTRINE)
    blocks.extend(EMBEDDING_DOCTRINE)
    blocks.extend(SELF_CONFIG_DOCTRINE)
    blocks.extend(GHOST_FOCUS)
    try:
        from auro_native_llm.ghost.doctrine import all_ghost_training_blocks

        blocks.extend(all_ghost_training_blocks())
    except Exception:
        pass
    # paired skill cards
    for s in skills:
        blocks.append(s["text"])
    # synthesis cards
    skill_names = ", ".join(s["name"] for s in skills[:40])
    blocks.append(
        "SKILL_CATALOG: Auro specializes in skills: " + skill_names
        + ". Route mesie-hub for spectral; mesie-embed for vectors; mesie-generate for PSD/FAS; "
        "mesie-match for ranking; mesie-validate for schema levels 1-6; mesie-neuroaix for connectome; "
        "mesie-octopus for eight-arm control; mesie-solus-organism for mini brain/heart."
    )
    blocks.append(
        "SELF_CONFIG_PROTOCOL: "
        "1) probe_mesie_install 2) attach_mesie_runtime(model_id) 3) inject_repo_sdks "
        "4) readiness M2 gate 5) long_harness coding+reasoning 6) save_mind specialized."
    )
    blocks.append(
        "GENERATIVE_PROTOCOL: "
        "prompt → retrieve embeddings → reason intelligence protocol → synthesize → "
        "validate spectral/schema if data → execute code asserts → absorb success."
    )
    blocks.append(
        "EMBEDDING_PROTOCOL: "
        "text/signal → SpectralVectorizer or HelixEncoder → optional ANNIndex search → "
        "compress hybrid → feed train Experience.embedding → multi-view CE."
    )
    return blocks


def run_specialization(cfg: Optional[SpecializeConfig] = None) -> Dict[str, Any]:
    cfg = cfg or SpecializeConfig()
    t0 = time.time()
    report: Dict[str, Any] = {
        "schema": "auro.specialize.v1",
        "focus": ["generative", "embedding", "self_config", "repo_skills"],
        "config": asdict(cfg),
    }

    from auro_native_llm.corpus.github_db import GitHubKnowledgeDB
    from auro_native_llm.mesie_runtime import attach_mesie_runtime, probe_mesie_install
    from auro_native_llm.organism.checkpoint import load_mind, save_mind
    from auro_native_llm.organism.self_train import Experience
    from auro_native_llm.sdk_runtime.injector import inject_repo_sdks

    print("[specialize] harvest skills…", flush=True)
    skills = harvest_skills(max_skills=cfg.max_skills, max_chars=cfg.max_skill_chars)
    report["skills_n"] = len(skills)
    report["skill_names"] = [s["name"] for s in skills]
    print(f"[specialize] skills={len(skills)}", flush=True)

    print("[specialize] GitHub knowledge DB…", flush=True)
    gdb = GitHubKnowledgeDB()
    report["github_docs"] = gdb.count()
    report["github_repos"] = len(gdb.repo_counts())
    print(f"[specialize] docs={gdb.count()} repos={report['github_repos']}", flush=True)

    resume = Path(cfg.resume_checkpoint)
    if not resume.exists():
        # fallback build
        from auro_native_llm.organism.family import build_mind

        print("[specialize] no resume — building lite mind", flush=True)
        mind = build_mind(cfg.model_id, lite=True, chrome_mock=True)
    else:
        print(f"[specialize] resume {resume}", flush=True)
        mind = load_mind(resume, chrome_mock=True)

    # Bind mesie transformers + SDKs (self-config path)
    mesie_probe = probe_mesie_install()
    mesie_health = attach_mesie_runtime(mind, lite=True, force_rebind=True)
    sdk_stats = inject_repo_sdks(mind, max_packages=200)
    report["mesie"] = {
        "version": mesie_probe.get("version"),
        "path": mesie_probe.get("path"),
        "n_capabilities_on": mesie_health.get("n_capabilities_on"),
        "capabilities_on": mesie_health.get("capabilities_on"),
        "intelligence_level": mesie_health.get("intelligence_level"),
    }
    report["sdk"] = {
        "n_packages": sdk_stats.get("n_packages"),
        "paths_injected": sdk_stats.get("paths_injected"),
    }
    print(
        f"[specialize] mesie={mesie_probe.get('version')} "
        f"caps={mesie_health.get('n_capabilities_on')} "
        f"sdk={sdk_stats.get('n_packages')}",
        flush=True,
    )

    trainer = mind.organs.trainer
    if trainer is None:
        raise RuntimeError("mind has no trainer organ")
    trainer.lr = cfg.lr
    trainer.batch_size = cfg.batch_size
    trainer.seq_len = cfg.seq_len
    trainer.capacity = max(trainer.capacity, 8192)
    ghost_blocks: List[str] = []
    try:
        from auro_native_llm.ghost.doctrine import all_ghost_training_blocks

        ghost_blocks = all_ghost_training_blocks()
    except Exception:
        pass
    trainer.seed_doctrine(
        GENERATIVE_DOCTRINE
        + EMBEDDING_DOCTRINE
        + SELF_CONFIG_DOCTRINE
        + GHOST_FOCUS
        + ghost_blocks
    )

    # Absorb specialization curriculum
    blocks = specialization_blocks(skills)
    n_abs = 0
    for b in blocks:
        emb = None
        try:
            if getattr(mind, "mesie_runtime", None) is not None:
                emb = mind.mesie_runtime.embed_text(b, dim=64)  # type: ignore[attr-defined]
        except Exception:
            emb = None
        kind = "skill"
        if b.startswith("GENERATIVE"):
            kind = "generative"
        elif b.startswith("EMBEDDING"):
            kind = "embedding"
        elif b.startswith("SELF_CONFIG"):
            kind = "self_config"
        elif b.startswith("GHOST"):
            kind = "ghost"
        elif b.startswith("SKILL"):
            kind = "skill"
        trainer.absorb(
            Experience(
                text=b[:2500],
                kind=kind,
                model_id=cfg.model_id,
                reward=0.92,
                embedding=emb,
                meta={"specialize": True},
            )
        )
        n_abs += 1

    # Absorb GitHub retrieval for skill queries
    for q in SKILL_QUERIES:
        try:
            hits = gdb.search(q, top_k=min(8, cfg.github_top_k))
        except Exception:
            hits = []
        for hit in hits:
            emb = None
            try:
                emb = gdb.embedder.embed_text(hit.text).tolist()
            except Exception:
                pass
            trainer.absorb(
                Experience(
                    text=hit.text[:2000],
                    kind="github_skill",
                    model_id=cfg.model_id,
                    reward=0.8 + 0.15 * max(0.0, min(1.0, float(getattr(hit, "score", 0.5)))),
                    embedding=emb,
                    meta={"repo": getattr(hit, "repo", ""), "query": q},
                )
            )
            n_abs += 1

    report["absorbed"] = n_abs
    report["buffer"] = len(trainer.buffer)
    print(f"[specialize] absorbed={n_abs} buffer={len(trainer.buffer)}", flush=True)

    tokenizer = mind.language.tokenizer
    # Build CE sequences from specialization + github blocks
    train_texts: List[str] = list(blocks)
    for q in SKILL_QUERIES[:10]:
        try:
            for hit in gdb.search(q, top_k=4):
                train_texts.append(hit.text[:1500])
        except Exception:
            pass

    seqs: List[List[int]] = []
    for t in train_texts:
        ids = tokenizer.encode(t, add_bos=True, add_eos=True, max_length=None)
        for i in range(0, max(1, len(ids) - 1), cfg.seq_len):
            chunk = ids[i : i + cfg.seq_len]
            if len(chunk) < 8:
                continue
            if len(chunk) < cfg.seq_len:
                chunk = chunk + [tokenizer.pad_id] * (cfg.seq_len - chunk.__len__())
            seqs.append(chunk[: cfg.seq_len])
    if not seqs:
        ids = tokenizer.encode(
            "MESIE generative embedding self-config specialized Auro",
            max_length=cfg.seq_len,
        )
        while len(ids) < cfg.seq_len:
            ids.append(tokenizer.pad_id)
        seqs = [ids[: cfg.seq_len]]

    print(f"[specialize] sequences={len(seqs)} train rounds={cfg.rounds}x{cfg.steps_per_round}", flush=True)

    history: List[Dict[str, Any]] = []
    rng = np.random.default_rng(7)
    steps0 = mind.language.train_steps

    for rnd in range(1, cfg.rounds + 1):
        focus = ["generative", "embedding", "self_config", "skills"][(rnd - 1) % 4]
        losses: List[float] = []
        print(f"[specialize] round {rnd}/{cfg.rounds} focus={focus}", flush=True)
        for step in range(1, cfg.steps_per_round + 1):
            pick = [
                seqs[int(i)]
                for i in rng.integers(0, len(seqs), size=min(cfg.batch_size, len(seqs)))
            ]
            arr = np.array(pick, dtype=np.int64)
            meaning = train_texts[step % len(train_texts)][:400]
            lr = cfg.lr * (0.998 ** ((rnd - 1) * cfg.steps_per_round + step))
            # slight focus boost
            if focus == "generative":
                lr *= 1.05
            elif focus == "embedding":
                lr *= 1.03
            m = mind.language.train_step(arr, arr, lr=lr, text_for_meaning=meaning)
            messy = trainer.train_on_model(mind.language, steps=cfg.messy_steps)
            losses.append(float(m.get("ce", m.get("loss", 0.0))))
            if step == 1 or step == cfg.steps_per_round or step % 10 == 0:
                print(
                    f"  [r{rnd} {step}/{cfg.steps_per_round}] ce={m.get('ce', m.get('loss', 0)):.4f} "
                    f"ppl={m.get('ppl', 0):.2f} messy={messy.get('last_loss')}",
                    flush=True,
                )

        # embedding contrast micro-loop: similar skills closer via joint train meaning
        emb_losses = []
        if cfg.embed_contrast_steps > 0 and getattr(mind, "mesie_runtime", None):
            rt = mind.mesie_runtime  # type: ignore[attr-defined]
            for _ in range(cfg.embed_contrast_steps):
                a = train_texts[int(rng.integers(0, len(train_texts)))]
                b = train_texts[int(rng.integers(0, len(train_texts)))]
                pair = f"EMBED_ALIGN\nA: {a[:300]}\nB: {b[:300]}\n"
                ids = tokenizer.encode(pair, add_bos=True, add_eos=True, max_length=cfg.seq_len)
                if len(ids) < cfg.seq_len:
                    ids = ids + [tokenizer.pad_id] * (cfg.seq_len - len(ids))
                arr = np.array([ids[: cfg.seq_len]], dtype=np.int64)
                m = mind.language.train_step(arr, arr, lr=cfg.lr * 0.5, text_for_meaning=pair[:200])
                emb_losses.append(float(m.get("ce", m.get("loss", 0))))
                try:
                    _ = rt.embed_text(a[:200], dim=64)
                except Exception:
                    pass

        # generative smoke
        gen = mind.generate(
            f"Specialize {focus}: explain how Auro uses MESIE for generation and embeddings.",
            max_new_tokens=48,
        )
        pulse = mind.pulse()

        row = {
            "round": rnd,
            "focus": focus,
            "mean_ce": float(np.mean(losses)) if losses else None,
            "last_ce": losses[-1] if losses else None,
            "embed_mean_ce": float(np.mean(emb_losses)) if emb_losses else None,
            "train_steps": mind.language.train_steps,
            "generate_ok": bool(getattr(gen, "ok", True)),
            "pulse_ok": bool(pulse.get("ok")) if isinstance(pulse, dict) else True,
            "params_live": mind.language.num_params,
        }
        history.append(row)
        print(
            f"  [r{rnd}] mean_ce={row['mean_ce']:.4f} steps={row['train_steps']}",
            flush=True,
        )

    # Final self-config probe + skill recall absorb
    try:
        from auro_native_llm.mesie_runtime import get_mesie_runtime

        stack14 = get_mesie_runtime("Auro-14B", lite=True, force_rebind=False)
        report["auro_14b_runtime"] = {
            "n_capabilities_on": stack14.health().get("n_capabilities_on"),
            "intelligence_level": stack14.health().get("intelligence_level"),
            "mesie_version": stack14.mesie_version,
        }
    except Exception as exc:
        report["auro_14b_runtime"] = {"error": str(exc)}

    out_dir = Path(cfg.output_dir) / cfg.output_name
    meta = save_mind(mind, out_dir)

    # multi-embed bank snapshot if available
    try:
        from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder

        emb = MultiMesieEmbedder()
        sample = train_texts[:64]
        rows = []
        for t in sample:
            try:
                if hasattr(emb, "embed_text"):
                    rows.append(np.asarray(emb.embed_text(t), dtype=np.float32).ravel())
                elif hasattr(emb, "embed"):
                    rows.append(np.asarray(emb.embed(t), dtype=np.float32).ravel())
            except Exception:
                continue
        if rows:
            # pad to same dim
            d = max(r.size for r in rows)
            mat = np.zeros((len(rows), d), dtype=np.float32)
            for i, r in enumerate(rows):
                mat[i, : r.size] = r
            np.savez_compressed(out_dir / "specialize_embed_bank.npz", vectors=mat)
            report["embed_bank"] = {
                "n": len(rows),
                "dim": int(d),
                "info": emb.info() if hasattr(emb, "info") else {},
            }
        else:
            report["embed_bank"] = {"ok": False, "error": "no embed rows"}
    except Exception as exc:
        report["embed_bank"] = {"ok": False, "error": str(exc)[:200]}

    report.update(
        {
            "ok": True,
            "history": history,
            "checkpoint": str(out_dir),
            "checkpoint_meta": meta,
            "num_params_live": mind.language.num_params,
            "train_steps": mind.language.train_steps,
            "train_steps_delta": mind.language.train_steps - steps0,
            "mean_ce_first": history[0]["mean_ce"] if history else None,
            "mean_ce_last": history[-1]["mean_ce"] if history else None,
            "elapsed_s": time.time() - t0,
            "specializations": {
                "generative": True,
                "embedding": True,
                "self_config": True,
                "skills": report["skill_names"],
            },
            "claim_boundary": (
                "Specialized live core trained on skills+doctrine+GitHub. "
                "Not a full 14B weight checkpoint; MESIE runtime features are process-bound."
            ),
        }
    )
    (out_dir / "SPECIALIZE_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    md = (
        f"# Auro Specialization Report\n\n"
        f"- Focus: **generative · embedding · self-config · repo skills**\n"
        f"- Skills absorbed: **{report['skills_n']}**\n"
        f"- GitHub docs: **{report['github_docs']}** / repos **{report['github_repos']}**\n"
        f"- MESIE: **{report['mesie'].get('version')}** caps **{report['mesie'].get('n_capabilities_on')}**\n"
        f"- Train steps delta: **{report['train_steps_delta']}** (total {report['train_steps']})\n"
        f"- CE first→last: **{report['mean_ce_first']:.4f} → {report['mean_ce_last']:.4f}**\n"
        f"- Params live: **{report['num_params_live']:,}**\n"
        f"- Checkpoint: `{out_dir}`\n"
        f"- Skills: {', '.join(report['skill_names'][:30])}…\n"
    )
    (out_dir / "SPECIALIZE_REPORT.md").write_text(md, encoding="utf-8")
    print(f"[specialize] saved {out_dir} delta_steps={report['train_steps_delta']}", flush=True)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Specialize Auro: generative+embed+self-config+skills")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--steps", type=int, default=40, dest="steps_per_round")
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_continual")
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--seq-len", type=int, default=128)
    args = p.parse_args(argv)
    cfg = SpecializeConfig(
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        resume_checkpoint=args.resume,
        lr=args.lr,
        seq_len=args.seq_len,
    )
    rep = run_specialization(cfg)
    print(
        json.dumps(
            {
                "ok": rep.get("ok"),
                "skills_n": rep.get("skills_n"),
                "train_steps_delta": rep.get("train_steps_delta"),
                "mean_ce_first": rep.get("mean_ce_first"),
                "mean_ce_last": rep.get("mean_ce_last"),
                "checkpoint": rep.get("checkpoint"),
                "mesie": rep.get("mesie"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if rep.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
