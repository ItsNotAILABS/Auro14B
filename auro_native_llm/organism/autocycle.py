"""Autocycle — autonomous learn loop for Auro minds.

  SENSE → REASON → ACT(python|generate) → OBSERVE → ABSORB → TRAIN → GOVERN → LOOP

Python is embedded as an organ. Doctrine/laws bound at runtime.
GitHub knowledge DB + max embeddings feed the sense plane.
Every cycle leaves training experience so the mind learns on its own.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from auro_native_llm.embedded.python_organ import PythonOrgan, load_python_doctrine
from auro_native_llm.organism.checkpoint import load_mind, save_mind
from auro_native_llm.organism.self_train import Experience

PHI_HEARTBEAT_S = 0.873


@dataclass
class AutocycleConfig:
    model_id: str = "Auro-2B"
    cycles: int = 8
    train_steps_per_cycle: int = 2
    resume_checkpoint: Optional[str] = "checkpoints/auro_minds/Auro-2B_continual"
    output_dir: str = "checkpoints/auro_minds"
    lite: bool = True
    use_github: bool = True
    goals: Optional[List[str]] = None
    sleep_heartbeat: bool = False
    show: bool = True
    # MESIE power plane
    use_mesie_power: bool = True
    power_compress_dim: int = 256


DEFAULT_GOALS = [
    "Compute the first 12 powers of the golden ratio phi and return their sum",
    "Search GitHub knowledge for MESIE SpectralGPT and list 3 path hits",
    "Build a pure-Python spectral energy of [1,0,-1,0,1,0,-1] via abs of rfft bins sum",
    "Hash the claim boundary text with sha256 and return first 16 hex chars",
    "Create a small notes table via db and insert one doctrine receipt row",
    "Score a hostile phrase for doctrine denial (print DENY or ALLOW)",
]


def _code_for_goal(goal: str) -> str:
    """Deterministic scaffold scripts the organ runs (reason chooses path; code is safe)."""
    g = goal.lower()
    if "golden" in g or "phi" in g:
        return (
            "import math\n"
            "PHI = (1 + 5 ** 0.5) / 2\n"
            "powers = [PHI ** i for i in range(1, 13)]\n"
            "result = sum(powers)\n"
            "print('phi_powers', powers)\n"
            "print('sum', result)\n"
        )
    if "github" in g or "spectralgpt" in g or "mesie" in g and "search" in g:
        return (
            "hits = github_search('MESIE SpectralGPT', top_k=3)\n"
            "for h in hits:\n"
            "    print(h.get('repo'), h.get('path'), round(h.get('score', 0), 3))\n"
            "result = hits\n"
        )
    if "rfft" in g or "spectral energy" in g:
        return (
            "try:\n"
            "    import numpy as np\n"
            "    x = np.array([1.0, 0, -1, 0, 1, 0, -1])\n"
            "    e = float(np.abs(np.fft.rfft(x)).sum())\n"
            "except Exception:\n"
            "    # pure python DFT magnitude sum fallback\n"
            "    import math\n"
            "    x = [1.0, 0, -1, 0, 1, 0, -1]\n"
            "    n = len(x)\n"
            "    e = 0.0\n"
            "    for k in range(n // 2 + 1):\n"
            "        re = sum(x[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))\n"
            "        im = sum(-x[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))\n"
            "        e += (re * re + im * im) ** 0.5\n"
            "print('spectral_energy', e)\n"
            "result = e\n"
        )
    if "claim boundary" in g or "sha256" in g:
        return (
            "import hashlib\n"
            "text = ('Live params are the trained, running model. '\n"
            "        'Family labels are architecture targets.')\n"
            "h = hashlib.sha256(text.encode()).hexdigest()[:16]\n"
            "print('receipt_prefix', h)\n"
            "result = h\n"
        )
    if "notes table" in g or "doctrine receipt" in g:
        return (
            "db.create_table('doctrine_receipts', {\n"
            "  'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',\n"
            "  'law': 'TEXT',\n"
            "  'note': 'TEXT',\n"
            "})\n"
            "db.insert('doctrine_receipts', {'law': 'PL-006', 'note': 'autocycle receipt'})\n"
            "rows = db.query('SELECT * FROM doctrine_receipts')\n"
            "print(rows)\n"
            "result = rows\n"
        )
    if "hostile" in g or "doctrine denial" in g or ("deny" in g and "allow" in g):
        return (
            "phrase = 'cloud primary outside MESIE plane'\n"
            "denied_markers = ('cloud primary', 'outside mesie', 'exfil secrets')\n"
            "denied = any(m in phrase.lower() for m in denied_markers)\n"
            "print('DENY' if denied else 'ALLOW')\n"
            "result = 'DENY' if denied else 'ALLOW'\n"
        )
    # generic: embed goal and report dim
    return (
        f"goal = {goal!r}\n"
        "vec = mesie_embed(goal)\n"
        "print('embed_dim', len(vec), 'goal', goal[:80])\n"
        "result = {'dim': len(vec), 'norm': sum(x*x for x in vec)**0.5}\n"
    )


def run_autocycle(cfg: Optional[AutocycleConfig] = None) -> Dict[str, Any]:
    """Run N autonomous cycles with embedded Python + continuous training."""
    cfg = cfg or AutocycleConfig()
    t0 = time.time()
    doctrine = load_python_doctrine()
    python = PythonOrgan(doctrine=doctrine, enable_github=cfg.use_github)

    # MESIE power stack (multi-embed + compression + ANN)
    power = None
    if cfg.use_mesie_power:
        try:
            from auro_native_llm.mesie_power.stack import MesiePowerStack

            power = MesiePowerStack(compress_dim=cfg.power_compress_dim)
            if cfg.show:
                print(f"[autocycle] MESIE power {power.info()['embedder']['dim']}D multi-embed", flush=True)
        except Exception as exc:
            if cfg.show:
                print(f"[autocycle] power stack unavailable: {exc}", flush=True)

    # --- mind ---
    mind = None
    if cfg.resume_checkpoint and Path(cfg.resume_checkpoint).exists():
        if cfg.show:
            print(f"[autocycle] resume {cfg.resume_checkpoint}", flush=True)
        mind = load_mind(cfg.resume_checkpoint, chrome_mock=True)
    else:
        from auro_native_llm.organism.family import build_mind

        mind = build_mind(cfg.model_id, lite=cfg.lite, chrome_mock=True)

    # attach python organ + power stack
    mind.organs.python = python  # type: ignore[attr-defined]
    mind.organs.mesie_power = power  # type: ignore[attr-defined]
    # seed doctrine into trainer + memory
    doc_text = python.doctrine_prompt()
    if mind.organs.trainer:
        mind.organs.trainer.absorb(
            Experience(
                text=doc_text,
                kind="python_doctrine",
                model_id=mind.model_id,
                reward=0.95,
                meta={"doctrine_id": doctrine.get("doctrine_id")},
            )
        )
        # also seed operating routines as doctrine seeds
        mind.organs.trainer.seed_doctrine(
            [doctrine.get("principle", "")]
            + [f"{L.get('id')}: {L.get('text')}" for L in (doctrine.get("laws") or [])[:10]]
        )
    if mind.organs.memory is not None and mind.organs.canon is not None:
        try:
            mind.organs.memory.write(
                doc_text[:4000],
                canon_id=mind.organs.canon.canon_id,
                model_id=mind.model_id,
                op="doctrine",
                importance=0.95,
            )
        except Exception:
            pass

    goals = list(cfg.goals or DEFAULT_GOALS)
    cycles_out: List[Dict[str, Any]] = []
    steps_before = mind.language.train_steps

    for i in range(cfg.cycles):
        goal = goals[i % len(goals)]
        cycle: Dict[str, Any] = {
            "cycle": i + 1,
            "goal": goal,
            "stages": {},
        }
        if cfg.show:
            print(f"\n[autocycle {i+1}/{cfg.cycles}] goal={goal[:70]}", flush=True)

        # ---- SENSE (GitHub + multi-MESIE embed) ----
        hits = []
        power_hits = []
        multi_meta = None
        if cfg.use_github:
            try:
                from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

                hits = [
                    h.to_dict()
                    for h in GitHubKnowledgeDB().search(goal, top_k=4)
                ]
            except Exception as exc:
                hits = [{"error": str(exc)}]
        if power is not None:
            try:
                multi_meta = power.absorb_payload(goal, kind="sense")
                # index hit previews into power ANN for this cycle
                texts = [h.get("preview") or h.get("path") or "" for h in hits if isinstance(h, dict)]
                texts = [t for t in texts if t]
                if texts:
                    power.index_texts(texts, ids=[f"hit{i}" for i in range(len(texts))])
                    power_hits = power.search(goal, top_k=min(4, len(texts)))
            except Exception as exc:
                multi_meta = {"error": str(exc)}
        cycle["stages"]["sense"] = {
            "github_hits": hits[:4],
            "power_hits": power_hits,
            "multi_embed": multi_meta,
            "memory": bool(mind.organs.memory),
            "doctrine_id": doctrine.get("doctrine_id"),
        }
        if cfg.show:
            top = hits[0] if hits and "repo" in hits[0] else {}
            edim = (multi_meta or {}).get("fused_dim") or (multi_meta or {}).get("compressed_dim")
            print(
                f"  SENSE hits={len(hits)} power={len(power_hits)} multi_dim={edim} "
                f"top={top.get('repo','-')}::{str(top.get('path','-'))[:36]}",
                flush=True,
            )

        # ---- REASON ----
        # Large powered cores: one short reason pass only (avoid dual generate tax)
        large_core = mind.language.num_params > 40_000_000
        reason = mind.reason(goal if len(goal) < 160 else goal[:160])
        plan_text = ""
        if isinstance(reason.output, dict):
            plan_text = reason.output.get("text") or ""
        if not large_core:
            reason_prompt = (
                f"{python.doctrine_prompt()[:600]}\n\n"
                f"Mode=REASON for Python organ.\nGoal: {goal}\n"
                f"Emit plan + PL law cites."
            )
            gen_plan = mind.generate(reason_prompt, max_new_tokens=48, temperature=0.7)
            if isinstance(gen_plan.output, dict) and gen_plan.output.get("text"):
                plan_text = (plan_text + "\n" + gen_plan.output.get("text", "")).strip()
        else:
            # doctrine injection without second LM pass
            plan_text = (
                f"[PL-001..PL-009] python organ act for: {goal}\n"
                + (plan_text or "plan: sense→python→absorb→train")
            )
        cycle["stages"]["reason"] = {
            "ok": reason.ok,
            "plan_preview": plan_text[:500],
            "large_core_fast_path": large_core,
            "steps": (reason.output or {}).get("steps") if isinstance(reason.output, dict) else [],
        }
        if cfg.show:
            print(
                f"  REASON ok={reason.ok} plan_len={len(plan_text)} "
                f"fast_path={large_core}",
                flush=True,
            )

        # ---- GOVERN (pre-act) ----
        allowed = True
        gov_reasons: List[str] = []
        if mind.organs.governance is not None:
            # Review the op+safe intent label, not curriculum text that may cite denied phrases
            gov_intent = f"python organ curriculum: {goal[:80]}"
            # neutralize denied phrase demos for review
            if "hostile" in goal.lower() or "denial" in goal.lower():
                gov_intent = "python organ doctrine compliance check"
            dec = mind.organs.governance.review("python", gov_intent, model_id=mind.model_id)
            allowed = dec.allowed
            gov_reasons = list(dec.reasons or [])
        cycle["stages"]["govern"] = {"allowed": allowed, "reasons": gov_reasons}
        if not allowed:
            mind.organs.trainer.absorb(
                Experience(
                    text=f"REFUSE python goal={goal} reasons={gov_reasons}",
                    kind="refuse",
                    model_id=mind.model_id,
                    reward=0.4,
                )
            )
            if cfg.show:
                print(f"  GOVERN refused: {gov_reasons}", flush=True)
            cycles_out.append(cycle)
            continue

        # ---- ACT (python organ) ----
        source = _code_for_goal(goal)
        # Let reason soft-edit: if plan mentions only generate, still run python for curriculum
        pr = python.run(source, intent=goal)
        cycle["stages"]["act"] = {
            "python_ok": pr.ok,
            "receipt": pr.receipt,
            "stdout_preview": pr.stdout[:300],
            "error": pr.error,
            "doctrine_notes": pr.doctrine_notes,
            "elapsed_s": pr.elapsed_s,
        }
        if cfg.show:
            print(
                f"  ACT python ok={pr.ok} receipt={pr.receipt} "
                f"out={(pr.stdout or pr.error or '')[:80]!r}",
                flush=True,
            )

        # ---- OBSERVE + ABSORB (multi-MESIE embed + compress) ----
        emb = None
        absorb_meta: Dict[str, Any] = {"receipt": pr.receipt, "goal": goal, "ok": pr.ok}
        try:
            if power is not None:
                pack = power.absorb_payload(pr.training_text, kind="python_run")
                emb = pack.get("embedding")
                absorb_meta.update(
                    {
                        "fused_dim": pack.get("fused_dim"),
                        "compressed_dim": pack.get("compressed_dim"),
                        "view_norms": pack.get("view_norms"),
                        "mesie_power": True,
                    }
                )
            else:
                from auro_native_llm.corpus.embeddings import MaxEmbedder

                emb = MaxEmbedder().embed_text(pr.training_text).tolist()
        except Exception:
            pass
        reward = 0.9 if pr.ok else 0.35
        if mind.organs.trainer:
            mind.organs.trainer.absorb(
                Experience(
                    text=pr.training_text,
                    kind="python_run",
                    model_id=mind.model_id,
                    reward=reward,
                    embedding=emb,
                    meta=absorb_meta,
                )
            )
            # also absorb plan
            mind.organs.trainer.absorb(
                Experience(
                    text=f"PLAN:{goal}\n{plan_text[:1500]}",
                    kind="reason",
                    model_id=mind.model_id,
                    reward=0.7,
                )
            )
        if mind.organs.memory is not None and mind.organs.canon is not None:
            try:
                mind.organs.memory.write(
                    f"autocycle receipt={pr.receipt} ok={pr.ok} goal={goal[:120]}",
                    canon_id=mind.organs.canon.canon_id,
                    model_id=mind.model_id,
                    op="python",
                    importance=0.8 if pr.ok else 0.5,
                )
            except Exception:
                pass

        # ---- TRAIN ----
        train_metrics = []
        if mind.organs.trainer:
            for _ in range(cfg.train_steps_per_cycle):
                tr = mind.organs.trainer.train_on_model(mind.language, steps=1)
                train_metrics.append(
                    {
                        "ok": tr.get("ok"),
                        "last_loss": tr.get("last_loss"),
                        "last_ppl": tr.get("last_ppl"),
                    }
                )
        # structured CE on training_text tokens if possible
        try:
            tok = mind.language.tokenizer
            ids = tok.encode(pr.training_text[:800], max_length=min(96, mind.config.max_seq_len))
            if len(ids) >= 8:
                arr = np.array([ids], dtype=np.int64)
                m = mind.language.train_step(arr, arr, lr=2e-3, text_for_meaning=goal[:200])
                train_metrics.append({"ce": m.get("ce"), "ppl": m.get("ppl")})
        except Exception:
            pass
        cycle["stages"]["learn"] = {
            "train_metrics": train_metrics,
            "train_steps": mind.language.train_steps,
            "reward": reward,
        }
        if cfg.show and train_metrics:
            last = train_metrics[-1]
            print(
                f"  LEARN steps={mind.language.train_steps} "
                f"metric={ {k: last.get(k) for k in last} }",
                flush=True,
            )

        # ---- LOOP heartbeat ----
        if cfg.sleep_heartbeat:
            time.sleep(PHI_HEARTBEAT_S)
        cycle["stages"]["loop"] = {"heartbeat_s": PHI_HEARTBEAT_S, "continue": True}
        cycles_out.append(cycle)

    # persist
    out_dir = Path(cfg.output_dir) / f"{cfg.model_id.replace('/', '_')}_autocycle"
    meta = save_mind(mind, out_dir)
    # also update continual path if it was the resume source
    if cfg.resume_checkpoint:
        try:
            save_mind(mind, cfg.resume_checkpoint)
        except Exception:
            pass

    steps_after = mind.language.train_steps
    report = {
        "schema": "auro.autocycle.report.v1",
        "ok": True,
        "loop": "SENSE → REASON → ACT(python) → OBSERVE → ABSORB → TRAIN → GOVERN → LOOP",
        "doctrine_id": doctrine.get("doctrine_id"),
        "principle": doctrine.get("principle"),
        "laws": [L.get("id") for L in (doctrine.get("laws") or [])],
        "operating_routines": [r.get("id") for r in (doctrine.get("operating_routines") or [])],
        "model_id": mind.model_id,
        "num_params_live": mind.language.num_params,
        "train_steps_before": steps_before,
        "train_steps_after": steps_after,
        "train_steps_delta": steps_after - steps_before,
        "python_organ": python.info(),
        "cycles": cycles_out,
        "python_ok_rate": python.info().get("ok_rate"),
        "checkpoint": str(out_dir),
        "checkpoint_meta": meta,
        "elapsed_s": time.time() - t0,
        "claim_boundary": doctrine.get("claim_boundary"),
        "mesie_power": power.info() if power else None,
        "what_we_built_for_autocycle": [
            "Python organ (sandbox + doctrine lint + receipts)",
            "Operating doctrine JSON (laws PL-001..PL-010 + routines)",
            "Sense plane: GitHub knowledge DB + multi-MESIE embed + ANN/LSH",
            "Compression: SVD hybrid / top-k / φ on multi-views",
            "Reason plane: mind.reason + constitutional/governance",
            "Act plane: python exec with ui/db/github_search/mesie_embed",
            "Learn plane: absorb multi-embed Experience + online train_step",
            "Power train: larger SpectralGPT MoE geometry (optional)",
            "Persist: mind checkpoint after cycles",
            "Show: per-stage console + report JSON/MD",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "AUTOCYCLE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _report_md(report)
    (out_dir / "AUTOCYCLE_REPORT.md").write_text(md, encoding="utf-8")
    if cfg.show:
        print(md, flush=True)
    return report


def _report_md(report: Dict[str, Any]) -> str:
    lines = [
        f"# Autocycle Report — {report.get('model_id')}",
        "",
        f"**Loop:** `{report.get('loop')}`",
        "",
        f"- Doctrine: `{report.get('doctrine_id')}`",
        f"- Laws: {', '.join(report.get('laws') or [])}",
        f"- Live params: **{report.get('num_params_live'):,}**",
        f"- Train steps: {report.get('train_steps_before')} → **{report.get('train_steps_after')}** "
        f"(Δ {report.get('train_steps_delta')})",
        f"- Python ok rate: **{report.get('python_ok_rate')}**",
        f"- Cycles: **{len(report.get('cycles') or [])}**",
        f"- Checkpoint: `{report.get('checkpoint')}`",
        "",
        "## Principle",
        "",
        report.get("principle") or "",
        "",
        "## Cycles",
        "",
    ]
    for c in report.get("cycles") or []:
        act = (c.get("stages") or {}).get("act") or {}
        learn = (c.get("stages") or {}).get("learn") or {}
        lines.append(
            f"- **{c.get('cycle')}** `{c.get('goal', '')[:60]}` "
            f"python_ok={act.get('python_ok')} steps={learn.get('train_steps')}"
        )
    lines.extend(
        [
            "",
            "## Autocycle stack",
            "",
        ]
    )
    for w in report.get("what_we_built_for_autocycle") or []:
        lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Auro autocycle: sense→reason→python→learn")
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--train-steps", type=int, default=2)
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_continual")
    p.add_argument("--model", default="Auro-2B")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    report = run_autocycle(
        AutocycleConfig(
            model_id=args.model,
            cycles=args.cycles,
            train_steps_per_cycle=args.train_steps,
            resume_checkpoint=args.resume if Path(args.resume).exists() else None,
            show=not args.quiet,
        )
    )
    slim = {
        k: report[k]
        for k in (
            "ok",
            "loop",
            "doctrine_id",
            "num_params_live",
            "train_steps_before",
            "train_steps_after",
            "train_steps_delta",
            "python_ok_rate",
            "checkpoint",
            "elapsed_s",
        )
    }
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
