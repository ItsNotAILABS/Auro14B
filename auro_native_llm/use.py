"""Usable entrypoint: think→answer, agents, chrome, neuro, medina parallel.

  python -m auro_native_llm.use "explain MESIE MoE"
  python -m auro_native_llm.use --team "build a small API"
  python -m auro_native_llm.use --multi-site
  python -m auro_native_llm.use --medina-shard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Use Auro mind: think, agents, portal, medina")
    p.add_argument("prompt", nargs="?", default="What is MESIE SpectralGPT and how do we train it?")
    p.add_argument("--model", default="Auro-2B")
    p.add_argument("--lite", action="store_true", default=True)
    p.add_argument("--full-core", action="store_true")
    p.add_argument("--team", action="store_true", help="Run multi-agent team on prompt")
    p.add_argument("--multi-site", action="store_true", help="Multi-site browser agents")
    p.add_argument("--teach", action="store_true", help="Mini brains teach domains")
    p.add_argument("--medina-shard", action="store_true", help="Show FSDP/ZeRO/tensor/pipeline plan")
    p.add_argument("--ready", action="store_true", help="Run NOVA promotion readiness (coding+reason)")
    p.add_argument("--code-harness", action="store_true", help="Run real coding harness only")
    p.add_argument("--long-harness", action="store_true", help="Long coding+reasoning harness + SDK inject")
    p.add_argument(
        "--mesie-runtime",
        action="store_true",
        help="Bind/probe installed mesie transformers into model runtime and print health",
    )
    p.add_argument(
        "--specialize",
        action="store_true",
        help="Train specialization: generative + embedding + self-config + all repo skills",
    )
    p.add_argument("--specialize-rounds", type=int, default=8)
    p.add_argument("--specialize-steps", type=int, default=40)
    p.add_argument(
        "--ghost",
        action="store_true",
        help="Run GHOST supervisor: policy + MESIE Ghost Node + receipts (+ LLM escalate)",
    )
    p.add_argument("--human-approved", action="store_true", help="Approve C4/C5 ghost actions")
    p.add_argument(
        "--google",
        action="store_true",
        help="Google virtual envelope: AI sandbox Chrome/Mail/Drive + collab workspace",
    )
    p.add_argument("--google-surface", default="status", help="chrome|mail|drive|search|collab|sites|status")
    p.add_argument("--google-action", default="list")
    p.add_argument("--collab", action="store_true", help="Collab chat turn (prompt = your message)")
    p.add_argument(
        "--physics",
        action="store_true",
        help="Run real physics AI formulas (dispersion/coherence/Kuramoto/Landau) + train steps",
    )
    p.add_argument("--physics-steps", type=int, default=20)
    p.add_argument(
        "--discover",
        action="store_true",
        help="Capability-state discovery (8-step sequence + 8-status taxonomy; ban 'I can't')",
    )
    p.add_argument(
        "--dual",
        action="store_true",
        help="Python=AI + Julia=BRAIN distributed virtual physics cores (environment is not ceiling)",
    )
    p.add_argument("--brain-cores", type=int, default=0, help="Virtual physics cores (0=auto threads)")
    p.add_argument("--brain-steps", type=int, default=4)
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="MESIE Virtual Processor hybrid: deterministic Ghost/MESIE first, LLM only if justified",
    )
    p.add_argument(
        "--mesie-only",
        action="store_true",
        help="Force zero LLM escalation on hybrid path",
    )
    p.add_argument(
        "--hybrid-demo",
        action="store_true",
        help="Batch demo: most work calls skip LLM",
    )
    p.add_argument(
        "--power-stack",
        action="store_true",
        help="Deep physics+economic engines+algorithms+transformers coupled loop",
    )
    p.add_argument("--stack-rounds", type=int, default=5)
    p.add_argument("--stack-physics-steps", type=int, default=3)
    p.add_argument(
        "--github",
        action="store_true",
        help="GitHub access status (gh sign-in + MCP identity)",
    )
    p.add_argument("--github-repos", action="store_true", help="List repos for authenticated user/org")
    p.add_argument("--github-owner", default="", help="Owner/org for --github-repos")
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_continual")
    p.add_argument("--max-tokens", type=int, default=96)
    args = p.parse_args(argv)

    from auro_native_llm.organism.family import build_mind
    from auro_native_llm.organism.checkpoint import load_mind

    # GitHub access (CLI + MCP guidance) — no full mind required
    if args.github or args.github_repos:
        from auro_native_llm.github_access import GitHubAccess

        ga = GitHubAccess()
        if args.github_repos:
            owner = args.github_owner or None
            if not owner:
                who = ga.whoami()
                owner = who.get("login")
            out = {
                "status": ga.status(),
                "repos": ga.list_repos(owner=owner, limit=20),
                "orgs": ga.orgs(),
            }
        else:
            out = {
                "status": ga.status(),
                "orgs": ga.orgs() if ga.status().get("signed_in") else {},
                "how_to": ga.login_hint(),
            }
        print(json.dumps(out, indent=2, default=str)[:12000], flush=True)
        return 0 if out.get("status", {}).get("signed_in") or out.get("repos", {}).get("ok") else 2

    # Power stack: physics + econ + algorithms + transformers together
    if args.power_stack:
        from auro_native_llm.engines.orchestra import PowerStack

        mind_s = None
        resume = Path(args.resume)
        try:
            if resume.exists():
                print(f"[power-stack] loading {resume}", flush=True)
                mind_s = load_mind(resume, chrome_mock=True)
        except Exception as exc:
            print(f"[power-stack] mind optional ({exc})", flush=True)
        stack = PowerStack(mind_s)
        rep = stack.run(
            args.prompt,
            rounds=args.stack_rounds,
            physics_steps=args.stack_physics_steps,
        )
        # compact
        hist = rep.get("history") or []
        last = hist[-1] if hist else {}
        print(
            json.dumps(
                {
                    "ok": rep.get("ok"),
                    "rounds": rep.get("rounds"),
                    "engines": rep.get("engines"),
                    "last": {
                        "physics_energy": (last.get("physics") or {}).get("energy"),
                        "rg_coupling": (last.get("physics") or {}).get("coupling"),
                        "magnetization": (last.get("physics") or {}).get("metrics", {}).get(
                            "magnetization"
                        ),
                        "wealth": (last.get("economy") or {}).get("wealth"),
                        "utility": (last.get("economy") or {}).get("utility"),
                        "free_energy": (last.get("economy") or {}).get("free_energy"),
                        "excess_demand": (last.get("economy") or {}).get("excess_demand_l2"),
                        "ot_cost": (last.get("algorithms") or {}).get("ot_cost"),
                        "match_cost": (last.get("algorithms") or {}).get("match_cost"),
                        "route": last.get("route"),
                        "transformer": last.get("transformer"),
                        "latency_ms": last.get("latency_ms"),
                    },
                    "trajectory": [
                        {
                            "E": (h.get("physics") or {}).get("energy"),
                            "W": (h.get("economy") or {}).get("wealth"),
                            "route": ((h.get("route") or {}).get("route")),
                            "score": ((h.get("route") or {}).get("score")),
                        }
                        for h in hist
                    ],
                    "saved": rep.get("saved"),
                    "philosophy": rep.get("philosophy"),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0 if rep.get("ok") else 2

    # Hybrid MESIE Virtual Processor — killer use case
    if args.hybrid or args.hybrid_demo:
        from auro_native_llm.vproc.hybrid import HybridRuntime

        mind_h = None
        resume = Path(args.resume)
        try:
            if resume.exists():
                print(f"[hybrid] loading {resume}", flush=True)
                mind_h = load_mind(resume, chrome_mock=True)
        except Exception as exc:
            print(f"[hybrid] mind optional ({exc})", flush=True)
        rt = HybridRuntime(mind_h)
        if args.hybrid_demo:
            demo = rt.batch_demo()
            print(json.dumps(demo, indent=2), flush=True)
            return 0
        out = rt.execute(args.prompt, force_mesie_only=args.mesie_only, save=True)
        # compact view
        wc = out.get("work_call") or {}
        m = wc.get("metrics") or {}
        print(
            json.dumps(
                {
                    "ok": out.get("ok"),
                    "routing": m.get("routing"),
                    "llm_used": (wc.get("result") or {}).get("escalate"),
                    "escalate_reason": (wc.get("result") or {}).get("escalate_reason"),
                    "metrics": m,
                    "features": wc.get("features"),
                    "receipt_tip": wc.get("receipt_tip"),
                    "killer_use_case": out.get("killer_use_case"),
                    "saved": out.get("saved"),
                    "processor": out.get("processor"),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0 if out.get("ok") else 2

    # Dual organism can run with light mind or without heavy load
    if args.dual:
        from auro_native_llm.dual import DualOrganism

        mind_d = None
        resume = Path(args.resume)
        try:
            if resume.exists():
                print(f"[dual] loading mind {resume}", flush=True)
                mind_d = load_mind(resume, chrome_mock=True)
            else:
                print("[dual] AI without heavy resume — still runs", flush=True)
        except Exception as exc:
            print(f"[dual] mind optional ({exc}); AI still drives brain", flush=True)
        org = DualOrganism(mind_d, n_cores=args.brain_cores)
        health = org.health()
        print(json.dumps({"health": health}, indent=2, default=str)[:3000], flush=True)
        result = org.think(args.prompt, steps=args.brain_steps)
        # save
        outp = Path("artifacts/auro-dual")
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "LAST_DUAL_THINK.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, default=str)[:8000], flush=True)
        print(f"\n[dual] saved {outp / 'LAST_DUAL_THINK.json'}", flush=True)
        return 0 if result.get("ok") else 2

    # Capability discovery can run without loading a full mind
    if args.discover:
        from auro_native_llm.capability import run_discovery

        rep = run_discovery(save=True, probes=True)
        # compact console view
        print(
            json.dumps(
                {
                    "ok": rep.get("ok"),
                    "repo_root": rep.get("repo_root"),
                    "elapsed_s": rep.get("elapsed_s"),
                    "n_capabilities": rep.get("n_capabilities"),
                    "available": rep.get("available"),
                    "by_status": {
                        k: v for k, v in (rep.get("by_status") or {}).items() if v
                    },
                    "constraints": rep.get("constraints"),
                    "saved": rep.get("saved"),
                    "anti_pattern": rep.get("anti_pattern"),
                    "taxonomy": rep.get("taxonomy"),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if rep.get("ok") else 2

    lite = not args.full_core
    resume = Path(args.resume)
    # Full runtime only when a heavy flag needs portal/SDK/mesie bind
    need_full = any(
        [
            getattr(args, "google", False),
            getattr(args, "collab", False),
            getattr(args, "ghost", False),
            getattr(args, "mesie_runtime", False),
            getattr(args, "multi_site", False),
            getattr(args, "long_harness", False),
            getattr(args, "specialize", False),
            getattr(args, "power_stack", False),
        ]
    )
    if resume.exists():
        print(f"[use] loading {resume} full_runtime={need_full}", flush=True)
        mind = load_mind(resume, chrome_mock=True, full_runtime=need_full)
    else:
        print(f"[use] building {args.model} lite={lite}", flush=True)
        mind = build_mind(args.model, lite=lite and not need_full, chrome_mock=True)

    print(
        json.dumps(
            {
                "model_id": mind.model_id,
                "num_params_live": mind.language.num_params,
                "train_steps": mind.language.train_steps,
                "neuro": getattr(mind.language, "_neuro", None)
                and mind.language._neuro.core.info(),  # type: ignore[attr-defined]
                "capabilities_n": len(mind.info().get("capabilities") or []),
            },
            indent=2,
        ),
        flush=True,
    )

    if args.physics:
        from auro_native_llm.physics import get_physics_engine, EQUATIONS
        from auro_native_llm.physics.formulas import (
            dispersion_omega,
            wiener_coherence,
            resonance_score,
            kuramoto_order,
            kuramoto_step,
            landau_free_energy,
            spectral_action_density,
            text_to_physical_signal,
            spectrum_from_signal,
            PHI,
        )
        import numpy as np

        eng = get_physics_engine()
        # bind physics on loaded mind
        mind.language.physics = eng
        sig = text_to_physical_signal(args.prompt, 128)
        spec = spectrum_from_signal(sig)
        k = np.linspace(0, float(PHI), 64)
        w = dispersion_omega(k)
        coh, _ = wiener_coherence(sig, eng.embed_physics(args.prompt, 128))
        R = resonance_score(spec, np.abs(np.fft.rfft(sig)))
        th = np.angle(np.fft.rfft(sig))
        th2 = kuramoto_step(th, w[: th.size] if w.size >= th.size else np.resize(w, th.size))
        r, psi = kuramoto_order(th2)
        S, _ = spectral_action_density(spec)
        m = eng.embed_physics(args.prompt, 32)
        F = landau_free_energy(m, m)
        # real train steps with physics loss
        tok = mind.language.tokenizer
        ids = tok.encode(args.prompt + " MESIE physics φ coherence", add_bos=True, add_eos=True, max_length=64)
        while len(ids) < 64:
            ids.append(tok.pad_id)
        arr = np.array([ids[:64]], dtype=np.int64)
        hist = []
        for i in range(args.physics_steps):
            met = mind.language.train_step(arr, arr, lr=2e-3, text_for_meaning=args.prompt)
            hist.append(met)
            if i == 0 or i == args.physics_steps - 1 or (i + 1) % 5 == 0:
                print(
                    f"  [physics {i+1}/{args.physics_steps}] "
                    f"ce={met.get('ce',0):.4f} L={met.get('loss',0):.4f} "
                    f"coh={met.get('phys_coherence',0):.3f} r={met.get('phys_kuramoto_r',0):.3f} "
                    f"R={met.get('phys_resonance',0):.3f} lr={met.get('lr',0):.5f}",
                    flush=True,
                )
        rep = eng.report().to_dict()
        print(
            json.dumps(
                {
                    "scaffold": False,
                    "fake": False,
                    "equations": EQUATIONS if isinstance(EQUATIONS, dict) else rep.get("equations"),
                    "closed_form_demo": {
                        "dispersion_omega_mean": float(w.mean()),
                        "coherence": coh,
                        "resonance": R,
                        "kuramoto_r": r,
                        "kuramoto_psi": psi,
                        "spectral_action": float(S),
                        "landau_F": float(F),
                        "phi": float(PHI),
                    },
                    "train": {
                        "steps": args.physics_steps,
                        "ce_first": hist[0].get("ce") if hist else None,
                        "ce_last": hist[-1].get("ce") if hist else None,
                        "loss_first": hist[0].get("loss") if hist else None,
                        "loss_last": hist[-1].get("loss") if hist else None,
                        "physics_flag": hist[-1].get("physics") if hist else None,
                    },
                    "engine_report": rep,
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0

    if args.collab:
        out = mind.collab(args.prompt)
        print(json.dumps(out, indent=2, default=str)[:8000], flush=True)
        return 0 if out.get("ok") else 2

    if args.google:
        env = mind.google_envelope(chrome_mock=True)
        # demo bundle when surface=status
        if args.google_surface == "status" and args.google_action == "list":
            demo = {
                "health": env.health(),
                "search": mind.google("search", query="MESIE spectral intelligence"),
                "mail_compose": mind.google(
                    "mail",
                    "compose",
                    to="human@collab.local",
                    subject="Sandbox check-in",
                    body="AI Gmail sandbox is live. This is not your real Gmail.",
                ),
                "drive_create": mind.google(
                    "drive",
                    "create",
                    name="Collab Notes.md",
                    content="# Shared notes\nFrom AI sandbox Drive → collab project.\n",
                ),
                "open_search": mind.google("sites", name="search"),
                "collab_projects": mind.google("collab", "list"),
            }
            print(json.dumps(demo, indent=2, default=str)[:12000], flush=True)
            return 0
        kw = {}
        # map prompt into sensible kwargs
        if args.google_surface == "search":
            kw["query"] = args.prompt
        elif args.google_surface == "collab":
            kw["text"] = args.prompt
        elif args.google_surface in ("chrome", "browser"):
            kw["url"] = args.prompt if args.prompt.startswith("http") else "https://www.google.com/"
        elif args.google_surface in ("sites", "google"):
            kw["name"] = args.prompt if args.prompt in (
                "search", "mail", "drive", "docs", "calendar", "maps", "youtube", "scholar"
            ) else "search"
        elif args.google_surface == "mail" and args.google_action == "compose":
            kw["to"] = "human@collab.local"
            kw["subject"] = "From Auro"
            kw["body"] = args.prompt
        elif args.google_surface == "drive" and args.google_action == "create":
            kw["name"] = "note.md"
            kw["content"] = args.prompt
        out = mind.google(args.google_surface, args.google_action, **kw)
        print(json.dumps(out, indent=2, default=str)[:10000], flush=True)
        return 0 if out.get("ok", True) else 2

    if args.ghost:
        from auro_native_llm.ghost.supervisor import GhostSupervisor
        from auro_native_llm.ghost.pillars import pillars_health
        from auro_native_llm.ghost.doctrine import all_ghost_training_blocks
        from auro_native_llm.organism.self_train import Experience
        from pathlib import Path as _P

        # ensure ghost + mesie bound
        if getattr(mind, "ghost", None) is None:
            mind.ghost = GhostSupervisor(mind)  # type: ignore[attr-defined]
        try:
            from auro_native_llm.mesie_runtime import attach_mesie_runtime

            attach_mesie_runtime(mind, lite=True)
        except Exception:
            pass
        # absorb GHOST doctrine then train a short pulse
        if mind.organs.trainer:
            for b in all_ghost_training_blocks():
                mind.organs.trainer.absorb(
                    Experience(
                        text=b,
                        kind="ghost_doctrine",
                        model_id=mind.model_id,
                        reward=0.95,
                        meta={"pillar": "GHOST"},
                    )
                )
            train_rep = mind.organs.trainer.train_on_model(mind.language, steps=24)
        else:
            train_rep = {"ok": False}
        sup = mind.ghost  # type: ignore[attr-defined]
        out = sup.run(
            args.prompt,
            human_approved=args.human_approved,
            offline=True,
        ).to_dict()
        # save receipt chain
        art = _P("artifacts/auro-ghost")
        art.mkdir(parents=True, exist_ok=True)
        chain_path = art / "last_receipt_chain.json"
        chain_path.write_text(json.dumps(out.get("receipt_chain") or {}, indent=2), encoding="utf-8")
        (art / "last_ghost_run.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(
            json.dumps(
                {
                    "pillars": pillars_health(),
                    "ok": out.get("ok"),
                    "risk_class": out.get("risk_class"),
                    "policy": out.get("policy"),
                    "haunt_flags": out.get("haunt_flags"),
                    "used_llm": (out.get("outcome") or {}).get("used_llm"),
                    "n_steps": ((out.get("outcome") or {}).get("result") or {}).get("n_steps"),
                    "custody": (out.get("receipt_chain") or {}).get("custody"),
                    "claims_n": len(out.get("claims") or []),
                    "receipt_path": str(chain_path),
                    "train_pulse": train_rep,
                    "latency_ms": out.get("latency_ms"),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0 if out.get("ok") else 2

    if args.specialize:
        from auro_native_llm.train.specialize import SpecializeConfig, run_specialization

        rep = run_specialization(
            SpecializeConfig(
                rounds=args.specialize_rounds,
                steps_per_round=args.specialize_steps,
                resume_checkpoint=args.resume,
            )
        )
        print(
            json.dumps(
                {
                    "ok": rep.get("ok"),
                    "skills_n": rep.get("skills_n"),
                    "skill_names": (rep.get("skill_names") or [])[:40],
                    "train_steps_delta": rep.get("train_steps_delta"),
                    "train_steps": rep.get("train_steps"),
                    "mean_ce_first": rep.get("mean_ce_first"),
                    "mean_ce_last": rep.get("mean_ce_last"),
                    "checkpoint": rep.get("checkpoint"),
                    "mesie": rep.get("mesie"),
                    "sdk": rep.get("sdk"),
                    "elapsed_s": rep.get("elapsed_s"),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if rep.get("ok") else 2

    if args.mesie_runtime:
        from auro_native_llm.mesie_runtime import attach_mesie_runtime, probe_mesie_install

        probe = probe_mesie_install()
        health = attach_mesie_runtime(mind, lite=not args.full_core, force_rebind=True)
        # also bind Auro-14B stack for family claim path
        from auro_native_llm.mesie_runtime import get_mesie_runtime

        stack_14 = get_mesie_runtime("Auro-14B", lite=not args.full_core, force_rebind=True)
        gpt_probe = stack_14.spectral_gpt_forward_probe("MESIE spectral transformer pipeline")
        helix = stack_14.helix_encode([0.1 * i for i in range(32)])
        intel = stack_14.intelligence_reason(
            {"query": "spectral match readiness", "model_id": "Auro-14B"}
        )
        print(
            json.dumps(
                {
                    "probe": {
                        "installed": probe.get("installed"),
                        "version": probe.get("version"),
                        "path": probe.get("path"),
                        "n_ok": probe.get("n_ok"),
                        "n_checked": probe.get("n_checked"),
                        "modules": {
                            k: {"ok": v.get("ok"), "error": v.get("error")}
                            for k, v in (probe.get("modules") or {}).items()
                        },
                    },
                    "mind_runtime": {
                        "model_id": health.get("model_id"),
                        "n_capabilities_on": health.get("n_capabilities_on"),
                        "capabilities_on": health.get("capabilities_on"),
                        "spectral_gpt": health.get("spectral_gpt"),
                        "intelligence_level": health.get("intelligence_level"),
                        "connectome": health.get("connectome"),
                        "pretraining": health.get("pretraining"),
                        "errors": health.get("errors"),
                    },
                    "auro_14b_runtime": stack_14.health(),
                    "spectral_gpt_probe": gpt_probe,
                    "helix_probe": helix,
                    "intelligence_probe": {
                        "ok": intel.get("ok"),
                        "level": intel.get("level"),
                        "error": intel.get("error"),
                    },
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0 if probe.get("installed") and health.get("n_capabilities_on", 0) >= 8 else 2

    if args.ready:
        rep = mind.ready()
        r = rep.get("readiness") or {}
        print(
            json.dumps(
                {
                    "tier": r.get("tier"),
                    "ready": r.get("ready"),
                    "coding_pass_rate": r.get("coding_pass_rate"),
                    "reasoning_accuracy": r.get("reasoning_accuracy"),
                    "generation_usable": r.get("generation_usable"),
                    "blockers": r.get("blockers"),
                    "receipt_sha256": rep.get("receipt_sha256"),
                    "expansion_allowed": rep.get("expansion_allowed"),
                    "num_params_live": rep.get("num_params_live"),
                    "rule": rep.get("rule"),
                },
                indent=2,
            ),
            flush=True,
        )
        print("\n" + (Path("artifacts/auro-readiness/PROMOTION_RECEIPT.md").read_text(encoding="utf-8") if Path("artifacts/auro-readiness/PROMOTION_RECEIPT.md").exists() else ""), flush=True)
        return 0 if rep.get("expansion_allowed") else 2

    if args.code_harness:
        from auro_native_llm.intelligence.coding import run_coding_harness

        h = run_coding_harness(mind, output_path="artifacts/auro-readiness/coding-receipt.json")
        print(json.dumps(h["summary"], indent=2), flush=True)
        for row in h["results"]:
            print(f"  {row['task_id']}: passed={row['passed']} method={row['method']}", flush=True)
        return 0 if h["summary"]["pass_rate"] > 0 else 2

    if args.long_harness:
        from auro_native_llm.intelligence.long_harness import run_long_harnesses

        rep = run_long_harnesses(mind)
        print(
            json.dumps(
                {
                    "coding": rep["coding"]["summary"],
                    "reasoning": rep["reasoning"]["summary"],
                    "sdk": {
                        "n_packages": (rep.get("sdk_runtime") or {}).get("n_packages"),
                        "by_kind": (rep.get("sdk_runtime") or {}).get("by_kind"),
                        "paths_injected": (rep.get("sdk_runtime") or {}).get("paths_injected"),
                    },
                    "mesie_runtime": rep.get("mesie_runtime"),
                    "helpers": rep.get("helpers"),
                    "team": rep.get("team"),
                    "usable": rep.get("usable"),
                    "elapsed_s": rep.get("elapsed_s"),
                },
                indent=2,
            ),
            flush=True,
        )
        for row in rep["coding"]["results"]:
            mark = "OK" if row["passed"] else "FAIL"
            print(f"  [{mark}] {row['task_id']} method={row['method']}", flush=True)
        return 0 if rep.get("usable") else 2

    if args.medina_shard:
        from auro_native_llm.medina.parallel import build_sharder, hybrid_plan

        for mode in ("zero3_fsdp", "tensor", "pipeline", "hybrid_3d"):
            sh = build_sharder(mode, world_size=8 if mode == "hybrid_3d" else 4)
            if mode == "pipeline":
                sh.assign_pipeline_layers(int(mind.config.num_layers))
            rep = sh.shard_language_model(mind.language)
            print(f"\n=== MEDINA {mode} ===", flush=True)
            print(
                json.dumps(
                    {
                        "mode": rep["mode"],
                        "world_size": rep["world_size"],
                        "n_param_shards": rep["n_param_shards"],
                        "n_grad_shards": rep["n_grad_shards"],
                        "n_opt_shards": rep["n_opt_shards"],
                        "per_rank_nbytes": rep["approx_per_rank_nbytes"],
                        "pipeline": rep.get("pipeline"),
                        "torch_fsdp": rep.get("torch_fsdp"),
                    },
                    indent=2,
                ),
                flush=True,
            )
        print("\n=== hybrid plan world=8 ===", flush=True)
        print(json.dumps(hybrid_plan(8).to_dict(), indent=2), flush=True)
        return 0

    if args.teach:
        print(json.dumps(mind.teach_domains(steps_per_lesson=1), indent=2)[:3000], flush=True)
        return 0

    if args.multi_site:
        mind.portal_open(chrome_mock=True)
        out = mind.multi_site(
            args.prompt,
            ["https://example.com", "https://example.org", "https://www.wikipedia.org"],
            chrome_mock=True,
        )
        print(json.dumps({k: out[k] for k in out if k != "open"}, indent=2)[:4000], flush=True)
        return 0

    if args.team:
        from auro_native_llm.agents.manager import AgentManager

        mgr = AgentManager(mind)
        mind.organs.agent_manager = mgr  # type: ignore[attr-defined]
        rep = mgr.run_team(args.prompt)
        print(json.dumps(rep, indent=2, default=str)[:5000], flush=True)
        return 0

    # default: usable hybrid LLM (works — knowledge + tools + LM quality gate)
    result = mind.think_answer(args.prompt, max_new_tokens=args.max_tokens, prefer_lm=True)
    print("\n=== THINK ===\n", result.get("thinking", "")[:2000], flush=True)
    print("\n=== ANSWER ===\n", result.get("answer", "")[:3000], flush=True)
    print(
        "\n=== META ===\n",
        json.dumps(
            {
                "ok": result.get("ok"),
                "usable": result.get("usable", True),
                "method": result.get("method"),
                "lm_used": result.get("lm_used"),
                "num_params": result.get("num_params"),
                "model_id": result.get("model_id") or mind.model_id,
                "train_steps": mind.language.train_steps,
                "compute_plane": "MESIE",
                "latency_ms": result.get("latency_ms"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
