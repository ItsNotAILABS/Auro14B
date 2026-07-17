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
    p.add_argument("--resume", default="checkpoints/auro_minds/Auro-2B_continual")
    p.add_argument("--max-tokens", type=int, default=96)
    args = p.parse_args(argv)

    from auro_native_llm.organism.family import build_mind
    from auro_native_llm.organism.checkpoint import load_mind

    lite = not args.full_core
    resume = Path(args.resume)
    if resume.exists():
        print(f"[use] loading {resume}", flush=True)
        mind = load_mind(resume, chrome_mock=True)
    else:
        print(f"[use] building {args.model} lite={lite}", flush=True)
        mind = build_mind(args.model, lite=lite, chrome_mock=True)

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

    # default: think then answer (neuro + real generation)
    result = mind.think_answer(args.prompt, max_new_tokens=args.max_tokens)
    print("\n=== THINK ===\n", result.get("thinking", "")[:2000], flush=True)
    print("\n=== ANSWER ===\n", result.get("answer", "")[:2000], flush=True)
    print(
        "\n=== META ===\n",
        json.dumps(
            {
                "ok": result.get("ok"),
                "num_params": result.get("num_params"),
                "neuro": result.get("neuro"),
                "latency_ms": result.get("latency_ms"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
