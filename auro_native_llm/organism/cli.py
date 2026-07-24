"""CLI for embedded Auro minds (full organism, always training)."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="auro-mind", description="Auro embedded organism minds")
    p.add_argument("--model", default="Auro-2B")
    p.add_argument("--full-core", action="store_true", help="Non-lite language core")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Mind organ manifest")
    sub.add_parser("family", help="Build all family minds + manifest")
    g = sub.add_parser("think", help="Generate (trains mind)")
    g.add_argument("--prompt", required=True)
    r = sub.add_parser("reason", help="Reason (trains mind)")
    r.add_argument("--topic", required=True)
    c = sub.add_parser("code", help="Code (trains mind)")
    c.add_argument("--task", required=True)
    w = sub.add_parser("work", help="Work tools (trains mind)")
    w.add_argument("--objective", required=True)
    sub.add_parser("pulse", help="Autonomic train pulse")
    ch = sub.add_parser("chrome", help="Chrome organ")
    ch.add_argument("--action", default="dom")
    ch.add_argument("--url", default="https://example.com")

    vt = sub.add_parser("value-train", help="Real corpus train + loss proof + checkpoint")
    vt.add_argument("--steps", type=int, default=60)
    vt.add_argument("--batch-size", type=int, default=4)
    vt.add_argument("--output-dir", default="checkpoints/auro_minds")

    ld = sub.add_parser("load", help="Load mind checkpoint and think")
    ld.add_argument("--path", required=True)
    ld.add_argument("--prompt", default="Auro MESIE spectral mind")

    sub.add_parser("teach", help="Teach mind Monaco/Jupyter/search/MCP")
    uc = sub.add_parser("use-cases", help="Run 100 use-case suite for a domain")
    uc.add_argument("--domain", required=True, choices=["monaco", "jupyter", "search", "mcp"])
    uc.add_argument("--limit", type=int, default=100)

    prod = sub.add_parser(
        "production",
        help="Full loop: train→measure→save→load→work→keep learning",
    )
    prod.add_argument("--steps", type=int, default=50)
    prod.add_argument("--batch-size", type=int, default=4)
    prod.add_argument("--output-dir", default="checkpoints/auro_minds")
    prod.add_argument("--work-objective", default="browse https://example.com and read DOM")
    prod.add_argument("--pulses", type=int, default=3)

    args = p.parse_args(argv)
    from auro_native_llm.organism.family import build_mind, build_family, family_manifest

    lite = not args.full_core

    if args.cmd == "family":
        fam = build_family(lite=lite)
        print(json.dumps(family_manifest(fam), indent=2)[:14000])
        return 0

    if args.cmd == "value-train":
        from auro_native_llm.organism.value_train import ValueTrainConfig, run_value_training

        report = run_value_training(
            ValueTrainConfig(
                model_id=args.model,
                steps=args.steps,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                lite=lite,
            )
        )
        print(json.dumps({k: report[k] for k in (
            "valuable", "improved", "loss_before", "loss_after", "loss_delta_ce",
            "probes", "checkpoint", "num_params_live", "elapsed_s",
        )}, indent=2))
        return 0 if report.get("valuable") or report.get("improved") else 1

    if args.cmd == "load":
        from auro_native_llm.organism.checkpoint import load_mind

        mind = load_mind(args.path)
        print(json.dumps(mind.info(), indent=2))
        print(json.dumps(mind.generate(args.prompt, max_new_tokens=32).to_dict(), indent=2)[:4000])
        return 0

    if args.cmd == "use-cases":
        from auro_native_llm.embedded.monaco import MonacoOrgan
        from auro_native_llm.embedded.jupyter import JupyterOrgan
        from auro_native_llm.embedded.search import SearchOrgan
        from auro_native_llm.embedded.mcp_hub import MCPOrgan
        from auro_native_llm.embedded.runner import run_suite

        if args.domain == "monaco":
            organ = MonacoOrgan()
        elif args.domain == "jupyter":
            organ = JupyterOrgan()
        elif args.domain == "search":
            organ = SearchOrgan(offline=True)
        else:
            organ = MCPOrgan()
            organ.wire_from_mind_organs(
                monaco=MonacoOrgan(),
                jupyter=JupyterOrgan(),
                search=SearchOrgan(offline=True),
            )
        report = run_suite(args.domain, organ, limit=args.limit)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if args.cmd == "production":
        from auro_native_llm.organism.production import ProductionConfig, run_production_loop

        report = run_production_loop(
            ProductionConfig(
                model_id=args.model,
                steps=args.steps,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                lite=lite,
                work_objective=args.work_objective,
                keep_learning_pulses=args.pulses,
            )
        )
        print(
            json.dumps(
                {
                    "valuable": report["valuable"],
                    "loop": report["loop"],
                    "claim_boundary": report["claim_boundary"],
                    "num_params_live": report["num_params_live"],
                    "parameter_target": report["parameter_target"],
                    "value_proof": report["value_proof"],
                },
                indent=2,
            )
        )
        return 0 if report.get("valuable") else 1

    mind = build_mind(args.model, lite=lite)

    if args.cmd == "teach":
        print(json.dumps(mind.teach().to_dict(), indent=2)[:10000])
        return 0

    if args.cmd == "info":
        print(json.dumps(mind.info(), indent=2))
        return 0
    if args.cmd == "think":
        print(json.dumps(mind.generate(args.prompt).to_dict(), indent=2)[:8000])
        return 0
    if args.cmd == "reason":
        print(json.dumps(mind.reason(args.topic).to_dict(), indent=2)[:8000])
        return 0
    if args.cmd == "code":
        print(json.dumps(mind.code(args.task).to_dict(), indent=2)[:8000])
        return 0
    if args.cmd == "work":
        print(json.dumps(mind.work(args.objective).to_dict(), indent=2)[:10000])
        return 0
    if args.cmd == "pulse":
        print(json.dumps(mind.pulse(), indent=2))
        return 0
    if args.cmd == "chrome":
        if args.action == "navigate":
            print(json.dumps(mind.chrome("navigate", url=args.url).to_dict(), indent=2)[:6000])
        else:
            print(json.dumps(mind.chrome(args.action).to_dict(), indent=2)[:6000])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
