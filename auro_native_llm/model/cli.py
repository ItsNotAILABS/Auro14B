"""CLI: build, train, generate, job-submit for Auro text LM family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="auro-lm",
        description="Auro text LLM family — MESIE-native, MoE, meaning engines, first-class generate",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a native Auro LM and print info")
    b.add_argument("--model", default="Auro-2B")
    b.add_argument("--mode", default="dev", choices=["dev", "full"])

    t = sub.add_parser("train", help="Train on MESIE corpus")
    t.add_argument("--model", default="Auro-2B")
    t.add_argument("--mode", default="dev", choices=["dev", "full"])
    t.add_argument("--steps", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--seq-len", type=int, default=96)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--vocab-size", type=int, default=4096)
    t.add_argument("--output-dir", default="checkpoints/auro")

    g = sub.add_parser("generate", help="Generate text (load checkpoint if present)")
    g.add_argument("--model", default="Auro-2B")
    g.add_argument("--checkpoint", default=None, help="Path to checkpoint dir")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.85)
    g.add_argument("--mode", default="dev")

    j = sub.add_parser("job", help="Submit MESIE training-fabric pretrain job")
    j.add_argument("--model", default="Auro-2B")
    j.add_argument("--mode", default="dev")
    j.add_argument("--steps", type=int, default=40)
    j.add_argument("--execute", action="store_true", help="Run train now")
    j.add_argument("--all", action="store_true", help="Submit all family lanes")

    i = sub.add_parser("info", help="Show family architecture table")
    i.add_argument("--mode", default="dev")

    args = p.parse_args(argv)

    if args.cmd == "info":
        from auro_native_llm.model.config import (
            all_family_ids,
            family_config,
            family_scale_table,
            list_mesie_presets,
        )
        from auro_native_llm.model.auro_lm import AuroLanguageModel

        rows = []
        for mid in all_family_ids():
            cfg = family_config(mid, mode=args.mode)
            # Build live params for MESIE tiny/small tiers only (safe on laptop)
            live = None
            if args.mode == "dev" and mid in ("Auro-2B", "Auro-4B"):
                m = AuroLanguageModel.build(mid, mode=args.mode)
                live = m.num_params
            rows.append(
                {
                    "model_id": mid,
                    "tier": cfg.tier,
                    "parameter_target": cfg.parameter_target,
                    "mesie_preset": cfg.mesie_preset,
                    "hidden_dim": cfg.hidden_dim,
                    "layers": cfg.num_layers,
                    "heads": cfg.num_heads,
                    "ffn_dim": cfg.ffn_dim,
                    "experts": cfg.num_experts,
                    "top_k": cfg.top_k_experts,
                    "use_moe": cfg.use_moe,
                    "use_cross_modal": cfg.use_cross_modal,
                    "use_spectral_encoder": cfg.use_spectral_encoder,
                    "positional_encoding": cfg.positional_encoding,
                    "activation": cfg.activation,
                    "normalization": cfg.normalization,
                    "num_params_live_sample": live,
                    "compute_plane": "MESIE",
                }
            )
        print(
            json.dumps(
                {
                    "family": rows,
                    "mesie_presets": list_mesie_presets(),
                    "scale_table": family_scale_table(),
                    "arsenal": [
                        "MoE",
                        "cross_modal",
                        "spectral_encoder",
                        "rotary",
                        "swiglu",
                        "rms_norm",
                        "qk_norm",
                        "gqa",
                        "multi_task_heads",
                        "meaning",
                        "spectral_fusion",
                    ],
                    "mathematics": "phi/golden",
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "build":
        from auro_native_llm.model.auro_lm import AuroLanguageModel

        model = AuroLanguageModel.build(args.model, mode=args.mode)
        print(json.dumps(model.info(), indent=2))
        return 0

    if args.cmd == "train":
        from auro_native_llm.model.train import TrainConfig, train_language_model

        report = train_language_model(
            TrainConfig(
                model_id=args.model,
                mode=args.mode,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                learning_rate=args.lr,
                vocab_size=args.vocab_size,
                output_dir=args.output_dir,
            )
        )
        slim = {k: v for k, v in report.items() if k not in ("history",)}
        print(json.dumps(slim, indent=2)[:4000])
        return 0

    if args.cmd == "generate":
        from auro_native_llm.model.auro_lm import AuroLanguageModel
        from auro_native_llm.model.checkpoint import load_checkpoint

        ckpt = args.checkpoint
        if ckpt is None:
            guess = Path("checkpoints/auro") / args.model
            if guess.exists():
                ckpt = str(guess)
        if ckpt:
            model = load_checkpoint(ckpt)
        else:
            model = AuroLanguageModel.build(args.model, mode=args.mode)
        result = model.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.cmd == "job":
        from auro_native_llm.model.jobs import submit_family_jobs, submit_pretrain_job

        if args.all:
            results = submit_family_jobs(mode=args.mode, steps=args.steps, execute=args.execute)
            print(json.dumps(results, indent=2)[:8000])
        else:
            result = submit_pretrain_job(
                args.model, mode=args.mode, steps=args.steps, execute=args.execute
            )
            print(json.dumps(result, indent=2)[:8000])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
