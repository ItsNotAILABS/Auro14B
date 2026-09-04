"""Train the first six AURO local lanes with one reproducible command.

This starts real dev-geometry training from the repository corpus. It produces
checkpoint artifacts locally; promotion still requires evaluation and hashes.

Example:
  python scripts/train_first_six.py --steps 200 --output-dir checkpoints/auro_first_six
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from auro_native_llm.model.train import TrainConfig, train_language_model

FIRST_SIX = ("Auro-156K", "Auro-320M", "Auro-640M", "Auro-1B", "Auro-2B", "Auro-3B")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(FIRST_SIX))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--output-dir", default="checkpoints/auro_first_six")
    p.add_argument("--corpus-root", default=".")
    args = p.parse_args()
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = sorted(set(models) - set(FIRST_SIX))
    if unknown:
        p.error(f"unknown first-six lanes: {unknown}")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for model_id in models:
        started = time.time()
        report = train_language_model(TrainConfig(
            model_id=model_id, mode="dev", steps=args.steps,
            batch_size=args.batch_size, seq_len=args.seq_len,
            vocab_size=args.vocab_size, output_dir=str(root / model_id),
            corpus_root=args.corpus_root, multi_repo=False,
            extra={"release_lane": "first-six", "claim_boundary": "dev-geometry-only"},
        ))
        results.append({
            "model_id": model_id,
            "checkpoint": report.get("checkpoint"),
            "live_parameters": report.get("num_params"),
            "parameter_target": report.get("parameter_target"),
            "steps": args.steps,
            "elapsed_s": round(time.time() - started, 3),
            "checkpoint_evidence": "local-candidate-not-promoted",
        })
    receipt = {
        "schema": "auro.first-six.training-receipt.v1",
        "models": results,
        "corpus_root": args.corpus_root,
        "claim_boundary": "real local dev-geometry checkpoints; not full parameter-target training or public promotion",
    }
    (root / "first-six-training-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
