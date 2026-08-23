from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from auro_native_llm.model.checkpoint import load_checkpoint
from auro_native_llm.model.train import TrainConfig, train_language_model

FAMILY = ("Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_manifest(root: Path) -> dict[str, Any]:
    allowed = {".md", ".txt", ".py", ".js", ".ts", ".json"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if any(part in {".git", "node_modules", "checkpoints", "artifacts", "dist"} for part in path.parts):
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "auro.training.corpus_manifest.v1", "files": files, "sha256": digest}


def artifact_digest(directory: Path) -> dict[str, str]:
    names = ("weights.npz", "layers.npz", "tokenizer.json", "config.json", "meta.json", "constitutional_manifest.json")
    return {name: sha256_file(directory / name) for name in names if (directory / name).exists()}


def retrain_model(model_id: str, root: Path, output: Path, steps: int, seq_len: int, max_files: int, max_chars: int) -> dict[str, Any]:
    started = time.time()
    cfg = TrainConfig(model_id=model_id, mode="dev", steps=steps, batch_size=1, seq_len=seq_len,
                      learning_rate=3e-3, vocab_size=4096, seed=42, output_dir=str(output), corpus_root=str(root),
                      report_every=max(1, steps), multi_repo=False, max_corpus_files=max_files,
                      max_corpus_chars=max_chars, extra={"source_seed": "training/brain_ai_seed_report.md", "promotion": "PROTOTYPE"})
    report = train_language_model(cfg)
    checkpoint = Path(report["checkpoint"])
    restored = load_checkpoint(checkpoint, allow_quarantined=True)
    restoration_sample = restored.generate("BRAIN-AI deterministic state, browser intelligence, secure execution:",
                                           max_new_tokens=12, temperature=0.0)
    return {
        "model_id": model_id,
        "mode": "dev",
        "parameter_target": int(report["parameter_target"]),
        "live_parameters": int(report["num_params"]),
        "steps": steps,
        "final_metrics": report["history"][-1] if report["history"] else {},
        "checkpoint": str(checkpoint),
        "checkpoint_files": artifact_digest(checkpoint),
        "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
        "restoration_verified": bool(restoration_sample.text),
        "restoration_sample": restoration_sample.text[:240],
        "elapsed_s": round(time.time() - started, 3),
        "claim_boundary": {"full_parameter_target_trained": False, "dev_geometry_trained": True, "production_promoted": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(FAMILY))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=120_000)
    parser.add_argument("--output", default="artifacts/retrain")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = corpus_manifest(root)
    requested = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = sorted(set(requested) - set(FAMILY))
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")
    results = [retrain_model(model_id, root, out / "checkpoints", args.steps, args.seq_len, args.max_files, args.max_chars)
               for model_id in requested]
    receipt = {
        "schema": "auro.portfolio.retrain_receipt.v1",
        "created_at": int(time.time()),
        "brain_ai_seed": "training/brain_ai_seed_report.md",
        "corpus_manifest": manifest,
        "models": results,
        "all_restored": all(item["restoration_verified"] for item in results),
        "truth_boundary": "This receipt proves executable dev-geometry training and checkpoint restoration only. It does not claim that the 2B/4B/8B/14B/100B parameter targets were fully trained.",
    }
    receipt["evidence_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = out / "portfolio-retrain-receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"receipt": str(target), "models": len(results), "all_restored": receipt["all_restored"], "evidence_sha256": receipt["evidence_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
