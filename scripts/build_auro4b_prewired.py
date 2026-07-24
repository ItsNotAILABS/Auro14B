#!/usr/bin/env python3
"""Build and audit an Auro-4B structured-prewired native checkpoint.

The command creates matched baseline and structured model births with identical
geometry and seed, then records deterministic tensor hashes and distribution
statistics. Full training and comparative benchmarks remain separate governed
steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auro_native_llm.model.auro4b import build_auro4b, write_birth_certificate
from auro_native_llm.model.prewiring import PrewiringConfig


def tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def tensor_stats(value: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(value.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "l2_norm": float(np.linalg.norm(array)),
        "sha256": tensor_sha256(value),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an auditable Auro-4B structured birth")
    parser.add_argument("--output", default="artifacts/auro4b-prewiring")
    parser.add_argument("--seed", type=int, default=873539)
    parser.add_argument("--full-geometry", action="store_true", help="Use the full Auro-4B family geometry; expensive")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    overrides: Dict[str, Any] = {}
    if not args.full_geometry:
        overrides = {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "head_dim": 16,
            "ffn_dim": 128,
            "vocab_size": 512,
            "max_seq_len": 128,
            "num_experts": 2,
            "top_k_experts": 1,
            "continuous_dim": 16,
            "spectral_input_dim": 64,
            "num_kv_heads": 2,
            "seed": args.seed,
        }

    baseline, _ = build_auro4b(structured=False, **overrides)
    candidate, receipt = build_auro4b(
        structured=True,
        prewiring=PrewiringConfig(seed=args.seed),
        **overrides,
    )

    baseline_embedding = baseline.core.embedding.token_embeddings
    candidate_embedding = candidate.core.embedding.token_embeddings
    delta = candidate_embedding - baseline_embedding

    report = {
        "schema": "auro.prewiring.ab.v1",
        "model_id": "Auro-4B",
        "compute_plane": "MESIE",
        "native": True,
        "geometry": "full" if args.full_geometry else "smoke",
        "seed": args.seed,
        "baseline": tensor_stats(baseline_embedding),
        "structured": tensor_stats(candidate_embedding),
        "delta": tensor_stats(delta),
        "receipt": receipt.to_dict() if receipt else None,
        "promotion": {
            "status": "NOT_EVALUATED",
            "required": [
                "equal-token training comparison",
                "gradient-flow comparison",
                "memory and tool-routing evaluation",
                "polyglot coding execution",
                "multimodal alignment",
                "official benchmark campaign",
            ],
        },
    }

    report_path = output / "ab-birth-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_birth_certificate(candidate, output / "birth-certificate.json", receipt)

    print(json.dumps({
        "ok": True,
        "report": str(report_path),
        "birth_certificate": str(output / "birth-certificate.json"),
        "baseline_sha256": report["baseline"]["sha256"],
        "structured_sha256": report["structured"]["sha256"],
        "delta_l2_norm": report["delta"]["l2_norm"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
