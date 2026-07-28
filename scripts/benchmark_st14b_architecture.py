#!/usr/bin/env python3
"""Emit reproducible AURO ST-14B architecture evidence.

This is an architecture-contract benchmark, not an H100 performance claim. It
computes parameter geometry, KV-cache requirements, quantization targets, and
promotion gates into a machine-readable receipt.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from auro_native_llm.model.st14b import (
    ST14BArchitecture,
    build_st14b_config,
    st14b_quantization_matrix,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/st14b/architecture-evidence.json",
        help="Path for the architecture evidence receipt",
    )
    args = parser.parse_args()

    spec = ST14BArchitecture()
    config = build_st14b_config()
    payload = {
        "schema": "auro.st14b.evidence.v1",
        "generated_at": int(time.time()),
        "architecture_report": spec.report(),
        "auro_config": config.to_dict(),
        "quantization_matrix": st14b_quantization_matrix(),
        "benchmark_program": {
            "required_runtimes": ["pytorch", "transformers", "vllm", "tensorrt-llm", "llama.cpp"],
            "required_precisions": ["bf16", "fp8", "int8", "int4"],
            "required_metrics": [
                "prefill_tokens_per_second",
                "decode_tokens_per_second",
                "time_to_first_token_ms",
                "inter_token_latency_ms",
                "peak_gpu_memory_bytes",
                "kv_cache_bytes",
                "model_flops_utilization",
                "quality_delta_vs_bf16",
            ],
            "required_workloads": [
                "prompt_128_decode_128",
                "prompt_2048_decode_256",
                "prompt_8192_decode_256",
                "concurrency_1",
                "concurrency_8",
                "concurrency_32",
            ],
        },
        "truth_boundary": (
            "This artifact proves configuration geometry and theoretical KV-cache reduction only. "
            "It does not prove a trained checkpoint, H100 latency, throughput, MFU, memory-bandwidth "
            "reduction, or quality. Those require exact runtime receipts."
        ),
    }
    payload["evidence_hash"] = "0x" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
