"""Benchmark MAESI FastSpectralCompute — loop vs batch matrix cosine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie.sdk import FastSpectralCompute


def main() -> None:
    refs = [load_reference_record(n) for n in list_references()[:30]]
    bench = FastSpectralCompute.benchmark_match(refs, n_repeat=200)
    out = ROOT / "deliverables" / "MAESI_Fast_Compute_Benchmark.json"
    payload = {
        "n_items": bench.n_items,
        "loop_match_ms": bench.loop_match_ms,
        "batch_match_ms": bench.batch_match_ms,
        "speedup_ratio": bench.speedup_ratio,
        "embed_batch_ms": bench.embed_batch_ms,
        "ann_query_ms": bench.ann_query_ms,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Items: {bench.n_items}")
    print(f"Loop:  {bench.loop_match_ms:.2f} ms/match")
    print(f"Batch: {bench.batch_match_ms:.2f} ms total")
    print(f"Speedup: {bench.speedup_ratio:.1f}x")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()