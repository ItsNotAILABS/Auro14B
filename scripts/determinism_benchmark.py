"""Demonstrate deterministic, fast matching / ranking / generation in MESIE."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_reference_record
from mesie import GenerationConfig, generate_psd, match_records
from mesie.matching.ranking import rank_candidates


def timed(fn, repeats: int = 100):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    elapsed = (time.perf_counter() - t0) / repeats
    return out, elapsed


def main() -> None:
    ref = load_reference_record("earthquake_psd_reference")
    cand = load_reference_record("structural_fas_reference")
    cfg = GenerationConfig(seed=42, target_frequency=np.linspace(0.1, 50, 128), amplitude_shape="power_law")

    _, match_s = timed(lambda: match_records(ref, cand))
    _, rank_s = timed(lambda: rank_candidates(ref, [cand, load_reference_record("rotdnn_reference")]))
    g1, gen_s = timed(lambda: generate_psd(cfg))

    g2 = generate_psd(cfg)
    m_repeat = match_records(ref, cand)

    print("=== MESIE determinism & speed benchmark ===\n")
    print(f"match_records (avg over 100):     {match_s*1e6:.1f} µs")
    print(f"rank_candidates (avg over 100):   {rank_s*1e6:.1f} µs")
    print(f"generate_psd (avg over 100):      {gen_s*1e6:.1f} µs")
    print()
    print(f"generate_psd seed=42 run1 id:     {g1.record_id}")
    print(f"generate_psd seed=42 run2 match:  {np.allclose(g1.components[0].amplitude, g2.components[0].amplitude)}")
    print(f"match composite (repeat):           {m_repeat.composite_score:.10f}")
    print()
    print("Why deterministic: GenerationConfig.seed fixes numpy RNG; matchers use")
    print("pure numpy interpolation/metrics (no unseeded randomness in core path).")


if __name__ == "__main__":
    main()