#!/usr/bin/env python3
"""Benchmark HIM feline-inspired neuromorphic dynamics with deterministic receipts.

Results are normalized CEU/control metrics, not physical energy measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auro_native_llm.brain.feline_neuromorphic import FelineNeuromorphicEngine
from auro_native_llm.brain.fused import HIMBrain as BaseHIMBrain


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def scenario_drives(name: str, cycle: int, regions: tuple[str, ...]) -> tuple[dict[str, float], float, float]:
    drives = {region: 0.0 for region in regions}
    if name == "quiet":
        drives["DLPFC_L"] = 0.08
        return drives, 0.15, 0.05
    if name == "visual_orienting":
        drives["SC"] = 1.0 if cycle % 8 == 0 else 0.18
        drives["V1"] = 0.92 if cycle % 8 in {0, 1} else 0.12
        drives["THL_L"] = 0.55 if cycle % 8 == 0 else 0.05
        return drives, 0.82, 0.88 if cycle % 8 == 0 else 0.22
    if name == "executive":
        for region in ("DLPFC_L", "ACC", "PPC_L", "HPC_R"):
            drives[region] = 0.62
        return drives, 0.72, 0.28
    if name == "sustained_overload":
        for region in regions:
            drives[region] = 0.86
        return drives, 0.95, 0.70
    raise ValueError(name)


def run_scenario(name: str, cycles: int) -> dict:
    regions = tuple(region.abbreviation for region in BaseHIMBrain().regions)
    engine = FelineNeuromorphicEngine(regions)
    energy: list[float] = []
    rates: list[float] = []
    sparsity: list[float] = []
    synaptic_events: list[int] = []
    pressure: list[float] = []
    orienting_cycles: list[int] = []

    for index in range(cycles):
        drives, salience, novelty = scenario_drives(name, index, regions)
        result = engine.cycle(drives, salience=salience, novelty=novelty)
        energy.append(result.energy_ceu)
        rates.append(result.spike_rate)
        sparsity.append(result.sparsity)
        synaptic_events.append(result.synaptic_events)
        pressure.append(result.energy_pressure)
        if result.orienting_burst:
            orienting_cycles.append(index)

    cfg = engine.config
    dense_reference_per_cycle = (
        len(regions) * (cfg.idle_energy_ceu + cfg.integration_energy_ceu + cfg.spike_energy_ceu)
        + len(engine.synapses) * cfg.synaptic_event_energy_ceu
    )
    observed_mean = statistics.mean(energy)
    return {
        "scenario": name,
        "cycles": cycles,
        "region_count": len(regions),
        "synapse_count": len(engine.synapses),
        "mean_spike_rate": round(statistics.mean(rates), 8),
        "mean_sparsity": round(statistics.mean(sparsity), 8),
        "mean_synaptic_events": round(statistics.mean(synaptic_events), 8),
        "mean_energy_ceu": round(observed_mean, 8),
        "p95_energy_ceu": round(percentile(energy, 0.95), 8),
        "max_energy_pressure": round(max(pressure), 8),
        "orienting_burst_count": len(orienting_cycles),
        "dense_reference_ceu_per_cycle": round(dense_reference_per_cycle, 8),
        "normalized_energy_vs_dense_reference": round(observed_mean / dense_reference_per_cycle, 8),
        "physical_energy_claim": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--output", default="evidence/neuromorphic-benchmark.json")
    args = parser.parse_args(argv)
    if args.cycles < 8:
        raise SystemExit("--cycles must be >= 8")

    scenarios = [run_scenario(name, args.cycles) for name in ("quiet", "visual_orienting", "executive", "sustained_overload")]
    body = {
        "schema": "auro.neuromorphic-benchmark.v1",
        "energy_unit": "normalized_compute_energy_unit_not_joule",
        "cycles_per_scenario": args.cycles,
        "scenarios": scenarios,
        "acceptance": {
            "quiet_sparse": next(item for item in scenarios if item["scenario"] == "quiet")["mean_sparsity"] >= 0.80,
            "visual_orienting_detected": next(item for item in scenarios if item["scenario"] == "visual_orienting")["orienting_burst_count"] > 0,
            "energy_bounded_vs_dense_reference": all(item["normalized_energy_vs_dense_reference"] <= 1.0 for item in scenarios),
            "no_physical_energy_claim": True,
        },
    }
    body["passed"] = all(body["acceptance"].values())
    body["receipt_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0 if body["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
