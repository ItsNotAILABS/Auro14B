"""50-day backward + 50-day forward orbital-edge spectral analysis.

Uses earthquake PSD reference as seismic coupling anchor. 'Orbital edges' are
modeled as band-limited spectral transients at harmonics of a 16.4-day orbital
period (approx. LEO-ish) plus a 50-day long cycle — matched and ranked with MESIE.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie import load_record, match_records, validate_record
from mesie.cognitive import SpectralAnomalyAdapter
from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.matching.ranking import rank_candidates

# Orbital parameters (tunable)
ORBITAL_PERIOD_DAYS = 16.4
LONG_CYCLE_DAYS = 50.0
EDGE_THRESHOLD = 0.72  # phase-gate for edge events
BACK_DAYS = 50
FWD_DAYS = 50


@dataclass
class DayResult:
    day_offset: int
    phase: str  # "history" | "forecast"
    is_orbital_edge: bool
    earthquake_match: float
    best_reference: str
    best_score: float
    anomaly_score: float
    is_anomaly: bool


def _orbital_edge_gate(day_index: float) -> bool:
    """True when orbital phase crosses edge threshold (rising limb)."""
    p_short = (day_index % ORBITAL_PERIOD_DAYS) / ORBITAL_PERIOD_DAYS
    p_long = (day_index % LONG_CYCLE_DAYS) / LONG_CYCLE_DAYS
    combined = 0.6 * np.sin(2 * np.pi * p_short) + 0.4 * np.sin(2 * np.pi * p_long)
    return float(combined) >= EDGE_THRESHOLD


def _apply_orbital_edge(base_amp: np.ndarray, freq: np.ndarray, day_index: float, rng: np.random.Generator) -> np.ndarray:
    """Inject orbital-edge energy in 0.02–0.5 Hz band on edge days."""
    amp = base_amp.copy().astype(float)
    if not _orbital_edge_gate(day_index):
        # slow drift only
        drift = 1.0 + 0.02 * np.sin(2 * np.pi * day_index / LONG_CYCLE_DAYS)
        return amp * drift

    mask = (freq >= 0.02) & (freq <= 0.5)
    edge_boost = 1.0 + 0.35 * rng.uniform(0.8, 1.2)
    amp[mask] *= edge_boost
    amp += rng.normal(0, 0.005, size=amp.shape)
    return np.maximum(amp, 1e-12)


def record_for_day(
    template: MultiElementRecord,
    day_index: int,
    phase_label: str,
    rng: np.random.Generator,
) -> MultiElementRecord:
    """Build daily spectral record from earthquake template + orbital perturbation."""
    comp = template.components[0]
    freq = comp.frequency.copy()
    amp = _apply_orbital_edge(np.abs(comp.amplitude), freq, day_index, rng)
    edge = _orbital_edge_gate(day_index)
    return MultiElementRecord(
        record_id=f"orbital_day_{int(day_index):+04d}_{phase_label}",
        components=[
            SpectralComponent(
                name="orbital_coupled_ns",
                frequency=freq,
                amplitude=amp,
                units=comp.units or "g^2/Hz",
                domain="frequency",
                metadata={"orbital_edge": edge, "day_index": day_index},
            )
        ],
        representation="psd",
        metadata=template.metadata,
        lineage=["earthquake_psd_reference", "orbital_edge_model"],
    )


def analyze() -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    earthquake = load_reference_record("earthquake_psd_reference")
    catalog = {name: load_reference_record(name) for name in list_references()}

    # Fit anomaly baseline on non-edge historical normals (synthetic quiet days)
    anomaly = SpectralAnomalyAdapter(threshold=2.5)
    baseline_days = [
        record_for_day(earthquake, float(d), "baseline", rng)
        for d in range(-80, -60)
        if not _orbital_edge_gate(float(d))
    ]
    anomaly.fit_baseline(baseline_days)

    results: List[DayResult] = []

    # --- Last 50 days (backward from today = day 0) ---
    for d in range(-BACK_DAYS, 0):
        rec = record_for_day(earthquake, float(d), "history", rng)
        v = validate_record(rec)
        if not v.is_valid:
            continue
        eq_match = match_records(earthquake, rec).composite_score
        ranked = rank_candidates(rec, list(catalog.values()))
        best = ranked[0]
        anom = anomaly.score_anomaly(rec)
        results.append(
            DayResult(
                day_offset=d,
                phase="history",
                is_orbital_edge=_orbital_edge_gate(float(d)),
                earthquake_match=eq_match,
                best_reference=best.candidate_id,
                best_score=best.composite_score,
                anomaly_score=anom,
                is_anomaly=anomaly.is_anomaly(rec),
            )
        )

    history = [r for r in results if r.phase == "history"]
    edge_days = [r for r in history if r.is_orbital_edge]
    normal_days = [r for r in history if not r.is_orbital_edge]

    # Learn forward model from history: linear trend of earthquake match vs day
    if len(history) >= 5:
        xs = np.array([r.day_offset for r in history], dtype=float)
        ys = np.array([r.earthquake_match for r in history], dtype=float)
        slope, intercept = np.polyfit(xs, ys, 1)
        mean_anom_edge = float(np.mean([r.anomaly_score for r in edge_days])) if edge_days else 0.0
        mean_anom_normal = float(np.mean([r.anomaly_score for r in normal_days])) if normal_days else 0.0
    else:
        slope, intercept = 0.0, 0.8
        mean_anom_edge, mean_anom_normal = 0.0, 0.0

    # --- Next 50 days (forward forecast) ---
    forecasts: List[DayResult] = []
    for d in range(1, FWD_DAYS + 1):
        predicted_match = float(np.clip(slope * d + intercept, 0, 1))
        predicted_edge = _orbital_edge_gate(float(d))
        # synthetic forecast record for ranking/anomaly
        rec = record_for_day(earthquake, float(d), "forecast", rng)
        ranked = rank_candidates(rec, list(catalog.values()))
        best = ranked[0]
        anom = anomaly.score_anomaly(rec)
        forecasts.append(
            DayResult(
                day_offset=d,
                phase="forecast",
                is_orbital_edge=predicted_edge,
                earthquake_match=predicted_match,
                best_reference=best.candidate_id,
                best_score=best.composite_score,
                anomaly_score=anom,
                is_anomaly=anom > mean_anom_normal * 1.5,
            )
        )

    future_edges = [f for f in forecasts if f.is_orbital_edge]
    alert_days = [f.day_offset for f in forecasts if f.is_orbital_edge and f.is_anomaly]

    report = {
        "model": {
            "orbital_period_days": ORBITAL_PERIOD_DAYS,
            "long_cycle_days": LONG_CYCLE_DAYS,
            "edge_threshold": EDGE_THRESHOLD,
            "anchor": "earthquake_psd_reference",
        },
        "history_summary": {
            "days": len(history),
            "orbital_edge_days": len(edge_days),
            "mean_eq_match_edge": float(np.mean([r.earthquake_match for r in edge_days])) if edge_days else None,
            "mean_eq_match_normal": float(np.mean([r.earthquake_match for r in normal_days])) if normal_days else None,
            "mean_anomaly_edge": mean_anom_edge,
            "mean_anomaly_normal": mean_anom_normal,
        },
        "forward_model": {
            "match_trend_slope_per_day": float(slope),
            "match_trend_intercept": float(intercept),
            "predicted_orbital_edge_days": [f.day_offset for f in future_edges],
            "alert_days_anomaly_elevated": alert_days,
        },
        "recognition_rules": {
            "edge_detected_when": "orbital phase gate >= threshold (short + long harmonic)",
            "coupling_weak_when": "earthquake_match drops > 0.1 below 50-day mean",
            "forward_alert_when": "forecast day is orbital edge AND anomaly > 1.5x normal baseline",
        },
        "history": [asdict(r) for r in history],
        "forecast": [asdict(r) for r in forecasts],
    }
    return report


def main() -> None:
    report = analyze()
    out = ROOT / "scripts" / "orbital_edge_50d_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    h = report["history_summary"]
    f = report["forward_model"]
    print("=== MESIE Orbital Edge Analysis (50d back + 50d forward) ===\n")
    print(f"History days analyzed:     {h['days']}")
    print(f"Orbital edge days (back):  {h['orbital_edge_days']}")
    print(f"Mean EQ match (edge):      {h['mean_eq_match_edge']:.4f}" if h["mean_eq_match_edge"] else "")
    print(f"Mean EQ match (normal):    {h['mean_eq_match_normal']:.4f}" if h["mean_eq_match_normal"] else "")
    print(f"Match trend (slope/day):   {f['match_trend_slope_per_day']:.6f}")
    print(f"Future orbital edge days:  {f['predicted_orbital_edge_days'][:12]}... ({len(f['predicted_orbital_edge_days'])} total)")
    print(f"Forward alert days:        {f['alert_days_anomaly_elevated'][:15]}")
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()