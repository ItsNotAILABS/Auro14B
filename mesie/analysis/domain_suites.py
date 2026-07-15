"""Cross-domain MESIE analysis: terrain, robotics, orbital, power, seismic."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _ensure_root() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


@dataclass
class SuiteResult:
    domain: str
    title: str
    records_used: List[str]
    reasoning: Dict[str, Any]
    matches: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    control_commands: List[str]
    logic_rules_fired: int
    plain_conclusion: str
    power_data: Dict[str, Any] = field(default_factory=dict)
    orbital_data: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


def _record_from_modes(
    record_id: str,
    modes: List[Dict[str, Any]],
    *,
    freq_key: str = "frequency_Hz",
    amp_key: str = "typical_amplitude_pT",
    representation: str = "psd",
    lineage: Optional[List[str]] = None,
) -> Any:
    from mesie.core.records import MultiElementRecord, SpectralComponent

    freqs = np.asarray([float(m[freq_key]) for m in modes], dtype=np.float64)
    amps = np.asarray([float(m.get(amp_key, m.get("amplitude", 1.0))) for m in modes], dtype=np.float64)
    return MultiElementRecord(
        record_id=record_id,
        components=[SpectralComponent(name="modes", frequency=freqs, amplitude=amps, units="linear")],
        representation=representation,
        lineage=lineage or [],
    )


def _terrain_record(rng: np.random.Generator) -> Any:
    _ensure_root()
    from data import load_reference_record
    from mesie.core.records import MultiElementRecord, SpectralComponent

    structural = load_reference_record("structural_fas_reference")
    comp = structural.components[0]
    freq = comp.frequency.copy()
    amp = np.abs(comp.amplitude).astype(float)
    rough = 1.0 + 0.12 * rng.uniform(0.5, 1.5) * np.exp(-((freq - 2.5) ** 2) / 4.0)
    amp = np.maximum(amp * rough, 1e-12)
    return MultiElementRecord(
        record_id="terrain_coupled_fas",
        components=[
            SpectralComponent(
                name="surface_motion",
                frequency=freq,
                amplitude=amp,
                units=comp.units or "m/s",
                metadata={"terrain_roughness": float(rng.uniform(0.3, 0.9))},
            )
        ],
        representation="fas",
        lineage=["structural_fas_reference", "terrain_model"],
    )


class DomainSuiteRunner:
    """Run MESIE reasoning suites per operational domain."""

    def __init__(self, seed: int = 42) -> None:
        _ensure_root()
        from mesie.internal_api import InternalRouter
        from mesie.octopus import OctopusController, OctopusConfig

        self.rng = np.random.default_rng(seed)
        self.router = InternalRouter()
        self.octopus = OctopusController(config=OctopusConfig(movement_steps=3))

    def _reason_and_match(
        self,
        primary: Any,
        catalog: List[Any],
        *,
        domain: str,
    ) -> tuple:
        from mesie import match_records, validate_record

        v = validate_record(primary)
        matches = []
        for cand in catalog[:6]:
            m = match_records(primary, cand)
            matches.append(
                {
                    "candidate_id": m.candidate_id,
                    "score": round(m.composite_score, 4),
                }
            )
        matches.sort(key=lambda x: x["score"], reverse=True)
        intel = self.router.call("intelligence", "reason", {"record": primary})
        mem = self.router.call("intelligence", "memory", {"record": primary})
        return v, matches, intel, mem

    def run_terrain(self) -> SuiteResult:
        t0 = time.perf_counter()
        _ensure_root()
        from data import list_references, load_reference_record

        terrain = _terrain_record(self.rng)
        catalog = [load_reference_record(n) for n in list_references()]
        from mesie import match_records

        v, matches, intel, mem = self._reason_and_match(terrain, catalog, domain="terrain")
        eq = load_reference_record("earthquake_psd_reference")
        eq_score = match_records(terrain, eq).composite_score
        report = self.octopus.run_standard_cycle(terrain, candidate=eq)
        plain = (
            f"Terrain-coupled FAS valid={v.is_valid}. "
            f"Strongest match: {matches[0]['candidate_id']} ({matches[0]['score']}). "
            f"Ground coupling to seismic anchor: {eq_score:.3f}. "
            f"{report.plain_summary}"
        )
        return SuiteResult(
            domain="terrain",
            title="Terrain & structural ground coupling",
            records_used=[terrain.record_id, "earthquake_psd_reference", "structural_fas_reference"],
            reasoning=intel.to_dict() if intel.ok else {},
            matches=matches[:5],
            metrics={
                "validation_level": v.level,
                "seismic_coupling_score": round(eq_score, 4),
                "terrain_roughness": terrain.components[0].metadata.get("terrain_roughness"),
            },
            control_commands=report.control.get("data", {}).get("commands", []),
            logic_rules_fired=report.logic.get("data", {}).get("count", 0),
            plain_conclusion=plain,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def run_robotics(self) -> SuiteResult:
        t0 = time.perf_counter()
        _ensure_root()
        from data import load_benchmark, load_reference_record

        vib = load_reference_record("vibration_monitoring_reference")
        bench = load_benchmark("spectral_classification_benchmark")
        from mesie.io.loaders import load_record

        samples = bench.get("samples", [])[:12]
        candidates = []
        for s in samples:
            raw = {
                "record_id": s.get("sample_id", "bench"),
                "representation": "single",
                "components": [
                    {
                        "name": "ch",
                        "frequency": s["frequencies"],
                        "amplitude": s["amplitudes"],
                    }
                ],
            }
            candidates.append(load_record(raw))

        from mesie.matching.ranking import rank_candidates

        ranked = rank_candidates(vib, candidates, top_k=5)
        fault = next((r for r in ranked if r.score < 0.55), ranked[-1])
        intel = self.router.call("intelligence", "reason", {"record": vib})
        report = self.octopus.run_standard_cycle(vib, candidate=load_record(candidates[0]))
        plain = (
            f"Robotics baseline (pump vibration) ranked {len(ranked)} machine states. "
            f"Likely fault candidate: {fault.candidate_id} (score {fault.score:.3f}). "
            f"Intelligence: {intel.data.get('conclusion', 'n/a')}. "
            f"{report.plain_summary}"
        )
        return SuiteResult(
            domain="robotics",
            title="Robotics & machinery condition monitoring",
            records_used=[vib.record_id] + [r.candidate_id for r in ranked[:3]],
            reasoning=intel.to_dict() if intel.ok else {},
            matches=[{"candidate_id": r.candidate_id, "score": round(r.score, 4)} for r in ranked],
            metrics={
                "machine_type": getattr(vib.metadata, "machine_type", None) if getattr(vib, "metadata", None) else "pump",
                "fault_threshold": 0.55,
                "top_score": round(ranked[0].score, 4),
            },
            control_commands=report.control.get("data", {}).get("commands", []),
            logic_rules_fired=report.logic.get("data", {}).get("count", 0),
            plain_conclusion=plain,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def run_orbital(self) -> SuiteResult:
        t0 = time.perf_counter()
        _ensure_root()
        from scripts.orbital_edge_50d_analysis import analyze as orbital_analyze
        from mesie.edge.hz_ladder import HzLadder
        from mesie.edge.satellite_nodes import ORBITAL_TIERS, SatelliteEdgeNode

        orb_report = orbital_analyze()
        ladder = HzLadder()
        tier4 = ladder.get_tier(4)
        nodes = [SatelliteEdgeNode(node_id=f"node_{t.name}", orbital_tier=t) for t in ORBITAL_TIERS[:4]]
        link_budgets = []
        for n in nodes[:3]:
            loss = n.path_loss_to_ground_dB()
            link_budgets.append(
                {
                    "node": n.node_id,
                    "tier": n.orbital_tier.name,
                    "orbital_hz": round(n.orbital_tier.orbital_frequency_Hz, 8),
                    "max_contact_s": round(n.contact_window_s(), 1),
                    "path_loss_dB": round(loss, 2),
                    "doppler_max_Hz": round(n.doppler_at_max_rate(), 1),
                    "hz_tier4_center_GHz": round(tier4.center_frequency_Hz / 1e9, 3) if tier4 else None,
                }
            )

        from data import load_reference_record

        eq = load_reference_record("earthquake_psd_reference")
        report = self.octopus.run_standard_cycle(eq)
        h = orb_report["history_summary"]
        f = orb_report["forward_model"]
        plain = (
            f"Orbital: {h['orbital_edge_days']} edge days in last 50d; "
            f"{len(f['predicted_orbital_edge_days'])} edge days forecast. "
            f"Alerts on days {f['alert_days_anomaly_elevated'][:8]}. "
            f"LEO link margins computed for {len(link_budgets)} nodes. "
            f"{report.plain_summary}"
        )
        return SuiteResult(
            domain="orbital",
            title="Orbital edge, satellite nodes & seismic coupling",
            records_used=["earthquake_psd_reference", "orbital_50d_synthetic"],
            reasoning=report.memory.get("reason", {}).get("data", {}),
            matches=[],
            metrics={
                "history_edge_days": h["orbital_edge_days"],
                "mean_eq_match_edge": h.get("mean_eq_match_edge"),
                "forecast_edge_count": len(f["predicted_orbital_edge_days"]),
                "alert_days": f["alert_days_anomaly_elevated"],
            },
            control_commands=report.control.get("data", {}).get("commands", []),
            logic_rules_fired=report.logic.get("data", {}).get("count", 0),
            plain_conclusion=plain,
            orbital_data={
                "orbital_50d": {
                    "model": orb_report["model"],
                    "history_summary": h,
                    "forward_model": {
                        k: f[k]
                        for k in (
                            "match_trend_slope_per_day",
                            "predicted_orbital_edge_days",
                            "alert_days_anomaly_elevated",
                        )
                    },
                },
                "satellite_link_budgets": link_budgets,
            },
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def run_power(self) -> SuiteResult:
        t0 = time.perf_counter()
        _ensure_root()
        from data import list_library, load_library
        from mesie.edge.hz_ladder import HzLadder

        sch = load_library("schumann_resonances")
        modes = sch.get("schumann_resonances", {}).get("modes", [])
        sch_rec = _record_from_modes(
            "power_schumann",
            modes,
            lineage=["schumann_resonances"],
        )
        em = load_library("electromagnetic_bands")
        band_modes = []
        for _name, b in list(em.get("radio_bands", {}).items())[:12]:
            f0 = (float(b["frequency_low_Hz"]) + float(b["frequency_high_Hz"])) / 2.0
            band_modes.append({"frequency_Hz": f0, "typical_amplitude_pT": 1.0})

        em_rec = None
        if band_modes:
            em_rec = _record_from_modes("power_em_bands", band_modes, lineage=["electromagnetic_bands"])

        ladder = HzLadder()
        tier_powers = [
            {
                "tier": t.tier_id,
                "name": t.name,
                "center_Hz": t.center_frequency_Hz,
                "max_data_rate_Mbps": round(t.max_data_rate_bps / 1e6, 2),
                "latency_ms": t.typical_latency_ms,
            }
            for t in ladder.tiers[:5]
        ]

        from data import load_reference_record

        catalog = [sch_rec]
        if em_rec:
            catalog.append(em_rec)
        catalog.append(load_reference_record("vibration_monitoring_reference"))
        v, matches, intel, _ = self._reason_and_match(sch_rec, catalog, domain="power")
        report = self.octopus.run_standard_cycle(sch_rec)
        geo = sch.get("earth_geophysical_frequencies", {}).get("frequencies", [])[:5]
        plain = (
            f"Power/eco-Hz: Schumann {len(modes)} modes embedded; "
            f"Hz-ladder tiers 0-4 characterized. "
            f"Intelligence: {intel.data.get('conclusion', 'n/a')}. "
            f"Best spectral neighbor: {matches[0]['candidate_id'] if matches else 'n/a'}. "
            f"{report.plain_summary}"
        )
        return SuiteResult(
            domain="power",
            title="Power, Schumann & electromagnetic ladder",
            records_used=["schumann_resonances", "electromagnetic_bands"],
            reasoning=intel.to_dict() if intel.ok else {},
            matches=matches[:5],
            metrics={"schumann_modes": len(modes), "validation_level": v.level},
            control_commands=report.control.get("data", {}).get("commands", []),
            logic_rules_fired=report.logic.get("data", {}).get("count", 0),
            plain_conclusion=plain,
            power_data={
                "hz_ladder_tiers": tier_powers,
                "schumann_peak_hz": [m["frequency_Hz"] for m in modes[:3]],
                "geophysical_refs": [{"name": g["name"], "Hz": g["frequency_Hz"]} for g in geo],
                "libraries_loaded": list_library(),
            },
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def run_seismic(self) -> SuiteResult:
        t0 = time.perf_counter()
        _ensure_root()
        from data import list_references, load_reference_record
        from mesie import match_records

        names = list_references()
        records = {n: load_reference_record(n) for n in names}
        matrix = []
        keys = list(records.keys())
        for a in keys:
            for b in keys:
                if a >= b:
                    continue
                s = match_records(records[a], records[b]).composite_score
                matrix.append({"a": a, "b": b, "score": round(s, 4)})
        matrix.sort(key=lambda x: x["score"], reverse=True)
        eq = records["earthquake_psd_reference"]
        intel = self.router.call("intelligence", "reason", {"record": eq})
        report = self.octopus.run_standard_cycle(eq)
        plain = (
            f"Seismic suite: {len(keys)} references cross-matched. "
            f"Strongest pair score {matrix[0]['score'] if matrix else 0} ({matrix[0]['a']} vs {matrix[0]['b']}). "
            f"{report.plain_summary}"
        )
        return SuiteResult(
            domain="seismic",
            title="Seismic & structural reference cross-analysis",
            records_used=keys,
            reasoning=intel.to_dict() if intel.ok else {},
            matches=matrix[:6],
            metrics={"pair_count": len(matrix), "reference_count": len(keys)},
            control_commands=report.control.get("data", {}).get("commands", []),
            logic_rules_fired=report.logic.get("data", {}).get("count", 0),
            plain_conclusion=plain,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    def run_all(self) -> Dict[str, Any]:
        suites = [
            self.run_terrain(),
            self.run_robotics(),
            self.run_orbital(),
            self.run_power(),
            self.run_seismic(),
        ]
        return {
            "engine": "MESIE",
            "version": "0.2.1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suite_count": len(suites),
            "total_elapsed_ms": round(sum(s.elapsed_ms for s in suites), 1),
            "suites": [asdict(s) for s in suites],
            "executive_summary": [s.plain_conclusion for s in suites],
        }


def run_all_suites(seed: int = 42) -> Dict[str, Any]:
    return DomainSuiteRunner(seed=seed).run_all()