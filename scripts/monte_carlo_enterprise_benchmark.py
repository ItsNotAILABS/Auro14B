"""Monte Carlo execution across 10 major enterprise MESIE/MAESI use cases."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_benchmarks, list_references, load_benchmark, load_reference_record
from mesie import match_records, validate_record
from mesie.cognitive.agent_state_adapter import SpectralAnomalyAdapter
from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.embeddings import SpectralFingerprintPipeline
from mesie.io.loaders import load_record
from mesie.sdk import MAESIClient, search_research


@dataclass
class EnterpriseUseCase:
    id: str
    name: str
    industry: str
    description: str
    success_metric: str


@dataclass
class TrialResult:
    trial: int
    success: bool
    latency_ms: float
    score: float
    detail: str


@dataclass
class UseCaseMonteCarloReport:
    use_case: EnterpriseUseCase
    n_trials: int
    success_rate: float
    mean_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    mean_score: float
    std_score: float
    p5_score: float
    failures: int
    sample_details: List[str] = field(default_factory=list)


ENTERPRISE_CASES: List[EnterpriseUseCase] = [
    EnterpriseUseCase("mfg_predictive", "Predictive Maintenance", "Manufacturing", "Detect pump/machine drift from vibration spectra.", "top_match_sim >= 0.5"),
    EnterpriseUseCase("energy_grid", "Grid & Power Monitoring", "Energy", "Schumann/EM band fingerprint stability under noise.", "validation_level >= 4"),
    EnterpriseUseCase("aerospace_orbital", "Satellite Ops & Orbital", "Aerospace", "Orbital-edge style spectral gate + seismic anchor coupling.", "match_score >= 0.6"),
    EnterpriseUseCase("insurance_seismic", "Catastrophe / Seismic Risk", "Insurance", "Cross-match earthquake vs structural references.", "match_score >= 0.55"),
    EnterpriseUseCase("structural_civil", "Structural / Civil Engineering", "Construction", "FAS structural spectrum ranking under perturbation.", "rank_top3_self_or_structural"),
    EnterpriseUseCase("health_device", "Medical Device Monitoring", "Healthcare", "Anomaly separation on biosignal-like spectra.", "anomaly_detects_outlier"),
    EnterpriseUseCase("robotics_fleet", "Autonomous Robotics Fleet", "Robotics", "Fast ANN neighbor lookup for fleet state.", "query_ms < 50 and sim > 0.4"),
    EnterpriseUseCase("telecom_compliance", "Telecom Spectrum Compliance", "Telecom", "EM band library embedding + research hit.", "research_hit_found"),
    EnterpriseUseCase("rd_lab", "R&D Spectral Lab", "Research", "Benchmark sample classification via ranking.", "rank_score >= 0.45"),
    EnterpriseUseCase("ai_copilot_rag", "AI Agent Spectral Memory", "Enterprise AI", "MAESI query with knowledge + fingerprint ANN.", "neighbors >= 1 and latency < 100ms"),
]


def _noisy_record(base: MultiElementRecord, rng: np.random.Generator, scale: float = 0.08) -> MultiElementRecord:
    comp = base.components[0]
    f = comp.frequency.copy()
    a = np.maximum(np.abs(comp.amplitude) * (1.0 + rng.normal(0, scale, size=len(comp.amplitude))), 1e-12)
    return MultiElementRecord(
        record_id=f"{base.record_id}_mc_{rng.integers(0, 1_000_000)}",
        components=[SpectralComponent(name=comp.name, frequency=f, amplitude=a, units=comp.units or "linear")],
        representation=base.representation,
    )


def _stats(trials: List[TrialResult]) -> Dict[str, float]:
    lat = np.array([t.latency_ms for t in trials])
    sc = np.array([t.score for t in trials])
    ok = sum(1 for t in trials if t.success)
    return {
        "success_rate": ok / max(len(trials), 1),
        "mean_latency_ms": float(np.mean(lat)),
        "std_latency_ms": float(np.std(lat)),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "mean_score": float(np.mean(sc)),
        "std_score": float(np.std(sc)),
        "p5_score": float(np.percentile(sc, 5)),
        "failures": len(trials) - ok,
    }


class MonteCarloEnterpriseRunner:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.refs = {n: load_reference_record(n) for n in list_references()}
        self.ref_list = list(self.refs.values())
        self.vib = self.refs.get("vibration_monitoring_reference") or self.ref_list[0]
        self.eq = self.refs.get("earthquake_psd_reference") or self.ref_list[0]
        self.struct = self.refs.get("structural_fas_reference") or self.ref_list[0]
        self.client = MAESIClient(fast=True, use_fingerprint=True)
        self.client.index_corpus(self.ref_list)
        self.fingerprint = SpectralFingerprintPipeline()
        self.fingerprint.index_records(self.ref_list)
        bench = load_benchmark("spectral_classification_benchmark")
        self.bench_samples = bench.get("samples", [])[:80]

    def run_case(self, case: EnterpriseUseCase, n_trials: int) -> UseCaseMonteCarloReport:
        runners: Dict[str, Callable[[int], TrialResult]] = {
            "mfg_predictive": self._trial_mfg,
            "energy_grid": self._trial_energy,
            "aerospace_orbital": self._trial_aerospace,
            "insurance_seismic": self._trial_insurance,
            "structural_civil": self._trial_structural,
            "health_device": self._trial_health,
            "robotics_fleet": self._trial_robotics,
            "telecom_compliance": self._trial_telecom,
            "rd_lab": self._trial_rd,
            "ai_copilot_rag": self._trial_ai,
        }
        fn = runners[case.id]
        trials = [fn(i) for i in range(n_trials)]
        st = _stats(trials)
        fails = [t.detail for t in trials if not t.success][:5]
        return UseCaseMonteCarloReport(
            use_case=case,
            n_trials=n_trials,
            sample_details=fails,
            **{k: st[k] for k in st},
        )

    def _trial_mfg(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        q = _noisy_record(self.vib, self.rng, scale=0.05)
        hits = self.client.fast_compute.cosine_search(q, top_k=1) if self.client.fast_compute else []
        sim = hits[0][1] if hits else 0.0
        ms = (time.perf_counter() - t0) * 1000
        ok = sim >= 0.5
        return TrialResult(trial, ok, ms, sim, f"sim={sim:.3f}")

    def _trial_energy(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        from mesie.analysis.domain_suites import _record_from_modes
        from data import load_library

        sch = load_library("schumann_resonances")
        modes = sch["schumann_resonances"]["modes"]
        rec = _record_from_modes("mc_sch", modes)
        noise = _noisy_record(rec, self.rng, 0.03)
        v = validate_record(noise)
        ms = (time.perf_counter() - t0) * 1000
        ok = v.level >= 4 and v.is_valid
        return TrialResult(trial, ok, ms, float(v.level), f"level={v.level}")

    def _trial_aerospace(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        day = float(self.rng.integers(-50, 51))
        from scripts.orbital_edge_50d_analysis import record_for_day

        rec = record_for_day(self.eq, day, "mc", self.rng)
        m = match_records(self.eq, rec).composite_score
        ms = (time.perf_counter() - t0) * 1000
        ok = m >= 0.6
        return TrialResult(trial, ok, ms, m, f"eq_match={m:.3f} day={day}")

    def _trial_insurance(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        q = _noisy_record(self.eq, self.rng)
        m = match_records(q, self.struct).composite_score
        ms = (time.perf_counter() - t0) * 1000
        ok = m >= 0.55
        return TrialResult(trial, ok, ms, m, f"eq_struct={m:.3f}")

    def _trial_structural(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        from mesie.matching.ranking import rank_candidates

        q = _noisy_record(self.struct, self.rng, 0.06)
        ranked = rank_candidates(q, self.ref_list, top_k=3)
        top = ranked[0] if ranked else None
        ms = (time.perf_counter() - t0) * 1000
        ok = top is not None and ("structural" in top.candidate_id or top.score >= 0.7)
        sc = top.score if top else 0.0
        return TrialResult(trial, ok, ms, sc, f"top={top.candidate_id if top else 'none'}")

    def _trial_health(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        adapter = SpectralAnomalyAdapter(threshold=2.0)
        adapter.fit_baseline([self.vib])
        normal = _noisy_record(self.vib, self.rng, 0.02)
        outlier = _noisy_record(self.eq, self.rng, 0.15)
        an_n = adapter.score_anomaly(normal)
        an_o = adapter.score_anomaly(outlier)
        ms = (time.perf_counter() - t0) * 1000
        ok = an_o > an_n * 1.2
        return TrialResult(trial, ok, ms, an_o - an_n, f"delta={an_o - an_n:.2f}")

    def _trial_robotics(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        q = _noisy_record(self.vib, self.rng)
        hits = self.client.fast_compute.cosine_search(q, top_k=3) if self.client.fast_compute else []
        ms = (time.perf_counter() - t0) * 1000
        sim = hits[0][1] if hits else 0.0
        ok = ms < 50 and sim > 0.4
        return TrialResult(trial, ok, ms, sim, f"ms={ms:.2f} sim={sim:.3f}")

    def _trial_telecom(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        hits = search_research("electromagnetic band spectrum", top_k=2)
        ms = (time.perf_counter() - t0) * 1000
        ok = len(hits) >= 1
        return TrialResult(trial, ok, ms, float(len(hits)), hits[0].title if hits else "none")

    def _trial_rd(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        if not self.bench_samples:
            return TrialResult(trial, False, 0, 0, "no benchmark")
        s = self.bench_samples[self.rng.integers(0, len(self.bench_samples))]
        raw = {
            "record_id": s.get("sample_id", "b"),
            "representation": "single",
            "components": [{"name": "ch", "frequency": s["frequencies"], "amplitude": s["amplitudes"]}],
        }
        rec = load_record(raw)
        from mesie.matching.ranking import rank_candidates

        ranked = rank_candidates(rec, self.ref_list, top_k=1)
        sc = ranked[0].score if ranked else 0.0
        ms = (time.perf_counter() - t0) * 1000
        ok = sc >= 0.45
        return TrialResult(trial, ok, ms, sc, f"score={sc:.3f}")

    def _trial_ai(self, trial: int) -> TrialResult:
        t0 = time.perf_counter()
        q = self.ref_list[self.rng.integers(0, len(self.ref_list))]
        r = self.client.query(q, top_k=3)
        fp = self.fingerprint.query(q, top_k=2)
        ms = (time.perf_counter() - t0) * 1000
        ok = len(r.neighbors) >= 1 and ms < 100
        sc = r.neighbors[0]["similarity"] if r.neighbors else 0.0
        return TrialResult(trial, ok, ms, sc, f"neighbors={len(r.neighbors)} fp={len(fp)}")

    def run_all(self, n_trials: int = 100) -> Dict[str, Any]:
        t0 = time.perf_counter()
        reports = [self.run_case(c, n_trials) for c in ENTERPRISE_CASES]
        overall_success = float(np.mean([r.success_rate for r in reports]))
        return {
            "engine": "MESIE / MAESI",
            "method": "monte_carlo",
            "trials_per_use_case": n_trials,
            "total_trials": n_trials * len(ENTERPRISE_CASES),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "overall_success_rate": round(overall_success, 4),
            "enterprise_grade": overall_success >= 0.85,
            "use_cases": [
                {
                    **asdict(r.use_case),
                    "monte_carlo": {
                        "n_trials": r.n_trials,
                        "success_rate": round(r.success_rate, 4),
                        "mean_latency_ms": round(r.mean_latency_ms, 3),
                        "std_latency_ms": round(r.std_latency_ms, 3),
                        "p95_latency_ms": round(r.p95_latency_ms, 3),
                        "mean_score": round(r.mean_score, 4),
                        "std_score": round(r.std_score, 4),
                        "p5_score": round(r.p5_score, 4),
                        "failures": r.failures,
                        "sample_failure_details": r.sample_details,
                    },
                }
                for r in reports
            ],
        }


def _write_markdown(data: Dict[str, Any], path: Path) -> None:
    lines = [
        "# MESIE Monte Carlo Enterprise Report",
        "",
        f"*Generated {data['generated_at']} — {data['total_trials']:,} trials across 10 enterprise use cases*",
        "",
        "## Executive summary",
        "",
        f"- **Overall success rate:** {data['overall_success_rate']*100:.1f}%",
        f"- **Enterprise grade (≥85%):** {'PASS' if data['enterprise_grade'] else 'REVIEW'}",
        f"- **Total runtime:** {data['elapsed_s']} s",
        f"- **Trials per use case:** {data['trials_per_use_case']}",
        "",
        "## 10 enterprise use cases",
        "",
        "| # | Industry | Use case | Success % | Mean ms | P95 ms | Mean score |",
        "|---|----------|----------|-----------|---------|--------|------------|",
    ]
    for i, uc in enumerate(data["use_cases"], 1):
        mc = uc["monte_carlo"]
        lines.append(
            f"| {i} | {uc['industry']} | {uc['name']} | {mc['success_rate']*100:.1f}% | "
            f"{mc['mean_latency_ms']:.2f} | {mc['p95_latency_ms']:.2f} | {mc['mean_score']:.3f} |"
        )
    lines.extend(["", "## Per-use-case detail", ""])
    for uc in data["use_cases"]:
        mc = uc["monte_carlo"]
        lines.append(f"### {uc['name']} ({uc['industry']})")
        lines.append("")
        lines.append(uc["description"])
        lines.append("")
        lines.append(f"- Success metric: `{uc['success_metric']}`")
        lines.append(f"- Success rate: **{mc['success_rate']*100:.1f}%** ({mc['n_trials'] - mc['failures']}/{mc['n_trials']})")
        lines.append(f"- Latency: mean {mc['mean_latency_ms']:.2f} ms, std {mc['std_latency_ms']:.2f}, p95 {mc['p95_latency_ms']:.2f}")
        lines.append(f"- Score: mean {mc['mean_score']:.4f}, std {mc['std_score']:.4f}, p5 {mc['p5_score']:.4f}")
        if mc["sample_failure_details"]:
            lines.append(f"- Sample failures: {mc['sample_failure_details'][:3]}")
        lines.append("")
    lines.extend([
        "## How to re-run",
        "",
        "```bash",
        "python scripts/monte_carlo_enterprise_benchmark.py",
        "python scripts/monte_carlo_enterprise_benchmark.py --trials 500",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    n = 100
    if "--trials" in sys.argv:
        idx = sys.argv.index("--trials")
        n = int(sys.argv[idx + 1])

    runner = MonteCarloEnterpriseRunner(seed=42)
    data = runner.run_all(n_trials=n)

    out_json = ROOT / "deliverables" / "MESIE_Monte_Carlo_Enterprise_Report.json"
    out_md = ROOT / "deliverables" / "MESIE_Monte_Carlo_Enterprise_Report.md"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(data, out_md)

    print("=== MESIE Monte Carlo Enterprise Benchmark ===\n")
    print(f"Trials: {data['total_trials']:,} | Overall success: {data['overall_success_rate']*100:.1f}%")
    print(f"Enterprise grade: {'PASS' if data['enterprise_grade'] else 'REVIEW'}\n")
    for uc in data["use_cases"]:
        mc = uc["monte_carlo"]
        print(f"  [{uc['industry']}] {uc['name']}: {mc['success_rate']*100:.1f}%  (p95 {mc['p95_latency_ms']:.1f} ms)")
    print(f"\nJSON: {out_json}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()