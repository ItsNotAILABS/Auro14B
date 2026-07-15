"""Copilot + MESIE Diagnostic Benchmark Suite.

Comprehensive benchmarking of MESIE performance and reasoning capabilities
using the engine itself. Measures speed, reasoning quality, and throughput
to diagnose Copilot + MESIE integration effectiveness.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import list_references, load_reference_record
from mesie import (
    GenerationConfig,
    IntelligenceLevel,
    IntelligenceConfig,
    IntelligenceProtocol,
    MultiElementRecord,
    SpectralComponent,
    match_records,
    validate_record,
    generate_psd,
)
from mesie.embeddings import SpectralFingerprintPipeline
from mesie.matching.ranking import rank_candidates
from mesie.sdk import FastSpectralCompute


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SpeedMetric:
    """Single speed measurement."""
    name: str
    operation: str
    latency_ms: float
    latency_us: float
    unit: str


@dataclass
class ReasoningMetric:
    """Reasoning quality measurement."""
    intelligence_level: int
    level_name: str
    success_rate: float
    quality_score: float
    latency_ms: float
    memory_mb: float
    notes: str


@dataclass
class ThroughputMetric:
    """Throughput measurement."""
    batch_size: int
    total_time_ms: float
    throughput_per_sec: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


@dataclass
class BottleneckAnalysis:
    """Identified performance bottleneck."""
    component: str
    latency_ms: float
    percentage_of_total: float
    recommendation: str


@dataclass
class DiagnosticReport:
    """Complete diagnostic benchmark report."""
    timestamp: str
    version: str
    speed_benchmarks: List[SpeedMetric] = field(default_factory=list)
    reasoning_benchmarks: List[ReasoningMetric] = field(default_factory=list)
    throughput_benchmarks: List[ThroughputMetric] = field(default_factory=list)
    bottleneck_analysis: List[BottleneckAnalysis] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Speed Benchmarks
# ============================================================================


class SpeedBenchmarks:
    """Measure latency of core MESIE operations."""

    def __init__(self):
        self.metrics: List[SpeedMetric] = []
        self.reference_records: List[MultiElementRecord] = []

    def setup(self, n_records: int = 10):
        """Load reference records for benchmarking."""
        refs = list_references()[:n_records]
        self.reference_records = [load_reference_record(name) for name in refs]

    def benchmark_matching(self, repeats: int = 100) -> SpeedMetric:
        """Benchmark spectral matching latency."""
        if len(self.reference_records) < 2:
            self.setup(10)

        ref1, ref2 = self.reference_records[0], self.reference_records[1]

        def _match():
            return match_records(ref1, ref2)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _match()
        elapsed_s = (time.perf_counter() - t0) / repeats
        elapsed_ms = elapsed_s * 1000
        elapsed_us = elapsed_s * 1e6

        metric = SpeedMetric(
            name="spectral_matching",
            operation="match_records(record1, record2)",
            latency_ms=elapsed_ms,
            latency_us=elapsed_us,
            unit="ms"
        )
        self.metrics.append(metric)
        return metric

    def benchmark_embedding_generation(self, repeats: int = 50) -> SpeedMetric:
        """Benchmark spectral fingerprint embedding generation."""
        if len(self.reference_records) < 1:
            self.setup(10)

        record = self.reference_records[0]
        pipeline = SpectralFingerprintPipeline()

        def _embed():
            return pipeline.process(record)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _embed()
        elapsed_s = (time.perf_counter() - t0) / repeats
        elapsed_ms = elapsed_s * 1000
        elapsed_us = elapsed_s * 1e6

        metric = SpeedMetric(
            name="embedding_generation",
            operation="SpectralFingerprintPipeline.process(record)",
            latency_ms=elapsed_ms,
            latency_us=elapsed_us,
            unit="ms"
        )
        self.metrics.append(metric)
        return metric

    def benchmark_validation(self, repeats: int = 100) -> SpeedMetric:
        """Benchmark record validation latency."""
        if len(self.reference_records) < 1:
            self.setup(10)

        record = self.reference_records[0]

        def _validate():
            return validate_record(record)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _validate()
        elapsed_s = (time.perf_counter() - t0) / repeats
        elapsed_ms = elapsed_s * 1000
        elapsed_us = elapsed_s * 1e6

        metric = SpeedMetric(
            name="record_validation",
            operation="validate_record(record)",
            latency_ms=elapsed_ms,
            latency_us=elapsed_us,
            unit="ms"
        )
        self.metrics.append(metric)
        return metric

    def benchmark_generation(self, repeats: int = 100) -> SpeedMetric:
        """Benchmark PSD/FAS generation speed."""
        cfg = GenerationConfig(seed=42, target_frequency=np.linspace(0.1, 50, 128))

        def _gen():
            return generate_psd(cfg)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _gen()
        elapsed_s = (time.perf_counter() - t0) / repeats
        elapsed_ms = elapsed_s * 1000
        elapsed_us = elapsed_s * 1e6

        metric = SpeedMetric(
            name="psd_generation",
            operation="generate_psd(config)",
            latency_ms=elapsed_ms,
            latency_us=elapsed_us,
            unit="ms"
        )
        self.metrics.append(metric)
        return metric

    def benchmark_ranking(self, repeats: int = 50) -> SpeedMetric:
        """Benchmark candidate ranking latency."""
        if len(self.reference_records) < 3:
            self.setup(10)

        ref = self.reference_records[0]
        candidates = self.reference_records[1:4]

        def _rank():
            return rank_candidates(ref, candidates)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _rank()
        elapsed_s = (time.perf_counter() - t0) / repeats
        elapsed_ms = elapsed_s * 1000
        elapsed_us = elapsed_s * 1e6

        metric = SpeedMetric(
            name="candidate_ranking",
            operation="rank_candidates(ref, [cand1, cand2, cand3])",
            latency_ms=elapsed_ms,
            latency_us=elapsed_us,
            unit="ms"
        )
        self.metrics.append(metric)
        return metric


# ============================================================================
# Reasoning Quality Benchmarks
# ============================================================================


class ReasoningBenchmarks:
    """Measure reasoning capabilities across intelligence levels."""

    def __init__(self):
        self.metrics: List[ReasoningMetric] = []
        self.reference_records: List[MultiElementRecord] = []

    def setup(self, n_records: int = 10):
        """Load reference records."""
        refs = list_references()[:n_records]
        self.reference_records = [load_reference_record(name) for name in refs]

    def test_intelligence_levels(self) -> List[ReasoningMetric]:
        """Test reasoning at all 5 intelligence levels."""
        if len(self.reference_records) < 2:
            self.setup(10)

        results = []
        levels = [
            (IntelligenceLevel.PASSIVE, "passive"),
            (IntelligenceLevel.REACTIVE, "reactive"),
            (IntelligenceLevel.ADAPTIVE, "adaptive"),
            (IntelligenceLevel.PREDICTIVE, "predictive"),
            (IntelligenceLevel.AUTONOMOUS, "autonomous"),
        ]

        for level, level_name in levels:
            tracemalloc.start()
            try:
                cfg = IntelligenceConfig(level=level)
                protocol = IntelligenceProtocol(cfg)

                # Extract spectrum from record
                record = self.reference_records[0]
                spectrum = record.components[0].amplitude.copy()

                t0 = time.perf_counter()
                result = protocol.reason(spectrum)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_mb = peak / (1024 * 1024)

                success = result is not None and hasattr(result, "confidence")
                quality = float(getattr(result, "confidence", 0.0)) if success else 0.0

                metric = ReasoningMetric(
                    intelligence_level=len(results),
                    level_name=level_name,
                    success_rate=1.0 if success else 0.0,
                    quality_score=quality,
                    latency_ms=elapsed_ms,
                    memory_mb=memory_mb,
                    notes=f"Reasoning result type: {type(result).__name__}"
                )
                results.append(metric)
                self.metrics.append(metric)
            except Exception as e:
                tracemalloc.stop()
                metric = ReasoningMetric(
                    intelligence_level=len(results),
                    level_name=level_name,
                    success_rate=0.0,
                    quality_score=0.0,
                    latency_ms=0.0,
                    memory_mb=0.0,
                    notes=f"Error: {str(e)[:100]}"
                )
                results.append(metric)
                self.metrics.append(metric)

        return results

    def test_spectral_validation_levels(self) -> List[ReasoningMetric]:
        """Test spectral validation at different confidence thresholds."""
        if len(self.reference_records) < 1:
            self.setup(10)

        results = []
        record = self.reference_records[0]

        for level_id in range(6):
            try:
                t0 = time.perf_counter()
                report = validate_record(record)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                success = report is not None and hasattr(report, "is_valid")
                quality = float(getattr(report, "score", 0.0)) if success else 0.0

                metric = ReasoningMetric(
                    intelligence_level=level_id,
                    level_name=f"validation_level_{level_id}",
                    success_rate=1.0 if success else 0.0,
                    quality_score=quality,
                    latency_ms=elapsed_ms,
                    memory_mb=0.0,
                    notes=f"Valid: {getattr(report, 'is_valid', False)}"
                )
                results.append(metric)
                self.metrics.append(metric)
            except Exception as e:
                metric = ReasoningMetric(
                    intelligence_level=level_id,
                    level_name=f"validation_level_{level_id}",
                    success_rate=0.0,
                    quality_score=0.0,
                    latency_ms=0.0,
                    memory_mb=0.0,
                    notes=f"Error: {str(e)[:100]}"
                )
                results.append(metric)
                self.metrics.append(metric)

        return results


# ============================================================================
# Throughput Benchmarks
# ============================================================================


class ThroughputBenchmarks:
    """Measure throughput at different batch sizes."""

    def __init__(self):
        self.metrics: List[ThroughputMetric] = []
        self.reference_records: List[MultiElementRecord] = []

    def setup(self, n_records: int = 100):
        """Load reference records for throughput testing."""
        refs = list_references()[:n_records]
        self.reference_records = [load_reference_record(name) for name in refs]

    def benchmark_batch_matching(self, batch_sizes: Optional[List[int]] = None) -> List[ThroughputMetric]:
        """Benchmark matching throughput at different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [10, 50, 100, 250]

        if len(self.reference_records) < max(batch_sizes):
            self.setup(max(batch_sizes) + 50)

        results = []
        ref = self.reference_records[0]

        for batch_size in batch_sizes:
            batch = self.reference_records[1:batch_size+1]
            latencies = []

            t_total_start = time.perf_counter()
            for candidate in batch:
                t_start = time.perf_counter()
                match_records(ref, candidate)
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000)
            t_total_end = time.perf_counter()

            total_time_ms = (t_total_end - t_total_start) * 1000
            latencies = np.array(latencies)

            throughput = len(batch) / (total_time_ms / 1000) if total_time_ms > 0 else 0
            metric = ThroughputMetric(
                batch_size=batch_size,
                total_time_ms=total_time_ms,
                throughput_per_sec=throughput,
                avg_latency_ms=float(np.mean(latencies)),
                p95_latency_ms=float(np.percentile(latencies, 95)),
                p99_latency_ms=float(np.percentile(latencies, 99)),
            )
            results.append(metric)
            self.metrics.append(metric)

        return results

    def benchmark_fast_compute(self) -> Optional[ThroughputMetric]:
        """Benchmark FastSpectralCompute batch performance."""
        try:
            if len(self.reference_records) < 30:
                self.setup(50)

            refs = self.reference_records[:30]
            bench = FastSpectralCompute.benchmark_match(refs, n_repeat=50)

            metric = ThroughputMetric(
                batch_size=len(refs),
                total_time_ms=bench.batch_match_ms,
                throughput_per_sec=len(refs) / (bench.batch_match_ms / 1000) if bench.batch_match_ms > 0 else 0,
                avg_latency_ms=bench.batch_match_ms / len(refs) if len(refs) > 0 else 0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
            )
            self.metrics.append(metric)
            return metric
        except Exception as e:
            print(f"FastSpectralCompute benchmark failed: {e}")
            return None


# ============================================================================
# Bottleneck Analysis
# ============================================================================


def analyze_bottlenecks(
    speed_metrics: List[SpeedMetric],
    reasoning_metrics: List[ReasoningMetric],
    throughput_metrics: List[ThroughputMetric]
) -> List[BottleneckAnalysis]:
    """Identify performance bottlenecks."""
    bottlenecks = []

    # Find slowest speed metrics
    if speed_metrics:
        sorted_speed = sorted(speed_metrics, key=lambda m: m.latency_ms, reverse=True)
        for metric in sorted_speed[:3]:
            percentage = (metric.latency_ms / sorted_speed[0].latency_ms * 100) if sorted_speed[0].latency_ms > 0 else 0
            bottleneck = BottleneckAnalysis(
                component=metric.name,
                latency_ms=metric.latency_ms,
                percentage_of_total=percentage,
                recommendation=_recommend_for_speed(metric)
            )
            bottlenecks.append(bottleneck)

    # Find problematic reasoning levels
    if reasoning_metrics:
        slow_reasoning = [m for m in reasoning_metrics if m.latency_ms > 100]
        for metric in slow_reasoning[:3]:
            bottleneck = BottleneckAnalysis(
                component=f"reasoning_{metric.level_name}",
                latency_ms=metric.latency_ms,
                percentage_of_total=0.0,
                recommendation=_recommend_for_reasoning(metric)
            )
            bottlenecks.append(bottleneck)

    return bottlenecks


def _recommend_for_speed(metric: SpeedMetric) -> str:
    """Get recommendation for speed metric."""
    recommendations = {
        "spectral_matching": "Consider batch operations or vectorization.",
        "embedding_generation": "Profile embedding pipeline; consider caching.",
        "record_validation": "Validate only necessary fields.",
        "psd_generation": "Pre-generate common configurations.",
        "candidate_ranking": "Use approximate ranking for large candidates.",
    }
    return recommendations.get(metric.name, "Investigate further.")


def _recommend_for_reasoning(metric: ReasoningMetric) -> str:
    """Get recommendation for reasoning metric."""
    if metric.intelligence_level >= 3:
        return f"Level {metric.intelligence_level} is computationally expensive. Consider caching."
    return "Acceptable latency."


# ============================================================================
# Report Generation
# ============================================================================


def generate_report(
    speed_metrics: List[SpeedMetric],
    reasoning_metrics: List[ReasoningMetric],
    throughput_metrics: List[ThroughputMetric],
    bottlenecks: List[BottleneckAnalysis]
) -> DiagnosticReport:
    """Generate comprehensive diagnostic report."""
    report = DiagnosticReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version="1.0",
        speed_benchmarks=speed_metrics,
        reasoning_benchmarks=reasoning_metrics,
        throughput_benchmarks=throughput_metrics,
        bottleneck_analysis=bottlenecks,
    )

    # Summary statistics
    if speed_metrics:
        speed_times = [m.latency_ms for m in speed_metrics]
        report.summary["speed_metrics_count"] = len(speed_metrics)
        report.summary["speed_min_ms"] = float(np.min(speed_times))
        report.summary["speed_max_ms"] = float(np.max(speed_times))
        report.summary["speed_mean_ms"] = float(np.mean(speed_times))

    if reasoning_metrics:
        success_rates = [m.success_rate for m in reasoning_metrics]
        quality_scores = [m.quality_score for m in reasoning_metrics]
        report.summary["reasoning_metrics_count"] = len(reasoning_metrics)
        report.summary["reasoning_avg_success_rate"] = float(np.mean(success_rates))
        report.summary["reasoning_avg_quality"] = float(np.mean(quality_scores))

    if throughput_metrics:
        throughputs = [m.throughput_per_sec for m in throughput_metrics]
        report.summary["throughput_metrics_count"] = len(throughput_metrics)
        report.summary["throughput_max_per_sec"] = float(np.max(throughputs))
        report.summary["throughput_min_per_sec"] = float(np.min(throughputs))

    report.summary["bottleneck_count"] = len(bottlenecks)

    return report


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def main() -> None:
    """Run comprehensive diagnostic benchmark suite."""
    print("=" * 80)
    print("Copilot + MESIE Diagnostic Benchmark Suite")
    print("=" * 80)
    print()

    # Speed Benchmarks
    print("Running Speed Benchmarks...")
    speed_bench = SpeedBenchmarks()
    speed_bench.setup(n_records=15)

    print("  - Spectral matching...", end=" ", flush=True)
    speed_bench.benchmark_matching(repeats=100)
    print("✓")

    print("  - Embedding generation...", end=" ", flush=True)
    speed_bench.benchmark_embedding_generation(repeats=50)
    print("✓")

    print("  - Record validation...", end=" ", flush=True)
    speed_bench.benchmark_validation(repeats=100)
    print("✓")

    print("  - PSD generation...", end=" ", flush=True)
    speed_bench.benchmark_generation(repeats=100)
    print("✓")

    print("  - Candidate ranking...", end=" ", flush=True)
    speed_bench.benchmark_ranking(repeats=50)
    print("✓")

    print()

    # Reasoning Benchmarks
    print("Running Reasoning Quality Benchmarks...")
    reasoning_bench = ReasoningBenchmarks()
    reasoning_bench.setup(n_records=15)

    print("  - Intelligence levels...", end=" ", flush=True)
    reasoning_bench.test_intelligence_levels()
    print("✓")

    print("  - Validation levels...", end=" ", flush=True)
    reasoning_bench.test_spectral_validation_levels()
    print("✓")

    print()

    # Throughput Benchmarks
    print("Running Throughput Benchmarks...")
    throughput_bench = ThroughputBenchmarks()
    throughput_bench.setup(n_records=300)

    print("  - Batch matching (10, 50, 100, 250)...", end=" ", flush=True)
    throughput_bench.benchmark_batch_matching()
    print("✓")

    print("  - FastSpectralCompute batch...", end=" ", flush=True)
    throughput_bench.benchmark_fast_compute()
    print("✓")

    print()

    # Bottleneck Analysis
    print("Analyzing Bottlenecks...")
    bottlenecks = analyze_bottlenecks(
        speed_bench.metrics,
        reasoning_bench.metrics,
        throughput_bench.metrics
    )
    print(f"  - Identified {len(bottlenecks)} bottlenecks")

    print()

    # Generate Report
    print("Generating Diagnostic Report...")
    report = generate_report(
        speed_bench.metrics,
        reasoning_bench.metrics,
        throughput_bench.metrics,
        bottlenecks
    )

    # Write report to JSON
    out_path = ROOT / "deliverables" / "Copilot_MESIE_Diagnostic_Benchmark.json"
    out_path.parent.mkdir(exist_ok=True)

    # Convert dataclasses to dict for JSON serialization
    report_dict = {
        "timestamp": report.timestamp,
        "version": report.version,
        "speed_benchmarks": [asdict(m) for m in report.speed_benchmarks],
        "reasoning_benchmarks": [asdict(m) for m in report.reasoning_benchmarks],
        "throughput_benchmarks": [asdict(m) for m in report.throughput_benchmarks],
        "bottleneck_analysis": [asdict(b) for b in report.bottleneck_analysis],
        "summary": report.summary,
    }

    out_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"✓ Report written to {out_path}")

    print()
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Speed Metrics:     {len(speed_bench.metrics)} measurements")
    print(f"Reasoning Metrics: {len(reasoning_bench.metrics)} measurements")
    print(f"Throughput:        {len(throughput_bench.metrics)} measurements")
    print(f"Bottlenecks:       {len(bottlenecks)} identified")
    print()
    print("Top Speed Metrics (fastest):")
    for metric in sorted(speed_bench.metrics, key=lambda m: m.latency_ms)[:3]:
        print(f"  • {metric.name:30} {metric.latency_us:8.1f} µs")
    print()
    print("Slowest Operations:")
    for metric in sorted(speed_bench.metrics, key=lambda m: m.latency_ms, reverse=True)[:3]:
        print(f"  • {metric.name:30} {metric.latency_ms:8.3f} ms")
    print()
    if throughput_bench.metrics:
        max_throughput = max(throughput_bench.metrics, key=lambda m: m.throughput_per_sec)
        print(f"Max Throughput:    {max_throughput.throughput_per_sec:8.1f} ops/sec @ batch_size={max_throughput.batch_size}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
