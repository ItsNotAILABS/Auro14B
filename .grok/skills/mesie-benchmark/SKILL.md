---
name: mesie-benchmark
description: >
  Comprehensive benchmarking suite for MESIE architecture including DRACO-equivalent diagnostic evaluation, determinism, and performance. Triggers: benchmark, determinism, speed, draco, diagnostic. Use for /mesie-benchmark or MESIE/MAESI/NeuroAIX evaluation tasks.
---

# mesie-benchmark

Native MESIE / MAESI / NeuroAIX skill — **Quality Assurance, Diagnostics & Benchmarks**.

## When to use

- Determinism and match/embed throughput
- DRACO-equivalent diagnostic evaluation of architecture
- Category-wise performance breakdown
- Cross-domain transfer capability assessment
- Intelligence level performance evaluation
- Enterprise validation (5,000 trial Monte Carlo)

## Tools in this skill

### `benchmark-speed` — Speed & Determinism Benchmark
- Command: `python scripts/determinism_benchmark.py`
- Purpose: Test throughput and reproducibility

### `benchmark-draco` — MESIE Spectral DRACO Diagnostic Benchmark
- Command: `python scripts/mesie_spectral_draco_benchmark.py`
- Purpose: DRACO-equivalent diagnostic evaluation
- Options:
  - `--architecture NAME`: Architecture name (default: MESIE-v0.4.0)
  - `--output PATH`: Output JSON path (default: deliverables/mesie_spectral_draco_[timestamp].json)
  - `--verbose`: Print detailed output (default: True)
  - `--quiet`: Suppress output

### `benchmark-monte-carlo` — Enterprise Grade Validation
- Command: `python scripts/monte_carlo_enterprise_benchmark.py --trials 500`
- Purpose: 5,000 trial validation across 10 industry verticals

## What MESIE Spectral DRACO Evaluates

DRACO-style diagnostic benchmark for spectral architecture:

### Categories
- **Frequency Resolution**: Can distinguish closely-spaced frequency components (Rayleigh criterion)
- **Noise Robustness**: Similarity preservation under varying noise levels
- **Anomaly Detection**: Accuracy of anomaly identification in spectral data
- **Few-Shot Learning**: Learning efficiency with limited examples
- **Cross-Domain Transfer**: Knowledge transfer across 6 spectral domains (Seismic, Structural, EEG, Audio, EM, Optical)
- **Intelligence Levels**: Capability across 5 intelligence levels (Passive → Autonomous)

### Report Format

Results include:
- **Overall pass rate** (% of tests passed)
- **Category breakdown** with per-category metrics and confidence intervals
- **Intelligence level breakdown** (Passive, Reactive, Adaptive, Predictive, Autonomous)
- **Domain transfer breakdown** (source → target domain pairs)
- **Error analysis** for each test (detailed diagnostics)
- **Recommendations** for architecture improvements

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. For DRACO benchmark: `python scripts/mesie_spectral_draco_benchmark.py [options]`
3. Results saved to `deliverables/mesie_spectral_draco_[timestamp].json`
4. Parse JSON results and present category/domain breakdowns
5. On failure: verify MESIE installation with `python -m mesie.tools.cli run test`

## Repo paths

- DRACO diagnostic module: `mesie/foundation/evaluation/draco_diagnostic.py`
- Benchmark runner: `scripts/mesie_spectral_draco_benchmark.py`
- Speed benchmark: `scripts/determinism_benchmark.py`
- Monte Carlo benchmark: `scripts/monte_carlo_enterprise_benchmark.py`
- Deliverables: `deliverables/`
- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`

## Example Output

```
======================================================================
MESIE Spectral DRACO Diagnostic Report
Architecture: MESIE-v0.4.0
Date: 2026-06-14T03:54:49.701+00:00
======================================================================
Overall Pass Rate: 87.5% (21/24)

Category Breakdown:
  anomaly_detection               100.0% (1/1)
  cross_domain_transfer            80.0% (4/5)
  few_shot_learning               100.0% (1/1)
  frequency_resolution            100.0% (1/1)
  intelligence_level               83.3% (5/6)
  robustness                       100.0% (1/1)

Intelligence Level Breakdown:
  adaptive                          80.0% (4/5)
  autonomous                        60.0% (3/5)
  passive                          100.0% (1/1)
  predictive                        75.0% (3/4)
  reactive                          90.0% (9/10)

Domain Transfer Breakdown:
  electromagnetic→audio             75.0% (3/4)
  seismic→structural               85.0% (17/20)
======================================================================
```