# MESIE Monte Carlo Enterprise Report

*Generated 2026-06-06T21:15:23Z — 5,000 trials across 10 enterprise use cases*

## Executive summary

- **Overall success rate:** 100.0%
- **Enterprise grade (≥85%):** PASS
- **Total runtime:** 5.37 s
- **Trials per use case:** 500

## 10 enterprise use cases

| # | Industry | Use case | Success % | Mean ms | P95 ms | Mean score |
|---|----------|----------|-----------|---------|--------|------------|
| 1 | Manufacturing | Predictive Maintenance | 100.0% | 0.23 | 0.25 | 0.978 |
| 2 | Energy | Grid & Power Monitoring | 100.0% | 0.16 | 0.19 | 6.000 |
| 3 | Aerospace | Satellite Ops & Orbital | 100.0% | 0.37 | 0.41 | 0.860 |
| 4 | Insurance | Catastrophe / Seismic Risk | 100.0% | 0.37 | 0.43 | 0.635 |
| 5 | Construction | Structural / Civil Engineering | 100.0% | 1.41 | 1.52 | 0.846 |
| 6 | Healthcare | Medical Device Monitoring | 100.0% | 0.57 | 0.63 | 74.429 |
| 7 | Robotics | Autonomous Robotics Fleet | 100.0% | 0.22 | 0.24 | 0.960 |
| 8 | Telecom | Telecom Spectrum Compliance | 100.0% | 0.02 | 0.02 | 2.000 |
| 9 | Research | R&D Spectral Lab | 100.0% | 1.37 | 1.46 | 0.736 |
| 10 | Enterprise AI | AI Agent Spectral Memory | 100.0% | 5.99 | 6.15 | 1.000 |

## Per-use-case detail

### Predictive Maintenance (Manufacturing)

Detect pump/machine drift from vibration spectra.

- Success metric: `top_match_sim >= 0.5`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.23 ms, std 0.03, p95 0.25
- Score: mean 0.9782, std 0.0466, p5 0.9001

### Grid & Power Monitoring (Energy)

Schumann/EM band fingerprint stability under noise.

- Success metric: `validation_level >= 4`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.16 ms, std 0.08, p95 0.19
- Score: mean 6.0000, std 0.0000, p5 6.0000

### Satellite Ops & Orbital (Aerospace)

Orbital-edge style spectral gate + seismic anchor coupling.

- Success metric: `match_score >= 0.6`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.37 ms, std 0.08, p95 0.41
- Score: mean 0.8601, std 0.0203, p5 0.7971

### Catastrophe / Seismic Risk (Insurance)

Cross-match earthquake vs structural references.

- Success metric: `match_score >= 0.55`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.37 ms, std 0.05, p95 0.43
- Score: mean 0.6351, std 0.0020, p5 0.6320

### Structural / Civil Engineering (Construction)

FAS structural spectrum ranking under perturbation.

- Success metric: `rank_top3_self_or_structural`
- Success rate: **100.0%** (500/500)
- Latency: mean 1.41 ms, std 0.16, p95 1.52
- Score: mean 0.8460, std 0.0024, p5 0.8419

### Medical Device Monitoring (Healthcare)

Anomaly separation on biosignal-like spectra.

- Success metric: `anomaly_detects_outlier`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.57 ms, std 0.07, p95 0.63
- Score: mean 74.4288, std 6.4887, p5 51.4530

### Autonomous Robotics Fleet (Robotics)

Fast ANN neighbor lookup for fleet state.

- Success metric: `query_ms < 50 and sim > 0.4`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.22 ms, std 0.01, p95 0.24
- Score: mean 0.9597, std 0.0664, p5 0.8293

### Telecom Spectrum Compliance (Telecom)

EM band library embedding + research hit.

- Success metric: `research_hit_found`
- Success rate: **100.0%** (500/500)
- Latency: mean 0.02 ms, std 0.00, p95 0.02
- Score: mean 2.0000, std 0.0000, p5 2.0000

### R&D Spectral Lab (Research)

Benchmark sample classification via ranking.

- Success metric: `rank_score >= 0.45`
- Success rate: **100.0%** (500/500)
- Latency: mean 1.37 ms, std 0.06, p95 1.46
- Score: mean 0.7364, std 0.0404, p5 0.6760

### AI Agent Spectral Memory (Enterprise AI)

MAESI query with knowledge + fingerprint ANN.

- Success metric: `neighbors >= 1 and latency < 100ms`
- Success rate: **100.0%** (500/500)
- Latency: mean 5.99 ms, std 1.68, p95 6.15
- Score: mean 1.0000, std 0.0000, p5 1.0000

## How to re-run

```bash
python scripts/monte_carlo_enterprise_benchmark.py
python scripts/monte_carlo_enterprise_benchmark.py --trials 500
```
