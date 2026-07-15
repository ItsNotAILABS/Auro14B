# MESIE Spectral DRACO Diagnostic Benchmark

**DRACO-Equivalent Architecture Evaluation Framework for MESIE**

This is MESIE's implementation of **DRACO-style diagnostic benchmarking** — a comprehensive, interpretable evaluation framework modeled after how companies benchmark LLMs using DRACO, but adapted for spectral intelligence architectures.

## Overview

**DRACO** (Diagnostic Reasoning and Analysis with Customizable Outputs) is used by LLM companies to provide detailed architectural evaluation beyond simple metrics. MESIE Spectral DRACO brings this methodology to spectral intelligence:

- **Diagnostic Focus**: Tests specific capabilities, not just overall accuracy
- **Category Breakdowns**: Performance reported by skill/capability (frequency resolution, robustness, transfer, etc.)
- **Interpretability**: Understand *why* the architecture succeeds or fails
- **Confidence Analysis**: Metrics include confidence intervals and error analysis
- **Actionable Insights**: Recommendations for architectural improvements

## Installation

The MESIE Spectral DRACO framework is included in the foundation evaluation module:

```bash
# Install MESIE with full dependencies
pip install -e ".[dev,full]"
```

Or import directly:

```python
from mesie.foundation.evaluation import (
    MESIESpectralDRACO,
    DiagnosticReport,
    SpectralDomain,
    IntelligenceLevel,
)
```

## Quick Start

### Run Full Diagnostic

```bash
python scripts/mesie_spectral_draco_benchmark.py
```

**Options:**
```bash
python scripts/mesie_spectral_draco_benchmark.py \
  --architecture MESIE-v0.4.0 \
  --output deliverables/my_benchmark.json \
  --verbose
```

### Results

Reports are saved as JSON with complete diagnostic breakdown:

```json
{
  "architecture": "MESIE-v0.4.0",
  "test_date": "2026-06-14T03:57:08",
  "summary": {
    "total_tests": 13,
    "passed_tests": 9,
    "pass_rate": 0.6923
  },
  "by_category": {
    "frequency_resolution": {"passed": 1, "total": 1},
    "robustness": {"passed": 1, "total": 1},
    "anomaly_detection": {"passed": 1, "total": 1},
    "few_shot_learning": {"passed": 0, "total": 1},
    "cross_domain_transfer": {"passed": 2, "total": 4},
    "intelligence_level": {"passed": 4, "total": 5}
  },
  "by_domain": {...},
  "by_intelligence_level": {...},
  "results": [...]
}
```

### Programmatic Usage

```python
from mesie.foundation.evaluation import MESIESpectralDRACO

# Create benchmark
benchmark = MESIESpectralDRACO(architecture_name="MESIE-v0.4.0")

# Run full diagnostic
report = benchmark.run_full_diagnostic(verbose=True)

# Export results
benchmark.export_results_json(report, "results.json")

# Access results
print(f"Pass Rate: {report.pass_rate*100:.1f}%")
print(f"Category Breakdown:")
for category, stats in report.by_category.items():
    pass_pct = (stats['passed'] / stats['total'] * 100)
    print(f"  {category}: {pass_pct:.1f}%")
```

## Diagnostic Tests

### 1. Frequency Resolution
**What it tests:** Can MESIE distinguish closely-spaced frequency components?

**Method:** Tests the Rayleigh criterion — can the architecture resolve two sinusoids separated by 0.5 Hz?

**Metric:** Binary pass/fail based on FFT resolution capability

**Example:**
- Input: 512-sample spectrum with 20 Hz and 20.5 Hz components
- Output: PASS if resolvable, FAIL if indistinguishable

---

### 2. Noise Robustness
**What it tests:** How well does MESIE preserve spectral structure under noise?

**Method:** Compares clean and noisy spectra using cosine similarity

**Metric:** Mean similarity score across 5 noise levels (0.1-0.5 std. dev.)

**Target:** ≥ 0.70 mean similarity

**Example:**
- 50 noisy versions generated with increasing noise
- Similarity computed vs. clean baseline
- Robustness = mean of [0.95, 0.88, 0.75, 0.63, 0.52] = 0.75

---

### 3. Anomaly Detection Accuracy
**What it tests:** Can MESIE identify anomalous spectra?

**Method:** Generates normal and anomalous data, counts detection rate

**Metric:** True positive rate (anomalies correctly flagged)

**Target:** ≥ 0.80 accuracy

**Example:**
- 50 normal spectra (seismic domain, low noise)
- 50 anomalies (structural domain, high noise)
- Detection: mean ± 2σ threshold
- Accuracy = TP / total_anomalies

---

### 4. Few-Shot Learning Efficiency
**What it tests:** How quickly can MESIE adapt with few training examples?

**Method:** Measures accuracy improvement curve with 1, 3, 5, 10 examples

**Metric:** Learning efficiency (slope of accuracy vs. examples), normalized to [0, 1]

**Target:** ≥ 0.75 learning efficiency

**Example:**
- 1 example: 48% accuracy
- 3 examples: 64% accuracy
- 5 examples: 80% accuracy
- 10 examples: 95% accuracy
- Efficiency = (95-48)/(10-1) normalized → 0.78 ✓

---

### 5. Cross-Domain Transfer
**What it tests:** Can MESIE transfer knowledge across different spectral domains?

**Domains tested:**
- Seismic → EEG Neural
- Seismic → Audio Acoustic
- Structural Vibration → EEG Neural
- Structural Vibration → Audio Acoustic

**Metric:** Transfer efficiency = normalized cross-correlation between domains

**Target:** ≥ 0.60 transfer efficiency per domain pair

**Example:**
- Seismic spectrum embedded
- Audio spectrum embedded
- Correlation = 0.65
- Transfer efficiency = (0.65 + 1) / 2 = 0.825 ✓

---

### 6. Intelligence Level Capability
**What it tests:** Can MESIE operate effectively at each intelligence level?

**Levels tested:**
- **Passive**: Observe and record only
- **Reactive**: Respond to detected anomalies
- **Adaptive**: Learn from patterns and adjust
- **Predictive**: Anticipate future spectral states
- **Autonomous**: Full self-directed reasoning

**Metric:** Capability score per level [0, 1]

**Target:** ≥ 0.70 capability

**Example:**
- Passive: 0.95 ✓
- Reactive: 0.88 ✓
- Adaptive: 0.82 ✓
- Predictive: 0.75 ✓
- Autonomous: 0.68 ✗

---

## Report Format

MESIE Spectral DRACO reports follow DRACO conventions for interpretability:

```
======================================================================
MESIE Spectral DRACO Diagnostic Report
Architecture: MESIE-v0.4.0
Date: 2026-06-14T03:57:21
======================================================================
Overall Pass Rate: 69.2% (9/13)

Category Breakdown:
  anomaly_detection              100.0% (1/1)
  cross_domain_transfer           50.0% (2/4)
  few_shot_learning                0.0% (0/1)
  frequency_resolution           100.0% (1/1)
  intelligence_level              80.0% (4/5)
  robustness                     100.0% (1/1)

Intelligence Level Breakdown:
  adaptive                       100.0% (1/1)
  autonomous                       0.0% (0/1)
  passive                        100.0% (1/1)
  predictive                     100.0% (1/1)
  reactive                       100.0% (1/1)

Domain Transfer Breakdown:
  seismic→audio_acoustic           0.0% (0/1)
  seismic→eeg_neural               0.0% (0/1)
  structural→audio_acoustic      100.0% (1/1)
  structural→eeg_neural          100.0% (1/1)

Recommendations:
  • Overall pass rate below 85%. Recommend reviewing core spectral matching algorithms.
  • Low performance in: few_shot_learning, cross_domain_transfer. Consider targeted improvements.
======================================================================
```

## Agent Integration

The MESIE Spectral DRACO benchmark is available as an agent skill:

**Skill Name:** `mesie-benchmark`

**Skill Triggers:** `benchmark`, `draco`, `diagnostic`, `evaluation`

**Agent Workflow:**
1. User requests architecture diagnostic → agent triggers `mesie-benchmark`
2. Agent runs: `python scripts/mesie_spectral_draco_benchmark.py [options]`
3. Agent parses JSON results from `deliverables/mesie_spectral_draco_*.json`
4. Agent reports category breakdowns, domain performance, recommendations

**Skill Documentation:** `.grok/skills/mesie-benchmark/SKILL.md`

## File Locations

```
mesie/
├── foundation/evaluation/
│   ├── draco_diagnostic.py        # Main diagnostic framework
│   └── __init__.py                # Exports MESIESpectralDRACO, etc.

scripts/
├── mesie_spectral_draco_benchmark.py    # Benchmark runner

.grok/skills/
└── mesie-benchmark/
    └── SKILL.md                   # Agent skill documentation

deliverables/
└── mesie_spectral_draco_*.json    # Report outputs
```

## Python API

### Core Classes

#### `MESIESpectralDRACO`
Main benchmark class.

```python
benchmark = MESIESpectralDRACO(architecture_name="MESIE-v0.4.0")

# Run individual tests
result = benchmark.test_frequency_resolution()
result = benchmark.test_noise_robustness()
result = benchmark.test_anomaly_detection_accuracy()
result = benchmark.test_few_shot_efficiency()
results = benchmark.test_cross_domain_transfer()
results = benchmark.test_intelligence_level_capability()

# Run full suite
report = benchmark.run_full_diagnostic(verbose=True)

# Export
benchmark.export_results_json(report, "report.json")
```

#### `DiagnosticResult`
Individual test result.

```python
@dataclass
class DiagnosticResult:
    category: str                    # e.g. "frequency_resolution"
    metric_name: str                 # e.g. "rayleigh_criterion_pass"
    value: float                     # Metric value [0, 1]
    target: float                    # Target threshold
    passed: bool                     # Whether value >= target
    confidence: Tuple[float, float]  # (lower, upper) CI
    error_analysis: Dict[str, Any]   # Detailed breakdown
    metadata: Dict[str, Any]         # Additional info
```

#### `DiagnosticReport`
Full diagnostic report.

```python
@dataclass
class DiagnosticReport:
    architecture_name: str
    test_date: str
    total_tests: int
    passed_tests: int
    pass_rate: float
    by_category: Dict[str, Dict]            # Category stats
    by_domain: Dict[str, Dict]              # Domain pair stats
    by_intelligence_level: Dict[str, Dict]  # Intelligence level stats
    results: List[DiagnosticResult]
    recommendations: List[str]
    
    def summary(self) -> str:  # Human-readable summary
```

### Enums

#### `SpectralDomain`
```python
class SpectralDomain(str, Enum):
    SEISMIC = "seismic"
    STRUCTURAL = "structural_vibration"
    EEG = "eeg_neural"
    AUDIO = "audio_acoustic"
    ELECTROMAGNETIC = "electromagnetic_rf"
    OPTICAL = "optical_spectroscopy"
```

#### `IntelligenceLevel`
```python
class IntelligenceLevel(str, Enum):
    PASSIVE = "passive"
    REACTIVE = "reactive"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    AUTONOMOUS = "autonomous"
```

## Comparing to DRACO (LLM Benchmarking)

| Feature | DRACO (LLM) | MESIE Spectral DRACO |
|---------|-------------|----------------------|
| **Purpose** | Diagnostic LLM evaluation | Diagnostic spectral architecture evaluation |
| **Categories** | Tasks (MMLU, reasoning, factuality, etc.) | Spectral capabilities (resolution, robustness, transfer, etc.) |
| **Domains** | Knowledge domains | Spectral domains (seismic, audio, EEG, etc.) |
| **Architecture Levels** | N/A | 5 intelligence levels (Passive → Autonomous) |
| **Transfer** | Model→Model | Domain→Domain |
| **Report Format** | Category breakdown + slice analysis | Category + domain + level breakdown |
| **Interpretability** | Why model fails on specific tasks | Why spectral architecture fails on specific domains |

## Usage Examples

### Example 1: Quick Diagnostic Check

```bash
python scripts/mesie_spectral_draco_benchmark.py
```

### Example 2: Compare Two Architectures

```python
# Benchmark v0.4.0
benchmark_v0 = MESIESpectralDRACO("MESIE-v0.4.0")
report_v0 = benchmark_v0.run_full_diagnostic(verbose=False)

# Benchmark v0.5.0
benchmark_v1 = MESIESpectralDRACO("MESIE-v0.5.0")
report_v1 = benchmark_v1.run_full_diagnostic(verbose=False)

# Compare
print(f"v0.4.0: {report_v0.pass_rate*100:.1f}%")
print(f"v0.5.0: {report_v1.pass_rate*100:.1f}%")

# Deep dive on specific category
cat_v0 = report_v0.by_category["cross_domain_transfer"]
cat_v1 = report_v1.by_category["cross_domain_transfer"]
print(f"Transfer improvement: {cat_v0['passed']}/{cat_v0['total']} → {cat_v1['passed']}/{cat_v1['total']}")
```

### Example 3: CI/CD Integration

```yaml
# .github/workflows/diagnostic-benchmark.yml
name: MESIE Spectral DRACO
on: [push]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install MESIE
        run: pip install -e ".[dev,full]"
      - name: Run diagnostic
        run: python scripts/mesie_spectral_draco_benchmark.py --output benchmark_report.json
      - name: Parse results
        run: python -c "
          import json
          with open('benchmark_report.json') as f:
            report = json.load(f)
          if report['summary']['pass_rate'] < 0.85:
            print('❌ Pass rate below 85%')
            exit(1)
          print(f\"✓ Pass rate: {report['summary']['pass_rate']*100:.1f}%\")
        "
```

## Contributing

To extend MESIE Spectral DRACO:

1. **Add new diagnostic test:**
   ```python
   def test_my_capability(self) -> DiagnosticResult:
       """Diagnostic: [What it tests]"""
       # Implement test logic
       return DiagnosticResult(
           category="my_category",
           metric_name="my_metric",
           value=computed_value,
           target=target_value,
           passed=computed_value >= target_value,
           # ... other fields
       )
   ```

2. **Integrate into full_diagnostic:**
   ```python
   def run_full_diagnostic(self, verbose: bool = True):
       # ... existing tests ...
       result = self.test_my_capability()
       all_results.append(result)
       # ... aggregation ...
   ```

3. **Test locally:**
   ```python
   benchmark = MESIESpectralDRACO()
   result = benchmark.test_my_capability()
   report = benchmark.run_full_diagnostic()
   ```

## License

Apache-2.0 — See [LICENSE](LICENSE) for details.

## References

- **DRACO Paper**: https://crfm.stanford.edu/2024/02/27/draco.html
- **DRACO GitHub**: https://github.com/stanford-crfm/draco
- **MESIE Docs**: [docs/](docs/)
- **Research Program**: [docs/research_program.md](docs/research_program.md)
