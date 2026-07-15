# MESIE / MAESI — 50 Major Chaos + Technical Tests

**Result:** 50/50 passed (100.0%)
**Runtime:** 466 ms on laptop virtual chip

## Summary

| Lane | Passed | Total |
|------|--------|-------|
| Chaos | 25 | 25 |
| Technical | 25 | 25 |

## Chaos engineering (25)

| ID | Test | Domain | ms | Status |
|----|------|--------|-----|--------|
| C01 | Noisy vibration match resilience | robustness | 0.4 | PASS |
| C02 | Flatline spectrum validation | edge_case | 0.1 | PASS |
| C03 | Amplitude spike self-match | edge_case | 0.2 | PASS |
| C04 | Bus reject empty validate payload | fault_injection | 0.1 | PASS |
| C05 | Bus unknown engine graceful fail | fault_injection | 0.1 | PASS |
| C06 | Rank empty candidate pool | edge_case | 0.0 | PASS |
| C07 | Rank single candidate | edge_case | 0.3 | PASS |
| C08 | Fingerprint noisy neighbor retrieval | retrieval | 31.0 | PASS |
| C09 | Octopus cycle under cross-domain noise | orchestration | 1.6 | PASS |
| C10 | Anomaly adapter outlier separation | control | 0.3 | PASS |
| C11 | Embed tiny 4-point spectrum | scale | 0.1 | PASS |
| C12 | Match 2048-point spectrum | scale | 0.3 | PASS |
| C13 | Seed-stable noisy match band | determinism | 1.1 | PASS |
| C14 | MAESI query under noise SLA | sla | 13.6 | PASS |
| C15 | TF salient under seismic noise | signal | 3.0 | PASS |
| C16 | Bus match stress 20 rounds | stress | 4.9 | PASS |
| C17 | Multi-component self-match | structure | 0.2 | PASS |
| C18 | Rank corpus after heavy perturbation | retrieval | 0.8 | PASS |
| C19 | Fingerprint bus index+query | bus | 12.7 | PASS |
| C20 | Octopus engine roster integrity | orchestration | 0.0 | PASS |
| C21 | Micro-amplitude validation tolerance | edge_case | 0.1 | PASS |
| C22 | Anomaly baseline vs noisy self | control | 0.2 | PASS |
| C23 | Fingerprint repeated noisy queries | stress | 74.8 | PASS |
| C24 | Control engine low-sim commands | control | 0.1 | PASS |
| C25 | MAESI fast compute speedup under load | performance | 153.7 | PASS |

## Technical validation (25)

| ID | Test | Domain | ms | Status |
|----|------|--------|-----|--------|
| T01 | Technical: STFT Spectrogram | time_frequency | 0.3 | PASS |
| T02 | Technical: Salient TF Peaks | signal_processing | 2.9 | PASS |
| T03 | Technical: LSH Spectral Hash | ann_retrieval | 0.1 | PASS |
| T04 | Technical: ANN Cosine Rerank | ann_retrieval | 0.1 | PASS |
| T05 | Technical: Pump Vibration Baseline | robotics | 0.0 | PASS |
| T06 | Technical: Anomaly vs Baseline | robotics | 0.2 | PASS |
| T07 | Technical: Schumann Eco-Hz | power_systems | 0.3 | PASS |
| T08 | Technical: EM Band Ladder | power_systems | 0.0 | PASS |
| T09 | Technical: LEO Contact Window | orbital_mechanics | 0.0 | PASS |
| T10 | Technical: Orbital Edge Gate | orbital_mechanics | 0.0 | PASS |
| T11 | Technical: Earthquake PSD Anchor | seismic | 0.0 | PASS |
| T12 | Technical: RotDNN Orientation | seismic | 0.0 | PASS |
| T13 | Technical: Structural FAS | structural | 0.0 | PASS |
| T14 | Technical: Spectral Vectorizer | spectral_ml | 0.1 | PASS |
| T15 | Technical: Fingerprint Pipeline | spectral_ml | 22.2 | PASS |
| T16 | Technical: Octopus Multi-Arm Control | robotics | 0.1 | PASS |
| T17 | Technical: Internal API Bus | signal_processing | 0.0 | PASS |
| T18 | Technical: Cross-Domain Transfer | spectral_ml | 2.6 | PASS |
| T19 | Technical: Intelligence Protocol | spectral_ml | 0.2 | PASS |
| T20 | Technical: Hz Virtual Chip | power_systems | 120.7 | PASS |
| T21 | TF→Salient integration chain | time_frequency | 2.9 | PASS |
| T22 | Research knowledge ANN hit | ann_retrieval | 0.0 | PASS |
| T23 | Technical matrix projection | spectral_ml | 0.0 | PASS |
| T24 | Multi-domain technical coverage | architecture | 0.0 | PASS |
| T25 | MAESI full stack query | product | 13.5 | PASS |