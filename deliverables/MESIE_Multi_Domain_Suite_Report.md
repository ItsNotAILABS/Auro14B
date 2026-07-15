# MESIE Multi-Domain Analysis Report

*Generated 2026-06-04T07:43:09Z — engine v0.2.1*

## Executive summary

1. Terrain-coupled FAS valid=True. Strongest match: ref-structural-fas-001 (0.845). Ground coupling to seismic anchor: 0.630. Record valid=True. Match score=0.630. Control issued ['investigate_match', 'trigger_alert']. Logic fired 2 rule(s). Intelligence says anomaly_detected. Workflow complete=True.
2. Robotics baseline (pump vibration) ranked 5 machine states. Likely fault candidate: earthquake-003 (score 0.549). Intelligence: anomaly_detected. Record valid=True. Match score=0.515. Control issued ['investigate_match', 'trigger_alert']. Logic fired 4 rule(s). Intelligence says anomaly_detected. Workflow complete=True.
3. Orbital: 7 edge days in last 50d; 6 edge days forecast. Alerts on days []. LEO link margins computed for 3 nodes. Record valid=True. Match score=1.000. Control issued ['none']. Logic fired 0 rule(s). Intelligence says normal_operation. Workflow complete=True.
4. Power/eco-Hz: Schumann 7 modes embedded; Hz-ladder tiers 0-4 characterized. Intelligence: normal_operation. Best spectral neighbor: power_schumann. Record valid=True. Match score=1.000. Control issued ['none']. Logic fired 0 rule(s). Intelligence says normal_operation. Workflow complete=True.
5. Seismic suite: 4 references cross-matched. Strongest pair score 0.7866 (earthquake_psd_reference vs rotdnn_reference). Record valid=True. Match score=1.000. Control issued ['none']. Logic fired 0 rule(s). Intelligence says normal_operation. Workflow complete=True.

## Domain suites

### Terrain & structural ground coupling (`terrain`)

**Conclusion:** Terrain-coupled FAS valid=True. Strongest match: ref-structural-fas-001 (0.845). Ground coupling to seismic anchor: 0.630. Record valid=True. Match score=0.630. Control issued ['investigate_match', 'trigger_alert']. Logic fired 2 rule(s). Intelligence says anomaly_detected. Workflow complete=True.

- Records: terrain_coupled_fas, earthquake_psd_reference, structural_fas_reference
- Runtime: 7 ms
- Metrics:
  - validation_level: 6
  - seismic_coupling_score: 0.6296
  - terrain_roughness: 0.5633270638512314
- Top matches:
  - `ref-structural-fas-001` → 0.845
  - `ref-earthquake-psd-001` → 0.6296
  - `ref-rotdnn-001` → 0.6174
  - `ref-vibration-monitor-001` → 0.4646

### Robotics & machinery condition monitoring (`robotics`)

**Conclusion:** Robotics baseline (pump vibration) ranked 5 machine states. Likely fault candidate: earthquake-003 (score 0.549). Intelligence: anomaly_detected. Record valid=True. Match score=0.515. Control issued ['investigate_match', 'trigger_alert']. Logic fired 4 rule(s). Intelligence says anomaly_detected. Workflow complete=True.

- Records: ref-vibration-monitor-001, earthquake-011, earthquake-005, earthquake-003
- Runtime: 16 ms
- Metrics:
  - machine_type: None
  - fault_threshold: 0.55
  - top_score: 0.5667
- Top matches:
  - `earthquake-011` → 0.5667
  - `earthquake-005` → 0.56
  - `earthquake-003` → 0.5487
  - `earthquake-002` → 0.5485

### Orbital edge, satellite nodes & seismic coupling (`orbital`)

**Conclusion:** Orbital: 7 edge days in last 50d; 6 edge days forecast. Alerts on days []. LEO link margins computed for 3 nodes. Record valid=True. Match score=1.000. Control issued ['none']. Logic fired 0 rule(s). Intelligence says normal_operation. Workflow complete=True.

- Records: earthquake_psd_reference, orbital_50d_synthetic
- Runtime: 167 ms
- Metrics:
  - history_edge_days: 7
  - mean_eq_match_edge: 0.8070476756412107
  - forecast_edge_count: 6
  - alert_days: []
- Satellite nodes:
  - node_LEO_300: contact 297.1s, loss 163.35 dB
  - node_LEO_550: contact 476.5s, loss 168.62 dB
  - node_LEO_780: contact 624.2s, loss 171.65 dB

### Power, Schumann & electromagnetic ladder (`power`)

**Conclusion:** Power/eco-Hz: Schumann 7 modes embedded; Hz-ladder tiers 0-4 characterized. Intelligence: normal_operation. Best spectral neighbor: power_schumann. Record valid=True. Match score=1.000. Control issued ['none']. Logic fired 0 rule(s). Intelligence says normal_operation. Workflow complete=True.

- Records: schumann_resonances, electromagnetic_bands
- Runtime: 6 ms
- Metrics:
  - schumann_modes: 7
  - validation_level: 6
- Top matches:
  - `power_schumann` → 1.0
  - `power_em_bands` → 0.6366
  - `ref-vibration-monitor-001` → 0.6083
- Hz ladder (power/comm):
  - Tier 0 ELF/Schumann: 7.83e+00 Hz
  - Tier 1 VLF: 1.50e+04 Hz
  - Tier 2 HF: 1.40e+07 Hz
  - Tier 3 UHF/Terrestrial: 1.58e+09 Hz

### Seismic & structural reference cross-analysis (`seismic`)

**Conclusion:** Seismic suite: 4 references cross-matched. Strongest pair score 0.7866 (earthquake_psd_reference vs rotdnn_reference). Record valid=True. Match score=1.000. Control issued ['none']. Logic fired 0 rule(s). Intelligence says normal_operation. Workflow complete=True.

- Records: earthquake_psd_reference, rotdnn_reference, structural_fas_reference, vibration_monitoring_reference
- Runtime: 6 ms
- Metrics:
  - pair_count: 6
  - reference_count: 4
- Top matches:
  - `earthquake_psd_reference` vs `rotdnn_reference` → 0.7866
  - `earthquake_psd_reference` vs `structural_fas_reference` → 0.6358
  - `rotdnn_reference` vs `vibration_monitoring_reference` → 0.6291
  - `rotdnn_reference` vs `structural_fas_reference` → 0.6204

## How to re-run

```bash
python scripts/run_multi_domain_suites.py
python scripts/orbital_edge_50d_analysis.py
```
