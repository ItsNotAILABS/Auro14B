---
name: mesie-maesi
description: >
  2 native tools for maesi. Triggers: batch match, fast compute, maesi, maesi sdk, neuroaix sdk, speedup, virtual chip. Use for /mesie-maesi or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-maesi

Native MESIE / MAESI / NeuroAIX skill — **MAESI SDK & Knowledge**.

## When to use

- Run MAESI v1.1: laws, elements, bio, technical + research knowledge, fast compute.
- Batch matrix cosine vs loop match — virtual-chip throughput.

## Tools in this skill

### `maesi` — MAESI SDK Run
- Command: `python scripts/run_maesi_sdk.py`
- Deliverable: `deliverables/MAESI_SDK_Run_Report.json`

### `fast-compute` — Fast Compute Benchmark
- Command: `python scripts/run_fast_compute_benchmark.py`
- Deliverable: `deliverables/MAESI_Fast_Compute_Benchmark.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`