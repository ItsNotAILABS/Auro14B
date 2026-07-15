---
name: mesie-data
description: >
  2 native tools for data. Triggers: benchmarks, bundled data, data quality, fix data, references, repair references. Use for /mesie-data or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-data

Native MESIE / MAESI / NeuroAIX skill — **MESIE Core Spectral Engine**.

## When to use

- Load references, benchmarks, spectral library.
- Repair bundled reference JSON (negative amplitudes, schema drift).

## Tools in this skill

### `bundled-data` — Load Bundled Data
- Command: `python examples/09_load_bundled_data.py`

### `fix-data` — Fix Reference Data
- Command: `python scripts/fix_reference_data.py`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`