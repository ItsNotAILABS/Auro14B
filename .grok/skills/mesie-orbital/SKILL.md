---
name: mesie-orbital
description: >
  50 days back + 50 forward orbital-edge analysis. Triggers: 50 day, orbital, satellite edge. Use for /mesie-orbital or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-orbital

Native MESIE / MAESI / NeuroAIX skill — **Domain & Enterprise Analysis**.

## When to use

- 50 days back + 50 forward orbital-edge analysis.

## Tools in this skill

### `orbital` — Orbital Edge 50d
- Command: `python scripts/orbital_edge_50d_analysis.py`
- Deliverable: `scripts/orbital_edge_50d_report.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`