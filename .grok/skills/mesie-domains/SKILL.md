---
name: mesie-domains
description: >
  Terrain, robotics, orbital, power, seismic analysis suites. Triggers: domains, power, robotics, seismic suites, terrain. Use for /mesie-domains or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-domains

Native MESIE / MAESI / NeuroAIX skill — **Domain & Enterprise Analysis**.

## When to use

- Terrain, robotics, orbital, power, seismic analysis suites.

## Tools in this skill

### `domains` — Multi-Domain Suites
- Command: `python scripts/run_multi_domain_suites.py`
- Deliverable: `deliverables/MESIE_Multi_Domain_Suite_Report.md`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`