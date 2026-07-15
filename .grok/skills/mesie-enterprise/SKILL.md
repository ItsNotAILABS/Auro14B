---
name: mesie-enterprise
description: >
  Monte Carlo benchmark across 10 enterprise use cases. Triggers: enterprise, monte carlo, sla benchmark. Use for /mesie-enterprise or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-enterprise

Native MESIE / MAESI / NeuroAIX skill — **Domain & Enterprise Analysis**.

## When to use

- Monte Carlo benchmark across 10 enterprise use cases.

## Tools in this skill

### `monte-carlo` — Monte Carlo Enterprise
- Command: `python scripts/monte_carlo_enterprise_benchmark.py --trials 200`
- Deliverable: `deliverables/MESIE_Monte_Carlo_Enterprise_Report.md`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`