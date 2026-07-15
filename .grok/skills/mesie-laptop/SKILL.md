---
name: mesie-laptop
description: >
  Virtual chip framing + embedded library report. Triggers: laptop, research report, virtual chip. Use for /mesie-laptop or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-laptop

Native MESIE / MAESI / NeuroAIX skill — **Domain & Enterprise Analysis**.

## When to use

- Virtual chip framing + embedded library report.

## Tools in this skill

### `laptop` — Laptop Research Report
- Command: `python scripts/embed_spectral_library.py && python scripts/generate_laptop_research_report.py`
- Deliverable: `deliverables/MESIE_Laptop_Research_Report.md`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`