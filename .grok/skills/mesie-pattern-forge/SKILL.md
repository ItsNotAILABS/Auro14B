---
name: mesie-pattern-forge
description: >
  Your X-ray math caretaker — z-depth, spectral decompose, phi-harmonics, fully local. Triggers: pattern forge, solus math, spectral decompose, xray, z-depth. Use for /mesie-pattern-forge or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-pattern-forge

Native MESIE / MAESI / NeuroAIX skill — **SOLUS Local Math AI Caretakers**.

## When to use

- Your X-ray math caretaker — z-depth, spectral decompose, phi-harmonics, fully local.

## Tools in this skill

### `pattern-forge` — SOLUS Pattern Forge
- Command: `python scripts/run_solus_math_caretakers.py`
- Deliverable: `deliverables/SOLUS_Math_Caretakers_Report.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`