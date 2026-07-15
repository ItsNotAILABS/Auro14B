---
name: mesie-validate
description: >
  Validate spectral JSON against MESIE schema levels 1-6. Triggers: quality check, schema, validate. Use for /mesie-validate or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-validate

Native MESIE / MAESI / NeuroAIX skill — **MESIE Core Spectral Engine**.

## When to use

- Validate spectral JSON against MESIE schema levels 1-6.

## Tools in this skill

### `validate` — Validate Record
- Command: `python examples/01_load_and_validate.py`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`