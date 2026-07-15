---
name: mesie-polyglot
description: >
  Polyglot test/use/integration: Python, Rust, Julia, Motoko, TypeScript + vector + AIS + 3rd-party AI. Triggers: ais, julia, motoko, polyglot, rust, third party ai, typescript, vector bridge. Use for /mesie-polyglot or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-polyglot

Native MESIE / MAESI / NeuroAIX skill — **Octopus & Internal API**.

## When to use

- Polyglot test/use/integration: Python, Rust, Julia, Motoko, TypeScript + vector + AIS + 3rd-party AI.

## Tools in this skill

### `ais-polyglot` — AISVectorPolyglot Suite
- Command: `python scripts/run_ais_polyglot_suite.py`
- Deliverable: `deliverables/AISVectorPolyglot_Integration_Report.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`