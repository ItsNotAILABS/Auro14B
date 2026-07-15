---
name: mesie-octopus
description: >
  Eight-arm controller; EMBED/MATCH default to AISVectorPolyglot (vector + Rust/Python). Triggers: eight arms, internal api, multi-arm, octopus, polyglot arms. Use for /mesie-octopus or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-octopus

Native MESIE / MAESI / NeuroAIX skill — **Octopus & Internal API**.

## When to use

- Eight-arm controller; EMBED/MATCH default to AISVectorPolyglot (vector + Rust/Python).

## Tools in this skill

### `octopus` — Octopus Engineering
- Command: `python examples/11_octopus_internal_api.py`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`