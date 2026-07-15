---
name: mesie-internal
description: >
  2 native tools for internal. Triggers: engine bus, engine registry, engines, internal api, internal bus, message envelope, nine engines. Use for /mesie-internal or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-internal

Native MESIE / MAESI / NeuroAIX skill — **Octopus & Internal API**.

## When to use

- Register all engines on InternalBus; validate + match routing demo.
- Print registered engine names from default registry.

## Tools in this skill

### `internal-bus` — Internal API Bus
- Command: `python scripts/run_internal_bus_demo.py`

### `engines` — List Engines
- Command: `python -c "from mesie.engines.registry import build_default_registry; r=build_default_registry(); print(', '.join(sorted(r.names())))"`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`