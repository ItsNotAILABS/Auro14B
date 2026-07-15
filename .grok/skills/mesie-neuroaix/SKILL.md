---
name: mesie-neuroaix
description: >
  2 native tools for neuroaix. Triggers: brain regions, cognitive, connectome, memory adapter, neuroaix. Use for /mesie-neuroaix or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-neuroaix

Native MESIE / MAESI / NeuroAIX skill — **NeuroAIX Connectome & Cognitive**.

## When to use

- MAESI observation encoder + 3D connectome brain demo.
- Spectral memory adapter and agent state.

## Tools in this skill

### `neuroaix` — NeuroAIX Connectome
- Command: `python examples/08_3d_connectome_brain.py`

### `cognitive` — Cognitive Memory
- Command: `python examples/07_cognitive_memory_adapter.py`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`