---
name: mesie-generate
description: >
  3 native tools for generate. Triggers: generate fas, generate psd, orientation, rotdnn, synthetic psd. Use for /mesie-generate or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-generate

Native MESIE / MAESI / NeuroAIX skill — **MESIE Core Spectral Engine**.

## When to use

- Synthesize power spectral density record (seeded).
- Synthesize Fourier amplitude spectrum.
- RotDNN orientation spectral workflow.

## Tools in this skill

### `generate-psd` — Generate PSD
- Command: `python examples/03_generate_psd.py`

### `generate-fas` — Generate FAS
- Command: `python examples/04_generate_fas.py`

### `rotdnn` — RotDNN Workflow
- Command: `python examples/05_rotdnn_workflow.py`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`