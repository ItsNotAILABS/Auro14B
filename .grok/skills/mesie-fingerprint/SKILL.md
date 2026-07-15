---
name: mesie-fingerprint
description: >
  TF → salient → LSH → approximate nearest neighbors. Triggers: ann, fingerprint, lsh, salient, time-frequency. Use for /mesie-fingerprint or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-fingerprint

Native MESIE / MAESI / NeuroAIX skill — **Embedding, Fingerprint & ANN Retrieval**.

## When to use

- TF → salient → LSH → approximate nearest neighbors.

## Tools in this skill

### `fingerprint` — Fingerprint ANN Pipeline
- Command: `python examples/13_fingerprint_ann_pipeline.py`
- Deliverable: `library/spectral_index.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`