---
name: mesie-embed-library
description: >
  2 native tools for embed-library. Triggers: embed folder, embed library, index corpus, my library. Use for /mesie-embed-library or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-embed-library

Native MESIE / MAESI / NeuroAIX skill — **Embedding, Fingerprint & ANN Retrieval**.

## When to use

- Embed all references + benchmarks into spectral_index.json.
- Embed your JSON folder into my_spectral_index.json.

## Tools in this skill

### `embed-library` — Embed Full Library
- Command: `python scripts/embed_spectral_library.py`
- Deliverable: `library/spectral_index.json`

### `embed-mine` — Embed User Library
- Command: `python scripts/embed_my_library.py <folder> --octopus`
- Deliverable: `library/my_spectral_index.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`