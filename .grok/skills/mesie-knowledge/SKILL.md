---
name: mesie-knowledge
description: >
  Search technical + research catalogs. Triggers: knowledge, research, technical library. Use for /mesie-knowledge or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-knowledge

Native MESIE / MAESI / NeuroAIX skill — **MAESI SDK & Knowledge**.

## When to use

- Search technical + research catalogs.

## Tools in this skill

### `knowledge` — Research Knowledge Search
- Command: `python -c "from mesie.sdk import search_research; print(search_research('LSH ANN', 5))"`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`