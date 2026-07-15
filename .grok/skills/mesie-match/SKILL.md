---
name: mesie-match
description: >
  2 native tools for match. Triggers: compare spectra, match, rank, rank candidates, retrieval, similarity, top-k. Use for /mesie-match or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-match

Native MESIE / MAESI / NeuroAIX skill — **MESIE Core Spectral Engine**.

## When to use

- Compare two spectral records with composite scoring.
- Rank a query spectrum against a candidate pool (composite scoring).

## Tools in this skill

### `match` — Match Two Spectra
- Command: `python examples/02_match_two_records.py`

### `rank` — Rank Candidates
- Command: `python scripts/run_rank_demo.py`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`