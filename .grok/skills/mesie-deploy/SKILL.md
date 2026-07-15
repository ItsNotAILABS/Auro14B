---
name: mesie-deploy
description: >
  2 native tools for deploy. Triggers: catalog, cloudflare, deploy, edge api, skills map, tool list, worker. Use for /mesie-deploy or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-deploy

Native MESIE / MAESI / NeuroAIX skill — **Deploy & Edge**.

## When to use

- Edge API deploy guide and worker health.
- Write JSON catalog of all native tools and skills.

## Tools in this skill

### `cloudflare` — Cloudflare Worker
- Command: `type docs\cloudflare.md`

### `catalog` — Export Tool Catalog
- Command: `python -m mesie.tools.cli catalog`
- Deliverable: `deliverables/MESIE_Native_Tools_Catalog.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`