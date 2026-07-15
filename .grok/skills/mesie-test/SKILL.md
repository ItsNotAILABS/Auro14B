---
name: mesie-test
description: >
  2 native tools for test. Triggers: ci, pytest, sdk test drive, smoke test, test. Use for /mesie-test or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-test

Native MESIE / MAESI / NeuroAIX skill — **Quality Assurance & Benchmarks**.

## When to use

- Run full pytest suite.
- End-to-end SDK smoke with report JSON.

## Tools in this skill

### `test` — Run Test Suite
- Command: `python -m pytest tests/ -q`

### `sdk-drive` — SDK Test Drive
- Command: `python scripts/sdk_test_drive.py`
- Deliverable: `scripts/sdk_test_drive_report.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`