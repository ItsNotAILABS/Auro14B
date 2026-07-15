---
name: mesie-logic-prover
description: >
  Your local math proof caretaker — mini heart + mini brain, zero 3rd party. Triggers: logic prover, proof, solus math, theorem. Use for /mesie-logic-prover or MESIE/MAESI/NeuroAIX tasks.
---

# mesie-logic-prover

Native MESIE / MAESI / NeuroAIX skill — **SOLUS Local Math AI Caretakers**.

## When to use

- Your local math proof caretaker — mini heart + mini brain, zero 3rd party.

## Tools in this skill

### `logic-prover` — SOLUS Logic Prover
- Command: `python scripts/run_solus_math_caretakers.py`
- Deliverable: `deliverables/SOLUS_Math_Caretakers_Report.json`

## Agent workflow

1. `cd` to repo root: `Multi-Element-Spectral-Intelligence-Engine-MESIE-`
2. Run via unified CLI: `python -m mesie.tools.cli run <tool-id>`
3. Or run the command above directly.
4. Read deliverable path if present; summarize results for the user.
5. On failure: run `python -m mesie.tools.cli run test` to verify environment.

## Repo paths

- Tools registry: `mesie/tools/registry.py`
- CLI: `python -m mesie.tools.cli list`