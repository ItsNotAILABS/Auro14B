# SOVEREIGN Organism Trial

The trial is a deterministic control-loop benchmark for the AURO/SOVEREIGN
architecture. It is intentionally runnable without a large checkpoint and
does not claim to validate trained model quality.

## Run

```bash
python benchmarks/sovereign_organism_trial.py --episodes 48
python benchmarks/sovereign_organism_trial.py --episodes 48 --seed 23 --json
```

The same seed produces the same episode set, actions, state transitions, and
receipt hashes.

## Conditions

- `frozen`: equal specialist voting, no adaptive state.
- `adaptive`: bounded Hebbian state and memory updates.
- `organism`: adaptive state plus council routing, distractor handling,
  memory, homeostatic energy gating, and authority gates.

## Episode contract

Every episode contains a multi-step task, conflicting specialist observations,
a changing hazard, a distractor, an energy budget, a fault opportunity, an
authority requirement, and a measurable action outcome.

## Metrics

The JSON report emits task success, uncertainty calibration, useful
disagreement, fault recovery, memory precision, forgetting events, energy,
latency, receipt replay equivalence, and unauthorized-action rate.

## Claim boundary

This is a falsifiable runtime benchmark. It is not evidence that Auro-156K,
Auro-320M, Auro-640M, Auro-1B, Auro-2B, or Auro-3B has a trained checkpoint.
The next adapter is to replace the deterministic specialists with each local
model while preserving the episode seed, receipt schema, and three conditions.

## First-six execution target

The first six lanes must expose the same `generate/score` adapter:

1. Auro-156K — atomic routing/classification seed
2. Auro-320M — atomic triage and retrieval filter
3. Auro-640M — code/evidence specialist
4. Auro-1B — private assistant and tool planner
5. Auro-2B — router and spectral triage
6. Auro-3B — structured coding and planning

Until exact checkpoints exist, each lane may run with the repository's
MESIE/SpectralGPT dev architecture or a declared compatible local checkpoint,
but the report must identify the active lane and checkpoint hash.
