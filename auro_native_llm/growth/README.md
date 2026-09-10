# Auro exponential growth ("career ladder") — mechanics

This package implements the **mechanics** of progressive growth for the Auro
SpectralGPT core (RMSNorm, RoPE, GQA attention with per-head QK-norm, SwiGLU
FFN, soft-routed MoE on selected layers, tied embeddings):

- `core.py` — minimal NumPy autograd + the trainable core transformer +
  exact parameter counting + Adam. Gradient-checked against finite
  differences (12 probes, max relative error 3.3e-05).
- `operators.py` — Net2Net-style **function-preserving** growth operators:
  - `grow_width`: double attention heads (GQA-aware) + QK-norm gains + SwiGLU
    width at fixed model dim (neuron splitting / head duplication with
    halved output projections).
  - `grow_depth`: double the layer count by stacking copies whose residual
    branches (`W_o`, `W_d`, expert `W_d`) are zero-initialized, so each new
    block is the identity at growth time.
  - `grow_moe`: convert a dense FFN into a 2-expert MoE (router zero-init →
    uniform routing → identical output) or duplicate experts of an existing
    MoE layer (router columns duplicated → probability mass halves → identical
    mixture).
- `schedule.py` — exponential milestone schedule (2x per rung). The 2B-lane
  destination uses the published honest numbers (total 4,251,025,408 /
  active 1,497,464,832): **15 doublings from 156K**. `plan_growth` picks the
  operator combo landing closest to each 2x target, because exact doubling
  is not always reachable (embeddings don't grow); ideal vs. achieved sizes
  are both logged.
- `career.py` — the runner: **train → checkpoint-as-new-model → grow →
  continue**. Every rung is saved under `checkpoints/rung-N/` as a new model
  (`weights.npz`, `config.json`, `rung.json` with parent/growth provenance,
  signed `receipt.json`). Supports `--resume`: continues from the latest
  *completed* rung instead of restarting, so runs compound.
- `receipts.py` — signed rung receipts (canonical JSON + sha256), following
  the repo's receipt convention.

## What this is and is not

**Is:** a verified implementation of function-preserving growth mechanics —
the operators provably (and numerically, max|diff| ≤ 1.8e-07) preserve the
model function at growth time, the schedule compounds exponentially, and a
real training loop shows loss continuing to decrease after each growth step
on synthetic data.

**Is not:** a claim that growth yields capability or intelligence. Loss on
synthetic data measures *optimization mechanics* — that the grown model
keeps learning — nothing more. "Grows 2x per milestone" describes parameter
mass, not competence.

## Compute reality

Real training to 2B parameters needs a real training budget: data at
trillion-token scale, distributed optimization, and eval infrastructure none
of which exists in this package. What remains for a real career ladder:

1. **The curriculum** — what data, in what order, with what done-criteria
   per rung (`RungCriteria.loss_target` is intentionally unset; the
   mechanics don't get to define "done").
2. **Hard-routed MoE** — this core uses soft routing (differentiable,
   simple); production uses top-k hard routing, whose growth operators need
   capacity-factor handling.
3. **The extended architecture** — the full MESIE model adds spectral and
   cross-modal modules; production growth extends these operators to them.
4. **Scale engineering** — the NumPy core here is a mechanics reference, not
   a training system. The ladder's destination is the real trainer.

## Open items log

- **Fixed 2026-09-10:** schedule overshoot — `rung_targets` emitted a rung
  past the 2B target then appended the target after it (non-monotonic, 17
  rungs vs `num_rungs() == 15`); now capped at the target, monotonic,
  `len == num_rungs + 1`, 16 rungs / 15 doublings as the README states.
- **Fixed 2026-09-10:** MoE router duplication — `grow_moe` on an existing
  MoE layer duplicated router *rows* `(d, E) -> (2d, E)` (forward crash);
  now duplicates *columns* `(d, E) -> (d, 2E)`, which halves each pair's
  probability mass and preserves the mixture (halving the *weights* would
  be wrong — softmax is not scale-invariant). Dense→MoE conversion also now
  creates `cfg.experts` experts instead of a hardcoded 2, fixing a latent
  KeyError on configs that already had E > 2.
- **Fixed 2026-09-10:** Auro-156K checkpoint integration — rung-0 init now
  discovers a real 156K checkpoint (`$AURO_156K_CHECKPOINT_DIR`,
  `<repo>/checkpoints/auro-156k/`) with strict key/shape validation, and
  falls back to random init with a clear log line when absent (it is absent:
  no trained 156K checkpoint ships with the repo).
- **Added 2026-09-10:** resume/compounding — `--resume` continues from the
  latest completed rung; runs compound instead of restarting.
- **Fixed 2026-09-10:** training memory growth — the NumPy autograd graph is
  a web of reference cycles (~0.5GB of transient graph per step at rung 2)
  that CPython's automatic cyclic GC could not keep up with, so RSS grew
  without bound and the OOM killer silently SIGKILLed rung-2 training (the
  earlier pipe through `tail` masked the real exit code). `train_rung` now
  collects the per-step graph explicitly; RSS stays flat and the fix is
  numerically a no-op.
- **Still open:** the coding milestone's RL trainer around
  `ModuleTask.evaluate` (loop primitive exists, trainer does not);
  `run_powershell`'s real execution path is untested on machines without
  PowerShell (only graceful degradation is covered).

## Run the mechanics demo

```bash
cd /path/to/Auro14B
~/workspace/venvs/itsnotai/bin/python -m auro_native_llm.growth.career \
    --growths 2 --steps 250 --batch 32 --out auro_native_llm/growth/checkpoints
```

Trains rung-0 (~115K params, "156K-scale"), grows twice via the scheduled
plans, and checkpoints each rung as a new model with a signed receipt.

### Resume: runs compound

```bash
~/workspace/venvs/itsnotai/bin/python -m auro_native_llm.growth.career \
    --resume --growths 1 --out auro_native_llm/growth/checkpoints
```

`--resume` scans `--out` for the latest *completed* rung (weights + config +
signed receipt; interrupted rungs without a receipt are ignored), loads it,
grows first — exactly as the loop's growth block would — and continues the
ladder from there. `--growths` then counts *additional* growths, rung
numbering continues, earlier checkpoints are left byte-identical, and each
new rung's receipt chains back to the resumed rung. Every run is better than
the last; nothing restarts from scratch.

### Rung-0 initialization: the Auro-156K checkpoint

`career.py` tries to initialize rung 0 from a real Auro-156K checkpoint
before falling back to random init. Honest status: **no trained Auro-156K
checkpoint ships with this repo** — `native_llm/configs/family/auro_156k.json`
is an architecture target (`status: architecture-target-not-trained-checkpoint`),
and the only real trained weights in the repo are HIM-native-v0 (~147K, a
different model). When a real 156K checkpoint exists, drop it at one of:

1. `$AURO_156K_CHECKPOINT_DIR/` (env override, checked first),
2. `<repo>/checkpoints/auro-156k/`,
3. `auro_native_llm/growth/checkpoints/auro-156k/`,

as `weights.npz` (growth-core key layout) + `config.json` (growth-core
schema, or the repo's `auro.native_llm.model_config` schema — mapped
best-effort with documented assumptions). Every key and shape is validated;
a mismatch raises a clear error and the run falls back to random init with
a logged warning. `--no-156k-init` skips discovery entirely. The loader never
fabricates weights.

## Tests

```bash
~/workspace/venvs/itsnotai/bin/python -m pytest tests/test_auro_growth.py -v
```

Covers: function preservation per operator (and chained), schedule math
(15 doublings 156K → 2B), param-count agreement, checkpoint round-trip,
and a tiny end-to-end train→grow→train mechanics run.

## Curriculum: the five training domains

`curriculum.py` defines the domain milestones that ride the ladder. Each
milestone has explicit done-criteria (metric, threshold, eval set, minimum
rungs); a rung may carry several active milestones at once.

| Milestone | Rungs | Done-criteria | Data |
|---|---|---|---|
| General mathematics | 0–4 | accuracy ≥ 0.85 on 1000 held-out | **synthetic-v1** |
| Ancient geometry (Euclid, phi) | 0–5 | accuracy ≥ 0.85 on 1000 held-out | **synthetic-v1** |
| Physics (kinematics, F=ma, units) | 3–8 | accuracy ≥ 0.80 on 1000 held-out | **synthetic-v1** |
| Architecture (proportion, bays, arches) | 5–10 | accuracy ≥ 0.80 on 1000 held-out | **synthetic-v1** |
| Coding with execution (tool use) | 8–14 | pass rate ≥ 0.70 on 200 sandbox-executed module tasks | sandbox-interactive-v1 |

**Synthetic labeling, stated plainly:** the math/physics/geometry/architecture
problem generators are synthetic (`synthetic-v1`). They exist for *mechanics
verification* — proving the curriculum machinery works: problems generate,
verifiers accept true answers and reject wrong ones, gates open and close.
They are NOT training data for a real model. Real training needs real
corpora plus a real tokenizer; the gate *shape* (metric, threshold, eval
set) is designed to survive that swap.

## Tool use: execution feedback in the loop

The coding milestone is **not static code text**. `exec_sandbox.py` embeds
real runtimes in the training loop:

- `run_python(code)` / `run_powershell(code)` — subprocess execution with
  timeout, stdout/stderr capture, and output caps. PowerShell degrades
  gracefully (`available=False`) where no runtime is installed.
- `ModuleTask` — a whole module with a `# __AURO_HOLE__` marker plus its
  tests. The model completes/extends/debugs the module; `evaluate()` runs
  the completion against the tests in the sandbox and reports pass/fail
  with the observed output. The gate judges *behavior*, not form.

The loop: sample a module task at the rung's difficulty → model generates
a completion → sandbox runs it → reward = test pass rate → the observed
stdout/stderr becomes the feedback signal for the next attempt. The data
is generated by interaction, not scraped as text.

**Isolation honesty:** the sandbox here is process-level isolation (fresh
temp dir, timeout, output caps) — loop mechanics, not a security boundary.
Real training runs untrusted model-generated code inside proper jails
(containers/seccomp/gVisor, no network). Never point this module at
adversarial code without that layer.

## Tests

```bash
~/workspace/venvs/itsnotai/bin/python -m pytest tests/test_auro_curriculum.py -v
```

Covers: generators produce verifiable problems (true answers pass, wrong
answers fail), milestone coverage of rungs 0–14 with done-criteria on every
milestone, sandbox execution/output-capture/timeout behavior, PowerShell
graceful degradation, and module-task judging (reference completion passes,
broken completion fails).
