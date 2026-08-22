# AURO Model Family

**Composable native model family and governed checkpoint/orchestration architecture on the MESIE compute plane.**

AURO is a family, not one checkpoint. Atomic specialists, micro models, core models and larger orchestration lanes share runtime, context, checkpoint and evidence conventions while remaining separately versioned artifacts.

## Family map

| Model | Class | Role |
|---|---|---|
| **Auro-156K / HIM-native-v0** | Atomic | specialization seed and small independently composable model |
| **Auro-2B** | Micro | compact model/runtime lane with multiple specialization paths |
| **Auro-4B** | Micro | native larger micro-model architecture and training lane |
| **Auro-8B** | Core | general reasoning/synthesis lane |
| **Auro-14B** | Orchestrator | larger coordination/training target |
| **Auro-100B** | Frontier | distributed architecture target |

Canonical classes:

```text
Atomic        < 1B
Micro         1B – <5B
Core          5B – <10B
Orchestrator  10B – <30B
Frontier      30B+
```

## Checkpoint inventory

Model architecture, local/private checkpoints and promoted release artifacts are tracked separately. Inventory local checkpoint evidence before selecting a runtime artifact:

```bash
python scripts/inventory_auro_checkpoints.py \
  --root checkpoints/auro_minds \
  --output evidence/local-checkpoint-inventory.json
```

This produces a concrete inventory of weight files, hashes and manifests available on the current machine.

## Context architecture

AURO supports two complementary context planes.

### Persistent logical context

`ContextEngine` uses a persistent SQLite/WAL + FTS knowledge plane to retain a large logical corpus while injecting only a bounded working set into a model call.

```bash
python -m auro_native_llm.use --colony --colony-germs 40 \
  --colony-context 500000 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Use the persistent context bank for this task."
```

### Governed accepted-context envelope

Long-context runtimes can accept a larger logical envelope, select relevant historical chunks plus recent context, record hashes/truncation and send a bounded dense working set to the underlying model.

The important operational distinction is:

```text
logical context retained != dense tokens sent to one transformer call
```

## Model orchestration

AURO supports specialist composition rather than requiring every capability to live inside one monolithic checkpoint.

```text
request
  -> capability/context analysis
  -> model/specialist selection
  -> bounded inference
  -> optional specialist council
  -> evaluation
  -> artifact/receipt
```

This allows many small specialists to remain individually trainable and traceable while a higher-level lane coordinates them.

## NEXUS federation

[`ecosystem.surface.json`](ecosystem.surface.json) declares the model-family surface to NEXUS.

Primary integration actions include:

```text
family.describe
checkpoint.inventory
context.retrieve
model.infer
council.plan
release.evidence
```

Typical inputs:

```text
nexus.task.v1
nexus.context-pack.v1
nexus.budget.v1
nexus.policy-decision.v1
```

Typical outputs:

```text
nexus.health.v1
nexus.telemetry.v1
nexus.artifact.v1
nexus.release-evidence.v1
```

## Production model lifecycle

```text
architecture
 -> tokenizer/data constitution
 -> train/fine-tune
 -> checkpoint
 -> inventory + hashes
 -> benchmark
 -> compatibility gate
 -> package
 -> runtime selection
 -> telemetry/evaluation
 -> promoted release
```

Every promoted checkpoint should have a model/config identity, weight hashes, tokenizer/config reference, training/evaluation metadata and a reproducible load path.

## Verification

Use repository-specific training/runtime tests for the lane being changed, then run the checkpoint inventory and NEXUS compatibility checks.

From NEXUS:

```bash
python tools/validate_ecosystem_protocols.py
python tools/validate_ecosystem_registry.py
python tools/production_gate.py
```

## Documentation

- [`docs/MODEL_FAMILY.md`](docs/MODEL_FAMILY.md) — family structure and naming
- checkpoint/evidence directories — model-specific manifests and receipts
- runtime/source directories — inference/training/context implementation

## Ecosystem

- [AURO / MESIE Runtime](https://github.com/ItsNotAILABS/AURO) — provider-neutral runtime surface
- [NEXUS](https://github.com/ItsNotAILABS/nexus) — protocols and routing
- [POCKET](https://github.com/ItsNotAILABS/pocket) — user/tenant/product host
- [POCKET Agent](https://github.com/ItsNotAILABS/pocket-agent) — long-running model consumer/orchestrator
- [Medina Memory](https://github.com/ItsNotAILABS/MedinaMemorySystems) — durable context and outcomes
- [MatDaemon](https://github.com/ItsNotAILABS/MatDaemon) — numerical compute

AURO is built around a simple production rule: **model families can be composable, but checkpoints, context and benchmark evidence remain explicit.**
