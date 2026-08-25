# AURO Sub-2B Release Train v1

## Purpose

This release train turns the existing AURO architecture ladder into an evidence-bound operator program for shipping the family through Auro-2B:

```text
Auro-156K
Auro-250M
Auro-500M
Auro-500M-SENSUS
Auro-500M-PRAXIS
Auro-500M-VERBUM
Auro-2B
```

The train does not equate a model name, configuration file, training command, output directory, or plausible weight file with a released checkpoint. Each identity must independently establish custody, provenance, evaluation, promotion, packaging, and rollback evidence.

## Current implementation surfaces

| Surface | Responsibility |
|---|---|
| `scripts/inventory_auro_checkpoints.py` | Audits local checkpoint candidates without trusting directory names |
| `auro_native_llm/release_train.py` | Converts observed evidence state into an ordered release plan |
| `scripts/build_sub2b_release_train.py` | Operator CLI for inventory plus plan generation |
| `auro_native_llm/model/train.py` | Fresh AURO family training entrypoint with explicit corpus controls |
| `scripts/checkpoint_promotion_gate.py` | Joins checkpoint verification, evidence artifacts, readiness, and promotion |
| `auro_native_llm/substrate/checkpoint_constitution.py` | Hashes checkpoint files and signs authorized promotion manifests |

## Seven independent release identities

### Base atomic lanes

- **Auro-156K** - routing seed, classification, JSON repair, tool selection, and high-multiplicity swarm cells.
- **Auro-250M** - phone/browser atomic expert for intent, retrieval filtering, structured transformation, code triage, memory consolidation, and semantic outlining.
- **Auro-500M** - edge worker and embedded specialist base for planning, code, evidence review, expansion, and consensus.

### Specialist triad

- **Auro-500M-SENSUS** - evidence, retrieval, factual review, context, and risk.
- **Auro-500M-PRAXIS** - code, tools, builds, debugging, and workflow execution.
- **Auro-500M-VERBUM** - writing, creativity, explanation, synthesis, and conversation.

A specialist name is not evidence of a separately trained model. Each specialist must provide either:

1. a distinct checkpoint and manifest; or
2. a promoted Auro-500M base checkpoint plus a versioned adapter whose files, training provenance, evaluation, and promotion are independently hash-bound.

### Parent lane

- **Auro-2B** - parent coordinator for the three 500M specialists, dynamically activated 250M/156K swarms, MESIE compute, tools, and final structured synthesis.

The composed council is not one 3.5B checkpoint. Every loaded checkpoint and adapter retains its own identity and measured parameter accounting.

## Evidence states

The inventory reports these states independently:

```text
artifact_present
manifest_present
tokenizer_custody
weight_hash_agreement
geometry_verified
integrity_verified
training_provenance_verified
evaluation_verified
promotion_signature_present
promotion_signature_verified
signed_promotion
promotion_ready
```

A 64-character string is no longer considered a valid promotion signature. Promotion is accepted only when the configured operator key verifies the HMAC over the exact manifest payload.

## Required checkpoint contents

A production candidate must include, directly or through a manifest-bound release directory:

- weight artifacts;
- tokenizer artifacts and immutable ID custody;
- architecture/configuration geometry;
- measured parameter count and parameter target;
- exact checkpoint ID and parent lineage;
- corpus and data-mixture manifest;
- signed training or adaptation receipt;
- training history and failure record;
- exact-checkpoint evaluation artifacts;
- product/API/browser execution smokes;
- model card and known limitations;
- rollback target and rollback validation;
- constitutional promotion manifest signed by the authorized operator.

## Release phases

### Phase 0 - Inventory and recovery

Run the audit against the actual private/local checkpoint root:

```bash
python scripts/inventory_auro_checkpoints.py \
  --root checkpoints/auro_minds \
  --output artifacts/sub2b-release/inventory.json
```

This identifies reusable weights, invalid or incomplete manifests, missing tokenizer custody, missing evaluations, and unverified promotion claims.

### Phase 1 - Corpus and tokenizer custody

Before training:

- freeze source and licensing records;
- deduplicate at document and semantic levels;
- record contamination controls;
- define train/validation/test boundaries;
- hash the final mixture manifest;
- audit tokenizer compression, byte behavior, control-token stability, and special-token ownership;
- preserve one tokenizer lineage across base and specialist variants unless an explicit compatibility migration is approved.

### Phase 2 - Train base lanes

Train Auro-156K, Auro-250M, and Auro-500M as independent exact checkpoints. Auro-2B is trained only after its intended atomic dependencies and evaluation contracts are fixed.

The current generic trainer supports fresh training and explicit corpus selection:

```bash
python -m auro_native_llm.model.train \
  --model Auro-250M \
  --mode full \
  --corpus-root /path/to/approved/corpus \
  --steps APPROVED_STEPS \
  --batch-size APPROVED_BATCH \
  --seq-len APPROVED_SEQUENCE_LENGTH \
  --vocab-size APPROVED_VOCAB_SIZE \
  --output-dir checkpoints/auro_release_candidates
```

The current generic trainer does **not** prove resume or distillation lineage. A fresh run is valid only when it is declared as such. Resume, teacher distillation, and adapter training need explicit entrypoints and receipts before they can be claimed.

### Phase 3 - Train the specialist triad

Create distinct SENSUS, PRAXIS, and VERBUM adapters or checkpoints from the promoted Auro-500M base. The adapter trainer must record:

- base checkpoint manifest hash;
- adapter dataset manifest;
- tokenizer compatibility;
- training command and environment;
- adapter files and hashes;
- specialist benchmark results;
- cross-specialist interference tests;
- rollback to the base model.

The current release plan intentionally marks the adapter trainer as missing rather than silently treating role prompts as trained specialists.

### Phase 4 - Exact-checkpoint evaluation

Each candidate receives model-only evaluation. The composed Auro-2B council additionally receives system evaluation.

Minimum evaluation families:

- perplexity and held-out loss;
- instruction following;
- reasoning and mathematics;
- code generation and executable repair;
- structured output validity;
- retrieval precision and citation grounding;
- tool selection and safe failure recovery;
- multi-turn conversation and continuity;
- creativity under constraints;
- mobile/browser memory and latency for atomic lanes;
- MoE routing balance and expert collapse;
- council ablations: 2B alone, one specialist, full triad, triad plus atomic swarm, triad plus MESIE;
- regression against the parent checkpoint and protected capabilities.

All results must identify the exact checkpoint and tokenizer hashes.

### Phase 5 - Human-authorized promotion

Run the exact checkpoint promotion gate:

```bash
python scripts/checkpoint_promotion_gate.py \
  CHECKPOINT_DIRECTORY \
  RELEASE_EVIDENCE.json \
  --output PROMOTION_RESULT.json
```

The promotion key must remain outside source control. Promotion is rejected when evidence is missing, hashes disagree, the checkpoint differs from the evidence manifest, or rollback is unverified.

### Phase 6 - Ship through 2B

The complete family release is ready only when:

```bash
python scripts/inventory_auro_checkpoints.py \
  --root RELEASE_CHECKPOINT_ROOT \
  --require-ship-through-2b
```

passes with the operator verification key available.

Packaging must include:

- full and quantized weights where supported;
- tokenizer and configuration;
- model card and license;
- SHA-256 manifest;
- training and evaluation receipts;
- promotion and rollback evidence;
- inference examples;
- hardware profiles;
- API compatibility;
- release notes and known limitations.

## Plan generation

Generate the current release train without executing it:

```bash
python scripts/build_sub2b_release_train.py \
  --checkpoint-root checkpoints/auro_minds \
  --corpus-manifest artifacts/corpus/manifest.json \
  --tokenizer-manifest artifacts/tokenizer/manifest.json \
  --output artifacts/sub2b-release/release-train.json
```

The plan includes:

- current evidence state per model;
- next action;
- prerequisites;
- explicit blockers;
- unapproved command templates;
- required output evidence;
- deterministic plan hash.

## Truth boundaries

The release train enforces the following:

- architecture configuration is not a trained checkpoint;
- a generated training command is not executed training;
- a successful process is not a promoted checkpoint;
- a model name is not checkpoint identity;
- a specialist prompt is not a trained adapter;
- a directory is not custody evidence;
- a local unverified signature is not authorization;
- a synthetic benchmark fixture is not official model quality;
- browser WebGPU source is not physical GPU execution;
- shipping the parent does not imply that its submodels are independently released.

## Immediate completion gates

1. Recover and audit every existing local checkpoint candidate.
2. Produce hash-bound corpus and tokenizer manifests.
3. Implement resume/distillation and specialist-adapter trainers with durable job receipts.
4. Execute bounded training on approved compute.
5. Run exact-checkpoint and council evaluations.
6. Promote each identity with a verified operator signature and rollback target.
7. Package and publish the seven-identity release only when the complete inventory gate passes.
