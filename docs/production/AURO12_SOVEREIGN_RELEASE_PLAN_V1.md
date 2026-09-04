# AURO × SOVEREIGN Twelve-Model Local Release Plan — V1

Status: release architecture
Date: 2026-09-04
Source family: ItsNotAILABS/Auro14B
Adaptive source: FreddyCreates/sovereign

## Release objective

Release a family of twelve local model lanes with exact model/config identity, tokenizer and checkpoint hashes, local launch path, benchmark and failure evidence, explicit parameter-count status, runtime memory/context contract, optional Hebbian and dynamic routing state, reproducible promotion, and rollback.

The family is a set of composable local models, not twelve claims that every lane is a fully trained foundation model.

## The weight distinction

### Base checkpoint weights

The trained model parameters. These are immutable for a versioned release and must have exact SHA-256 manifests.

### Runtime synaptic weights

Hebbian/LTP-LTD association values used by the orchestration or memory layer. These can adapt after outcomes, but they are not silently presented as changes to the base neural checkpoint.

### Dynamic routing weights

Per-request or per-task weights used to select specialists, fuse evidence, allocate attention, or choose a model lane. They may adapt during operation and must be logged in receipts.

This distinction is essential for reproducibility:

base checkpoint + tokenizer + runtime version + adaptive state + input = reproducible inference context

## Twelve release lanes

| Lane | Class | Primary role | Adaptive state |
| --- | --- | --- | --- |
| Auro-156K | Atomic | JSON repair, routing, extraction, terminology | optional |
| Auro-320M | Atomic | compact classifier and safety triage | optional |
| Auro-640M | Atomic | code/repository analysis specialist | optional |
| Auro-1B | Micro | private assistant and tool use | optional |
| Auro-2B | Micro | local routing, context retrieval, spectral triage | optional |
| Auro-3B | Micro | structured coding and planning | optional |
| Auro-4B | Micro | specialist council supervisor | optional |
| Auro-6B | Core | general reasoning and synthesis | optional |
| Auro-8B | Core | stronger general mission/agent model | optional |
| Auro-10B | Orchestrator | multi-model planning and evidence fusion | optional |
| Auro-14B | Orchestrator | HIM/Auro family coordination target | optional |
| Sovereign | Orchestrator/organism | strong governed council and adaptive intelligence layer | Hebbian + dynamic |

Names and sizes above are release slots until each exact checkpoint is inventoried and promoted. No lane should publish a parameter-count claim until its checkpoint evidence exists.

## Sovereign release position

Sovereign should be released as a strong system model, not only as a single weight file:

- local model or model adapter;
- multi-agent council;
- Hebbian memory;
- homeostatic exploration/exploitation;
- dynamic routing;
- policy and governance;
- receipts and state registry;
- polyglot engine interfaces.

Its strength should be demonstrated through measured tasks: reasoning accuracy, tool-use success, code repair, long-context retrieval, contradiction detection, adaptation after feedback, recovery after component loss, energy and latency per result, and reproducibility with and without adaptive state.

The phrase strong AI should describe capability evidence, not imply human-level general intelligence or consciousness.

## Adaptive learning contract

Start with the Sovereign adaptive loop:

outcome -> learning signal -> embedding update -> homeostasis -> next decision

For each update record:

- model lane;
- base checkpoint hash;
- adaptive-state version;
- pre-update state digest;
- outcome quality;
- novelty;
- learning rate;
- Hebbian/LTP-LTD delta;
- post-update state digest;
- rollback reference.

The first production rule should be append-only adaptive state with checkpoints. Never mutate the only copy of a model's learned state.

## Twelve-model operating modes

### Solo

One local model answers within a bounded context and budget.

### Specialist chain

Small models perform extraction, classification, spectral analysis, retrieval, planning, and verification in sequence.

### Council

Multiple models produce independent outputs; a fusion or verifier model compares evidence and disagreement.

### Sovereign orchestration

Sovereign selects lanes, routes context, applies policy, monitors adaptation, and emits the final answer plus receipts.

### Space mission mode

The family runs at the edge:

- atomic models on robots or sensors;
- micro models on rover or habitat computers;
- core model on local mission control;
- orchestrator on habitat or orbit;
- Earth receives semantic summaries and evidence bundles.

## Required artifacts per lane

Every public lane needs a model card, exact checkpoint or explicit architecture-only status, SHA-256 manifest, tokenizer files and round-trip test, architecture/config, training provenance, benchmark results, failure and limitation report, local clean-start instructions, runtime health endpoint, adaptive-state schema if enabled, license and attribution, promotion decision, and rollback artifact.

## Promotion gates

A lane is not promoted merely because its app loads.

Promotion requires exact artifact inventory, clean local load, deterministic smoke test, benchmark receipt, memory/context test, failure injection, safety and privacy review, adaptive-state replay if enabled, and no unsupported claims in README, UI, or model card.

## Initial implementation sequence

1. Inventory every actual Auro checkpoint and hash.
2. Create the twelve-lane manifest with status planned, architecture, candidate, or promoted.
3. Port Sovereign Hebbian and homeostasis interfaces into a provider-neutral runtime contract.
4. Define adaptive-state serialization and replay.
5. Benchmark an atomic lane, micro lane, core lane, Auro14B, and Sovereign.
6. Build the local council runner.
7. Publish model cards and receipts.
8. Promote lanes independently.

## Immediate next deliverable

Create a release manifest with all twelve lanes, but mark every lane based on evidence:

- promoted: exact verified weights and passing gates;
- candidate: weights exist but gates incomplete;
- architecture: design exists but weights are not verified;
- planned: release slot only.

This lets us move quickly without confusing a family roadmap with trained-model evidence.

## Source lineage

- Auro14B: https://github.com/ItsNotAILABS/Auro14B
- Auro model family: https://github.com/ItsNotAILABS/Auro14B/blob/main/docs/MODEL_FAMILY.md
- Auro product charter: https://github.com/ItsNotAILABS/Auro14B/blob/main/PRODUCT.md
- Sovereign: https://github.com/FreddyCreates/sovereign
- Sovereign adaptive intelligence: https://github.com/FreddyCreates/sovereign/blob/main/ADAPTIVE_INTELLIGENCE_IMPLEMENTATION.md
- Sovereign Hebbian decay model: https://github.com/FreddyCreates/sovereign/blob/main/src/frontend/src/models/b4/HebbianDecayModel.ts
