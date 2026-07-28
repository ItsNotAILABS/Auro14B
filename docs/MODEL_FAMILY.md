# AURO Model Family

AURO is a **family of composable native models**, not one model that only becomes useful at 14B.

The family is designed to release useful checkpoints throughout the training ladder. Every released checkpoint must carry its own tokenizer, hashes, evaluation receipt, model card, launch command and claim boundary.

## Canonical classes

| Class | Parameter range | AURO examples | Primary role |
|---|---:|---|---|
| **Atomic** | below 1B | Auro-156K and future specialized sub-1B checkpoints | Single-purpose intelligence units that can be multiplied, embedded and composed |
| **Micro** | 1B to below 5B | Auro-2B, Auro-4B | Standalone private assistants, routers, tool users, coders and domain specialists |
| **Core** | 5B to below 10B | Auro-8B | General reasoning, synthesis and planning |
| **Orchestrator** | 10B to below 30B | Auro-14B | Coordinates atomic and micro-model councils and longer workflows |
| **Frontier** | 30B and above | Auro-100B architecture target | Distributed research-scale architecture |

## Why `atomic`

A sub-1B checkpoint is not merely a smaller general model. Its design advantage is **multiplicity**. A user can run ten or twenty atomic AURO models, each trained for a narrow responsibility, and compose them into a colony or governed council.

Examples include:

- code review
- document extraction
- routing
- JSON repair
- repository triage
- safety classification
- spectral matching
- memory consolidation
- tool selection
- domain-specific terminology

Atomic models should be inexpensive to copy, specialize, evaluate, replace and run close to the data.

## Why `micro`

The 2B and 4B lanes are micro models. They remain compact enough for local and private deployment, but they can operate independently instead of only as one narrow component.

- **Auro-2B:** routing, tool use, spectral triage, private local assistant
- **Auro-4B:** coding, structured output, specialist planning, atomic-colony supervision

## Release ladder

| Model | Class | Repository status boundary |
|---|---|---|
| **Auro-156K** | Atomic | Reference architecture/checkpoint lane; downloadable claims require the exact weights and evidence bundle |
| **Auro-2B** | Micro | Checkpoint-specific evidence required before release claims |
| **Auro-4B** | Micro | Checkpoint-specific evidence required before release claims |
| **Auro-8B** | Core | Architecture/training target until a promoted checkpoint exists |
| **Auro-14B** | Orchestrator | Active training target; not a finished 14B checkpoint without promotion evidence |
| **Auro-100B** | Frontier | Architecture target only |

## Context architecture

AURO exposes a **294,912-token governed accepted-context envelope**. It combines deterministic chunking, hashes, salience retrieval, recency retention and a bounded dense MESIE working window.

This is not a claim that all 294,912 tokens enter one dense Softmax operation. Receipts distinguish accepted, retrieved, attended and truncated tokens.

## Release contract

A downloadable AURO checkpoint is considered release-ready only when the package contains:

1. exact checkpoint weights and SHA-256 manifest;
2. tokenizer with byte-perfect round-trip evidence;
3. architecture and runtime config;
4. training provenance and loss history;
5. checkpoint-specific benchmark and failure results;
6. clean-install and launch proof;
7. API or local inference smoke receipt;
8. model card with intended use and limitations;
9. promotion authorization and rollback evidence.

Family names are not checkpoint evidence. Every released variant is evaluated independently.
