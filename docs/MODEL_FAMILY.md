# AURO Model Family

AURO is a **family of composable native models**, not one model that only becomes useful at 14B. Every released checkpoint carries its own tokenizer, hashes, evaluation receipt, model card, launch command, promotion state, rollback target, and claim boundary.

## Canonical capacity classes

| Class | Parameter range | AURO examples | Primary role |
|---|---:|---|---|
| **Atomic** | below 1B | Auro-156K, Auro-250M, Auro-500M | Specialized intelligence units multiplied, embedded, composed, and deployed near data |
| **Micro** | 1B to below 5B | Auro-2B, Auro-4B | Standalone private assistants, routers, tool users, coders, and atomic-swarm supervisors |
| **Core** | 5B to below 10B | Auro-8B | General reasoning, synthesis, planning, and multi-domain work |
| **Orchestrator** | 10B to below 30B | Auro-14B | Coordinates atomic, micro, and core councils across longer workflows |
| **Frontier** | 30B and above | Auro-100B target | Distributed research-scale architecture |

Capacity is only one routing axis. Every checkpoint also advertises capabilities such as retrieval, code triage, tool planning, evidence review, writing, voice, vision, memory, or orchestration. The scheduler chooses the smallest evidence-backed checkpoint that can perform the role.

## Atomic ladder

### Auro-156K

The checked-in reference/open-weight lane and ultra-small specialization seed. Intended roles include routing, classification, JSON repair, tool selection, and style control. It can be multiplied in high-count swarms and deployed in WASM or embedded environments.

### Auro-250M

The phone/browser atomic lane. Intended roles include intent extraction, retrieval filtering, structured transformation, code triage, memory consolidation, and semantic outlining. It can run independently or as a child of Auro-500M and Auro-2B.

### Auro-500M

The strong edge worker and base for the Auro-2B triad:

- **Auro-500M-SENSUS** — evidence and perception;
- **Auro-500M-PRAXIS** — code, tools, and execution planning;
- **Auro-500M-VERBUM** — language, creative branching, explanation, and conversation.

The three names become distinct model claims only when separate checkpoints or hash-bound trained adapters exist. Role metadata alone remains a routing identity.

## Auro-2B hierarchical runtime

Auro-2B is both a standalone micro model and the parent of a hierarchical generation system:

```text
Auro-2B parent
   ↓
three concurrent Auro-500M specialists
   ↓
topic-scoped Auro-250M and Auro-156K swarms
   ↓
500M specialist reports
   ↓
three-way consensus
   ↓
Auro-2B final synthesis
   ↓
Python/WASM conversational fluidizer
```

MESIE processes the ingress, every specialist and atomic stage, and the final egress. Children receive bounded task capsules rather than the complete parent context. The runtime records estimated transport reduction, model identities, MESIE receipts, contracts, disagreement, and promotion blockers.

Model instances are accounted separately. Auro-2B plus three 500M specialists is a composed runtime, not one 3.5B checkpoint.

See [`AURO_2B_TRIAD_SWARM.md`](AURO_2B_TRIAD_SWARM.md).

## Why atomic

A sub-1B checkpoint is not merely a weaker general model. Its advantage is multiplicity and narrow optimization. A user can operate many atomic checkpoints, each trained for a bounded responsibility, and compose them into a colony, triad, governed council, MoE expert bank, browser worker, or phone agent.

Examples include:

- code review and patch triage;
- document extraction and retrieval filtering;
- routing and tool selection;
- JSON and schema repair;
- repository triage;
- safety classification;
- spectral matching;
- memory consolidation;
- evidence verification;
- language expansion and style control.

Atomic models should be inexpensive to copy, specialize, evaluate, replace, and run close to the data.

## Release ladder

| Model | Class | Repository status boundary |
|---|---|---|
| **Auro-156K / HIM-native-v0** | Atomic | Checked-in reference checkpoint lane; every packaged release still requires exact hashes and evidence |
| **Auro-250M** | Atomic | Architecture and training lane until an exact checkpoint passes mobile, browser, and swarm evaluations |
| **Auro-500M** | Atomic | Architecture and training lane; SENSUS/PRAXIS/VERBUM need distinct specialization evidence |
| **Auro-2B** | Micro | Existing local checkpoint lanes; public release claims require inventory and constitutional promotion of exact artifacts |
| **Auro-4B** | Micro | Native architecture and active checkpoint-production lane; full trained-weight promotion remains checkpoint-specific |
| **Auro-8B** | Core | Architecture and integration lane until a promoted native-family checkpoint exists |
| **Auro-14B** | Orchestrator | Active training/orchestration target; not a finished 14B release without exact evidence |
| **Auro-100B** | Frontier | Architecture target only |

## Context architecture

AURO preserves two cooperating planes:

- the persistent SQLite/WAL/FTS5 logical-memory system, configurable around 500K logical context and exercised above one million retained tokens;
- the governed 294,912-token accepted-context envelope that emits deterministic receipts and reduces the model-facing dense view.

Neither is a claim of 500K or 294K simultaneous dense attention.

## Checkpoint release contract

A downloadable AURO checkpoint is release-ready only when the package contains:

1. exact checkpoint weights and SHA-256 manifest;
2. tokenizer with byte-perfect round-trip evidence;
3. architecture and runtime configuration;
4. training provenance, data safety evidence, and loss history;
5. checkpoint-specific benchmark and failure results;
6. clean-install and launch proof;
7. API, local, mobile, or browser inference receipts as applicable;
8. model card with intended use and limitations;
9. signed promotion authorization and verified rollback evidence.

Family names, architecture arithmetic, successful fixture tests, and agent counts are not checkpoint evidence. Every released variant is evaluated independently.
