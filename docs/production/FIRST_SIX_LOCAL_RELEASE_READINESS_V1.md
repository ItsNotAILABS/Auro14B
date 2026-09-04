# First Six Local Release Readiness v1

Date: 2026-09-04

## Decision

The first six AURO lanes are being made usable as small local intelligence
cells before any large-model promotion:

- Auro-156K
- Auro-320M
- Auro-640M
- Auro-1B
- Auro-2B
- Auro-3B

“Usable” has two separate meanings:

1. **Runtime usable**: the lane has a stable local adapter, bounded context,
   deterministic seed support, JSON output, receipt emission, and can run with
   a mini/SML-compatible backend.
2. **Checkpoint released**: the exact named tensors, tokenizer, hashes,
   provenance, benchmarks, and model card have been published.

The repository currently supports the first category through the native MESIE
dev architecture and family contracts. The second category remains
checkpoint-specific and must not be inferred from the lane name.

## Shared local contract

Every first-six lane must implement:

```text
load(manifest, backend, device)
generate(prompt, context, seed, limits)
score(candidate, task)
snapshot()
receipt()
```

The adapter must report:

- lane ID and runtime mode;
- backend and device;
- active checkpoint identity, or `architecture-scaffold`;
- parameter target versus live parameter count;
- tokenizer identity;
- adaptive-state version;
- memory policy;
- receipt hash.

## SML/mini-LLM profiles

These profiles are deployment envelopes, not claims about final model
parameter counts:

| Profile | Purpose | Required behavior |
|---|---|---|
| `mini-cpu` | laptop/CPU development | bounded sequence, deterministic sampling, JSON receipts |
| `mini-webgpu` | browser/edge | same adapter and receipt schema |
| `sml-quantized` | small local quantized checkpoint | explicit quantization metadata and replay seed |
| `organism-cell` | one specialist in SOVEREIGN | no tool authority by default; council-mediated action |

A lane can be useful as a specialist before it is a strong general assistant.
The first six should prioritize routing, classification, retrieval, coding
triage, structured transformation, tool planning, and memory operations.

## Lane readiness

| Lane | Primary role | Runtime target | Checkpoint evidence |
|---|---|---|---|
| Auro-156K | routing/classification seed | mini-cpu, webgpu, organism-cell | required |
| Auro-320M | triage/retrieval filter | mini-cpu, webgpu, quantized | required |
| Auro-640M | code/evidence specialist | mini-cpu, quantized, organism-cell | required |
| Auro-1B | private assistant/tool planner | mini-cpu, quantized | required |
| Auro-2B | router/spectral triage | mini-cpu, quantized, organism-cell | required |
| Auro-3B | structured coding/planning | mini-cpu, quantized | required |

## Acceptance gate for each lane

A lane is marked `runnable-dev` only after:

- local load succeeds without network access;
- one prompt produces structured output;
- deterministic seed produces the same receipt;
- empty/malformed input is rejected safely;
- context limit is enforced;
- adaptive state can be disabled;
- memory can be exported and deleted;
- unauthorized tools are denied;
- SOVEREIGN Organism Trial adapter passes;
- CPU or WebGPU latency and memory are recorded.

A lane is marked `promoted-checkpoint` only after the full AURO promotion
contract: exact weights, tokenizer, SHA-256 manifest, training provenance,
benchmarks, failures, model card, license review, and rollback evidence.

## Why this matters

The first six are the multiplicity layer of SOVEREIGN. They should be cheap to
copy, specialize, replace, and run close to private data. Auro-14B does not
need to be present for this layer to provide value. The organism can operate
as a council of small cells, with SOVEREIGN routing, memory, MESIE signals,
homeostasis, and authority controls around them.

The SOVEREIGN Organism Trial is the common falsifiable test for all six lanes.
