# AURO

**A composable native model family built on the MESIE compute plane.**

AURO is not one model waiting to become useful at 14B. It is a living family of trained checkpoints, active training lanes, specialist atomic models, micro models, and larger orchestration architectures that share one governed runtime.

> Continuity rule: before changing a family claim, inspect checkpoint manifests, training receipts, runtime code, and prior context systems. Architecture names, local checkpoint names, and promoted downloadable releases are related but not interchangeable.

## Family continuity

| Model | Class | Current repository state |
|---|---|---|
| **Auro-156K / HIM-native-v0** | Atomic | Real open-weight reference checkpoint and executable specialization seed; 146,576 measured learned parameters in the checked-in release |
| **Auro-2B** | Micro | Existing model lane with physics, HIM-SFT, specialized, and continual checkpoint paths used by the runtime; exact local Grok-produced weights must be verified from their checkpoint manifest rather than inferred from GitHub because `checkpoints/auro_minds/` is local/gitignored |
| **Auro-4B** | Micro | Active next-model lane already underway: native 4B geometry, structured prewiring, MoE architecture, checkpoint constitution binding, active/stored parameter accounting, and CI receipts are merged; full trained-weight promotion remains checkpoint-specific |
| **Auro-8B** | Core | General reasoning lane and compatible checkpoint-backed inference integrations exist; native-family promotion remains evidence-bound |
| **Auro-14B** | Orchestrator | Training and orchestration target, not represented as a finished promoted 14B checkpoint |
| **Auro-100B** | Frontier | Architecture target only |

### Canonical terminology

- **Atomic:** below 1B parameters. Small, independently specialized units intended to be multiplied and composed.
- **Micro:** 1B to below 5B. Compact standalone models such as Auro-2B and Auro-4B.
- **Core:** 5B to below 10B. General reasoning and synthesis.
- **Orchestrator:** 10B to below 30B. Coordinates atomic, micro, and core councils.
- **Frontier:** 30B and above. Distributed research architecture.

The atomic lane is not a temporary substitute for the larger models. A user can create ten or twenty specialized atomic checkpoints or adapters, keep each lineage separate, and coordinate them through AURO/HIM/NOVA.

Read [`docs/MODEL_FAMILY.md`](docs/MODEL_FAMILY.md).

## Verify the local 2B before making claims

The repository contains the 2B architecture/runtime path and references local checkpoints including:

- `Auro-2B_physics`
- `Auro-2B_him_sft`
- `Auro-2B_specialized`
- `Auro-2B_continual`

Those directories are normally local and gitignored. Run the evidence inventory on the machine that contains Grok's checkpoint work:

```bash
python scripts/inventory_auro_checkpoints.py \
  --root checkpoints/auro_minds \
  --output evidence/local-checkpoint-inventory.json
```

The inventory reports exact weight files, hashes, manifests, 2B candidates, and whether the local evidence bundle is complete. It does not downgrade an existing 2B merely because private/local weights are absent from GitHub.

## Context architecture: two cooperating planes

AURO has **two different context mechanisms**. Neither replaces the other.

### 1. 500k logical context / million-token virtualization

`auro_native_llm.production_fleet.context_engine.ContextEngine` is a persistent SQLite/WAL + FTS5 knowledge plane. It stores overlapping source-tagged chunks, deduplicates documents, ranks by lexical relevance, importance, and recency, and injects only a bounded working set into each model call.

The original public-alpha work supported a configurable **512 to 300,000 retrieved-token budget**, which is why the runtime exposed a **500k logical context** configuration. The store itself was tested above one million logical tokens while injecting a much smaller evidence pack. This is virtualized retrieval, not one transformer Softmax over 500k tokens.

```bash
python -m auro_native_llm.use --colony --colony-germs 40 \
  --colony-context 500000 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Use the persistent logical context bank to answer this task."
```

The context engine keeps separate accounting for:

- total logical tokens retained in the knowledge plane;
- retrieved/injected tokens for the current call;
- source chunk IDs and provenance;
- token budget;
- hash-linked ingest and retrieval receipts.

### 2. 294,912-token governed accepted-context envelope

`AuroLongContextModel` accepts up to **294,912 tokens** into a deterministic governed envelope, selects historical chunks plus the recent tail, records hashes and truncation, and sends a bounded dense working set to the underlying model.

```python
from auro_native_llm import AuroLongContextModel

model = AuroLongContextModel.build(model_id="Auro-2B", mode="dev")
result = model.prepare_context(token_ids, query_token_ids=query_ids)
print(result.receipt)
```

This is not a claim that all 294,912 tokens enter one dense attention operation. It is also not a replacement for the persistent 500k logical context bank. The intended stack is:

```text
persistent logical memory (500k+ / million-token tested)
        -> ranked source-grounded working set
        -> governed 294,912-token accepted envelope
        -> bounded dense MESIE/model attention
```

## Four-atomic specialization experiment

The repository now includes an executable reference framework in `auro_native_llm/atomic_colony.py`.

```python
from auro_native_llm.atomic_colony import AtomicColony

colony = AtomicColony.repository_audit_colony(
    base_checkpoint="checkpoints/open/HIM-native-v0"
)

receipt = colony.run(
    "Audit a repository change for code, evidence, contradictions, and continuity.",
    executor=lambda specialist, task: run_atomic_checkpoint(
        checkpoint=specialist.base_checkpoint,
        system_instruction=specialist.instruction,
        task=task,
        adapter_path=specialist.adapter_path,
    ),
)
```

The reference colony creates four independent identities:

1. **Retriever atom** — locates relevant repository evidence.
2. **Code-reader atom** — reconstructs implementation and dependencies.
3. **Red-team atom** — finds regressions, contradictions, and unsupported claims.
4. **Consolidator atom** — produces the final continuity-preserving result.

The framework emits a deterministic experiment receipt and preserves the shared base-checkpoint lineage. Declaring four roles is executable routing specialization; claiming four weight-specialized models additionally requires four trained checkpoints or adapters with their own manifests and evaluations.

## Auro-4B is already underway

The native 4B lane includes:

- 32 layers;
- 3,072 hidden width;
- 24 query heads and 8 KV heads;
- 10,240 SwiGLU width;
- GQA, RoPE, RMSNorm, and structured residual metadata;
- approximately 4.03B active parameters in the dense-equivalent accounting;
- 8-expert, top-2 MoE policy with larger stored capacity;
- structured prewiring and birth receipts;
- checkpoint-constitution integration;
- architecture and CI gates.

The next 4B work is not “create a 4B architecture.” It is checkpoint production: corpus admission, full training, resumability, tokenizer/checkpoint packaging, evaluations, failure analysis, and promotion evidence.

## Install

```bash
git clone https://github.com/ItsNotAILABS/Auro14B.git
cd Auro14B
python -m pip install -e .
```

Python 3.10+ is required. Optional surfaces use Julia, Node.js, Cloudflare Workers, browser ONNX/WASM, and mobile runtimes.

## Run the local model surface

```bash
export PYTHONPATH=.
python -m auro_native_llm.use \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Explain the AURO model and context architecture."
```

Agentic loop:

```bash
python -m auro_native_llm.use --him --him-germs 20 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Inspect this repository and plan the next training cycle."
```

## Checkpoint release standard

A promoted downloadable checkpoint requires:

1. exact weights and SHA-256 manifest;
2. tokenizer and byte-perfect round-trip audit;
3. architecture/runtime configuration;
4. corpus provenance and training history;
5. checkpoint-specific evaluations and failure samples;
6. clean-install and launch proof;
7. API/local inference smoke receipt;
8. intended-use and limitations model card;
9. signed promotion and rollback evidence.

Quarantined checkpoints are rejected by default. Local/private checkpoints can still exist and be used without being falsely described as a public promoted release.

## Runtime surfaces

| Surface | Location | Purpose |
|---|---|---|
| Native model | `auro_native_llm/` | model, training, specialization, context, checkpoint, and organism runtime |
| Persistent context | `auro_native_llm/production_fleet/context_engine.py` | 500k+ logical memory virtualization and bounded source injection |
| Long-context envelope | `auro_native_llm/context/` | governed 294,912-token accepted context and receipts |
| Browser runtime | `browser-runtime/` | local ONNX/WASM inference with no silent remote fallback |
| Cloudflare operator | `workers/auro-platform/` | governed remote runtime and UI |
| Mobile runtime | `mobile-runtime/` | Expo client and device-sense integration |
| Web3 surface | `him-web3/` | server-held RPC credentials and read-only chain tools |

## Validation

```bash
python -m pytest -q \
  tests/test_context_engine.py \
  tests/test_long_context_envelope.py \
  tests/test_atomic_colony.py \
  tests/test_model_taxonomy.py \
  tests/test_auro4b_architecture.py
```

## Architecture

```text
Atomic checkpoints/adapters (<1B) ─┐
Auro-2B / Auro-4B micro models ────┼── governed colony / NOVA council
Auro-8B core model ─────────────────┘              │
                                                   ├── Auro-14B orchestrator target
persistent 500k+ logical context ── retrieval ─────┤
294,912 accepted-context envelope ─ dense bound ───┤
MESIE + constitutional receipts ───────────────────┘
```

## Project identity

- Organization: [ItsNotAILABS](https://github.com/ItsNotAILABS)
- Repository: [ItsNotAILABS/Auro14B](https://github.com/ItsNotAILABS/Auro14B)
- Compute plane: **MESIE**
- Family: **AURO**
- License: see [`LICENSE`](LICENSE)
