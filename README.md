# AURO

**A composable native model family built on the MESIE compute plane.**

AURO is not one model waiting to become useful at 14B. It is a release ladder of specialized **atomic**, **micro**, **core**, and **orchestrator** models that can run independently or combine into governed colonies and councils.

> Repository boundary: model names describe family lanes and architecture targets. A trained checkpoint is marketed only when its exact weights, tokenizer, hashes, evaluation receipt and promotion evidence are present.

## Family at a glance

| Model | Class | Intended role | Current claim boundary |
|---|---|---|---|
| **Auro-156K** | Atomic | Specialization seed, embedded agent, colony unit | Reference atomic lane; exact downloadable checkpoint claims require its evidence bundle |
| **Auro-2B** | Micro | Router, tool user, spectral triage, private assistant | Checkpoint-specific evidence required |
| **Auro-4B** | Micro | Coding, structured output, specialist planning, colony supervisor | Checkpoint-specific evidence required |
| **Auro-8B** | Core | General reasoning, synthesis and planning | Architecture/training target |
| **Auro-14B** | Orchestrator | Multi-model coordination and council chair | Active training target; not represented as a finished 14B checkpoint |
| **Auro-100B** | Frontier | Research-scale distributed architecture | Architecture target only |

### Canonical terminology

- **Atomic:** below 1B parameters. Small, replaceable, highly specialized units designed to be multiplied and composed.
- **Micro:** 1B to below 5B. Compact standalone models for private assistants, tools, coding and domain work.
- **Core:** 5B to below 10B. General reasoning and synthesis.
- **Orchestrator:** 10B to below 30B. Coordinates atomic and micro-model councils.
- **Frontier:** 30B and above. Distributed research architecture.

A user should be able to download ten or twenty atomic AURO models, specialize them for different responsibilities, and run them as one governed system. The small-model lane is a primary product strategy, not a temporary compromise.

Read the full family contract in [`docs/MODEL_FAMILY.md`](docs/MODEL_FAMILY.md).

## What is implemented

- MESIE-native causal language-model family and training surfaces
- Mixture-of-experts family policy
- Atomic Auro-156K configuration lane
- 2B, 4B, 8B, 14B and 100B architecture lanes
- checkpoint constitution, quarantine and signed promotion evidence
- tokenizer, training, generation and checkpoint APIs
- multi-agent/colony runtime surfaces
- governed browser, mobile and Cloudflare runtime projects
- **294,912-token governed accepted-context envelope**
- deterministic context chunk hashes, salience retrieval and continuity receipts
- bounded dense MESIE working context with explicit claim boundaries

## Context

AURO accepts up to **294,912 tokens** through a governed context envelope. The envelope records:

- accepted tokens
- dense working tokens
- retrieved tokens
- selected chunks
- truncation
- deterministic hashes

This is not a claim that 294,912 tokens enter one dense Softmax operation. The dense MESIE window remains bounded while older context is retained and retrieved through the governed envelope.

```python
from auro_native_llm import AuroLongContextModel

model = AuroLongContextModel.build(model_id="Auro-2B", mode="dev")
result = model.prepare_context(token_ids, query_token_ids=query_ids)
print(result.receipt)
```

## Install

```bash
git clone https://github.com/ItsNotAILABS/Auro14B.git
cd Auro14B
python -m pip install -e .
```

Python 3.10+ is required. Optional surfaces use Julia, Node.js, Cloudflare Workers and browser ONNX runtimes.

## Run the local model surface

```bash
export PYTHONPATH=.
python -m auro_native_llm.use \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Explain the AURO atomic model strategy."
```

Agentic loop:

```bash
python -m auro_native_llm.use --him --him-germs 20 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Inspect this repository and plan the next training cycle."
```

Colony composition:

```bash
python -m auro_native_llm.use --colony --colony-germs 20 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "Route this task through specialized atomic agents."
```

## Python API

```python
from auro_native_llm.model import (
    AuroLanguageModel,
    ModelClass,
    classify_parameter_count,
    family_config,
    release_ladder,
)

print(classify_parameter_count(156_000) is ModelClass.ATOMIC)
print(classify_parameter_count(2_000_000_000) is ModelClass.MICRO)
print(release_ladder())

config = family_config("Auro-2B", mode="dev")
model = AuroLanguageModel(config)
print(model.info())
```

## Training ladder

The repository supports development and training workflows across the family. Architecture labels are never substituted for trained-weight evidence.

```bash
# Build provenance-bound corpus
python scripts/build_unified_training_corpus.py \
  --mesie-root /path/to/MESIE \
  --sovereign-root /path/to/sovereign

# 14B training lane
python scripts/train_14b.py \
  --sovereign-root /path/to/sovereign \
  --corpus-jsonl artifacts/auro14b-corpus/corpus.jsonl \
  --rounds 2 --steps 4
```

A smoke run validates the pipeline, not model quality:

```bash
python scripts/train_14b.py --smoke \
  --sovereign-root /path/to/sovereign \
  --corpus-jsonl artifacts/auro14b-corpus/corpus.jsonl \
  --output checkpoints/auro_minds/Auro-14B-smoke \
  --rounds 1 --steps 1 --seq-len 32
```

## Checkpoint release standard

A downloadable checkpoint is release-ready only when it includes:

1. exact weights and SHA-256 manifest;
2. tokenizer and round-trip audit;
3. architecture/runtime configuration;
4. corpus provenance and training history;
5. checkpoint-specific evaluations and failure samples;
6. clean-install and launch proof;
7. API/local inference smoke receipt;
8. intended-use and limitations model card;
9. signed promotion and rollback evidence.

Quarantined checkpoints are rejected by default. Promotion requires signed evidence rather than boolean environment flags.

## Runtime surfaces

| Surface | Location | Purpose |
|---|---|---|
| Native Python model | `auro_native_llm/` | model, training, context and checkpoint runtime |
| Browser runtime | `browser-runtime/` | local ONNX/WASM inference with no silent remote fallback |
| Cloudflare operator | `workers/auro-platform/` | governed remote runtime and UI |
| Mobile runtime | `mobile-runtime/` | Expo client and device-sense integration |
| Web3 surface | `him-web3/` | server-held RPC credentials and read-only chain tools |

## Validation

Focused gates cover:

- AURO family architecture and MoE policy
- Auro-4B native behavior
- checkpoint constitutional authorization
- 294,912-token context envelope and claim boundaries
- tokenizer and model runtime surfaces

Run focused tests locally:

```bash
python -m pytest -q tests/test_long_context.py tests/test_checkpoint_constitution.py
```

## Architecture

```text
Atomic models (<1B) ─┐
Micro models (2B/4B) ├── governed colony / council ── Auro-14B orchestrator
Core model (8B) ─────┘                 │
                                      ├── MESIE compute
                                      ├── constitutional checkpoints
                                      ├── 294,912-token context envelope
                                      └── browser / mobile / API surfaces
```

## Project identity

- Organization: [ItsNotAILABS](https://github.com/ItsNotAILABS)
- Repository: [ItsNotAILABS/Auro14B](https://github.com/ItsNotAILABS/Auro14B)
- Compute plane: **MESIE**
- Family: **AURO**
- License: see [`LICENSE`](LICENSE)
