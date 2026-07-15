# Auro Native LLM Spine

**Status:** production scaffold, not a completed trained checkpoint  
**Repository:** `ItsNotAILABS/Auro14B`  
**Base system:** MESIE / Multi-Element Spectral Intelligence Engine  
**Target:** Auro native text-generation model family with a 14B development lane and 200B frontier training lane

## What this is

This directory turns the existing MESIE/Auro substrate into a native LLM program. The repo already contains spectral intelligence, transformer language, embeddings, intelligence protocols, training/inference concepts, and PyTorch/Transformers optional dependencies. The Auro Native LLM Spine adds the missing production contract for text generation:

- model-family charter
- 14B development lane
- 200B target architecture lane
- tokenizer and data protocol
- training stages
- evaluation gates
- serving contract
- NOVA root feeder manifest
- proof receipts and public/private boundaries

## What this is not

This commit does **not** claim that a 200B model has already been trained. A 200B text-generation checkpoint requires:

- large curated pretraining corpus
- tokenizer training and validation
- distributed GPU/accelerator training
- checkpoint storage
- evaluation receipts
- safety/alignment review
- serving infrastructure
- release governance

The scaffold defines how Auro becomes that system without making false checkpoint claims.

## Native model family

| Lane | Purpose | Status |
|---|---|---|
| `Auro-14B-Dev` | practical development model family and internal integration target | scaffolded |
| `Auro-14B-Instruct` | instruction-tuned development lane | planned |
| `Auro-200B-Base` | frontier native pretraining lane | architecture target |
| `Auro-200B-Instruct` | supervised + preference tuned assistant lane | architecture target |
| `Auro-200B-NOVA` | NOVA-routed internal model lane with receipts and gates | architecture target |

## Relationship to MESIE

MESIE is not discarded. It becomes Auro's structured intelligence substrate:

- spectral transformer research informs architecture and token-routing experiments
- intelligence protocols become evaluation and memory/control layers
- MESIE embeddings become auxiliary structured-signal context inputs
- MESIE training/inference docs become the foundation for a broader text-generation pipeline

## Root feed

Auro14B feeds NOVA root through:

```text
integrations/feeder-repos/auro14b/
models/auro/native-llm/
protocols/model-feeds/auro14b/
```

## Primary commands after implementation

```bash
python -m auro_native_llm.tokenizer.train --config native_llm/configs/tokenizer_200b.json
python -m auro_native_llm.data.build_corpus --config native_llm/configs/data_mixture_200b.json
python -m auro_native_llm.train.pretrain --config native_llm/configs/auro_14b_dev.json
python -m auro_native_llm.eval.run --config native_llm/configs/eval_gates.json
python -m auro_native_llm.serve.local --config native_llm/configs/serving_contract.json
```

These commands are contracts for the next code pass; they are not represented as already working entrypoints until the Python modules are added and tested.
