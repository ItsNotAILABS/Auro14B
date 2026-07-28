# AURO ST-14B Efficiency Profile

**Status:** PROTOTYPE architecture contract  
**Profile:** `auro.st14b.efficiency.v1`  
**Role:** Dense serving-optimized AURO execution lane

## Purpose

The ST-14B research is integrated as a dedicated AURO serving profile rather than silently replacing the existing AURO/HIM/MESIE organism architecture.

The portfolio boundary is:

```text
NOVA / Chimeria cognition and governance
                ↓
MedinaMemorySystems retrieval and continuity
                ↓
AURO-ST-14B dense inference core
                ↓
NOVA governed tools, robots and remote execution
                ↓
durable receipts and evidence
```

This preserves AURO's broader architecture while creating a measurable 14B-class decoder lane optimized for inference economics.

## Canonical geometry

| Property | Value |
|---|---:|
| Model ID | `AURO-ST-14B` |
| Dense parameter estimate | 14,339,691,520 |
| Parameter target | 14.2B |
| Layers | 40 |
| Hidden dimension | 5,120 |
| Query heads | 40 |
| KV heads | 5 |
| Head dimension | 128 |
| GQA ratio | 8:1 |
| SwiGLU intermediate dimension | 18,432 |
| Vocabulary | 128,000 |
| Dense context | 8,192 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |

The supplied research alternated between a 64-to-8 example and a 40-to-5 specification. The implementation adopts the internally consistent table geometry: 40 query heads and 5 KV heads, which preserves the same 8:1 grouping ratio.

## KV-cache contract

AURO-ST-14B stores keys and values at the five native KV heads before query-head expansion.

At batch size 1, BF16 precision and 8,192 tokens:

- GQA KV cache: 838,860,800 bytes, or 800 MiB
- MHA-equivalent KV cache: 6,710,886,400 bytes, or 6,400 MiB
- theoretical reduction: 87.5%

This is a geometry result, not a measured memory-bandwidth result. The research claim of 42% lower memory-bandwidth demand remains a benchmark target until profiled on an exact runtime and checkpoint.

## Runtime lanes

| Lane | Decision | Required evidence |
|---|---|---|
| PyTorch BF16 | REFERENCE | exact checkpoint, quality baseline and memory trace |
| Transformers | BENCHMARK | generation parity and cache behavior |
| vLLM | ADOPT TARGET | paged KV cache, batching, TTFT and throughput receipts |
| TensorRT-LLM FP8 | BENCHMARK | H100 runtime receipt and accuracy delta |
| INT8 | PLANNED | calibration receipt and exact quality evaluation |
| INT4 AWQ/GPTQ/GGUF | RESEARCH | quality floor, memory and local-runtime comparison |
| llama.cpp | BENCHMARK | GGUF conversion and platform matrix |

## Required benchmark matrix

Every promoted checkpoint must record:

- exact model and tokenizer hashes
- runtime name, version and commit
- hardware identity and driver versions
- precision and quantization method
- prompt and generation lengths
- concurrency and batch policy
- TTFT
- inter-token latency
- prefill throughput
- decode throughput
- peak device memory
- measured KV-cache use
- quality delta against BF16
- receipt hash and immutable artifact location

Minimum workloads:

```text
prompt 128  → decode 128
prompt 2048 → decode 256
prompt 8192 → decode 256
concurrency 1, 8 and 32
```

## Research claims converted into gates

The following values are **targets**, not current AURO results:

- 42% memory-bandwidth reduction
- 1.8x throughput improvement
- 64.2% prefill MFU
- TTFT below 12 ms at a 2,048-token prompt
- 142 tokens/second per stream at concurrency 32

They become publishable only when an exact checkpoint and runtime generate reproducible receipts.

## Why this profile is dense

The existing AURO family includes MESIE MoE, cross-modal, spectral, meaning, adaptation and organism layers. Those features remain important, but mixing them into the ST-14B parameter claim would make the 14B geometry ambiguous.

The ST profile therefore disables core MoE, cross-modal and spectral blocks inside the serving core. Higher cognition, memory, tool use and embodiment remain composable through NOVA, Chimeria, MedinaMemorySystems and AURO's external organs.

This yields a clean separation:

```text
cognition envelope ≠ serving-core parameter count
```

## Implemented artifacts

- `auro_native_llm/model/st14b.py`
- `scripts/benchmark_st14b_architecture.py`
- `tests/test_st14b_efficiency.py`
- `.github/workflows/auro-st14b-efficiency.yml`

The benchmark harness emits `auro.st14b.evidence.v1` with a deterministic evidence hash and explicit claim boundary.

## Promotion ladder

```text
PROTOTYPE architecture contract
→ micro-instantiated geometry test
→ tokenizer and checkpoint training plan
→ exact BF16 checkpoint
→ Transformers parity
→ vLLM benchmark
→ TensorRT-LLM FP8 benchmark
→ INT8 and INT4 quality studies
→ production candidate
```

No stage may inherit the evidence of a later stage by documentation alone.
