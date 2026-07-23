# Auro-4B Competitive Evaluation Gate

Auro-4B may only be described as outperforming another system after a pinned,
reproducible comparison using the exact checkpoint hash, equal tool access,
equal budgets, recorded prompts, executable receipts, and failure analysis.

## Required lanes

- polyglot code generation and execution;
- repository-native engineering;
- long-context retrieval and synthesis;
- multimodal reasoning;
- governed tool use;
- civilization-architecture transfer;
- latency, memory, energy estimate, and cost per validated task.

## Target manifest

Record the official comparison-model name, version, provider, access date,
context limit, modalities, enabled tools, sampling settings, price, and official
documentation source. If the exact requested target cannot be verified, mark the
run `TARGET_UNRESOLVED` rather than substituting another model.

## Promotion rule

Auro-4B is promoted only when:

1. the exact Auro checkpoint and tokenizer hashes are recorded;
2. all executable claims have test receipts;
3. aggregate performance is higher on a meaningful task set;
4. no critical lane falls below the release threshold;
5. model-only and full-system results are reported separately;
6. all failures and unsupported tasks are disclosed.

## Required artifacts

- target-manifest.json
- task-manifest.jsonl
- auro-results.jsonl
- comparison-results.jsonl
- execution-receipts/
- scorecard.json
- failure-analysis.md
- benchmark-receipt.json

Current status: `SPEC_DEFINED_NOT_EXECUTED`.
