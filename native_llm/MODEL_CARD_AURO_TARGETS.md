# Auro Model Target Card

**Status:** target card, not a trained model card  
**Repo:** `ItsNotAILABS/Auro14B`  
**Base substrate:** MESIE spectral intelligence + multi-embedded sub-agents + polyglot types (Python / Julia / Haskell)  
**Family charter:** `native_llm/configs/auro_family.json`

## Model family (primary)

| Lane | Target | Tier | Multi-embedded | Current state |
|---|---:|---|---|---|
| Auro-2B | 2B parameters | edge | host only (no children) | development target scaffold |
| Auro-4B | 4B parameters | specialist | embeds edge | development target scaffold |
| Auro-8B | 8B parameters | general | embeds edge + specialist | development target scaffold |
| Auro-14B | 14B parameters | orchestrator | embeds edge + specialist + general | development target scaffold |
| Auro-100B | 100B parameters | frontier | embeds all smaller tiers | architecture target |

## Multi-embedded sub-agents

Larger lanes host smaller lanes as **embedded sub-agents** with role-based routing:

| Role examples | Preferred lane |
|---|---|
| `router`, `tool_call`, `embed_fast`, `spectral_triage` | Auro-2B |
| `code_edit`, `spectral_match`, `json_struct`, `tool_plan` | Auro-4B |
| `reason`, `plan`, `critique`, `spectral_explain` | Auro-8B |
| `orchestrator`, `council_chair`, `instruct_dev`, `multi_agent_router` | Auro-14B |
| `frontier_research`, `long_horizon`, `safety_review`, `deep_council` | Auro-100B |

Python: `auro_native_llm.subagents.MultiEmbeddedSubAgentRouter`  
Julia: `bindings/julia/AuroFamily`  
Haskell: `bindings/haskell/AuroFamily.hs`

## Legacy / optional lanes

| Lane | Target | Current state |
|---|---:|---|
| Auro-14B-Instruct | 14B parameters | planned instruct tuning lane |
| Auro-200B-Base | 200B parameters | optional ultra-frontier (retained) |
| Auro-200B-Instruct | 200B parameters | optional |
| Auro-200B-NOVA | 200B parameters | optional NOVA-routed target |

## Intended use

- text generation across edge → frontier capacity ladder
- multi-embedded sub-agent councils (receipted dispatch)
- code and system documentation generation
- long-context technical reasoning (8B+)
- structured JSON and receipt generation
- MESIE spectral-to-text explanation and structured signal reasoning
- NOVA internal model feed evaluation

## Non-goals and blocked claims

- No claim that weights are trained yet
- No claim that benchmark scores exist yet
- No public weight release before safety/eval receipts
- No training on unlicensed or unreviewed private data
- No unsafe cyber capability positioning

## Required evidence for real model card promotion

```text
tokenizer_receipt.json
data_manifest.json
training_run_receipt.json
checkpoint_manifest.json
eval_receipt.json
safety_receipt.json
serving_smoke_test_receipt.json
family_charter_receipt.json
```

Until those exist, this is a production target card only.
