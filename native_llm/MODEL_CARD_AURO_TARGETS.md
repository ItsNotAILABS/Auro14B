# Auro Model Target Card

**Status:** target card, not a trained model card  
**Repo:** `ItsNotAILABS/Auro14B`  
**Base substrate:** MESIE spectral intelligence + transformer/intelligence protocols  

## Model lanes

| Lane | Target | Current state |
|---|---:|---|
| Auro-14B-Dev | 14B parameters | development target scaffold |
| Auro-14B-Instruct | 14B parameters | planned instruct tuning lane |
| Auro-200B-Base | 200B parameters | architecture target |
| Auro-200B-Instruct | 200B parameters | architecture target |
| Auro-200B-NOVA | 200B parameters | NOVA-routed internal target |

## Intended use

- text generation
- code and system documentation generation
- long-context technical reasoning
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
```

Until those exist, this is a production target card only.
