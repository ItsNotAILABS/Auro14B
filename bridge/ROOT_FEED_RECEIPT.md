# Auro14B Root Feed Receipt

**Status:** scaffold feed receipt  
**Feeder:** `ItsNotAILABS/Auro14B`  
**Root:** `ItsNotAILABS/NOVA-private-root`  
**Lane:** native LLM / model feed  

## Feeded artifacts

```text
native_llm/README.md
native_llm/configs/auro_14b_dev.json
native_llm/configs/auro_200b_target.json
native_llm/configs/tokenizer_200b.json
native_llm/configs/data_mixture_200b.json
native_llm/configs/eval_gates.json
native_llm/configs/serving_contract.json
auro_native_llm/
bridge/feeder-manifest.json
```

## Claim boundary

This receipt does not claim a trained 200B checkpoint. It records that the repo now has the scaffold, commands, configs, and feeder contract needed to begin a real native LLM production program.

## Required promotion receipts

- tokenizer receipt
- data manifest receipt
- pretraining loss receipt
- checkpoint manifest
- evaluation receipt
- safety receipt
- serving smoke-test receipt
- NOVA root import receipt
