# Auro production contract

## Loop

```text
train → measure → save → load → work → keep learning
```

## Claim boundary (non-negotiable)

| Statement | Meaning |
|-----------|---------|
| **Live params** | The trained, **running** executable core (what `num_params_live` reports) |
| **Family labels** (2B / 4B / 8B / 14B / 100B) | Architecture **targets** for scaled cores — not a claim that full dense weights exist |
| **Value** | Proven by **holdout CE drop** + **working tools** + **durable checkpoint** — not marketing |

## Command

```bash
python -m auro_native_llm.organism.production --model Auro-2B --steps 50
# or
python -m auro_native_llm.organism.cli production --model Auro-2B --steps 50
```

Writes under `checkpoints/auro_minds/<model>/`:

- `language/` — weights + tokenizer  
- `memory.json` / `trainer.json` / `mind_meta.json`  
- `VALUE_REPORT.json` + `VALUE_REPORT.md`  
- `PRODUCTION_LOOP.json` + `PRODUCTION_LOOP.md`  

## After production

```bash
python -m auro_native_llm.organism.cli load --path checkpoints/auro_minds/Auro-2B
python -m auro_native_llm.organism.cli work --objective "browse https://example.com"
```

Loaded minds **keep learning** on every act (absorb + train_step).
