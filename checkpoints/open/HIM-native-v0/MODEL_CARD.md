# HIM-native-v0

HIM-native-v0 is a compact, fully local open-weight causal language-model checkpoint. It proves Auro's complete tokenizer → corpus → training → weights → loader → generation → API path without third-party base weights or provider distillation.

## Measured facts

- Architecture: context-MLP causal language model
- Learned parameters: 146,576 float32 scalars
- Tokenizer: immutable UTF-8 byte fallback, 272 IDs, zero unknown tokens
- Unique corpus pass: 4,109 tokens
- Optimizer token presentations: 115,200
- Training steps: 1,200
- Training loss: 5.6062 → 0.0776
- Held-out loss: 3.5374; perplexity: 34.3771
- Base model: none; seeded local initialization
- Weights: `weights.npz.b64`, decoded by `OpenHIM.load`

## Intended use

Architecture validation, open-weight pipeline testing, tokenizer audits, local inference integration, API smoke tests, and continued training research.

## Not ready for

General-purpose assistant use, factual reliance, coding autonomy, safety-critical decisions, financial execution, or claims of 14B parameters. The observed generation quality is still poor. `Auro14B` remains a family target label, not this checkpoint's measured parameter count.

## Accounting boundary

Training tokens and parameters are different quantities. `optimizer_tokens_seen` counts token presentations consumed by updates. `num_parameters` counts learned scalar weights. Neither should be relabeled as the other.

