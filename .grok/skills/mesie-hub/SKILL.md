---
name: mesie-hub
description: >
  Master hub for MESIE, MAESI, and NeuroAIX. Routes to all native skills and tools.
  Triggers: mesie, maesi, neuroaix, spectral engine, octopus, fingerprint, monte carlo.
  Use when user asks to run MESIE or needs the tool catalog.
---

# MESIE / MAESI / NeuroAIX Hub

Unified native suite: **33 tools**, **24 skills** (incl. hub).

```bash
python -m mesie.tools.cli list
python -m mesie.tools.cli run <tool-id>
```

## Skills map

| Skill | Tools |
|-------|-------|
| `/mesie-benchmark` | benchmark |
| `/mesie-data` | bundled-data, fix-data |
| `/mesie-deploy` | cloudflare, catalog |
| `/mesie-domains` | domains |
| `/mesie-embed` | embed |
| `/mesie-embed-library` | embed-library, embed-mine |
| `/mesie-enterprise` | monte-carlo |
| `/mesie-fingerprint` | fingerprint |
| `/mesie-generate` | generate-psd, generate-fas, rotdnn |
| `/mesie-internal` | internal-bus, engines |
| `/mesie-knowledge` | knowledge |
| `/mesie-laptop` | laptop |
| `/mesie-logic-prover` | logic-prover |
| `/mesie-maesi` | maesi, fast-compute |
| `/mesie-match` | match, rank |
| `/mesie-neuroaix` | neuroaix, cognitive |
| `/mesie-octopus` | octopus |
| `/mesie-orbital` | orbital |
| `/mesie-pattern-forge` | pattern-forge |
| `/mesie-polyglot` | ais-polyglot |
| `/mesie-solus-organism` | solus-organism |
| `/mesie-test` | test, sdk-drive |
| `/mesie-validate` | validate |

## Quick enterprise stack

1. `/mesie-embed-library` — index corpus
2. `/mesie-fingerprint` — TF + LSH + ANN
3. `/mesie-octopus` — eight-arm workflow
4. `/mesie-maesi` — knowledge + fast compute
5. `/mesie-enterprise` — Monte Carlo 10 use cases

Regenerate skills: `python -m mesie.tools.cli skills`