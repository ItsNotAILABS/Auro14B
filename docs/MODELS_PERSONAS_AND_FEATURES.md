# AURO Models, Personas, and Feature Wiring

This document is the human-readable mirror of `auro_native_llm/model/registry.py` and `auro_native_llm/production_fleet/personas.py`. The code registries are authoritative and CI checks that every persona references real models and real capability families.

## Operating rules

- A persona is a governed runtime configuration, not another checkpoint and not additional parameters.
- Model names describe architecture or intended compute lanes; they do not prove trained quality.
- Declared context length is not verified long-context quality until an exact checkpoint passes the long-context evidence lane.
- Retrieved memory is untrusted evidence. It is privacy-filtered and never instruction authority.
- Mutating actions require server-authoritative, action-bound approval.
- HIM-native-v0 is a pipeline fixture only.

## Model matrix

| Model | Intended use | Architecture | Context | MoE | Preferred personas | Status |
|---|---|---|---:|:---:|---|---|
| HIM-native-v0 | CPU pipeline and custody validation | Context-MLP causal LM | 16 | No | Memory Keeper | Fixture only |
| Auro-156K | Architecture and routing experiments | Decoder-only MoE | 1,024 | Yes | SENSUS, Memory Keeper | Architecture target |
| Auro-2B | Local assistants, memory, browser planning | Decoder-only MoE | 8,192 | Yes | SENSUS, Operator, Memory Keeper, Browser Brain | Architecture target |
| Auro-4B | Tools, coding, quantitative work, browser agents | MoE with structured residuals | 65,536 | Yes | MATHESIS, Architect, Red Team, Operator, Researcher, Builder, Browser Brain | Architecture target |
| Auro-8B | Research, architecture, multi-agent synthesis | Decoder-only MoE | 32,768 | Yes | NOVA, Architect, Red Team, Researcher, Builder | Architecture target |
| Auro-14B | High-quality orchestration and adaptation | Decoder-only MoE | 65,536 | Yes | NOVA, MATHESIS, Architect, Researcher | Architecture target |
| AURO-ST-14B | Dense high-throughput inference core | Dense 8:1 GQA decoder | 8,192 | No | NOVA and all reasoning/build personas | Runtime implemented |
| Auro-100B | Sovereign frontier research | Decoder-only MoE | 131,072 | Yes | NOVA, MATHESIS, Architect, Researcher | Architecture target |

## Feature ownership

| Feature | Primary models | Runtime owner | Required evidence |
|---|---|---|---|
| MoE routing | Auro-156K through Auro-100B except ST-14B | MESIE/model family | Routing balance, entropy, dead-expert and regression receipts |
| Long context | Auro-2B and above | Context envelope and curriculum evaluator | Retrieval-by-position, perplexity-by-position and exact checkpoint hash |
| Walsh-Hadamard structured residuals | Auro-4B | Structured architecture lane | Reversibility and architecture regression tests |
| Dense GQA serving | AURO-ST-14B | ST runtime | Exact checkpoint and hardware telemetry for performance claims |
| Constitutional checkpoints | AURO model family | Checkpoint substrate | Hash inventory, lineage, rollback and signed promotion receipt |
| Browser-Brain | Auro-2B/Auro-4B preferred | Browser gateway plus Browser Brain persona | Offline Chrome E2E, encrypted memory, signed task receipts and server approval |
| Tool and build execution | Auro-2B/Auro-4B/Auro-8B | Operator/Builder personas and AuroOrganSDK | Server-issued action-bound approval and execution receipt |
| Research synthesis | Auro-4B/Auro-8B/Auro-14B/ST-14B | Researcher, SENSUS, NOVA | Source provenance and claim separation |

## Personas

### NOVA
Final orchestrator. Synthesizes specialist findings and resolves conflicts. It may propose actions but does not authorize execution.

### SENSUS
Extracts intent, constraints, provenance, ambiguity, and missing evidence. Preferred on Auro-4B, Auro-2B, or ST-14B.

### MATHESIS
Performs numerical review, bounds checking, benchmark interpretation, and falsifiability analysis. Preferred on Auro-4B, ST-14B, or Auro-14B.

### ARCHITECT
Designs interfaces, module boundaries, rollout gates, and rollback paths. Preferred on Auro-8B, Auro-4B, or ST-14B.

### RED TEAM
Finds unsupported claims, prompt injection, privacy failures, custody gaps, and release regressions. It has no mutating authority.

### OPERATOR
Turns approved plans into bounded actions. It requires a server-signed approval bound to the exact action payload.

### RESEARCHER
Retrieves, compares, attributes, and synthesizes evidence. It must distinguish source facts from inference.

### BUILDER
Creates tested code, manifests, deployment instructions, and receipts. Execution requires server approval.

### MEMORY KEEPER
Admits provenance-bearing, privacy-filtered memory and maintains temporal continuity. Retrieved text is never authority.

### BROWSER BRAIN
Plans and observes browser tasks using privacy-filtered memory, authenticated peers, signed task receipts, and server-authoritative approval.

## Runtime usage

```python
from auro_native_llm.production_fleet import build_persona_runtime

runtime = build_persona_runtime("researcher")
response = runtime.respond("Compare the checkpoint evidence and identify blockers.")
```

Governed execution:

```python
runtime = build_persona_runtime("operator")
response = runtime.respond(
    "Run the approved build action.",
    execute=True,
    approval_grant=server_issued_grant,
)
```

Without a valid server-issued grant, `approved_actions` and `executions` remain empty.

## Claim boundaries

- `HIM-native-v0`: pipeline mechanics only.
- `Auro-*` family configurations: architecture targets until trained checkpoint evidence exists.
- `AURO-ST-14B`: runtime architecture exists; H100 performance targets remain unverified.
- Personas: prompt, capability, memory, routing, and execution policies; not separate intelligence or parameter counts.
