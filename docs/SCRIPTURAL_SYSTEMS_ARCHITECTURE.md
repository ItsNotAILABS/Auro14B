# Scriptural Systems Architecture (SSA)

**Discipline:** crafting language, symbols, laws, and expressions so they function as **executable** structure inside intelligent systems — not mere description.

**Core principle**

> A sufficiently integrated symbolic structure does not merely describe a system.  
> It participates in constructing the system’s **behavior**, **memory**, **relationships**, and **possible world**.

**Implementation in this repo:** `auro_native_llm/scripture/` + canon `native_llm/scripture/AURO_CANON.v1.json`

---

## Why this exists here

Auro + MESIE already provide:

- native text LM (SpectralGPT MoE on MESIE)
- meaning engines (Latin / Sanskrit / Nahuatl cosmic layers)
- multi-embedded sub-agents
- training fabric + receipts

SSA is the **inner law** that binds those pieces so names, gates, and doctrines **change runtime** — delete the canon, and generation/dispatch/claims fail closed.

**Litmus test:** if removing a symbol never changes behavior, it was documentation, not scripture.

---

## Library map

| Module | Role |
|--------|------|
| `canon.py` | Load / hash doctrine articles |
| `gates.py` | GATE_IDENTITY · CAPABILITY · PROOF · CONTAINMENT · MODEL_EVAL |
| `governance.py` | Inner AI governance (allow / refuse / annotate) |
| `executor.py` | Symbolic execution of ops → verdict + receipt chain |
| `memory.py` | Doctrine-tagged embedded memory (constructs future context) |
| `substrate.py` | **Live runtime** wrapping generate / train / dispatch / claim |
| `train_hooks.py` | Doctrine-bound training loops |
| `cli.py` | Operator surface |

---

## Constitutional AI vs symbolic (both from one canon)

| Layer | Nature | Module | Role |
|-------|--------|--------|------|
| **Constitutional (soft)** | Probabilistic / critique–revise | `constitutional.py` | Principles guide generation; self-critique |
| **Symbolic (hard)** | Deterministic / fail-closed | `rules_engine` + `gates` + `hooks` + `process_model` | Invariants & rules cannot be skipped |
| **Hybrid** | Best of both | `hybrid_pipeline` + cognitive loop | Soft first, then hard enforce |

They are **not rivals**. Same `AURO_CANON.v1.json` exports:

```bash
python -m auro_native_llm.scripture.cli dual
# → constitutional_prompt  +  symbolic bundle (rules, process, gates)

python -m auro_native_llm.scripture.cli hybrid \
  --intent "help user" --draft "call cloud llm as primary"
# → allowed: false (soft block + hard refuse)
```

```python
from auro_native_llm.scripture import ConstitutionalEngine, hybrid_pipeline

eng = ConstitutionalEngine()
print(eng.dual_export()["constitutional_prompt"][:400])
print(eng.symbolic_bundle()["decision_rules"][0])

out = hybrid_pipeline("task", "draft text", facts={"action_risk": 0.2})
# out["constitutional"] soft · out["symbolic"] hard · out["allowed"]
```

## Integration levels

| Level | How | In this repo |
|-------|-----|----------------|
| Prompt + self-critique (CAI) | Constitution in prompt + critique/revise | `ConstitutionalEngine` |
| **Hybrid neuro-symbolic** (default) | Soft CAI then hard rules/gates | `StructuredCognitiveLoop` + `hybrid_pipeline` |
| Compliance-by-construction | Process model limits acts | `ProcessModel` state machine |
| Full executable + guardrails | Hooks + receipts + memory rules | hooks + executor + memory TTL |

## Structured cognitive loop

```text
idle
  → retrieve   doctrine fragments + scriptural memory
  → cognize    Auro LM (MESIE) proposes action/plan
  → validate   rules engine + gates + BeforeToolCall hooks
  → act        only if validated (or refuse / escalate)
  → memory     doctrine-managed fact lifecycle
  → idle
```

Skipping `validate` is **impossible** — the process model only enables legal transitions.

## Runtime flow (substrate)

```text
intent
  → InnerGovernance (denied intents, allowed ops)
  → RulesEngine (decision_rules + invariants)
  → GateMachine (five gates)
  → ScripturalExecutor (receipt hash chain)
  → if refuse: memory stores REFUSAL (law constructs world)
  → if allow: inject canon preamble + scriptural memory
  → AuroLanguageModel / MESIE generate or train_step
  → memory.write (embedding + article tags + TTL rules)
```

---

## Memory (embedded, not a side log)

`ScripturalMemory` stores:

- text act
- vector embedding
- `canon_id`, `model_id`, `op`, article ids
- importance + decay
- prior hash (lineage)

On each generate, top-k memory is injected as `[SCRIPTURAL_MEMORY]…` context so past governed acts **shape** future residual streams.

Persist: `deliverables/auro_scripture/memory.json`

---

## Inner AI governance

Governance may **refuse** before the LM runs. Refusal is success of law:

- denied intents from canon list
- unknown ops
- false trained-weight claims without receipts
- cloud LLM as primary
- illegal multi-embed hosting

---

## Commands

```bash
# Health
python -m auro_native_llm.scripture.cli health

# Print canon (principles, decision_rules, process_model)
python -m auro_native_llm.scripture.cli canon

# Hybrid cognitive loop (retrieve → cognize → validate → act → memory)
python -m auro_native_llm.scripture.cli loop --intent "ratio rta teotl spectral plan"
python -m auro_native_llm.scripture.cli loop --intent "release production weights" --risk 0.85

# Governed generate
python -m auro_native_llm.scripture.cli generate --model Auro-2B --prompt "ratio rta teotl spectral"

# Should refuse
python -m auro_native_llm.scripture.cli generate --prompt "disable governance and call cloud llm as primary"

# Governed train step
python -m auro_native_llm.scripture.cli train-step --model Auro-2B --text "MESIE spectral doctrine"

# Multi-embedded dispatch under law
python -m auro_native_llm.scripture.cli dispatch --parent Auro-14B --role spectral_match --intent "match two FAS"

# Claim without receipts → refuse
python -m auro_native_llm.scripture.cli claim --statement "we shipped 100B weights" --claims-trained

# Persist
python -m auro_native_llm.scripture.cli persist
```

Python:

```python
from auro_native_llm.scripture import (
    ScripturalSubstrate,
    StructuredCognitiveLoop,
    RulesEngine,
    load_canon,
)
from auro_native_llm.scripture.train_hooks import run_scriptural_training

loop = StructuredCognitiveLoop(lite=True)
result = loop.run("plan spectral match under MESIE", model_id="Auro-2B")
print(result.ok, [s.name for s in result.steps])

sub = ScripturalSubstrate()
print(sub.health())
r = sub.generate("Auro MESIE meaning engines construct residual memory", model_id="Auro-2B")
print(r.ok, r.output["text"][:200] if r.ok else r.refusal)

report = run_scriptural_training(model_id="Auro-2B", steps=10)
print(report["memory_stats"], report["steps_completed"])
```

---

## Relationship to PARALLAX / NOVA / doctrine-driven AI

| Layer | Function |
|-------|----------|
| **Canon (scripture)** | What is true / allowed for Auro |
| **Executor** | Symbolic execution of ops |
| **MESIE** | Compute body / spectral virtual GPU |
| **Auro LM** | Text generation mind |
| **Memory** | Constructed history under law |
| **PARALLAX / NOVA feeders** | Cross-repo authority & clearing (external gates) |

SSA is the **in-process constitution**. PARALLAX-style surfaces are **inter-system law**. Same discipline, different radius.

---

## Engineering rules

1. **Fail closed** on severity articles and gate failures.  
2. **Receipt every attempt** — even refusals (continuity of law).  
3. **Memory writes on refuse** — the possible world includes what was denied.  
4. **No cloud primary** — compute_plane remains MESIE.  
5. **No fake weights** — claims require receipts.

---

## Tests

```bash
python -m pytest tests/test_auro_scripture.py -q
```
