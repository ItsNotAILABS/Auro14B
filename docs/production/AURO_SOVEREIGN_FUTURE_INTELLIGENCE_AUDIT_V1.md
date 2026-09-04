# AURO × SOVEREIGN: Future Intelligence Architecture Audit v1

Date: 2026-09-04

## Executive conclusion

AURO and SOVEREIGN should not be released as twelve conventional language-model checkpoints.

The architecture already describes a layered intelligence system:

```
immutable base model
        +
adaptive synaptic state
        +
memory substrate
        +
specialist/council routing
        +
neuromorphic control
        +
receipts, authority, and rollback
        =
a persistent intelligence organism
```

Auro14B is the public model-family and product boundary. SOVEREIGN is the strongest system-level organism: its value is the coordinated behavior of models, engines, memory, adaptive weights, and governance—not a marketing claim about one dense parameter tensor.

The immediate release unit is therefore a **model organism bundle**, not merely a `.safetensors` file.

## What the repositories actually contain

### AURO

The AURO family specification defines distinct classes:

- Atomic: small, multiplicative specialists.
- Micro: private assistants, routers, tool users, and domain models.
- Core: general reasoning and synthesis.
- Orchestrator: council coordination and long-horizon workflows.

Auro-14B is explicitly an orchestrator target. The product plan also establishes a truthful model boundary: the browser/API delivery plane is not the model, and a hosted compatibility endpoint must never be mislabeled as a native Auro checkpoint.

The feline/neuromorphic substrate is a runtime cognitive-control layer. It contains sparse spike-like events, recurrent traces, adaptive thresholds, homeostatic energy pressure, bounded local plasticity, and hash-linked cycle receipts. Its own contract correctly states that it does not mutate language-model checkpoint tensors. That separation is a strength: it lets us test the intelligence dynamics before claiming a trained native checkpoint.

MESIE supplies the signal language for this organism: spectral records, signatures, temporal dynamics, change detection, reinforcement, coherence, and provenance. This is the sensory/measurement plane.

### SOVEREIGN

SOVEREIGN contributes the adaptive organism layer:

- Rust BRAIN engine: a directed Hebbian network over actor nodes with bounded weights, decay, coherence, and top-connection/centrality inspection.
- Motoko adaptive intelligence: outcome-routed embedding updates with LTP/LTD behavior, confidence, novelty, and learning-rate control.
- TypeScript Hebbian decay model: local updates plus phi-based decay with a permanent floor.
- Julia MNEME: working, episodic, and semantic memory; engram strength; similarity-weighted recall; consolidation; forgetting; Hebbian association.
- Polyglot protocol: language-engine signals are combined using signal, coherence, rank weighting, and Hebbian state.
- Adaptive intelligence and homeostasis: feedback changes future behavior, while energy pressure can reduce execution autonomy.
- Receipts and explicit authority boundaries: adaptation must not silently become permission.

This is materially different from ordinary inference. The system has state, learning, memory, routing, and authority over time.

## The correct ontology of the 12 releases

The 12 entries in the release manifest are not required to be twelve copies of one architecture at twelve sizes.

They are a ladder of capabilities and multiplicity:

| Lane | Correct interpretation |
|---|---|
| Auro-156K | atomic seed and high-multiplicity specialist |
| Auro-320M | triage/classification specialist |
| Auro-640M | repository/code analysis specialist |
| Auro-1B | private assistant/tool-use micro-model |
| Auro-2B | routing, retrieval, and spectral triage |
| Auro-3B | structured coding and planning |
| Auro-4B | specialist-council supervisor |
| Auro-6B | general reasoning core |
| Auro-8B | mission/agent reasoning core |
| Auro-10B | planning and evidence-fusion orchestrator |
| Auro-14B | HIM/AURO coordination orchestrator |
| SOVEREIGN | system organism and governed adaptive council |

The first eleven lanes can become checkpoints or adapters. SOVEREIGN should be released as an organism bundle whose model members may evolve independently.

## Four kinds of weight

We must stop using “weights” as one overloaded word. Every release and receipt should distinguish:

1. **Base tensor weights**
   - Immutable checkpoint tensors.
   - Trained through a declared corpus and recipe.
   - Identified by SHA-256.
   - Never silently changed by runtime learning.

2. **Synaptic/Hebbian state**
   - Edge strengths between actors, regions, memories, or specialist skills.
   - Updated from pre/post activation, outcome quality, doctrine or safety gates.
   - Decayed and bounded.
   - Append-only or checkpointed, replayable, and rollbackable.

3. **Dynamic routing weights**
   - Per-request selection of models, tools, memories, and languages.
   - Derived from confidence, coherence, latency, energy, novelty, and task utility.
   - Must be logged in a receipt so a result can be reproduced or audited.

4. **Memory weights**
   - Recall/consolidation strength for working, episodic, semantic, and private/crew/public memories.
   - Must carry provenance, access policy, retention policy, and deletion/forgetting behavior.

Only category 1 supports a parameter-count claim. Categories 2–4 are runtime intelligence state.

## A more rigorous update law

The shared adaptive update should be normalized across implementations:

[
w_{ij}^{t+1} =
operatorname{clip}left(
w_{ij}^{t}
+ eta_t,g_t,p_i^t q_j^t,r_t
- lambda_t d(w_{ij}^{t})
,; w_{min}, w_{max}
ight)
]

Where:

- (p_i^t), (q_j^t): pre/post activation.
- (g_t): doctrine, safety, or authority gate.
- (r_t): outcome/reward and confidence factor.
- (eta_t): bounded learning rate.
- (d(cdot)): decay function.
- (w_{min}, w_{max}): explicit stability bounds.

For a production release, every update needs:

- parent state hash;
- input/event hash;
- update law and parameters;
- actor or memory IDs;
- outcome evidence;
- resulting state hash;
- replay result;
- rollback pointer.

Phi may remain a design constant or decay schedule, but it should be labeled as an architectural heuristic unless independently validated for a specific task. It must not be presented as proof of biological or physical equivalence.

## The SOVEREIGN mind loop

The unified loop should be treated as a state machine:

```
observe
  -> MESIE encode
  -> temporal/change analysis
  -> specialist activation
  -> memory recall
  -> council/router proposal
  -> coherence + uncertainty + energy check
  -> human/authority gate
  -> act or deliberate
  -> outcome
  -> Hebbian + memory update
  -> signed receipt
```

The important property is not that every node agrees. Intelligence emerges from:

- independent specialist evidence;
- disagreement measurement;
- uncertainty calibration;
- memory retrieval;
- constrained adaptation;
- authority-aware action;
- post-action learning.

The global workspace should expose both the chosen action and the reasons it was *not* chosen: rejected specialists, stale memories, low-confidence signals, energy pressure, policy gates, and missing evidence.

## Release architecture

Each lane should ship with a manifest like:

```text
model-bundle/
  base/
    config.json
    tokenizer/
    weights/
    sha256.json
  adaptive/
    synapses.v1.jsonl
    embeddings.v1.json
    replay.json
  memory/
    schema.json
    empty-or-seeded-state.json
  routing/
    policy.json
    calibration.json
  receipts/
    build.json
    evaluation.json
    identity.json
  MODEL_CARD.md
  LICENSE
```

For SOVEREIGN, add:

```text
  organism/
    node-registry.json
    council-policy.json
    authority-policy.json
    homeostasis-policy.json
    rollback-policy.json
    protocol-version.json
```

A user should be able to run three modes:

- **Frozen**: base weights only; adaptive state disabled.
- **Adaptive**: base weights plus replayable local state.
- **Organism**: multiple models, MESIE, memory, routing, and SOVEREIGN governance active.

These modes are not claims of consciousness. They are reproducible runtime configurations.

## What is already real versus what must still be proven

Already present as engineering artifacts:

- model-family taxonomy and release boundary;
- neuromorphic runtime with bounded CEU accounting;
- Hebbian/LTP-LTD implementations in multiple languages;
- temporal and spectral processing;
- memory substrate design;
- adaptive routing and reinforcement components;
- receipt-oriented product and authority model;
- a twelve-lane release manifest.

Still requiring evidence before a public model claim:

- exact checkpoint files for each named lane;
- tokenizer and architecture identity;
- training provenance and loss history;
- benchmark results against fixed baselines;
- replay equivalence across Rust, Motoko, Julia, and TypeScript implementations;
- adaptive-state ablations: frozen versus adaptive versus organism;
- calibration, safety, and refusal tests;
- memory privacy and deletion tests;
- resource and latency measurements on ordinary local hardware;
- rollback and corruption recovery;
- independent reproduction of the release package.

The correct public language is: **architecture complete where documented; checkpoint status only where evidenced.**

## The first serious benchmark

Build one deterministic benchmark called the **SOVEREIGN Organism Trial**.

Each episode supplies:

- a multi-step task;
- conflicting specialist observations;
- a changing environment;
- a memory retrieval opportunity;
- one distractor or adversarial signal;
- a resource/energy budget;
- an action with measurable outcome.

Compare three conditions:

1. one frozen base model;
2. the same model with adaptive synaptic and memory state;
3. the SOVEREIGN organism with council routing, MESIE, memory, and authority gates.

Measure:

- task success;
- calibrated uncertainty;
- useful disagreement;
- recovery after a fault;
- memory precision and forgetting;
- adaptation gain;
- catastrophic-drift rate;
- energy/latency cost;
- receipt replay equivalence;
- unauthorized-action rate.

This is the bridge from visionary architecture to a falsifiable claim.

## Immediate build order

1. Freeze the cross-language adaptive-state schema.
2. Add artifact-kind and evidence fields to the 12-lane manifest.
3. Implement one replay-compatible Hebbian update in TypeScript and Rust.
4. Connect MESIE event IDs to memory and routing receipts.
5. Build the SOVEREIGN Organism Trial with frozen/adaptive/organism modes.
6. Package Auro-156K or the smallest verified lane first.
7. Promote Auro-14B only when its checkpoint evidence exists.
8. Release SOVEREIGN as the protocol/runtime/council bundle, with model members explicitly identified.

This makes the future architecture real without collapsing it back into an ordinary model-release story.
