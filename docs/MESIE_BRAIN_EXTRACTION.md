# MESIE brain extraction charter

## Product boundary

Auro14B is the HIM open-weight model family. MESIE is a source library during migration, not the permanent product namespace. The checkpoint, tokenizer, inference API, brain controller and agent runtime must continue to operate after `mesie/` is removed.

## Retained mechanisms

| MESIE mechanism | Permanent HIM destination | Readiness gate |
|---|---|---|
| 44-region functional topology | `auro_native_llm.brain.HIMBrain` | exact topology/count test |
| activation decay and recurrence | `HIMBrain.cycle` | bounded-state and persistence tests |
| coherence, anomaly and dominant system | brain-cycle result | deterministic routing tests |
| memory context | bounded working-memory gate | persistence test |
| biological homeostasis metaphor | bounded activation regulator | every value remains in `[0,1]` |
| cognitive perception-to-action loop | salience and route arbitration | execution remains separately authorized |
| legacy BRAIN state name | `HIMBrain.state` alias | compatibility test |

## Removal policy

Do not delete legacy MESIE files merely because a replacement exists. A legacy component becomes a deletion candidate only after its imports are inventoried, its useful behavior has a canonical owner, parity is tested, public names have aliases, and production gates pass without `mesie` installed. Domain applications and duplicated interfaces are removed in a later, separately reviewed cleanup PR.

This fusion changes inference-time cognition. It does not change or inflate the model's learned parameter count. Checkpoint improvement remains an explicit corpus, training, evaluation and promotion process.
