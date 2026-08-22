# AURO Exact-Checkpoint Release Promotion

A checkpoint is not production-ready because it exists, loads, hashes correctly, or has a strong architecture name. Production promotion requires evidence bound to the exact checkpoint being released.

## Promotion order

1. Verify immutable checkpoint artifacts with `auro_native_llm.open_weights.verify_checkpoint`.
2. Audit the tokenizer manifest with `scripts/tokenizer_audit.py`.
3. Pin release evidence artifact hashes in one release evidence manifest.
4. Score human-deliverable readiness with `scripts/readiness_score.py`.
5. Run `scripts/checkpoint_promotion_gate.py` against the exact checkpoint and exact evidence manifest.
6. Promote only when `promote=true` and no unresolved blocker remains.

## Required evidence classes

The promotion gate requires hashed artifacts for:

- tokenizer audit;
- corpus provenance / safety manifest;
- training report;
- official benchmark results;
- coding execution results;
- governed-execution denial tests;
- API chat smoke proof;
- browser chat smoke proof;
- clean-install/start proof;
- release model card.

Portability is also a readiness gate and should be represented in the readiness input with a concrete evidence reference.

## Human-deliverable readiness

The canonical threshold is 0.85. The weighted score alone is insufficient: every critical gate must independently meet the threshold and the unresolved blocker list must be empty.

Critical gates are checkpoint integrity, training provenance, tokenizer integrity, corpus provenance, official benchmarks, governed execution, API chat smoke, and clean install.

The readiness score is not MMLU, HumanEval, general intelligence, consciousness, or a model-quality percentage.

## Current baseline

`evidence/readiness-input.json` is deliberately conservative. It records known missing release evidence rather than fabricating readiness from source-code presence.

At the time this gate was introduced, the current repository ByteTokenizer already provided byte-level round trip and no unknown token, but its stable control-token set did not yet include `<mathesis>`, `<cain>`, or `<oro>`. The tokenizer audit therefore treats that inventory as incomplete for the next release family. Existing checkpoint control IDs must not be renumbered to fix this; migration must append new stable controls or introduce a versioned tokenizer while preserving old checkpoint loadability.

## Runner failure boundary

A GitHub Actions run that terminates before step 1 with no logs is infrastructure unavailability, not evidence that tests passed or failed. Release promotion remains blocked until equivalent executable evidence exists from a valid runner or another reproducible execution environment.
