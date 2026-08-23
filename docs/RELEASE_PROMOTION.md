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

## Tokenizer v2 migration

The release tokenizer is versioned rather than renumbered.

Tokenizer v1 remains:

- control IDs `0..15`;
- byte IDs `16..271`.

Tokenizer v2 preserves those IDs exactly and appends load-bearing architecture controls after the byte vocabulary:

- `<mathesis>` = `272`;
- `<cain>` = `273`;
- `<oro>` = `274`.

This placement is deliberate. Inserting controls before the byte range would shift every byte embedding/output row and make old checkpoints semantically incompatible. Appending the controls keeps old control IDs and byte IDs unchanged.

New `OpenHIMConfig` checkpoints use tokenizer v2 by default. A legacy checkpoint whose `config.json` has no `tokenizer_version` is interpreted as v1 during load. The loader verifies tensor shapes after selecting the correct tokenizer vocabulary.

New training receipts bind `tokenizer_version` and the SHA-256 of `tokenizer.json`. The checkpoint verifier checks the tokenizer manifest's byte-lossless/no-UNK invariant, immutable control-ID layout when present, receipt tokenizer version, and receipt tokenizer hash.

Generation suppresses every control-token ID, including the v2 extension IDs, so these architecture tokens cannot appear as invisible sampled output through the byte decoder.

## Current baseline

`evidence/readiness-input.json` is deliberately conservative. It records known missing release evidence rather than fabricating readiness from source-code presence.

Tokenizer v2 architecture is now present, but tokenizer readiness is still not promoted to passing until an exact release checkpoint is trained/saved with v2 and its tokenizer audit receipt is pinned in the release evidence manifest.

## Runner failure boundary

A GitHub Actions run that terminates before step 1 with no logs is infrastructure unavailability, not evidence that tests passed or failed. Release promotion remains blocked until equivalent executable evidence exists from a valid runner or another reproducible execution environment.
