# AURO / NOVA / Relay Evidence and Production Boundaries

This document is the canonical correction layer for architecture, checkpoint,
context, perception, observation, and continuous-training claims.

## Context planes

AURO preserves two different context systems. They are complementary and must
not be described as the same mechanism.

1. **Persistent logical context** uses SQLite/WAL/FTS5 storage and bounded
   retrieval. Its roughly 500K configured logical capacity, and prior tests with
   more than one million stored tokens, do not represent a transformer attention
   window. Each call receives a bounded evidence pack.
2. **Governed accepted-context envelope** accepts at most 294,912 token IDs,
   selects at most 8,192 historical tokens, and emits at most 32,768 tokens into
   the dense model-facing view. The envelope is deterministic and receipt-bearing,
   but it is not 294K-token dense attention.

## Model-family and Auro-4B claims

Family-wide MoE flags and expanded context values are architecture contracts.
They are not evidence that corresponding checkpoints were trained, that expert
routing is balanced, or that language quality holds at the declared context.

Auro-4B denotes approximately four billion active parameters per token under
its sparse routing contract. Stored expert capacity is larger. Both numbers must
be reported together. A model-family name is never substituted for a checkpoint
manifest, weight custody, tokenizer custody, evaluation, or promotion receipt.

## Checkpoint custody

`inventory_auro_checkpoints.py` now separates:

- artifact presence;
- manifest presence;
- manifest-to-weight and manifest-to-tokenizer hash agreement;
- declared geometry and parameter count;
- exact-checkpoint evaluation evidence;
- signed constitutional promotion.

A dummy manifest plus arbitrary weight bytes intentionally fails
`evidence_complete`. Local Auro-2B availability remains an operator claim until
this audit runs on the actual machine and reports `promotion_ready=true`.

## Relay and SignalLens

NEXUS Relay now signs normalized read receipts with
`RELAY_RECEIPT_SIGNING_KEY`. AURO rejects Relay learning evidence unless:

- issuer is `nexus-relay`;
- the HMAC signature verifies with `NEXUS_RELAY_RECEIPT_KEY`;
- the normalized content hash matches;
- citations contain valid HTTP(S) URLs.

The pinned egress request signature now includes a one-time nonce. The egress
service persists nonce claims in `RELAY_EGRESS_NONCE_STORE` and rejects replay,
including after process restart. Production deployment must mount that path on
durable storage; `/tmp` only protects across restarts while the filesystem is
retained.

`SignalLensRelayPerception` is the canonical NOVA adapter. It has no direct or
simulated fallback. Configuration proves only that wiring exists. A successful,
hash-verified Relay response is required before a run may state that the deployed
cross-repository perception path was observed.

## NOVA authorization and state

A nonempty `approval_id` is not authorization. `NovaRuntimeState` binds each
approval to:

- a durable session;
- a principal;
- the SHA-256 of the exact action;
- an expiry;
- one-time consumption.

Sessions, approvals, replay keys, and runtime receipts are stored in SQLite WAL.
Receipts are chained. Replayed events and reused approvals fail closed.

## HIM observation

The observation harness no longer silently swallows checkpoint-load failures.
If a candidate checkpoint exists and fails to load, the run fails. Lightweight
construction requires explicit authorization and is labelled
`lightweight_fixture`.

A passing observation requires every record to be successful, usable, receipt-
bearing, and part of a valid chain. Promotion observation additionally requires
a trusted HMAC signer. The report states explicitly that:

- no optimizer update occurred;
- same-session evidence is not durable cross-session memory;
- an observation trajectory is not checkpoint promotion;
- unsigned self-generated hashes do not establish external custody.

## Continuous training

The continuous fleet remains bounded and governed, but its execution plane now
includes `DurableJobQueue` and `run_continuous_worker.py`:

- SQLite WAL queue;
- atomic worker leases;
- expired-lease recovery;
- bounded retries;
- allowlisted training entrypoints;
- required resume checkpoint;
- subprocess result checks;
- `HIM_SFT_REPORT.json` verification;
- candidate artifact hashes;
- training execution receipt.

A job specification or queued record is not training completion. Completion is
recorded only after the declared checkpoint directory and evidence artifacts are
verified.

## Receipt custody

Continuous-agent receipts are hash-chained. Trusted workers can HMAC-sign them
with `AURO_RECEIPT_SIGNING_KEY` and identify custody through
`AURO_RECEIPT_SIGNER_ID`. Unsigned development receipts remain inspectable but
must not satisfy promotion.

## CI and GitHub status

The `Evidence Security Production Gate` attaches to every pull request and every
push to `main`; it does not depend on path filters. GitHub merge text is not proof
that checks ran. Claims about validation must cite an exposed workflow run,
completed job steps, and uploaded evidence artifact for the exact head SHA.
