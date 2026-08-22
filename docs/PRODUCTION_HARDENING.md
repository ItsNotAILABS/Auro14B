# AURO Production Hardening

This document describes the production authority, persistence, and evidence-durability controls added after the feline-inspired neuromorphic substrate landed on `main`.

## Production mode

Set `AURO_ENV=production` (or `AURO_PRODUCTION=1`) for a production launch. Production startup fails closed unless all three security secrets below are present, at least 32 characters long, and distinct from one another:

- `AURO_API_TOKEN`: bearer token for protected API routes.
- `AURO_EXECUTION_TOKEN`: operator authorization required before mutation-capable HTTP paths are accepted.
- `AURO_APPROVAL_HMAC_KEY`: server-side signing/verifying key for exact action approvals.

`GET /v1/health/ready` returns HTTP 503 when the production security contract or receipt-chain integrity is not ready. Liveness remains separate from readiness.

Secrets must not be exposed to model prompts, browser clients, end users, generated tool arguments, logs, or receipts.

## Security boundary

Mutating native capabilities no longer accept a caller-controlled `approved: bool` value. Mutation authority is represented by a signed server approval grant verified with `AURO_APPROVAL_HMAC_KEY`.

A newly issued server approval binds:

- schema and server authority;
- approval ID and subject;
- cryptographic nonce;
- the exact ordered action set;
- SHA-256 of that action set;
- not-before and expiry timestamps;
- HMAC-SHA256 signature.

Legacy v1 grants remain verification-compatible during migration. New grants are v2. Every approved action is consumed exactly once through an atomic replay marker. A valid signature is necessary but not sufficient for execution: the exact action must be in the signed set, within its time window, and unused.

### Replay state

`AURO_APPROVAL_REPLAY_DIR` controls where one-time action-consumption markers are stored. Default: `./state/approval-replay`.

The directory must be writable only by the AURO service identity and live on storage with atomic create semantics. Replay markers are written with restrictive permissions, fsynced, and can be pruned after their approvals expire. Consumption happens before dispatch to provide at-most-once mutation semantics. If a process dies after consumption but before a remote side effect can be conclusively observed, operators must treat the result as ambiguous and reconcile from downstream receipts/state rather than replay the grant.

## HTTP execution contract

`POST /v1/capabilities/call` behaves as follows:

- read-only capability: API authentication only;
- mutation-capable capability: API authentication + execution token; the server issues an exact action-bound approval grant internally and consumes it once;
- caller-provided boolean approval is not an authority source.

`POST /v1/respond` accepts `approval_grant` when `execute=true`.

`POST /v1/chat/completions` accepts `auro_approval_grant` when `auro_execute=true`.

These model execution routes require both the operator execution token and a server-signed grant matching the exact model-proposed action list. The model cannot mint, broaden, or reuse authority.

Browser worker state mutations are also operator-authorized: task claim and task completion require the execution token. Read-only task listing remains under normal API authentication.

## Neuromorphic persistence

Neuromorphic state persistence now uses:

- validation before mutation of live engine state;
- SHA-256 integrity sealing;
- finite/non-negative numeric checks;
- exact region inventory validation;
- edge inventory validation;
- timing-plasticity validation;
- exclusive temporary-file creation;
- file `fsync` before replacement;
- atomic `os.replace`;
- parent-directory `fsync` where supported;
- corrupt-state quarantine rather than partial restore.

A corrupt state does not silently pass. The brain starts from clean neuromorphic defaults, and `neuromorphic_persistence.degraded`, `last_error`, and `quarantined_path` expose the failure to operators. The stable public brain schema remains `him.brain.v2.neuromorphic`; persistence hardening does not gratuitously break that external identifier.

The neuromorphic substrate remains telemetry/control architecture. It does not grant execution authority, prove checkpoint quality improvements, imply biological equivalence, or establish physical energy efficiency.

## Receipt durability

The receipt ledger retains its existing verification response contract while hardening storage semantics:

- a receipt is durably appended before it is admitted into in-memory ledger state;
- append files use restrictive creation permissions;
- each append is `fsync`ed and the parent directory is synced where supported;
- startup rejects incomplete trailing records;
- startup rejects malformed or hash-chain-tampered records.

This preserves the receipt chain as evidence across normal process restarts and common torn-write failure modes. It does not turn the local ledger into a distributed consensus system.

## Verification

Focused regression tests:

```bash
pytest -q \
  tests/test_production_authority_hardening.py \
  tests/test_neuromorphic_persistence_hardening.py \
  tests/test_receipt_ledger_hardening.py \
  tests/test_feline_neuromorphic.py \
  tests/test_timing_plasticity.py \
  tests/test_neuromorphic_bridge.py
```

Standalone verifier:

```bash
python scripts/verify_production_hardening.py
```

The production hardening gate also compiles the package and enforces the invariant that `NativeCapabilities.call` has no caller-supplied `approved` parameter and that model execution routes carry signed grants into the runtime.

## Evidence boundary

A merge, successful import, or unit-test pass is not evidence that an AURO language checkpoint became more capable. Checkpoint promotion still requires exact-checkpoint training/evaluation evidence and the repository release/readiness gates. The controls in this hardening pass establish authority, replay, persistence, readiness, and operational evidence boundaries only.
