# AURO Production Hardening

This document describes the production authority and neuromorphic persistence controls added after the feline-inspired neuromorphic substrate landed on `main`.

## Security boundary

Mutating native capabilities no longer accept a caller-controlled `approved: bool` value. The only accepted mutation authority is a signed server approval grant verified with `AURO_APPROVAL_HMAC_KEY`.

A server approval binds:

- schema and authority;
- approval ID and subject;
- cryptographic nonce for v2 grants;
- the exact ordered action set;
- SHA-256 of that action set;
- not-before and expiry timestamps;
- HMAC-SHA256 signature.

Every approved action is consumed exactly once through an atomic replay marker. A valid signature is therefore necessary but not sufficient for execution: the action must also be a member of the signed set and unused.

### Required production secrets

- `AURO_API_TOKEN`: bearer token for protected API routes.
- `AURO_EXECUTION_TOKEN`: operator authorization required before mutation-capable HTTP paths are accepted.
- `AURO_APPROVAL_HMAC_KEY`: server-side signing/verifying key for exact action approvals.

`AURO_APPROVAL_HMAC_KEY` must not be exposed to model prompts, browser clients, end users, or generated tool arguments.

### Replay state

`AURO_APPROVAL_REPLAY_DIR` controls where one-time action consumption markers are stored. Default: `./state/approval-replay`.

The directory must be writable only by the AURO service identity and live on storage with atomic create semantics. Consumption happens before dispatch to provide at-most-once mutation semantics. If a process dies after consumption but before a remote side effect can be conclusively observed, operators must treat the action result as ambiguous and reconcile from downstream receipts/state rather than replay the grant.

## HTTP execution contract

`POST /v1/capabilities/call` behaves as follows:

- read-only capability: API authentication only;
- mutation-capable capability: API authentication + execution token; the server issues an exact action-bound approval grant internally and consumes it once;
- caller-provided boolean approval is not an authority source.

`POST /v1/respond` accepts `approval_grant` when `execute=true`.

`POST /v1/chat/completions` accepts `auro_approval_grant` when `auro_execute=true`.

These model execution routes require both the operator execution token and a server-signed grant matching the exact model-proposed action list. The model cannot mint or widen authority.

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

A corrupt state does not silently pass. The brain starts from clean neuromorphic defaults, and `neuromorphic_persistence.degraded`, `last_error`, and `quarantined_path` expose the failure to operators.

The neuromorphic substrate remains telemetry/control architecture. It does not grant execution authority, prove checkpoint quality improvements, imply biological equivalence, or establish physical energy efficiency.

## Verification

Focused regression tests:

```bash
pytest -q \
  tests/test_production_authority_hardening.py \
  tests/test_neuromorphic_persistence_hardening.py \
  tests/test_feline_neuromorphic.py \
  tests/test_timing_plasticity.py \
  tests/test_neuromorphic_bridge.py
```

Static/runtime-independent verifier:

```bash
python scripts/verify_production_hardening.py
```

The production hardening gate also compiles the package and enforces the invariant that `NativeCapabilities.call` has no caller-supplied `approved` parameter.

## Evidence boundary

A merge, successful import, or unit-test pass is not evidence that an AURO language checkpoint became more capable. Checkpoint promotion still requires exact-checkpoint training/evaluation evidence and the repository release/readiness gates. The controls in this hardening pass establish authority, replay, persistence, and operational correctness boundaries only.
