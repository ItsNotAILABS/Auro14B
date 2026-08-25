# AURO Multi-Task Runtime v1

## Purpose

The AURO task runtime turns a large objective into a durable, inspectable body of work. It supports multiple dependent tasks, long-running workers, explicit review stages, progress events, artifact delivery, retries, pause/resume, cancellation, and a final evidence bundle.

It does **not** expose private chain-of-thought. It records bounded reasoning summaries, decisions, assumptions, evidence references, validations, and artifacts.

```text
User objective
    ↓
Explicit task graph or Auro-2B council planning
    ↓
Durable SQLite/WAL run
    ↓
Ready tasks leased to bounded workers
    ↓
Research / code / build / test / writing / packaging
    ↓
Independent reviews and consensus stages
    ↓
Final synthesis
    ↓
Artifact quality gate
    ↓
Manifest + receipt + downloadable ZIP
```

## What changed

The former request path could answer one turn and the existing continuous queue could schedule one bounded training command. The new runtime adds a general-purpose task graph for work that may contain many deliverables and may continue across process restarts.

### Multi-task execution

A run may contain up to the operator-approved budget of tasks. Each task declares:

- an immutable step ID;
- title and objective;
- task kind;
- dependencies;
- required worker capabilities;
- priority;
- retry and timeout policy;
- artifact contract;
- validation contract;
- reasoning depth;
- risk class;
- approval requirement;
- metadata.

Cycles and unknown dependencies are rejected before the run is stored.

### Long-running work

Each executable task is leased to a worker through a one-time random lease token. Workers may:

- claim a ready task;
- create files in an isolated step workspace;
- send heartbeat renewals;
- emit bounded progress events;
- complete with structured output and artifacts;
- fail explicitly and request a bounded retry.

SQLite WAL state preserves runs, steps, attempts, leases, events, and artifacts across restarts. Expired leases are reissued only when attempts remain.

### Deeper reasoning without hidden chain-of-thought

`quality_mode` controls explicit reasoning stages:

| Mode | Graph behavior |
|---|---|
| `fast` | Requested tasks only |
| `standard` | Requested tasks plus final synthesis |
| `deep` | Independent review of each task plus final synthesis |
| `exhaustive` | Per-task review, final synthesis, and complete delivery quality gate |

A review step checks correctness, evidence, completeness, contradictions, security, and artifact validity. The final synthesis integrates outputs and preserves disagreements. The quality gate validates the delivery against the original objective.

These stages make the reasoning process deeper and auditable without exporting internal token-by-token model cognition.

## Data model

The runtime owns four durable tables:

```text
task_runs
  identity, objective, plan, budget, status, scope, result

task_steps
  dependencies, status, attempts, lease, approval, output, validation

task_artifacts
  content hash, media type, size, path, metadata, lineage

task_events
  ordered hash-linked state transitions and progress records
```

Every event binds the previous event hash. Final receipts bind:

- objective hash;
- plan hash;
- artifact-manifest hash;
- event-chain root;
- principal and organization;
- final status.

An optional `AURO_TASK_RECEIPT_HMAC_KEY` produces an E4 local signed receipt. It still does not establish E5 external custody.

## Creating a run

A caller may submit an explicit task graph:

```json
{
  "objective": "Research, build, test, document, and package the feature",
  "quality_mode": "exhaustive",
  "budget": {
    "max_steps": 64,
    "max_runtime_seconds": 172800,
    "max_artifact_bytes": 536870912,
    "max_total_attempts": 128
  },
  "tasks": [
    {
      "step_id": "research",
      "title": "Research",
      "objective": "Collect primary evidence and requirements",
      "kind": "research",
      "required_capabilities": ["research"],
      "artifacts": [
        {"name": "RESEARCH.md", "media_type": "text/markdown"}
      ]
    },
    {
      "step_id": "implementation",
      "title": "Implementation",
      "objective": "Implement the approved architecture",
      "kind": "code",
      "dependencies": ["research"],
      "required_capabilities": ["code"],
      "artifacts": [
        {"name": "implementation.patch", "media_type": "text/x-diff"}
      ]
    },
    {
      "step_id": "tests",
      "title": "Tests",
      "objective": "Run regression, security, and integration tests",
      "kind": "test",
      "dependencies": ["implementation"],
      "required_capabilities": ["test"],
      "artifacts": [
        {"name": "TEST_RESULTS.json", "media_type": "application/json"}
      ]
    }
  ]
}
```

The runtime expands this in exhaustive mode with:

```text
review:research
review:implementation
review:tests
final-synthesis
delivery-quality-gate
```

### Council planning

A caller may instead set:

```json
{
  "objective": "Produce the complete release package",
  "plan_with_council": true,
  "quality_mode": "deep",
  "deliverables": ["source", "tests", "documentation", "release archive"]
}
```

This path fails closed unless the Auro-2B council is configured and returns a valid task-plan contract. A conversational answer is not silently accepted as a plan.

## Worker protocol

### Claim

A worker advertises capabilities and requests a task:

```json
{
  "worker_id": "pocket-agent-01",
  "capabilities": ["research", "code", "test", "review", "synthesis"],
  "lease_seconds": 900
}
```

The response contains:

- task contract;
- lease owner;
- lease expiry;
- one-time lease token;
- isolated workspace path.

The lease token is stored only as a SHA-256 digest. A different worker or stale token cannot complete the step.

### Progress and heartbeat

A worker may report:

```json
{
  "percent": 45,
  "current_operation": "running integration tests",
  "completed_units": 18,
  "total_units": 40,
  "reasoning_summary": ["The initial implementation exposed a schema mismatch"]
}
```

Progress events are bounded and hash-linked. They are not interpreted as proof of successful execution.

### Completion

A worker completes with structured output and artifacts:

```json
{
  "output": {
    "summary": "Implementation and tests completed",
    "decisions": ["Preserved the existing public API"],
    "evidence_refs": ["artifact:TEST_RESULTS.json"]
  },
  "artifacts": [
    {
      "name": "TEST_RESULTS.json",
      "media_type": "application/json",
      "json": {"passed": 182, "failed": 0}
    },
    {
      "name": "release.zip",
      "media_type": "application/zip",
      "workspace_path": "dist/release.zip"
    }
  ],
  "validation": {"passed": true}
}
```

Inline artifacts are bounded. Larger outputs must be written into the step workspace and registered with `workspace_path`. Relative paths are checked to prevent traversal outside the workspace.

## Artifact delivery

Each stored artifact records:

```json
{
  "schema": "auro.task-artifact.v1",
  "artifact_id": "artifact-...",
  "run_id": "run-...",
  "step_id": "tests",
  "name": "TEST_RESULTS.json",
  "media_type": "application/json",
  "bytes": 1284,
  "sha256": "...",
  "relative_path": "artifacts/tests/TEST_RESULTS.json"
}
```

At terminal state the runtime emits:

```text
ARTIFACT_MANIFEST.json
RUN_RECEIPT.json
RUN_STATE.json       when a bundle is requested
EVENTS.json          when a bundle is requested
<all delivered artifacts>
<run-id>-delivery.zip
```

The ZIP is a delivery bundle, not proof that every contained claim is correct. The manifest and validations state the evidence available.

## Status transitions

### Run

```text
queued
running
paused
awaiting_approval
succeeded
partial
failed
cancelled
```

### Step

```text
pending
ready
awaiting_approval
leased
running
retry_wait
succeeded
failed
blocked
cancelled
skipped
```

A downstream step becomes blocked when a required dependency fails, is cancelled, or is blocked.

## Approval boundary

Risk classes 3–5 require approval by default. The task runtime does not accept a caller-supplied boolean or a non-empty approval ID. It requires a trusted `approval_verifier` callback supplied by the host runtime. Without one, the step remains `awaiting_approval`.

Approval verification belongs to the existing governed execution plane and should bind:

- operator;
- organization;
- run and step;
- exact action hash;
- runtime cell;
- filesystem and egress scope;
- expiry;
- nonce;
- receipt.

## Multi-user boundary

Every run is scoped to a `principal_id` and optional `organization_id`. Reads require matching scope. Worker mutation routes remain protected by the production execution token because a worker is a separate machine principal.

POCKET remains the canonical user, organization, entitlement, and RBAC authority. The AURO runtime preserves the supplied scope but does not replace POCKET authentication.

## Environment

```text
AURO_TASK_DB=state/task-runs.sqlite3
AURO_TASK_ARTIFACT_ROOT=state/task-artifacts
AURO_TASK_RECEIPT_HMAC_KEY=<32+ character operator secret>
AURO_TASK_RECEIPT_SIGNER=auro-task-runtime
```

The production HTTP API also requires the existing:

```text
AURO_API_TOKEN
AURO_EXECUTION_TOKEN
AURO_APPROVAL_HMAC_KEY
```

## API surface

```text
GET  /v1/task-runtime
GET  /v1/task-runs
POST /v1/task-runs
GET  /v1/task-runs/{run_id}
GET  /v1/task-runs/{run_id}/events
GET  /v1/task-runs/{run_id}/bundle.zip
POST /v1/task-runs/{run_id}/pause
POST /v1/task-runs/{run_id}/resume
POST /v1/task-runs/{run_id}/cancel
POST /v1/task-runs/{run_id}/claim
POST /v1/task-runs/{run_id}/steps/{step_id}/heartbeat
POST /v1/task-runs/{run_id}/steps/{step_id}/progress
POST /v1/task-runs/{run_id}/steps/{step_id}/complete
POST /v1/task-runs/{run_id}/steps/{step_id}/fail
```

API requests provide scope through:

```text
x-auro-principal-id
x-auro-organization-id   optional
```

Task creation and reads require API authentication. Worker claims, heartbeats, completions, failures, pause/resume, cancellation, and bundle mutation use the execution-authenticated path.

## Failure behavior

The runtime fails visibly when:

- a dependency graph is cyclic;
- an unknown dependency is referenced;
- a task exceeds the run budget;
- a lease expires;
- a worker or lease token does not match;
- a required artifact is missing;
- a validation contract fails;
- an artifact escapes the workspace;
- a high-risk step lacks verified approval;
- the council planning contract is malformed;
- a downstream dependency fails.

No failure is converted into a successful conversational answer.

## Evidence boundary

The runtime establishes source-level and deterministic local behavior when its focused tests pass. It does not by itself establish:

- that an external worker executed;
- that a language model reasoned correctly;
- that a generated artifact is commercially useful;
- that a deployment is live;
- that a checkpoint is trained or promoted;
- that a local signature has external custody.

Those claims require their own execution, validation, checkpoint, deployment, and custody evidence.
