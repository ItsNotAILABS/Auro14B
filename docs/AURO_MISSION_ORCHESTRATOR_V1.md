# AURO Mission Orchestrator v1

## Purpose

The Mission Orchestrator turns a single user objective into a durable graph of
bounded tasks that can run in parallel, survive process restarts, preserve
progress, produce user-facing artifacts, and stop cleanly when evidence or
execution requirements are not met.

It extends the existing Auro-2B council runtime. It does not replace HIM/NOVA,
MESIE, the model family, checkpoint custody, the continuous-improvement fleet,
or the approval-bound execution organs.

```text
User objective
    |
    v
Mission planner
    |
    +-- interpret scope
    +-- evidence review ---------+
    +-- solution development ----+-- can run concurrently
                                  |
                                  v
                             deliverables
                                  |
                                  v
                              red-team
                                  |
                                  v
                              synthesis
                                  |
                                  v
                         manifest + ZIP bundle
```

## What it adds

- explicit multi-task requests;
- default deep seven-stage mission plans;
- dependency-aware DAG scheduling;
- parallel execution of independent ready tasks;
- durable SQLite/WAL state;
- idempotent mission creation;
- task leases, heartbeats, expiry recovery, and bounded retry;
- pause, resume, and cancellation;
- organization and operator ownership;
- multi-pass Auro-2B council deliberation;
- bounded decision summaries instead of private chain-of-thought export;
- content-addressed artifacts;
- artifact manifests and downloadable ZIP bundles;
- a long-running worker mode and bounded API-triggered bursts.

## Reasoning model

A task may request one to six council passes.

```text
pass 1  independent analysis
pass 2  critique, alternatives, contradiction search
pass N  corrected final decision
```

Each pass is a new Auro-2B council turn. Every turn may activate SENSUS,
PRAXIS, VERBUM, Auro-250M workers, Auro-156K workers, and MESIE stage
analysis. The orchestrator stores:

- the final answer;
- a bounded reasoning/decision summary;
- alternatives considered;
- evidence and receipt references;
- confidence;
- blockers;
- artifact hashes.

It does **not** store or return hidden chain-of-thought.

## Default mission plan

When the caller supplies an objective but no explicit task graph, the planner
creates:

1. `interpret` — scope, assumptions, constraints, unknowns, success criteria;
2. `evidence` — strongest evidence, contradictions, risks, missing validation;
3. `solution` — implementation or practical work product;
4. `deliverables` — requested user-facing artifacts;
5. `red-team` — defects, unsafe assumptions, evidence gaps, usability problems;
6. `synthesize` — reconciled final result and handoff;
7. `package` — artifact manifest and ZIP bundle.

`evidence` and `solution` become ready together after `interpret`, allowing
parallel work without violating dependencies.

## Custom task graph

```json
{
  "title": "Release package",
  "objective": "Research, implement, test, and package the feature.",
  "max_parallel": 4,
  "idempotency_key": "release-package-2026-08-25",
  "tasks": [
    {
      "task_id": "research",
      "title": "Research the current system",
      "objective": "Inspect current implementation and produce a gap analysis.",
      "kind": "research",
      "depends_on": [],
      "reasoning_rounds": 3,
      "required_artifacts": [
        "research/gap-analysis.md",
        "research/gap-analysis.json"
      ]
    },
    {
      "task_id": "implement",
      "title": "Implement the change",
      "objective": "Create the production implementation and tests.",
      "kind": "implementation",
      "depends_on": ["research"],
      "reasoning_rounds": 3,
      "required_artifacts": ["implementation/report.md"]
    },
    {
      "task_id": "review",
      "title": "Independent review",
      "objective": "Red-team correctness, security, and product usability.",
      "kind": "review",
      "depends_on": ["implement"],
      "reasoning_rounds": 3,
      "required_artifacts": ["review/findings.md"]
    },
    {
      "task_id": "package",
      "title": "Package the result",
      "objective": "Create the artifact manifest and download bundle.",
      "kind": "package",
      "depends_on": ["review"]
    }
  ]
}
```

Cycles, duplicate task IDs, and missing dependencies are rejected before a
mission is written.

## Product API

Run the additive mission-aware server:

```bash
python -m auro_native_llm.production_fleet.mission_server \
  --host 127.0.0.1 \
  --port 8090
```

The existing HIM, council, model, context, capability, and receipt routes remain
available because `MissionHandler` subclasses the current production handler.

Required identity headers:

```text
x-auro-operator-id
x-auro-organization-id
```

Create a mission:

```bash
curl -s http://127.0.0.1:8090/v1/missions \
  -H "authorization: Bearer $AURO_API_TOKEN" \
  -H "content-type: application/json" \
  -H "x-auro-operator-id: alfredo" \
  -H "x-auro-organization-id: itsnotai" \
  -d '{
    "title":"Deep release analysis",
    "objective":"Analyze the release, improve it, red-team it, and deliver all artifacts.",
    "max_parallel":3,
    "deliverables":[
      "deliverables/release-report.md",
      "deliverables/release-report.json"
    ]
  }'
```

Run a bounded execution burst:

```bash
curl -s -X POST \
  http://127.0.0.1:8090/v1/missions/MISSION_ID/run \
  -H "authorization: Bearer $AURO_API_TOKEN" \
  -H "x-auro-execution-token: $AURO_EXECUTION_TOKEN" \
  -H "content-type: application/json" \
  -H "x-auro-operator-id: alfredo" \
  -H "x-auro-organization-id: itsnotai" \
  -d '{
    "worker_id":"operator-laptop",
    "max_tasks":20,
    "time_budget_seconds":900,
    "capabilities":["council","artifact-write","package"]
  }'
```

Inspect a mission:

```bash
curl -s http://127.0.0.1:8090/v1/missions/MISSION_ID \
  -H "authorization: Bearer $AURO_API_TOKEN" \
  -H "x-auro-operator-id: alfredo" \
  -H "x-auro-organization-id: itsnotai"
```

Download an artifact:

```bash
curl -o mission-artifacts.zip \
  http://127.0.0.1:8090/v1/missions/MISSION_ID/artifacts/mission-artifacts.zip \
  -H "authorization: Bearer $AURO_API_TOKEN" \
  -H "x-auro-operator-id: alfredo" \
  -H "x-auro-organization-id: itsnotai"
```

Pause, resume, or cancel:

```text
POST /v1/missions/{mission_id}/pause
POST /v1/missions/{mission_id}/resume
POST /v1/missions/{mission_id}/cancel
```

These mutation routes require the normal AURO execution token.

## Long-running worker

```bash
python -m auro_native_llm.tasks.worker \
  --worker-id operator-laptop \
  --max-tasks-per-burst 8 \
  --time-budget-seconds 240
```

The worker repeatedly finds queued/running missions and executes bounded
bursts. Mission state, task results, retries, decisions, and artifacts survive
worker restarts.

A one-pass mode is available for Cron, systemd timers, Cloudflare-governed
remote dispatch, or an external supervisor:

```bash
python -m auro_native_llm.tasks.worker --once
```

## Environment

```text
AURO_API_TOKEN
AURO_EXECUTION_TOKEN
AURO_APPROVAL_HMAC_KEY
AURO_COUNCIL_CONFIG_JSON or AURO_COUNCIL_CONFIG_PATH
AURO_COUNCIL_RECEIPT_HMAC_KEY
AURO_COUNCIL_RECEIPT_SIGNER
AURO_MISSION_DB
AURO_MISSION_ARTIFACT_ROOT
AURO_MISSION_MAX_ARTIFACT_BYTES
```

The council must be configured before reasoning/research/implementation tasks
can execute. Unconfigured council work fails visibly rather than falling back to
an unrelated model.

## Artifact custody

Every write is:

- confined to the mission root;
- protected against absolute paths and `..` traversal;
- performed through a temporary file followed by atomic replacement;
- hashed with SHA-256;
- recorded with media type, size, task, and label.

The final package includes `ARTIFACT_MANIFEST.json` and
`mission-artifacts.zip`.

The hash proves the bytes delivered by this runtime. It does not prove that the
artifact is factually correct, aesthetically good, deployed, signed by an
external custodian, or accepted by a customer.

## Evidence boundaries

This subsystem can establish source/test evidence for:

- task graph construction;
- dependency ordering;
- durable state transitions;
- tenant ownership checks;
- lease recovery;
- multi-pass council invocation;
- artifact production and hashing;
- bundle generation.

It does not by itself establish:

- trained checkpoint quality;
- successful external tool execution;
- production deployment;
- clean customer installation;
- correctness of generated content;
- external E5 evidence custody;
- completion while no worker or scheduler is running.

The durable worker makes long work resumable. It is not a promise of invisible
background execution without an actual running worker, service manager, Cron,
Cloudflare Workflow, or other supervisor.
