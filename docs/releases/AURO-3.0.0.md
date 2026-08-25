# AURO 3.0.0 — Independent Harness Fabric

AURO 3.0.0 adds durable, recursively composable work harnesses for multi-hour and multi-day deployed execution.

## Architecture

A harness is a complete independent work instance with its own:

- persistent state directory;
- objective and task DAG;
- role-specific agent roster;
- event hash chain;
- worker lease and heartbeat state;
- retries and resume point;
- child-harness lineage;
- result aggregation;
- self-written reusable skill artifacts.

A parent harness may fan out independent child harnesses. Children can be advanced by separate workers and survive process restart. The parent rejoins child results when the tree reaches a terminal state.

## Production surfaces

### Python

- `auro_native_llm.work.IndependentHarnessFabric`
- `auro_native_llm.work.HarnessOrchestrator`
- `auro_native_llm.work.HarnessSkillForge`
- `mesie.foundation.harness_bridge.MesieHarnessBridge`

### CLI

```bash
python -m auro_native_llm.work.harness_cli orchestrate "build and review a production subsystem"
python -m auro_native_llm.work.harness_cli list
python -m auro_native_llm.work.harness_cli advance-tree <HARNESS_ID>
python -m auro_native_llm.work.harness_cli pause <HARNESS_ID>
python -m auro_native_llm.work.harness_cli resume <HARNESS_ID>
```

### Resident worker

```bash
python -m auro_native_llm.work.harness_worker --poll-seconds 5 --cycles 1
```

The worker scans durable active harnesses, acquires leases, advances bounded cycles, persists state, and repeats. If it exits, another worker can resume from persisted state.

### Harness Control API

```bash
export AURO_HARNESS_TOKEN='replace-with-a-long-production-token'
python -m auro_native_llm.work.harness_server
```

Routes:

- `GET /health`
- `GET /v1/manifest`
- `GET /v1/harnesses`
- `GET /v1/harnesses/{id}`
- `POST /v1/harnesses`
- `POST /v1/orchestrate`
- `POST /v1/harnesses/{id}/run`
- `POST /v1/harnesses/{id}/advance-tree`
- `POST /v1/harnesses/{id}/fanout`
- `POST /v1/harnesses/{id}/task`
- `POST /v1/harnesses/{id}/pause`
- `POST /v1/harnesses/{id}/resume`
- `POST /v1/harnesses/{id}/cancel`
- `POST /v1/harnesses/{id}/aggregate`

In production, the harness API requires a token of at least 32 characters.

## Self-learning skills

Completed harnesses can distill successful work into versioned `SkillArtifact` records. Later harness planners retrieve matching active skills as evidence. Skills track observed outcomes and can be superseded or retired when later evidence is worse.

Generated skill artifacts are procedure data, not silently executable Python.

## MESIE integration

The embedded MESIE source now exposes `MesieHarnessBridge`, allowing MESIE-native runtimes to create complete persistent harness trees while MESIE remains the model compute plane.

## Verification status

Focused regression tests are committed in `tests/test_independent_harness_fabric.py` covering persistence, fan-out isolation, rejoin, skill distillation, pause/resume, and lease exclusion.

Execution was attempted from the available build container, but the environment could not resolve `github.com`, so the branch could not be cloned and pytest could not start. No test-pass claim is made for this release until execution infrastructure is available.

## Claim boundary

This release provides software architecture for long-running deployed workers. ChatGPT itself is not claiming to continue work in the background after the conversation ends. A deployed `harness_worker` process or another caller must advance harness cycles.
