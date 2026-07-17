# NOVA Production Fleet

This runtime turns Auro/Medina checkpoint endpoints into usable internal workers.

## Start Medina-Native-8B

Run the loopback service from `NATIVE-NOVA-PROTOCOL/models/medina-native-8b`, then:

```bash
export AURO_BASE_URL=http://127.0.0.1:8088/v1
export AURO_MODEL=medina-native-8b
export AURO_PARAMETER_COUNT=8200000000
python -m auro_native_llm.production_fleet.cli "Analyze my repositories and propose the next bounded build"
```

Start the BRAIN AI-facing loopback API:

```bash
python -m auro_native_llm.production_fleet.server --host 127.0.0.1 --port 8090
curl -X POST http://127.0.0.1:8090/v1/respond -H 'content-type: application/json' -d '{"message":"Create a plan and have the internal council verify it"}'
```

`AURO_PARAMETER_COUNT` must come from the loaded checkpoint/config. Remove it when unverified. Agent count never changes model parameter count.

## Cognitive cycle

NOVA runs five internal roles over the same real model endpoint: SENSUS, MATHESIS, architect, red team, and operator. It then requests a final synthesis. Responses contain an answer, concise reasoning summary, evidence, confidence, agent receipts, and proposed actions.

Actions are proposals by default. `--execute` only marks bounded MatDaemon/CAPSULA proposals as approved for a separate governed executor; the runtime never falsely claims execution.

## Integration ownership

- BRAIN AI: brain state, identity, continuity, agent UI.
- NOVA: policy, council routing, arbitration, receipts.
- Auro or Medina-Native-8B: checkpoint-backed text inference.
- MatDaemon: embedding similarity, memory and agent routing compute.
- CAPSULA: approved code/build capsule execution.

## Injected organ SDK

`AuroOrganSDK` gives the model one stable Python surface over all four systems:

```python
from auro_native_llm.production_fleet import AuroOrganSDK, NovaRuntime

sdk = AuroOrganSDK()
print(sdk.manifest())
answer = NovaRuntime(sdk=sdk).respond("Rank these memories, build the selected capsule, and report the evidence", execute=True)
```

Environment endpoints: `BRAIN_AI_URL`, `NOVA_URL`, `MATDAEMON_URL`, and `CAPSULA_URL`. MatDaemon calls must use its declared `matdaemon_*` tools. CAPSULA is restricted to session creation, file writes, runs, manifests, and deploy plans. BRAIN AI remains state/continuity input; NOVA owns authorization.

The native Auro checkpoint should not be the default chat model until fixed-prompt generation and holdout gates pass. Medina-Native-8B supplies the immediate usable inference lane while native Auro training continues.

## Native skills and MCP replacement layer

Auro now exposes one internal capability registry rather than requiring a separate install for every skill or MCP server. `GET /v1/capabilities` returns schemas, permissions, organs, and playbooks. `POST /v1/capabilities/call` invokes a capability with `{name, arguments, approved}`.

Built-ins cover BRAIN state, operator snapshots, memory ranking, matrix compute, CAPSULA build sessions, research, reasoning, building, and continuity. New capabilities join the same registry; the model does not need a new prompting convention or external plugin wrapper for each one. Mutating build calls remain approval-gated and all calls return receipts.
