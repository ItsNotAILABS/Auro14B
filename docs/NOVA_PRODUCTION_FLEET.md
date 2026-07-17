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

The native Auro checkpoint should not be the default chat model until fixed-prompt generation and holdout gates pass. Medina-Native-8B supplies the immediate usable inference lane while native Auro training continues.
