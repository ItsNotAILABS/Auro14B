# AURO-2B Council Runtime v1

The AURO-2B Council Runtime is the executable composition layer for:

```text
Auro-2B parent
  -> Auro-500M-SENSUS
  -> Auro-500M-PRAXIS
  -> Auro-500M-VERBUM
  -> topic-scoped Auro-250M and Auro-156K workers
  -> specialist synthesis
  -> triad consensus
  -> Auro-2B final synthesis
  -> Python/Pyodide conversational fluidizer
```

It extends the canonical AURO model family. It does not replace the existing
HIM/NOVA runtime, the MESIE compute plane, checkpoint custody, Browser-Brain,
or the family-wide MoE and context contracts.

## Production posture

The runtime is disabled unless an operator supplies an explicit endpoint
manifest. It never silently points all identities at the default model endpoint.

```text
AURO_COUNCIL_CONFIG_PATH=/absolute/path/council.json
# or
AURO_COUNCIL_CONFIG_JSON={...}

AURO_COUNCIL_RECEIPT_HMAC_KEY=<32+ character operator secret>
AURO_COUNCIL_RECEIPT_SIGNER=<stable signer identity>
```

Use the non-secret template:

```text
native_llm/configs/auro_2b_council.example.json
```

Validate the topology without making network calls:

```bash
python scripts/validate_auro2b_council_config.py \
  native_llm/configs/auro_2b_council.example.json
```

The example intentionally passes the topology gate with warnings because its
checkpoint and adapter fields are placeholders. A release configuration must
also pass:

```bash
python scripts/validate_auro2b_council_config.py \
  /secure/operator/council.json \
  --require-evidence
```

## Required identities

The configuration must contain exactly:

```text
Auro-2B
Auro-500M-SENSUS
Auro-500M-PRAXIS
Auro-500M-VERBUM
Auro-156K
Auro-250M
```

The three 500M specialists may share one verified base checkpoint only when each
has a distinct verified adapter. Merely assigning three names to one endpoint
does not establish three trained specialists.

Every identity records its own:

- parameter target;
- endpoint and served model name;
- exact checkpoint ID and SHA-256;
- optional adapter ID and SHA-256;
- measured parameter count when available;
- timeout and API-key reference.

Parameter counts remain separate. The council must never be represented as one
merged 3.5B checkpoint.

## Turn lifecycle

1. The Auro-2B parent receives the user turn.
2. MESIE creates an ingress receipt.
3. SENSUS, PRAXIS, and VERBUM run concurrently.
4. Each specialist receives topic-specific atomic tasks.
5. Atomic workers receive bounded capsules rather than the full parent context.
6. Every atomic stage receives a MESIE receipt.
7. Each 500M specialist synthesizes its atomic reports.
8. Every specialist independently reviews all three reports.
9. The three consensus votes preserve disagreement and evidence.
10. Auro-2B performs a final structured synthesis.
11. The deterministic Python/Pyodide fluidizer produces conversational text.
12. MESIE analyzes the final text and emits an egress receipt.
13. The runtime writes a hash-linked turn receipt and optionally signs it.

## API

The production server exposes authenticated routes:

```text
GET  /v1/council
POST /v1/council/respond
POST /v1/chat/completions  model=auro-2b-council
```

`GET /v1/council` returns configuration and custody status without exposing API
keys or secret values.

Native request:

```json
{
  "message": "Research the evidence, design the implementation, and explain it clearly.",
  "parent_context": "Optional bounded parent-only context"
}
```

OpenAI-compatible request:

```json
{
  "model": "auro-2b-council",
  "stream": false,
  "messages": [
    {"role": "user", "content": "Create a verified deployment plan."}
  ]
}
```

The OpenAI-compatible response includes an `auro` section containing evidence
class, blockers, atomic-agent counts, text-movement estimate, and the council
receipt. It explicitly records that the composition is not one checkpoint.

## Evidence classes

The runtime reports one of:

- `E2-execution-log` - the turn ran, but one or more response contracts failed;
- `E3-validated-output` - the response contracts passed, but signed complete
  checkpoint custody is not established;
- `E4-signed-receipt` - the local receipt is signed and all runtime blockers are
  absent.

E4 remains local signed custody. External or independently reproduced custody is
an E5 operation performed outside this runtime.

## Release blockers

A turn cannot become release evidence when any of these remain true:

- Auro-2B checkpoint custody is missing;
- a 500M specialist lacks checkpoint or adapter evidence;
- the three specialists do not have three distinct specialization proofs;
- Auro-156K or Auro-250M checkpoint custody is missing;
- an atomic task ran MESIE-only instead of invoking an atomic checkpoint;
- specialist, consensus, atomic, or parent response contracts failed;
- receipt signing is not configured.

These blockers do not prevent development experiments. They prevent development
experiments from being mislabeled as promoted model evidence.

## Security boundaries

- API authentication applies before council discovery or inference.
- API keys are referenced through environment-variable names and never returned.
- Parent context is bounded to 48,000 characters at the HTTP boundary.
- Atomic workers receive only task capsules; they do not receive parent context.
- No council response is permitted to authorize tools or computer mutation.
- The council is a reasoning composition. Computer execution remains governed by
  action-bound approvals and the runtime-cell system.
- Production mode requires a distinct 32+ character council receipt HMAC key
  when the council is configured.

## Required evaluation sequence

Before a public checkpoint release, run the same workload against:

1. Auro-2B alone;
2. Auro-2B plus one 500M specialist;
3. Auro-2B plus the 500M triad;
4. Auro-2B plus triad and atomic swarm;
5. the complete council with MESIE and fluidizer.

Measure:

- task accuracy and rubric quality;
- factual correction and uncertainty calibration;
- code-test success;
- creative diversity and conversational preference;
- context movement using exact tokenizer counts;
- wall-clock latency and throughput;
- per-model token use;
- expert utilization;
- disagreement detection;
- energy and memory on named hardware;
- failure and timeout behavior.

The current runtime emits the structure and evidence needed for those tests. It
does not contain fabricated benchmark results.
