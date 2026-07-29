# AURO-2B Triad Swarm

## Continuity

This runtime extends the existing AURO family. It does not replace:

- the checked-in HIM-native-v0/Auro-156K open-weight reference;
- local Auro-2B checkpoint lanes;
- native Auro-4B architecture and prewiring;
- the MESIE compute plane;
- `AtomicColony` specialization identities;
- the multi-embedded sub-agent router;
- NOVA agents, Browser-Brain, the persistent context engine, or the governed context envelope.

The new work fills the missing atomic capacity between 156K and 2B and connects it to the model runtime.

## Family slice

| Lane | Capacity class | Dual purpose |
|---|---|---|
| Auro-156K | Atomic | Very small routing, repair, classification, and style-control cells |
| Auro-250M | Atomic | Phone/WASM retrieval, transformation, code triage, and memory workers |
| Auro-500M | Atomic | Strong edge worker and base checkpoint for the three specialist identities |
| Auro-2B | Micro | Standalone private model and parent coordinator of the triad and dynamic atomic swarms |

The three 500M identities are:

- **Auro-500M-SENSUS** — evidence and perception;
- **Auro-500M-PRAXIS** — code, tools, and execution planning;
- **Auro-500M-VERBUM** — language, creative branching, explanation, and conversation.

A name alone is not weight specialization. SENSUS, PRAXIS, and VERBUM require independently identified checkpoints or a hash-bound base checkpoint plus distinct trained adapters before they count as three specialized models.

## Turn lifecycle

```text
User turn
   ↓
Auro-2B MESIE ingress analysis
   ↓
Three concurrent 500M specialist branches
   ├── SENSUS → topic-scoped 250M/156K swarm
   ├── PRAXIS → topic-scoped 250M/156K swarm
   └── VERBUM → topic-scoped 250M/156K swarm
   ↓
Each 500M synthesizes its atomic reports
   ↓
All three 500M models independently review the complete triad report
   ↓
Triad consensus reconciliation
   ↓
Auro-2B structured final synthesis
   ↓
Pure-Python / Pyodide-WASM fluidization
   ↓
Auro-2B MESIE egress analysis and receipt
   ↓
Conversational response
```

The parent retains the full conversation and context. Children receive bounded task capsules containing:

- a narrow objective;
- explicit constraints;
- evidence references;
- the requested role;
- an output budget;
- a capsule hash.

This reduces repeated context movement. The reported reduction is an estimate based on transport bytes divided by four. It is not an exact tokenizer, latency, or throughput benchmark.

## Model and parameter accounting

The runtime never calls the composed system a single 3.5B checkpoint.

A deployed turn may load:

- one independently identified Auro-2B checkpoint;
- three independently identified Auro-500M specialist checkpoints/adapters;
- zero or more independently identified Auro-250M and Auro-156K checkpoints;
- MESIE compute workers.

Each weight set retains its own parameter count, hash, tokenizer custody, training lineage, evaluation, and promotion status. Agent instances are never added to a model's parameter total.

## Configure exact model endpoints

`AURO_TRIAD_FLEET_JSON` is an object rather than the ordinary model-fleet list:

```json
{
  "main": {
    "model_id": "Auro-2B",
    "id": "auro-2b-local",
    "base_url": "http://127.0.0.1:8088/v1",
    "model": "Auro-2B",
    "checkpoint_id": "Auro-2B-release-candidate",
    "checkpoint_sha256": "<64 hex characters>",
    "measured_parameters": 2000000000,
    "provider": "repository-native-open-weights"
  },
  "specialists": [
    {
      "model_id": "Auro-500M-SENSUS",
      "id": "auro-500m-sensus",
      "base_url": "http://127.0.0.1:8091/v1",
      "model": "Auro-500M-SENSUS",
      "checkpoint_id": "Auro-500M-base",
      "checkpoint_sha256": "<64 hex characters>",
      "adapter_id": "sensus-v1",
      "adapter_sha256": "<64 hex characters>"
    },
    {
      "model_id": "Auro-500M-PRAXIS",
      "id": "auro-500m-praxis",
      "base_url": "http://127.0.0.1:8092/v1",
      "model": "Auro-500M-PRAXIS",
      "checkpoint_id": "Auro-500M-base",
      "checkpoint_sha256": "<64 hex characters>",
      "adapter_id": "praxis-v1",
      "adapter_sha256": "<64 hex characters>"
    },
    {
      "model_id": "Auro-500M-VERBUM",
      "id": "auro-500m-verbum",
      "base_url": "http://127.0.0.1:8093/v1",
      "model": "Auro-500M-VERBUM",
      "checkpoint_id": "Auro-500M-base",
      "checkpoint_sha256": "<64 hex characters>",
      "adapter_id": "verbum-v1",
      "adapter_sha256": "<64 hex characters>"
    }
  ],
  "atomics": [
    {
      "model_id": "Auro-250M",
      "role": "*",
      "id": "auro-250m",
      "base_url": "http://127.0.0.1:8094/v1",
      "model": "Auro-250M",
      "checkpoint_id": "Auro-250M-release-candidate",
      "checkpoint_sha256": "<64 hex characters>"
    },
    {
      "model_id": "Auro-156K",
      "role": "*",
      "id": "auro-156k",
      "base_url": "http://127.0.0.1:8095/v1",
      "model": "HIM-native-v0",
      "checkpoint_id": "Auro-156K",
      "checkpoint_sha256": "<64 hex characters>"
    }
  ]
}
```

Then start the production API:

```bash
export AURO_API_TOKEN='<private API token>'
export AURO_TRIAD_ENABLED=1
export AURO_TRIAD_FLEET_JSON="$(cat private/triad-fleet.json)"
python -m auro_native_llm.production_fleet.server --host 127.0.0.1 --port 8090
```

Native route:

```bash
curl http://127.0.0.1:8090/v1/triad/respond \
  -H "Authorization: Bearer $AURO_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Inspect this design and explain the strongest evidence-backed next step."}'
```

OpenAI-compatible route:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Authorization: Bearer $AURO_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"auro-2b-triad-swarm","messages":[{"role":"user","content":"Explain the architecture."}]}'
```

## Browser-Brain and Python/WASM

`browser-brain/src/python-wasm-fluidizer.js` loads a locally hosted Pyodide distribution and performs the final deterministic report-to-conversation pass in browser WASM. It does not install packages or contact a Python package index.

Place a pinned Pyodide distribution under:

```text
browser-brain/public/vendor/pyodide/
```

or pass another same-origin local location to `PythonWasmFluidizer`. If the local assets are absent, the WASM renderer fails rather than silently using a remote Python service.

Browser-Brain can invoke the triad with:

```javascript
const result = await brain.think(message, {triad: true});
```

## WebGPU training nodes

The existing NumPy AURO training path computes matrix gradients through `get_cuda_plane().matmul`. When the cluster is explicitly configured, those matrix products are transported to browser WebGPU workers.

Start the coordinator:

```bash
export AURO_WEBGPU_CLUSTER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m auro_native_llm.webgpu_cluster.coordinator \
  --host 127.0.0.1 --port 8765 --token "$AURO_WEBGPU_CLUSTER_TOKEN"
```

Serve the node page from a local HTTP origin:

```bash
python -m http.server 8770 --directory browser-brain/training
```

Open one or more tabs:

```text
http://127.0.0.1:8770/node.html?coordinator=http://127.0.0.1:8765&token=<token>
```

Enable it for training:

```bash
export AURO_WEBGPU_CLUSTER_URL=http://127.0.0.1:8765
export AURO_WEBGPU_CLUSTER_TOKEN=<token>
```

The coordinator's presence proves only that the transport is configured. Browser GPU use is established only by completed receipts with backend `browser-webgpu`.

## Tests

```bash
python -m pytest -q \
  tests/test_atomic_family_v2.py \
  tests/test_auro2b_triad_swarm.py \
  tests/test_webgpu_cluster.py \
  tests/test_python_wasm_fluidizer.py \
  tests/test_triad_production_api.py

python scripts/run_auro2b_triad_benchmark.py \
  --mode fixture \
  --output evidence/auro-2b-triad-benchmark.json
```

The default benchmark is a permanently quarantined mechanics fixture. Exact mode requires `AURO_TRIAD_FLEET_JSON` and real reachable model endpoints:

```bash
python scripts/run_auro2b_triad_benchmark.py --mode exact
```

## Release gate through Auro-2B

Shipping each checkpoint requires its own:

1. exact weights and hash manifest;
2. tokenizer and byte-perfect round-trip audit;
3. corpus provenance, deduplication, contamination, and secret-scan receipts;
4. training configuration, loss history, resume state, and optimizer evidence;
5. official benchmark results and failure samples;
6. mobile/browser/API portability receipts where applicable;
7. triad and atomic-swarm compatibility results;
8. signed constitutional promotion and rollback evidence.

Architecture compilation, fixture conversations, or successful WebGPU matmul jobs are not substitutes for trained checkpoint quality.
