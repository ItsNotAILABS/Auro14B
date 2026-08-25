# AURO Browser-WebGPU Training Fabric v1

The WebGPU Training Fabric turns explicitly enrolled browser tabs into bounded
float32 matrix workers for AURO. It is a real compute transport connected to the
same matrix API used by the repository training plane.

It is not a substitute for a complete distributed optimizer, model checkpoint,
or benchmark campaign. A browser receipt proves one returned matrix job under
the protocol; a model-training claim additionally requires the full training
and promotion evidence chain.

## Components

```text
Training process
  -> CudaPlane.matmul / train_step_linear
  -> WebGPUClusterClient
  -> persistent coordinator (SQLite/WAL)
  -> leased matrix job
  -> one browser WebGPU node
  -> result validation
  -> signed or unsigned receipt
  -> training process
```

Files:

```text
auro_native_llm/webgpu_cluster/coordinator.py
auro_native_llm/polyglot/webgpu_cluster.py
auro_native_llm/polyglot/cuda_plane.py
browser-brain/training/node.html
browser-brain/training/node.js
```

## Start the coordinator

For a loopback-only development coordinator:

```bash
python -m auro_native_llm.webgpu_cluster.coordinator \
  --host 127.0.0.1 \
  --port 8765 \
  --db state/auro-webgpu-cluster.sqlite
```

For signed local receipts:

```bash
export AURO_WEBGPU_RECEIPT_HMAC_KEY='<32-or-more-character-secret>'
export AURO_WEBGPU_CLUSTER_TOKEN='<32-or-more-character-secret>'
python -m auro_native_llm.webgpu_cluster.coordinator \
  --host 127.0.0.1 \
  --port 8765 \
  --db state/auro-webgpu-cluster.sqlite
```

A non-loopback bind is rejected unless a 32-character-or-longer shared token is
configured. For a remote deployment, place TLS and identity-aware access in
front of the coordinator and allow only the exact browser origins required by
the operator.

## Open a browser worker

Serve the Browser-Brain directory from a local HTTP origin or approved HTTPS
origin. WebGPU requires a secure context; ordinary `file://` loading is not the
supported path.

Open:

```text
browser-brain/training/node.html
```

Enter:

- coordinator URL;
- cluster token;
- a stable worker name;
- Start worker.

A token may be bootstrapped through the URL fragment for controlled local use:

```text
node.html#coordinator=http://127.0.0.1:8765&token=...&autostart=1
```

The page copies fragment values into `sessionStorage` and immediately removes
the fragment from the visible URL. The token is never read from the query
string and is never written to the activity log.

## Training-process configuration

```bash
export AURO_WEBGPU_CLUSTER_URL='http://127.0.0.1:8765'
export AURO_WEBGPU_CLUSTER_TOKEN='<cluster-token>'
export AURO_WEBGPU_CLUSTER_TIMEOUT='180'
export AURO_WEBGPU_CLUSTER_REQUIRE_WORKER='1'
```

Then the existing accelerated plane selects `webgpu_cluster` before CUDA, MPS,
CuPy, ChaosCUDA, or NumPy:

```python
from auro_native_llm.polyglot.cuda_plane import get_cuda_plane

plane = get_cuda_plane(refresh=True)
print(plane.info())
output = plane.matmul(a, b)
step = plane.train_step_linear(weights, inputs, targets, lr=1e-3)
```

If `AURO_WEBGPU_CLUSTER_REQUIRED=1`, inability to reach a ready cluster is a
hard failure instead of a fallback to another backend.

## Persistence and recovery

The coordinator stores in SQLite/WAL:

- queued jobs;
- binary float32 inputs;
- worker identities and capabilities;
- one-time lease tokens;
- lease attempts and expirations;
- validated result bytes;
- errors and completion timing;
- execution receipts and optional signatures.

After coordinator restart:

- queued jobs remain queued;
- expired leases return to the queue until their bounded attempt count is
  exhausted;
- completed results and receipts remain inspectable until retention pruning;
- old lease tokens cannot submit a result.

## Lease protocol

Each claimed job contains:

```text
job_id
worker_id
lease_token
lease_expires_at
attempt
max_attempts
input shapes and float32 payloads
```

While a WebGPU kernel is running, the browser renews the lease through:

```text
POST /lease/renew
```

The result route requires the exact worker and one-time lease token. A result is
rejected when the job is completed, reissued, expired, or belongs to another
worker.

## Coordinator API

```text
GET  /status
GET  /job?worker_id=...&wait=...&capabilities=...
GET  /job/{job_id}
GET  /receipt/{receipt_sha256}
POST /jobs
POST /matmul
POST /lease/renew
POST /result
POST /cancel
```

`POST /jobs` is asynchronous. `POST /matmul` waits for one completed result and
is the path used by `WebGPUClusterClient.matmul`.

## Receipt contents

A completion receipt includes:

- job and worker identity;
- lease attempt;
- matrix geometry;
- input and result SHA-256 digests;
- worker-reported backend;
- worker-capability digest;
- bounded worker evidence;
- elapsed time;
- completion time;
- receipt hash;
- optional HMAC signature and signer identity.

The coordinator validates output shape, byte count, and finite float32 values
before completing a job.

## Security boundaries

- Browser workers receive only the matrix job they lease.
- The coordinator does not send model source, prompts, credentials, datasets, or
  complete checkpoints unless higher-level training code explicitly encodes
  those values into a matrix job.
- Cluster tokens belong in environment variables or the worker's tab-local
  session state, not source control.
- CORS is denied except for loopback origins and exact operator allowlists.
- Non-loopback binding requires a strong token.
- Workers can report WebGPU metadata, but the protocol does not provide remote
  hardware attestation.
- A signed receipt is local E4 custody, not independent E5 reproduction.

## Claim boundary

A valid browser receipt supports this claim:

> The coordinator accepted a bounded matrix job and received a shape-valid,
> finite float32 result from the named worker under the browser-WebGPU protocol.

It does not by itself support these claims:

- an entire corpus trained successfully;
- a named AURO checkpoint was produced;
- the browser used a particular physical GPU;
- training was faster than a named baseline;
- distributed gradients were numerically equivalent;
- a checkpoint passed promotion.

Those require exact training manifests, optimizer state, loss history, resume
proof, model hashes, checkpoint custody, benchmark results, and promotion
receipts.

## Next scaling stage

The next production layer should add sharded tensor operations and optimizer
coordination rather than pretending independent matrix jobs equal distributed
LLM training. The required additions are:

1. tensor-shard manifests and deterministic partitioning;
2. per-shard dtype and numerical-tolerance contracts;
3. reduce/all-reduce semantics;
4. optimizer-state ownership;
5. gradient accumulation and step barriers;
6. worker capability scheduling;
7. checkpoint-resume receipts;
8. cross-worker numerical equivalence tests;
9. dataset and sample-order custody;
10. exact-checkpoint evaluation after training.
