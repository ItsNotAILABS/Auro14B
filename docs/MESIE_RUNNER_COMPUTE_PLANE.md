# MESIE Runner Compute Plane

MESIE is the compute plane for Auro and Medina model training. It is not a UI label or an offline benchmark wrapper.

## Compute contract

`mesie.compute` provides differentiable PyTorch modules that operate on transformer hidden states:

- FFT projection over the sequence axis
- non-linear spectral bands
- entropy, centroid, flatness, high-frequency ratio, and phase coherence
- cross-layer spectral drift loss
- gated spectral conditioning

These operations remain in the autograd graph and may contribute directly to model optimization.

## Runner contract

The CUDA image under `runner/` creates an ephemeral GitHub Actions runner with labels:

```text
self-hosted,linux,x64,gpu,mesie,spectral-transformer
```

At startup it discovers the physical GPU, memory, host, storage, architecture, and roles, then writes a hashed `mesie-compute-node/1.0` receipt before accepting work.

The runner requires a short-lived GitHub registration token. It cannot invent a GPU or register itself without repository or organization authorization.

## Launch

On an NVIDIA Docker host:

```bash
export GITHUB_REPOSITORY=ItsNotAILABS/Auro14B
export GITHUB_RUNNER_TOKEN='<short-lived-registration-token>'
docker compose -f runner/docker-compose.yml up --build
```

For Medina model training, set `GITHUB_REPOSITORY=ItsNotAILABS/NATIVE-NOVA-PROTOCOL` and retain the MESIE labels. The training workflow can then select the same compute plane with:

```yaml
runs-on: [self-hosted, linux, x64, gpu, mesie, spectral-transformer]
```

## Boundary

The repository defines and validates the runner image, discovery, receipts, and spectral algorithms. A running node still requires an actual NVIDIA host and a GitHub runner registration token issued by the repository or organization.
