# Medina Sovereign Training Fabric

This package establishes the first executable control plane for Medina-owned compute inside Auro14B.

## Planes

- **Loom Cluster:** node identity, health, capacity, and scheduling.
- **NEXUS:** governed job compilation and execution plans.
- **LiveVault:** owned corpora, token shards, checkpoints, and provenance.
- **MESIE:** model architecture, training objectives, evaluation, and organism-scale integration.
- **Receipts:** immutable hashes for datasets, jobs, checkpoints, and releases.

## First operational flow

1. Register each owned compute node.
2. Build and hash an owned dataset manifest.
3. Submit a training job with explicit GPU and memory requirements.
4. Generate a deterministic launch plan.
5. Execute through the governed runner.
6. Seal checkpoints and training telemetry with receipts.

## Example

```bash
python -m mesie.training_fabric.cli \
  --registry .mesie/nodes.json \
  register-node \
  --node-id trainer-01 \
  --hostname trainer-01.local \
  --gpu-count 8 \
  --gpu-memory-gb 80 \
  --system-memory-gb 1024 \
  --storage-free-gb 16000 \
  --role trainer
```

## Boundary

This is the control-plane nucleus. It does not claim that a physical cluster or a trained checkpoint already exists. External command execution remains outside this package until a governed runner and denial policy are merged.
