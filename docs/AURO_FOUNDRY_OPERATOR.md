# Auro Foundry Operator Runbook

Auro Foundry turns operator-authorized repositories and local source trees into a runnable local text-generation model.

## Install

```bash
python -m pip install -e ".[foundry,dev]"
```

Private repository discovery uses the GitHub CLI session already authenticated on the machine:

```bash
gh auth status
```

## One-command lifecycle

The smallest complete run uses the `micro` lane:

```bash
auro-foundry all \
  --org ItsNotAILABS \
  --workspace artifacts/auro-foundry \
  --preset micro \
  --steps 20
```

This performs:

1. repository discovery and shallow checkout
2. secret-like file exclusion and text redaction
3. content deduplication and provenance capture
4. repository-native byte BPE tokenizer training
5. packed train/validation token stream creation
6. causal Transformer optimization
7. periodic and final checkpoint creation
8. evaluation and training receipt emission

The final checkpoint is written under:

```text
artifacts/auro-foundry/runs/auro-owned-run/final.pt
```

## Resume training

```bash
auro-foundry resume \
  --config artifacts/auro-foundry/train-config.json \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt
```

Increase `max_steps` in the configuration before resuming when more optimization steps are required.

## Generate from the checkpoint

```bash
auro-foundry generate \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt \
  --prompt "Explain how NOVA, MESIE, and Auro connect."
```

## Start the chat server and browser UI

```bash
auro-foundry serve \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt \
  --host 127.0.0.1 \
  --port 8090 \
  --open-browser
```

Open `http://127.0.0.1:8090`.

OpenAI-compatible routes:

```text
GET  /health
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

## MESIE-governed execution

Run the same training command through MESIE's shell-free allowlisted execution bridge:

```bash
auro-foundry train-governed \
  --config artifacts/auro-foundry/train-config.json \
  --workdir . \
  --receipts artifacts/auro-foundry/mesie-receipts
```

## Model lanes

```text
micro   immediate CPU/GPU development and E2E proof
local   larger workstation model
14b     distributed 14B configuration
206.7b  dense target configuration
```

All lanes use the same corpus, tokenizer, model, checkpoint, generation, and serving code paths. Larger configurations require enough registered memory and compute to instantiate and optimize them; the software no longer substitutes validation receipts for training.

## Evidence artifacts

```text
corpus/manifest.json
tokenizer.json
dataset/dataset-manifest.json
runs/<run>/train-config.json
runs/<run>/step-*.pt
runs/<run>/best.pt
runs/<run>/final.pt
runs/<run>/training-receipt.json
end-to-end-receipt.json
mesie-receipts/<job>/execution-receipt.json
```
