# Auro Repository Consolidation Runbook

Auro Foundry now has two connected command surfaces:

- `auro-foundry`: clone/scan repositories, build corpus, train, resume, generate, and serve.
- `auro-consolidate`: register the Medina repository estate, expose worker plans, run native probes, run coding execution tests, dispatch official benchmarks, and create a talk-ready solidification receipt.

## 1. Install

```bash
python -m pip install -e ".[foundry,benchmarks,dev]"
gh auth login
```

Private repositories require an authenticated GitHub CLI session.

## 2. Feed the organization into Auro

The full organization scan is the direct corpus lane:

```bash
auro-foundry all \
  --org ItsNotAILABS \
  --workspace artifacts/auro-foundry \
  --preset local \
  --steps 1000
```

This uses Auro Foundry's existing repository discovery, secret-file exclusion, credential redaction, deduplication, provenance records, tokenizer training, packed datasets, checkpointing, and receipts.

The curated federation manifest is created with:

```bash
auro-consolidate manifest --workspace artifacts/auro-foundry
```

It registers NOVA, Auro, AURO, MedinaMemorySystems, NATIVE-NOVA-PROTOCOL, PRODUCTION-, Enterprise-OS-intelligence, cloudcolony, Chimeria, PARRALAX, PARALLAX Clearinghouse, NEUROSWARMAI, LOOM, CAPSULA, PhoneAI, ForgeBridge-MCP, organism-bots, nova-intelligence, MatDaemon, FABLEBREAKER, and CyberSecurity-AI.

## 3. Inspect worker plans

```bash
auro-consolidate workers --capability benchmark
auro-consolidate workers --capability code
auro-consolidate workers --capability release
```

The registry includes ORIGO, SENSUS, CORPUS, MATHESIS, CODEX, RTMX, PHAI, PORT, TEST, BENC, SACE, LAWX, SUCC, NOVA, CAIN, and ORO.

## 4. Native regression probes

```bash
auro-consolidate benchmark \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt \
  --workspace artifacts/auro-foundry
```

Custom cases may be supplied as JSON with `--cases cases.json`.

## 5. Coding and execution harness

```bash
auro-consolidate code-eval \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt \
  --workspace artifacts/auro-foundry
```

The harness extracts generated Python, executes without a shell in a temporary directory, applies an executable allowlist and timeout, caps output, and writes a receipt. MESIE's governed runner remains the outer execution boundary for production nodes.

## 6. Official benchmark bridge

Start Auro's API first:

```bash
auro-foundry serve \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt \
  --port 8090
```

Then dispatch EleutherAI lm-evaluation-harness profiles:

```bash
auro-consolidate official --task leaderboard
auro-consolidate official --task math
auro-consolidate official --task coding
auro-consolidate official --task instruction
```

Profiles map to MMLU, ARC-Challenge, HellaSwag, WinoGrande, GSM8K, Minerva Math, HumanEval, MBPP, and IFEval where supported by the installed lm-eval version. Results and samples are written under `artifacts/auro-foundry/benchmarks/lm-eval`.

## 7. Solidify and talk to Auro

```bash
auro-consolidate solidify \
  --checkpoint artifacts/auro-foundry/runs/auro-owned-run/final.pt \
  --workspace artifacts/auro-foundry \
  --serve \
  --open-browser
```

This writes:

```text
artifacts/auro-foundry/federation-manifest.json
artifacts/auro-foundry/worker-registry.json
artifacts/auro-foundry/benchmarks/native-receipt.json
artifacts/auro-foundry/benchmarks/coding-receipt.json
artifacts/auro-foundry/solidification-receipt.json
```

The browser opens at `http://127.0.0.1:8090` and talks to the mounted checkpoint through Auro's OpenAI-compatible chat endpoint.

## Evidence boundary

The harnesses and integration paths are executable. Published benchmark scores must come from completed receipts produced against a named checkpoint. The repository does not claim MMLU, GSM8K, HumanEval, MBPP, IFEval, or other scores until those runs finish.
