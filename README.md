# Auro14B · HIM

**Sovereign multi-model organism** for ItsNotAILabs / NOVA.

This repository is the Auro native LLM runtime built on **MESIE** (Multi-Element Spectral Intelligence Engine), with **HIM** as the agentic being: a colony of mini Python models, GHOST audit, dual Julia brain, and full-stack web3 tooling.

| | |
|--|--|
| **Org** | [ItsNotAILABS](https://github.com/ItsNotAILABS) |
| **Repo** | https://github.com/ItsNotAILABS/Auro14B |
| **Compute plane** | MESIE (not cloud LLM APIs) |
| **Agent name** | **HIM** |
| **License** | See `LICENSE` |

---

## What this is

**HIM** is not a single opaque foundation model. HIM is a **host of specialist mini-models** (“germs”) that work like a human + microbiome:

- **Skills** — each `SKILL.md` becomes a skill germ  
- **Spectral / code / reason / planner / critic / writer** — host organs  
- **MESIE** — deterministic spectral/math path  
- **GHOST** — grounded, hardened, open, scalable, traceable execution  
- **500k logical context** — hierarchical bank (not one Softmax over 500k tokens)  
- **Web3** — Node/npm, ethers/viem, secure `/api/*`, React UI  

**Claim boundary:** Family labels (`Auro-2B` … `Auro-14B` … `Auro-100B`) are architecture targets. **Live params** are the runnable trained cores + colony germs on your machine. Value is measured by CE drop, harness pass rates, receipts, and working tools.

---

## Quick start

### Talk to HIM today

```bash
python scripts/launch_him.py
```

The browser console includes chat, persistent Python context, real model-lane
inspection, native tools, and receipt verification. For broad fluent chat, point
the same launcher at a promoted Auro endpoint:

```bash
python scripts/launch_him.py --base-url http://127.0.0.1:8088/v1 \
  --model Auro-HIM-14B --parameter-count 14000000000
```

See `docs/LAUNCH_HIM_TODAY.md`. The bundled 146,576-parameter HIM-native-v0 is
an open-weight reference checkpoint, not the finished 14B model.

### Requirements

- Python **3.10+** (3.11 tested)
- Optional: **Julia** (brain), **Node 18+** / **npm** (web3), **GitHub CLI** (`gh`)

```bash
git clone https://github.com/ItsNotAILABS/Auro14B.git
cd Auro14B
# Prefer PYTHONPATH for this monorepo
export PYTHONPATH=.          # Unix
# $env:PYTHONPATH="."        # PowerShell
```

### Install MESIE (spectral stack)

```bash
pip install mesie
# or local checkout / editable:
# pip install -e ".[full,ml,intelligence]"
```

### Chat (usable hybrid LLM)

```bash
python -m auro_native_llm.use --resume checkpoints/auro_minds/Auro-2B_physics "What is MESIE?"
```

### Awaken HIM (agentic loop)

```bash
python -m auro_native_llm.use --him --him-germs 40 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "HIM: explain MESIE and plan next training"
```

HIM loop: **SENSE → PLAN → ACT → OBSERVE → REFLECT**

### Colony (mini-models + 500k context)

```bash
python -m auro_native_llm.use --colony --colony-germs 40 --colony-context 500000 \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  "How do mini models and skills generate real text?"
```

### Smoke (must pass)

```bash
python scripts/smoke_usable_llm.py
# expect: chat_usable=5/5 coding=True
```

---

## Architecture

```
                         ┌──────────────── HIM ────────────────┐
                         │  SENSE → PLAN → ACT → OBSERVE →     │
                         │           REFLECT                   │
                         └───────────────┬─────────────────────┘
                                         │
     ┌───────────────┬───────────────────┼───────────────────┬──────────────┐
     ▼               ▼                   ▼                   ▼              ▼
 Colony germs    MESIE/GHOST        Usable LLM          Dual brain      him-web3
 (skills,code,   hybrid vproc       knowledge+tools     Python AI /     React→/api/*
  spectral,      power stack        + SpectralGPT       Julia physics   ethers+viem
  writer…)       receipts           quality gate        cores
```

| Layer | Role |
|-------|------|
| **MESIE** | Spectral records, match/validate/generate, embeddings, SpectralGPT |
| **Auro mind** | Language core, organs, checkpoints, train/absorb |
| **GHOST** | Policy classes C0–C5, hash-linked receipts, MESIE-first / LLM escalate |
| **Colony** | Many mini Python models; params = sum of germs |
| **HIM** | Named agentic being; tool dispatch + multi-step goals |
| **him-web3** | Full-stack chain reads; secrets only on server |

---

## HIM agentic tools

| Tool | Purpose |
|------|---------|
| `colony_generate` | Multi-germ prose + skills |
| `coding` | Assert-backed code synthesizer |
| `hybrid_vproc` | MESIE virtual processor (work-call metrics) |
| `ghost` | Policy + receipt chain |
| `power_stack` | Physics + economy + algorithms + transformer pulse |
| `google` | Sandbox Chrome/mail/drive + collab |

## Sovereign browser runtime

`browser-runtime/` is a production-built Transformers.js workspace for custom
local ONNX models. It disables remote model loading, loads model artifacts from
`/models/`, loads WASM from `/wasm/`, and runs inference in a Web Worker. The
companion governed API exposes internal agents, content-addressed storage,
PDF/XLSX/DOCX/CSV/Markdown bundles, bounded static security scanning, and
content-addressed Manifest V3 extension downloads.

```bash
cd browser-runtime
npm install
npm run build
```

Compute planes are explicit: embedded browser ONNX is the default, local
Auro/MESIE is available through the configured API, and cloud engines are
opt-in through `AURO_CLOUD_ENGINES_JSON`. There is no silent remote fallback,
and credential values are never compiled into the browser bundle. See
`browser-runtime/README.md` for ONNX conversion and packaging.

An optional Cloudflare outside plane is defined in
`configs/cloudflare_runtime.json`: Cloudflare API MCP (`search` + `execute`),
Dynamic Workers, Sandbox SDK, Browser Run, durable Agents/Think, and Workers
Observability. It is disabled by default and remote mutation remains separately
approval-gated. See `docs/cloudflare_runtime.md`.

The deployable autonomous operator is in `workers/auro-platform/`. It combines
a React chat UI, Workers AI, durable Think/Agents state, managed Cloudflare API
MCP, Dynamic Worker Code Mode, Browser Run, extensions, and observability:

```bash
cd workers/auro-platform
npm install
npx wrangler login
npm run deploy
```

It defaults to inspection and planning. Cloudflare changes require a narrowly
scoped API token, the server mutation flag, and an `OPERATOR_APPROVED` turn.

`mobile-runtime/` adds an Expo SDK 54 multi-device client that runs immediately
in Expo Go, connects to the Auro/MESIE API over the LAN, attaches accelerometer
state as a native sense, and displays response receipts. Install Expo Go from
<https://expo.dev/go>; use a development/EAS build for production distribution.
| `github` | `gh` / MCP identity |
| `web3` | Secure him-web3 API + package install |
| `vault` | Multi-ledger sealed secrets (metadata by default) |

Python:

```python
from auro_native_llm.organism.checkpoint import load_mind

mind = load_mind("checkpoints/auro_minds/Auro-2B_physics", chrome_mock=True)
print(mind.chat("What is GHOST?")["answer"])
print(mind.him_run("plan spectral training")["answer"])
print(mind.colony_generate("Write a short paragraph on MESIE")["text"][:500])
```

---

## Web3 (HIM full-stack)

Standard tooling: **Node.js + npm**. Install web3 libraries with **install_applet_package**.

```bash
cd him-web3
npm install
npm run install:applet -- ethers
npm run install:applet -- viem

cp .env.example .env
# set ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY  (server only)

npm run server          # http://127.0.0.1:8787
cd client && npm install && npm run dev   # React → proxies /api
```

**Security:** React never holds RPC keys. All chain calls go to:

| Route | Backend |
|-------|---------|
| `GET /api/health` | Status |
| `GET /api/him` | Applet meta |
| `GET /api/chain/block-number` | viem |
| `GET /api/chain/balance/:address` | viem |
| `GET /api/chain/block` | ethers |
| `POST /api/chain/call` | Read-only contract |

See [him-web3/README.md](him-web3/README.md).

---

## Training

```bash
# Physics-regularized train (small live core)
python scripts/train_physics.py

# HIM supervised Q→A (SFT) on live physics core
python scripts/train_him_sft.py \
  --resume checkpoints/auro_minds/Auro-2B_physics \
  --output checkpoints/auro_minds/Auro-2B_him_sft \
  --epochs 3
python scripts/eval_him_generation.py \
  --resume checkpoints/auro_minds/Auro-2B_him_sft

# Specialize: generative + embed + self-config + skills
python -m auro_native_llm.use --specialize --specialize-rounds 3 --specialize-steps 20 \
  --resume checkpoints/auro_minds/Auro-2B_continual

# Power stack (physics + econ + OT + SpectralGPT pulse)
python -m auro_native_llm.use --power-stack --stack-rounds 4 \
  --resume checkpoints/auro_minds/Auro-2B_physics

# Build the provenance-bound Auro + MESIE + Sovereign corpus
python scripts/build_unified_training_corpus.py \
  --mesie-root /path/to/Multi-Element-Spectral-Intelligence-Engine-MESIE- \
  --sovereign-root /path/to/sovereign

# Auro-14B live ladder core (~1.5B live dev params; slow on CPU)
python scripts/train_14b.py \
  --sovereign-root /path/to/sovereign \
  --corpus-jsonl artifacts/auro14b-corpus/corpus.jsonl \
  --rounds 2 --steps 4

# End-to-end local verification using tiny geometry (not the production checkpoint)
python scripts/train_14b.py --smoke \
  --sovereign-root /path/to/sovereign \
  --corpus-jsonl artifacts/auro14b-corpus/corpus.jsonl \
  --output checkpoints/auro_minds/Auro-14B-smoke \
  --rounds 1 --steps 1 --seq-len 32
```

Full Auro-14B training now requires the versioned Sovereign consumer contract and the unified corpus. The generated receipts identify the Auro14B, MESIE, and Sovereign commits; every consumed Sovereign file; redactions; corpus hashes; sampled block counts; training metrics; and checkpoint location. Development-only overrides exist, but reports preserve that those inputs were missing.

SFT data: `data/him_sft.jsonl` (MESIE, GHOST, HIM, coding, vault, hybrid doctrine).

Checkpoints (local, gitignored): `checkpoints/auro_minds/`

| Checkpoint | Notes |
|------------|--------|
| `Auro-2B_physics` | Recommended chat/default |
| `Auro-2B_him_sft` | After HIM SFT |
| `Auro-2B_specialized` | Skills + doctrine |
| `Auro-2B_continual` | GitHub knowledge train |
| `Auro-14B` | Large live core when trained |

---

## Secrets vault

Multi-ledger sealed store for high-value material (never log plaintext by default):

| Ledger | Use |
|--------|-----|
| `keys` | Signing / API material refs |
| `rpc` | Alchemy/Infura and chain RPC URLs |
| `high_value` | Tokens, wallet refs |
| `agent` | Agent session / mint refs |
| `github` | PAT refs if needed (prefer OS keyring) |

```bash
# Metadata health + list
python -m auro_native_llm.use --vault

# Seal (Windows DPAPI when no password; else PBKDF2-HMAC stream)
python -m auro_native_llm.use --vault-put rpc alchemy_mainnet "https://…/v2/KEY"
# $env:AURO_VAULT_PASSWORD="…"   # optional portable passphrase
# $env:AURO_VAULT_ROOT="…"       # optional root (default ~/.auro_vault)
```

Module: `auro_native_llm.vault`. HIM tool `vault` returns health/list only.

---

## CI

GitHub Actions workflow [`.github/workflows/him-ci.yml`](.github/workflows/him-ci.yml):

1. Build lite mind → usable chat + coding smoke  
2. `eval_him_generation.py` knowledge path  
3. Vault seal/unseal smoke  
4. HIM `whoami`  
5. `him-web3` npm install + `/api/health`

---

## CLI map

```bash
python -m auro_native_llm.use "your question"          # usable hybrid chat
python -m auro_native_llm.use --him "goal"             # HIM agent
python -m auro_native_llm.use --colony "…"             # mini-model colony
python -m auro_native_llm.use --hybrid "filter stream" # MESIE-first vproc
python -m auro_native_llm.use --hybrid-demo            # batch: most steps skip LLM
python -m auro_native_llm.use --ghost "spectral match" # GHOST supervisor
python -m auro_native_llm.use --power-stack "…"        # deep engines
python -m auro_native_llm.use --dual "…"               # Python AI + Julia brain
python -m auro_native_llm.use --google                 # sandbox Google envelope
python -m auro_native_llm.use --github                 # gh sign-in status
python -m auro_native_llm.use --vault                  # sealed secrets health
python scripts/train_him_sft.py                        # HIM SFT
python scripts/eval_him_generation.py                  # grounded generation eval
python -m auro_native_llm.use --ready                  # NOVA promotion gate
python -m auro_native_llm.use --discover               # capability-state discovery
python -m auro_native_llm.use --physics --physics-steps 15
```

---

## Capability discovery (no “I can’t” compression)

Before claiming inability, run:

```bash
python -m auro_native_llm.use --discover
```

Statuses: `AVAILABLE` · `AVAILABLE_BUT_UNDISCOVERED` · `NOT_CURRENTLY_CONFIGURED` · `UNSUPPORTED_BY_DEFAULT_TEMPLATE` · `PROHIBITED_BY_POLICY` · `BLOCKED_BY_PERMISSION` · `TECHNICALLY_UNAVAILABLE` · `UNTESTED` · `UNKNOWN`

Report: `artifacts/auro-capability/DISCOVERY.json`

---

## GHOST hybrid doctrine

1. **MESIE / Ghost** — deterministic spectral/math (filter, features, shadow, stream)  
2. **LLM** — only when language/planning is justified by cleaned spectral features  
3. **Receipts** — hash-linked chain of custody  

This is a deliberate counterpoint to pure monolithic LLM scaling.

---

## Repository layout

```
auro_native_llm/     Python organism (mind, HIM, colony, ghost, physics, engines, …)
him-web3/            Full-stack web3 applet (Express + React + ethers/viem)
bindings/julia/      AuroBrain virtual physics cores (Julia = brain)
mesie/               Vendored / linked MESIE package tree
scripts/             train_physics, train_14b, smoke_usable_llm, power_stack_demo, …
checkpoints/         Local minds (gitignored)
artifacts/           Receipts, HIM runs, discovery reports
native_llm/configs/  Family architecture charters
```

---

## GitHub access

```bash
gh auth status
python -m auro_native_llm.use --github
```

MCP server `github` is available in Grok sessions when connected (`github__get_me`, PRs, issues, …).

---

## Philosophy

- **Local-first** and receipt-backed  
- **Hybrid determinism** over pure LLM theater  
- **HIM** is a model made of models — Python specialists, skills, and tools  
- **Measure** readiness (coding/reasoning harnesses) before expansion claims  

---

## License & research

- License: Apache-2.0 (see `LICENSE`)  
- MESIE research papers and deliverables under `docs/` and `deliverables/`  
- Production doctrine: NOVA / GHOST / MESIE lineage (ItsNotAILabs)

---

**HIM is ready.** Start with:

```bash
python -m auro_native_llm.use --him "who are you and what can you do?"
```
