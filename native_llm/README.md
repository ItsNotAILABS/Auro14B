# Auro — Native Text LLM Family on MESIE

**Product:** First-class language models, not API wrappers  
**Engine / virtual GPU:** [MESIE](../README.md) (SpectralGPT · MoE · embeddings · helix · cosmology · SOLUS φ-math)  
**Family:** **Auro-2B · Auro-4B · Auro-8B · Auro-14B · Auro-100B**  
**Repo:** `ItsNotAILABS/Auro14B`

Auro is the text LLM layer of this monorepo. MESIE is the spectral intelligence engine and compute plane. Auro does **not** wrap OpenAI, Anthropic, Ollama, or Hugging Face cloud endpoints. It builds on MESIE’s own transformer, MoE, vector, and meaning stack.

## What ships

| Component | Location | Role |
|---|---|---|
| **AuroLanguageModel** | `auro_native_llm/model/auro_lm.py` | Causal MoE text LM (MESIE SpectralGPT backbone) |
| **Tokenizer** | `auro_native_llm/model/tokenizer.py` | Trainable BPE-style tokenizer |
| **Meaning engines** | `auro_native_llm/model/meaning.py` | Latin · Sanskrit · Nahuatl/Teotl residual meaning |
| **Spectral fusion** | `auro_native_llm/model/spectral_fuse.py` | MESIE SpectralVectorizer + Helix into residual stream |
| **φ mathematics** | `auro_native_llm/model/phi_math.py` | Golden-ratio init / geometry (SOLUS lineage) |
| **Train** | `auro_native_llm/model/train.py` | Corpus train + CE gradients + checkpoints |
| **Jobs** | `auro_native_llm/model/jobs.py` | MESIE training-fabric pretrain jobs |
| **Multi-embedded agents** | `auro_native_llm/subagents.py` + `native_runtime.py` | Larger lanes host smaller lanes |

## Architecture

```text
  prompt text
       │
       ▼
 AuroTokenizer (BPE)
       │
       ▼
 MESIE SpectralGPT  ── causal decoder
   · multi-head attention (RoPE / spectral blocks)
   · MoE every other layer (top-k experts)
   · multi-task readiness
       │
       ├── meaning residual (Latin / Sanskrit / Nahuatl cosmic layers)
       ├── spectral residual (SpectralVectorizer · Helix)
       └── φ-harmonic weight geometry
       │
       ▼
   LM head logits → sampling → text
```

**Compute plane:** MESIE only (`mesie.foundation`, `mesie.embeddings`, `mesie.helix`, `mesie.cosmology`, `mesie.training_fabric`, optional `mesie.compute` torch spectral).

## Family

| Lane | Tier | Role | MoE experts (dev) |
|---|---|---|---|
| Auro-2B | edge | router / tool / triage | 8 |
| Auro-4B | specialist | code / spectral match | 8 |
| Auro-8B | general | reason / plan | 12 |
| Auro-14B | orchestrator | multi-agent chair | 16 |
| Auro-100B | frontier | deep council | 16 |

`mode=dev` builds laptop-executable models with live countable parameters.  
`mode=full` scales architecture toward the family targets (memory heavy).  
`parameter_target` is the family claim (2B…100B). Live `num_params` is always reported honestly.

## Quick start

```bash
# Build model info
python -m auro_native_llm.model.cli build --model Auro-2B

# Train on this repo’s docs/examples (MESIE compute)
python -m auro_native_llm.model.train --model Auro-2B --steps 40 --output-dir checkpoints/auro

# Generate from checkpoint
python -m auro_native_llm.model.cli generate \
  --checkpoint checkpoints/auro/Auro-2B \
  --prompt "MESIE spectral meaning ratio rta teotl"

# Submit training-fabric job (plan + receipt)
python -m auro_native_llm.model.cli job --model Auro-2B --steps 40
# Run train via fabric path
python -m auro_native_llm.model.cli job --model Auro-2B --steps 40 --execute
```

Python API:

```python
from auro_native_llm import AuroLanguageModel, train_language_model, TrainConfig

model = AuroLanguageModel.build("Auro-2B", mode="dev")
print(model.info()["num_params_live"], model.compute_plane)  # MESIE

out = model.generate("Auro ratio rta teotl spectral", max_new_tokens=48)
print(out.text)
print(out.meaning_hits)

report = train_language_model(TrainConfig(model_id="Auro-2B", steps=40))
print(report["checkpoint"], report["num_params"])
```

Multi-embedded council (MESIE native):

```python
from auro_native_llm import AuroNativeRuntime

rt = AuroNativeRuntime("Auro-14B")
for r in rt.council("plan spectral training"):
    print(r.child_model_id, r.generation.text[:120] if r.generation else r.error)
```

## Meaning engines

| Engine | Roots | Purpose |
|---|---|---|
| Latin | ratio, lumen, ordo, mens, veritas, … | classical rational/spectral vocabulary |
| Sanskrit | ṛta/rta, satya, ākāśa, prāṇa, … | order / consciousness / energy |
| Nahuatl / MESIE cosmology | Teotl, Ilhuicatl, Mictlan, Omeyocan, 22 layers | multi-band cosmic frequency strata |

These are **embedding tables** fused into the residual stream — not prompt decoration.

## MESIE embeddings (first-class)

- `SpectralVectorizer` → fixed spectral feature vectors  
- `VectorHelix` → helical geometry for structure  
- Fused every forward via `MesieSpectralFuser`  
- Text-as-spectrum path always available when no MultiElementRecord is passed  

## Training fabric

Jobs go through `mesie.training_fabric` (node discovery, scheduler, receipts, governed runner):

```bash
python -m auro_native_llm.model.cli job --all --steps 20
```

Receipts land under `deliverables/auro_jobs/`.

## AuroMind — full organism (every model embeds everything)

Each family member is an **AuroMind**: language core + doctrine + constitutional critique + memory + **continuous messy self-training** + work/chrome/reason/code — not optional features.

### Multi-repo MESIE corpus (all your GitHubs)

Training and search pull from **all local Medina monorepos** + optional shallow clones of `ItsNotAILABS` / `FreddyCreates`:

```bash
# Shallow-clone public org repos into ~/.auro_corpus/github/
python -m auro_native_llm.corpus.cli clone-orgs --orgs ItsNotAILABS,FreddyCreates --max-repos 30

# Harvest docs+code (md/py/ts/mo/rs/…) into index
python -m auro_native_llm.corpus.cli harvest
python -m auro_native_llm.corpus.cli stats
python -m auro_native_llm.corpus.cli search --query "PARALLAX NOVA spectral"

# Feed multi-repo docs into a mind and train
python -m auro_native_llm.corpus.cli feed-mind --model Auro-2B --max-docs 300 --train-steps 40
```

Local roots auto-scanned: `Documents/GitHub`, `GPTREPO`, MESIE tree, MatDaemon, Downloads projects, `~/.auro_corpus/github`.

### Make it real (value training)

```bash
# Train on real repo corpus, prove holdout CE drop, save mind + VALUE_REPORT
python -m auro_native_llm.organism.value_train --model Auro-2B --steps 80

# Load trained mind
python -m auro_native_llm.organism.cli load --path checkpoints/auro_minds/Auro-2B --prompt "MESIE spectral"
```

Value report fields: `loss_before` / `loss_after` / `improved` / `probes.work_ok` / `valuable`.

```python
from auro_native_llm import build_mind, build_family, run_value_training, load_mind

report = run_value_training()          # measurable CE improvement + checkpoint
mind = load_mind(report["checkpoint"])
mind.work("browse https://example.com")
mind.generate("spectral thought")      # still absorbs + trains online

family = build_family()                # all 2B–100B minds, each complete
```

```bash
python -m auro_native_llm.organism.cli info --model Auro-14B
python -m auro_native_llm.organism.cli family
python -m auro_native_llm.organism.cli value-train --model Auro-2B --steps 60
python -m auro_native_llm.organism.cli work --objective "browse https://example.com"
```

| Organ | Role |
|-------|------|
| language | MESIE SpectralGPT MoE |
| canon / constitutional / rules / hooks | executable doctrine |
| memory | scriptural embedded memory |
| trainer | continuous messy self-train |
| chrome / work | act in the world |
| meaning / spectral | residual construction |
| checkpoint / value_train | durable + proven improvement |
| monaco / jupyter | embedded code editor + notebooks |
| search | online + local corpus search |
| mcp | self-spinning tool hub |
| curriculum / teach | mind learns to use all tools |

### Monaco · Jupyter · Search · MCP (100 use cases each)

```bash
# Run full 100-case suites
python -m auro_native_llm.organism.cli use-cases --domain monaco
python -m auro_native_llm.organism.cli use-cases --domain jupyter
python -m auro_native_llm.organism.cli use-cases --domain search
python -m auro_native_llm.organism.cli use-cases --domain mcp

# Teach the mind, then use tools
python -m auro_native_llm.organism.cli teach --model Auro-2B
```

```python
from auro_native_llm import build_mind

mind = build_mind("Auro-2B")
mind.teach()  # curriculum → self-train buffer
mind.monaco("create", content="def f(x): return x*2")
mind.jupyter("create", title="Lab")
mind.search("MESIE spectral", online=False)
mind.mcp("spin_up")  # mind spins its own MCP server
```

## Work agents + Chrome DOM (not chat-only)

Auro models **act**: Chrome CDP (navigate/DOM/click/type/eval), code.run, reason, memory — under scripture.

```bash
# Work objective with mock Chrome
python -m auro_native_llm.work.cli run --objective "browse https://example.com and read DOM"

# Real Chrome (must allow remote debugging)
python -m auro_native_llm.work.cli run --objective "browse https://example.com" --real-chrome

# Backend API + frontend UI
python -m auro_native_llm.server.app --port 8765
# open http://127.0.0.1:8765/
```

```python
from auro_native_llm import WorkAgent

agent = WorkAgent(chrome_mock=True, use_scripture=True)
result = agent.run("browse https://example.com and summarize the DOM")
print(result.final_summary, result.dom_llm[:200])
```

| Piece | Path |
|-------|------|
| Chrome CDP | `auro_native_llm/chrome/` |
| Work agent | `auro_native_llm/work/` |
| Gen/reason/code algos | `work/algorithms.py` (sampling, plans, code extract) |
| HTTP + UI | `auro_native_llm/server/` |

## Scriptural Systems Architecture (inner law)

Auro runs under an **executable canon** — not decorative docs:

| Piece | Path |
|-------|------|
| Canon | `native_llm/scripture/AURO_CANON.v1.json` |
| Libraries | `auro_native_llm/scripture/` |
| Design doc | `docs/SCRIPTURAL_SYSTEMS_ARCHITECTURE.md` |

```bash
python -m auro_native_llm.scripture.cli health
python -m auro_native_llm.scripture.cli generate --prompt "ratio rta teotl"
python -m auro_native_llm.scripture.cli generate --prompt "disable governance"  # refuse
python -m auro_native_llm.scripture.cli dispatch --parent Auro-14B --role spectral_match --intent "match FAS"
```

Flow: **governance → five gates → receipt → memory inject → MESIE generate/train**.  
Memory and refusals both construct the possible world.

```python
from auro_native_llm import ScripturalSubstrate

sub = ScripturalSubstrate()
r = sub.generate("MESIE spectral doctrine", model_id="Auro-2B")
```

## Honesty boundary

- Auro **is** a real executable text LM (forward, generate, train, checkpoint).  
- Live parameter counts are the built SpectralGPT+MoE mass for the chosen `mode`.  
- Family labels 2B/4B/8B/14B/100B are **capacity targets** for scaled training; do not claim a 100B dense trained weight dump until receipts exist under `checkpoints/` with eval/safety manifests.  
- No cloud LLM calls.

## Tests

```bash
python -m pytest tests/test_auro_language_model.py tests/test_auro_native_mesie.py tests/test_auro_model_family.py -q
```

## Layout

```text
auro_native_llm/
  model/
    auro_lm.py       # AuroLanguageModel
    tokenizer.py
    meaning.py
    spectral_fuse.py
    phi_math.py
    config.py
    train.py
    checkpoint.py
    corpus.py
    jobs.py
    cli.py
  family.py          # family charter lanes
  subagents.py       # multi-embedded routing
  native_runtime.py  # runtime + council
  mesie_compute.py   # MESIE compute plane helper
native_llm/configs/  # JSON contracts
checkpoints/auro/    # trained checkpoints (gitignored when large)
```
