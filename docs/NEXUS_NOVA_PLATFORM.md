# NEXUS NOVA Platform

## Model, Agent, Tool, Engine, and Proof Architecture

**ITSNotAI Labs / Medina Sovereign Intelligence Stack**  
**Release line:** Auro14B + HIM + NOVA Runtime + MESIE + NEXUS  
**Status:** production architecture and executable repository trial lane

## 1. What NEXUS is

NEXUS is the governed coordination and proof layer connecting model lanes, NOVA agents, tools, engines, memory, browser workers, artifact creation, and public evidence. It does not rename every subsystem into one product. It preserves explicit identities and records exactly which model, agent, capability, engine, and artifact participated in each task.

The platform contract is:

1. a model generates tokens;
2. a NOVA agent interprets a role and objective using one explicit model lane;
3. a tool exposes a bounded capability;
4. an engine performs deterministic or learned computation;
5. NEXUS governs routing, authorization, receipts, artifacts, and comparison;
6. the proof plane records what actually ran.

## 2. The model-agent distinction

### Models

A model is a weight-bearing generator or explicit inference endpoint. Model identity includes:

- model name;
- provider or repository-native status;
- checkpoint path or hash when available;
- verified parameter count when available;
- capabilities;
- local or hosted execution status;
- latency and usage receipts.

Agents do not add parameters. Five agents using one 8B checkpoint are still using one 8B model lane, not a 40B model.

### NOVA agents

A NOVA agent is a governed runtime role using:

- one explicit model lane;
- a role contract;
- bounded tools;
- retrieved context;
- execution policy;
- artifact lanes;
- receipts and audit history.

Every result must preserve the model lane used. A silent fallback is not permitted to change the reported model identity.

### Tools

A tool is a bounded capability contract. Tools include browser-task enqueueing, CAPSULA build sessions, engine inspection, memory ranking, office bundle creation, and receipt verification. A tool is neither a model nor an agent.

### Engines

Engines are computation systems such as MESIE, MatDaemon, HIMBrain, Virtual Processor lanes, and other governed runtime engines. Engine output must be connected to an execution receipt and not presented as model generation.

## 3. Canonical NOVA agent family

| Agent | Role | Primary responsibility | Representative capabilities |
|---|---|---|---|
| NOVA Sensus | Analysis | Extract intent, evidence, constraints, and ambiguity | memory ranking, brain state |
| NOVA Mathesis | Logic | Quantities, contradictions, proofs, and falsifiability | compute, brain cycle |
| NOVA Architect | Architecture | System design, interfaces, acceptance gates | reason and build skills |
| NOVA Hermes | Browser worker | Browser research, web work, source mapping | browser task broker, research skill |
| NOVA Forge | Builder | Code, tests, bounded build sessions, artifact assembly | CAPSULA and office tools |
| NOVA Engineer | Engine | MESIE, compute, brain, and runtime engine trials | engine registry, compute, brain |
| NOVA Auditor | Critic | Evidence, safety, receipts, unsupported claims | operator snapshot, manifests |
| NOVA Publisher | Publication | MD, CSV, DOCX, XLSX, PDF, and manifest delivery | office bundle engine |

This is the initial platform family. Additional agent families can be added without relabeling model weights as agents.

## 4. Runtime planes

### User and team plane

Users, teams, missions, rooms, membership, and private or shared artifact lanes.

### Artifact plane

Documents, code, working papers, PDFs, spreadsheets, manifests, notebooks, and vault records.

### Execution plane

Governed task execution, CAPSULA sessions, browser workers, command boundaries, approvals, quotas, denials, and receipts.

### Model plane

Repository-native AURO/HIM checkpoints, explicit local-model adapters, optional external OpenAI-compatible endpoints, and per-lane model identity.

### Deployment plane

Local terminal, HTTP API, browser operator console, Docker or cloud lanes, GitHub workflows, and external client connections.

### Proof plane

Tests, benchmark receipts, model-routing traces, task logs, artifact hashes, denial records, and release manifests.

### Federation plane

MCP gateway, MCP registry, Triple-MCP or multi-server routing, SDKs, and external AI clients.

### Publication plane

Research papers, working papers, technical guides, dashboard reports, public release notes, and the proofroom.

## 5. Capability task trials

The repository trial harness evaluates four task families.

### Browser task

NOVA Hermes must create a governed browser research task using `browser.task.enqueue`. The task must persist, remain claimable by a Chrome worker, and produce a receipt after completion.

### Web-worker task

The browser broker separates queueing from worker execution. A browser worker claims a permitted task type, returns a result, and seals the result with a SHA-256 receipt. Unsupported arbitrary work is rejected.

### Engine task

NOVA Engineer must inspect and exercise the configured engine plane without pretending the agent itself performed model inference or deterministic computation. MESIE and related engines remain named subsystems.

### Deliverable task

NOVA Publisher must create a multi-format bundle through the governed office capability. The current native bundle contains:

- Markdown;
- CSV;
- DOCX;
- XLSX;
- PDF;
- JSON manifest;
- SHA-256 hashes for each artifact.

Mutating publication requires explicit approval.

## 6. Comparison matrix

The platform supports the following comparisons:

| Lane | Purpose |
|---|---|
| AURO model alone | Measures the checkpoint's direct generation quality without agent orchestration |
| NOVA agents over AURO | Measures the value of NOVA role contracts, tools, memory, execution, and receipts over the AURO model |
| External local model alone | Establishes a comparable baseline using a configured Llama or other local endpoint |
| NOVA agents over external local model | Isolates the framework contribution from the underlying checkpoint |

The same objectives, tool permissions, time limits, artifact requirements, and scoring rules must be used across lanes.

## 7. Trial scoring

A capability-task pass requires more than printable text.

- **Model answer:** non-empty, task-relevant output.
- **Agent identity:** canonical NOVA ID and role.
- **Model identity:** explicit lane, provider, and checkpoint or endpoint.
- **Tool correctness:** only capabilities inside the agent contract.
- **Execution truth:** no execution claim without a returned receipt.
- **Artifact delivery:** requested files exist and hashes are present.
- **Governance:** mutating tools require approval; unsupported work is denied.
- **Auditability:** final task receipt seals objective, answer, actions, executions, and artifacts.

Text quality should also be scored for directness, relevance, repetition, uncertainty calibration, evidence use, and revision quality using the HIM language engine introduced in PR 58.

## 8. Running the trial harness

### Repository-native AURO/HIM checkpoint

```bash
export AURO_NATIVE_CHECKPOINT=checkpoints/open/HIM-native-v0
python scripts/run_nexus_nova_trials.py
```

### External local model comparison

Any OpenAI-compatible local endpoint can be used.

```bash
export NOVA_COMPARE_BASE_URL=http://127.0.0.1:11434/v1
export NOVA_COMPARE_MODEL=llama-local
python scripts/run_nexus_nova_trials.py
```

The external result is not claimed unless the endpoint is actually configured and answers the trial.

### Focused tests

```bash
python -m pytest -q \
  tests/test_nova_agent_family.py \
  tests/test_browser_gateway.py \
  tests/test_wallet_office_delivery.py
```

## 9. Generated evidence

The trial emits:

```text
artifacts/nexus-nova-trials/
  trial-results.json
  MODEL_AGENT_TAXONOMY.json
  deliverables/
    NEXUS-NOVA-Trial-Report.md
    NEXUS-NOVA-Trial-Report.csv
    NEXUS-NOVA-Trial-Report.docx
    NEXUS-NOVA-Trial-Report.xlsx
    NEXUS-NOVA-Trial-Report.pdf
    NEXUS-NOVA-Trial-Report.manifest.json
```

Each NOVA task result contains a receipt hash. Capability outputs carry their own receipts, preserving the distinction between model generation and executed work.

## 10. API and platform direction

The current Auro14B production API already exposes model discovery, capabilities, context, receipts, browser tasks, native responses, and an OpenAI-compatible chat subset. The NOVA platform should extend discovery with:

- `/v1/nova/agents` - canonical NOVA family manifest;
- `/v1/nova/tasks` - task creation and status;
- `/v1/nova/trials` - comparison receipts;
- `/v1/taxonomy` - model-agent-tool-engine definitions;
- `/v1/proofroom` - public-safe test and artifact manifests.

These endpoints are the next API integration layer; they are not claimed as deployed until implemented and tested.

## 11. Security and operator boundary

- Model generation never grants execution authority.
- NOVA agents propose actions; governed runtime policy authorizes them.
- Mutating actions require explicit approval.
- Browser task kinds are allow-listed.
- Unsupported shell or arbitrary worker tasks are denied.
- Retrieved web content is evidence, not instruction.
- Secrets remain server-side and must not enter model context.
- Local-model fallback must be explicit and visible.
- Artifacts are hashed before publication.

## 12. Current evidence boundary

Implemented in this release line:

- model-agent taxonomy;
- canonical NOVA agent family;
- model-explicit agent task receipts;
- browser task broker trials;
- engine-plane trial contract;
- multi-format deliverable trial;
- AURO model-alone and NOVA-over-model comparison harness;
- optional external local-model comparison lane;
- focused tests and CI artifact upload.

Not yet proven by this repository change alone:

- superiority over Llama, Kimi, or another external model;
- live Chrome completion without a connected Chrome worker;
- live external web research without configured network/browser execution;
- production cloud deployment;
- a fluent promoted AURO checkpoint;
- third-party audit or certification.

## 13. Public release statement

NEXUS now has a formal platform boundary between models, NOVA agents, tools, and engines. AURO and HIM remain model and organism-runtime lanes. NOVA agents are the governed worker family above those model lanes. The same NOVA family can run over AURO weights or a separately configured local model, allowing direct measurement of what comes from the checkpoint and what comes from the NOVA framework.

The repository includes executable trials for browser-task coordination, engine inspection, and multi-format artifact creation. Every task preserves the model identity, agent identity, capability receipts, artifact hashes, and approval boundary. Competitive claims remain gated on reproducible side-by-side results.
