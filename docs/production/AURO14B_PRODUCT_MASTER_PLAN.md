# Auro14B Product Master Plan

## Mission

Ship Auro14B as an independent public product surface with its own identity, browser application, Cloudflare Worker API, GitHub Pages mirror, deployment gates, receipts, and truthful model-status boundary.

The product must remain useful before full Auro14B weights exist, but it must never relabel hosted or reference weights as Auro14B.

## Product surfaces

### 1. GitHub Pages — public release surface

Purpose:
- public product identity
- architecture and research summaries
- model status and readiness evidence
- API examples
- launch instructions
- release receipts and checksums
- static fallback when the live Worker is unavailable

Rules:
- no secrets
- no private prompts or credentials
- no proprietary deployment state
- no claim of trained Auro14B weights without checkpoint evidence

### 2. Cloudflare Worker — live application and API

Purpose:
- serve the interactive Auro14B browser application
- expose health, model-card, capability, chat, and receipt routes
- route requests to an explicitly configured inference provider
- preserve session and release receipts
- enforce request limits and safe failure modes

Initial endpoints:
- `GET /api/health`
- `GET /api/model-card`
- `GET /api/capabilities`
- `POST /api/chat`
- `GET /api/receipts`

Future endpoints:
- `POST /api/context/query`
- `POST /api/tools/plan`
- `POST /api/evaluate`
- `GET /api/releases/:id`

### 3. Model service boundary

The Worker is not the model. It is the public control and delivery plane.

Supported lanes:
1. `hosted-compatibility` — an identified hosted model used only to keep the application operational.
2. `auro-reference` — a small repository-native checkpoint used for pipeline validation.
3. `auro-promoted` — an exact Auro checkpoint that passed tokenizer, training, benchmark, portability, safety, and readiness gates.

Every response must expose the active lane and model identity. Only `auro-promoted` may be presented as an Auro model release.

### 4. Proof and release lane

Every release should contain:
- source commit
- build receipt
- Worker dry-run receipt
- static-site manifest
- secret-scan result
- model identity receipt
- checkpoint and tokenizer hashes when present
- benchmark versions and limits
- clean-start smoke proof
- unresolved blockers

## User experience

The first public app contains:
- Auro14B identity and status
- live connection state
- chat interface
- model-lane badge
- response metadata
- architecture panel
- release-evidence panel
- links to documentation and repository

The interface must make the distinction between product availability and native-model readiness visible, not hidden in fine print.

## Security boundary

- repository remains public
- secrets exist only in GitHub Actions secrets and Cloudflare secrets
- browser receives no provider or Cloudflare credentials
- no source maps in production
- no arbitrary upstream URL supplied by browser clients
- inference upstream is configured server-side
- request bodies are bounded
- responses use no-store and defensive headers
- logs and receipts never contain bearer tokens or secret values

## Deployment topology

```text
GitHub main
  |-- GitHub Pages workflow -> static public mirror
  |-- Cloudflare workflow -> Worker + static app
  |-- release gate -> manifests, tests, receipts

Browser
  |-- Pages: public documentation and fallback
  |-- Worker: interactive chat and API

Worker
  |-- hosted compatibility model, or
  |-- approved Auro inference service
```

## Delivery phases

### Phase 1 — independent product shell
- create static Auro14B application
- create Worker API
- add explicit model-lane reporting
- add local and CI validation
- add Pages deployment
- add protected Worker deployment

### Phase 2 — persistence and receipts
- add Durable Object session history
- hash-link response receipts
- publish sanitized release receipts
- add rate and abuse controls

### Phase 3 — real Auro integration
- define inference-service contract
- verify checkpoint/tokenizer identity at startup
- reject unknown model hashes
- expose exact promoted release metadata
- add streaming responses

### Phase 4 — evaluation and release certification
- official benchmark harness
- coding execution receipts
- browser/API portability tests
- safety and governed denial tests
- readiness scoring
- downloadable beginner launch bundle

### Phase 5 — product expansion
- repository context and retrieval
- agent and tool planning
- multi-user workspaces
- SDK and MCP surface
- local/private inference options

## Critical gates

A release cannot be called Auro14B-native unless all are true:
- exact checkpoint exists
- tokenizer exists and is lossless
- checkpoint and tokenizer hashes are published
- training configuration and loss history are preserved
- official benchmark results identify task versions and limits
- API and browser smoke tests use the exact checkpoint
- clean installation succeeds
- no unresolved critical safety or provenance blocker remains

## Current honest status

- independent product deployment: in implementation
- Cloudflare-hosted interactive compatibility lane: in implementation
- repository-native Auro reference lane: present elsewhere in repository, not promoted here
- trained Auro14B checkpoint: not verified
- Auro14B-native public model release: not ready

## Loop protocol

For every cycle:
1. select the lowest production gate
2. make one falsifiable change
3. run focused tests
4. run the full release gate
5. compare evidence
6. keep only improvements
7. emit a cycle receipt
8. continue until blocked by unavailable credentials, compute, or external infrastructure
