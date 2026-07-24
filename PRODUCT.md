# Auro Operator Console — Product Charter

## Product

**Public name:** Auro14B · HIM  
**Canonical title:** *AURUM ORGANISMUS XIV — Human–Intelligence Mesh*  
**Abbreviation:** AOX–HIM  
**Parent:** NOVA / MESIE  
**Inheritance:** NOVA governance → MESIE deterministic compute → HIM organism → Auro production fleet  
**Legacy aliases:** Auro, Auro14B, HIM (preserved; never silently renamed)

## Mission

Give one human operator a calm, legible surface for inspecting Auro's real model endpoint, internal council, native capability contracts, receipt chain, paper wallet, integrity vault, and Office delivery engine. The console makes evidence visible and keeps mutation behind explicit approval and an operator token.

## Users and jobs

- **Primary:** the sovereign operator running Auro locally, often reviewing dense runtime state quickly.
- **Secondary:** maintainers validating a release or investigating a failed capability.
- Inspect health and the configured model without implying unverified parameters.
- Ask Auro for a model-backed response and see bounded reasoning summaries, agents, actions, and receipts.
- Inspect native capabilities and their approval requirements.
- Verify the receipt chain and paper ledger.
- Create real deliverables only after deliberate approval.

## Product principles

1. **Evidence before theater.** A configured endpoint is not a trained checkpoint; an agent count is never a parameter count.
2. **Local-first authority.** Bind to loopback by default. Execution requires `AURO_EXECUTION_TOKEN` and explicit approval.
3. **Receipts are part of the result.** Model responses and capability calls produce hash-linked evidence.
4. **One organism, named organs.** BRAIN AI, NOVA, MatDaemon, CAPSULA, PARALLAX, and Native Office retain their identities and proof boundaries.
5. **Useful without secrets.** The console never stores or echoes credentials. Token input remains in memory for the active page only.
6. **Progressive disclosure.** Status first; technical JSON is available but does not dominate the interface.

## Runtime topology

`Operator → Auro Console → loopback HTTP server → NOVA runtime → internal council → organ SDK / native capabilities → receipt ledger`

The server may call a configured OpenAI-compatible local endpoint. The console itself performs no model inference and holds no private chain key.

## Proof boundaries

| System | Role | Proof boundary |
|---|---|---|
| Auro14B · HIM | operator-facing organism | only configured parameter counts marked verified; model family names are targets |
| NOVA | orchestration and synthesis | model-backed output plus bounded summaries; no private chain-of-thought |
| BRAIN AI | cognitive state adapter | reports adapter/runtime response only |
| MatDaemon | ranking and bounded compute | returned values and execution receipts |
| CAPSULA | governed build sessions | session files, run output, and manifests |
| PARALLAX | accounting boundary | paper/testnet credits only; no custody, signing, or live settlement |
| Integrity Vault | content integrity | content-addressed storage; not encrypted secret custody |
| Native Office | document production | generated MD/CSV/DOCX/XLSX/PDF plus SHA-256 manifest |

## Security and governance

- Default host is `127.0.0.1`; non-loopback deployment requires an explicit operator decision and an external TLS/auth proxy.
- Read endpoints do not require the execution token. Approved mutation and `execute=true` do.
- Browser assets are same-origin, dependency-free, and served with a restrictive Content Security Policy.
- No third-party analytics, remote fonts, cookies, localStorage, or token persistence.
- Request bodies are size-limited; errors are bounded; directory traversal is impossible because assets are embedded.
- Paper wallet cannot overdraft and uses double-entry postings. Live custody remains out of scope.

## Accessibility and performance

- WCAG 2.2 AA target: keyboard operation, visible focus, semantic landmarks, labeled controls, status announcements, and reduced-motion support.
- System fonts only. No blocking assets or client framework. Initial HTML/CSS/JS is served from the Python process.
- Responsive from 360 px through desktop; dense evidence remains horizontally scrollable rather than clipped.

## Release gates

A release is eligible only when:

- server and UI route tests pass;
- mutation remains fail-closed without token and approval;
- health, capabilities, receipt verification, and console smoke checks pass;
- CSP and defensive headers are present;
- no UI copy claims a checkpoint, parameter count, custody, encryption, or execution that evidence does not prove;
- the LLM Production Forge readiness record distinguishes product-surface readiness from benchmark intelligence.

## Explicit non-goals

- No claim that Auro14B currently contains 14B trained parameters.
- No autonomous financial execution, private-key custody, or mainnet settlement.
- No exposure of hidden chain-of-thought.
- No replacement of NOVA, MESIE, BRAIN AI, MatDaemon, CAPSULA, or PARALLAX names.

