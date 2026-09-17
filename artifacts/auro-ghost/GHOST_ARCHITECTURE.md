# GHOST in Auro14B

ItsNotAILabs / LoomMultiAI design philosophy — **implemented** in `auro_native_llm/ghost/`.

## Pillars

| Letter | Pillar | Implementation |
|--------|--------|----------------|
| **G** | Grounded Intelligence | `ClaimKind` + `label_claim` — operator / evidence / observation / inference / unresolved / deterministic MESIE |
| **H** | Hardened Operation | `PolicyGate` + `RiskClass` C0–C5, least-privilege scopes, persona cannot weaken |
| **O** | Open and Auditable | Inspectable receipt payloads: intent, policy, tool I/O, validation, hashes |
| **S** | Scalable Local-First | Offline default, remote gateway closed, MESIE local engines first |
| **T** | Traceable Receipts | `GhostReceiptChain` hash-linked custody answers + replay verify |

## Hybrid flow

```
Human intent
  → GhostSupervisor (MONDAY-light)
  → PolicyGate (risk class + scopes)
  → GhostAgentRuntime
       → MesieGhostNode (spectral, embed, helix, intelligence, connectome)
       → LLM escalate only if plan/language needed or MESIE incomplete
  → Haunt detector
  → Receipt chain + grounded claims
```

## CLI

```bash
python -m auro_native_llm.use --ghost "Embed spectral features then plan a match strategy"
python -m auro_native_llm.use --specialize   # trains generative+embed+self-config+GHOST+skills
```

## Python

```python
from auro_native_llm.organism.checkpoint import load_mind
mind = load_mind("checkpoints/auro_minds/Auro-2B_continual")
out = mind.ghost_run("Match two spectral streams with receipts")
print(out["custody"], out["haunt_flags"])
```

## Files

- `ghost/pillars.py` — doctrine + claim kinds  
- `ghost/policy.py` — risk classes / hardened scopes  
- `ghost/receipts.py` — hash chain of custody  
- `ghost/node.py` — MESIE Ghost Node (deterministic)  
- `ghost/agent.py` — Ghost Agent hybrid  
- `ghost/supervisor.py` — supervisory interface  
- `ghost/doctrine.py` — training blocks  

Receipts land in `artifacts/auro-ghost/`.
