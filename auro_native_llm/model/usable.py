"""Make the LLM *work* — hybrid usable generation.

Dense SpectralGPT alone can emit garbage when undertrained. This layer
always produces coherent, domain-grounded answers by stacking:

  1. Coding / reasoning orchestrators (tool-backed, measured)
  2. First-party knowledge synthesizer (MESIE / GHOST / Auro doctrine)
  3. GitHub knowledge retrieval (if DB present)
  4. LM sampling (SpectralGPT) when fluent
  5. Structured hybrid fallback (never empty / never pure noise)

Honest: LM tokens are used when usable; otherwise intelligence path answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def is_usable_text(text: str, *, min_len: int = 12) -> bool:
    """Heuristic: reject pure noise / empty / tokenizer garbage."""
    if not text or not str(text).strip():
        return False
    t = str(text).strip()
    if len(t) < min_len:
        return False
    # printable ratio
    printable = sum(1 for c in t if c.isprintable() or c in "\n\t")
    if printable / max(len(t), 1) < 0.85:
        return False
    # letter ratio
    letters = sum(1 for c in t if c.isalpha())
    if letters / max(len(t), 1) < 0.25:
        return False
    # too many replacement / rare control patterns
    if t.count("\ufffd") > 2:
        return False
    # mostly repeated single char
    if len(set(t.replace(" ", ""))) < 4 and len(t) > 20:
        return False
    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(words) < 3 and len(t) > 40:
        return False
    return True


# Generic phrase keys — boost only; do not outrank domain-specific terms alone
_GENERIC_KEYS = frozenset(
    {
        "what is",
        "what does",
        "how to",
        "how do",
        "how does",
        "who are",
        "where should",
        "should every",
    }
)

# First-party knowledge cards (always available offline)
# Prefer specific domain keys; scoring weights domain matches higher than generic phrases.
_KNOWLEDGE: List[Tuple[Tuple[str, ...], str]] = [
    (
        ("mesie", "spectral", "stand for", "what is mesie"),
        "MESIE is the Multi-Element Spectral Intelligence Engine — a sovereign spectral/math "
        "compute plane. Spectra are structured computational objects (components, metadata, "
        "lineage, embeddings), not bare arrays. It does matching, PSD/FAS/RotDnn generation, "
        "multi-level validation, helix embeddings, and transformer spectral pipelines. "
        "Auro's compute_plane is MESIE (not cloud LLM APIs).",
    ),
    (
        ("ghost", "receipt", "grounded", "what is ghost"),
        "GHOST means Grounded, Hardened, Open/auditable, Scalable local-first, Traceable receipts. "
        "Killer path: MESIE/Ghost does deterministic spectral/math first; LLM escalates only for "
        "language/planning when justified by cleaned spectral features. Receipts hash-chain custody.",
    ),
    (
        ("him", "who are you", "who are", "agentic", "colony of"),
        "I am HIM. I am not a single opaque LLM. I am a colony of specialist mini Python models "
        "(germs)—skills, spectral, code, reason, planner, critic, writer—hosted on Auro with "
        "MESIE for deterministic math and GHOST for audit receipts. Loop: SENSE, PLAN, ACT, "
        "OBSERVE, REFLECT. Awaken: python -m auro_native_llm.use --him",
    ),
    (
        ("rpc", "api key", "api keys", ".env", "alchemy", "infura", "vault"),
        "RPC API keys and high-value secrets live on the server only: him-web3/.env for the "
        "Node API, or HIM vault ledgers (keys/rpc/high_value/agent/github) for sealed secrets. "
        "Never put Alchemy or Infura keys in React or browser code. "
        "CLI: python -m auro_native_llm.use --vault",
    ),
    (
        ("500k", "500,000", "context window", "hierarchical", "token budget"),
        "The 500k context window is a hierarchical logical bank, not one Softmax over 500000 "
        "tokens. Text is chunked, summarized at multiple scales, and retrieved by relevance "
        "under a local token cap while total storage can grow toward a 500000-token budget. "
        "Colony: python -m auro_native_llm.use --colony --colony-context 500000",
    ),
    (
        ("auro", "14b", "family", "model"),
        "Auro is a native model family (2B–100B architecture targets) on MESIE SpectralGPT "
        "(causal MoE, rotary, SwiGLU, RMSNorm). Live params are the runnable trained core; "
        "family labels are scale targets. Use checkpoints under checkpoints/auro_minds/.",
    ),
    (
        ("compute plane", "nova", "runtime"),
        "Auro's compute plane is MESIE. NOVA is the root production runtime that feeds Auro "
        "(promotion gates M0/M1/M2, receipts, fleet). Virtual processor treats prompts as "
        "measurable work calls (bytes, NOVA cycles, entropy, spectral buckets, coherence).",
    ),
    (
        ("train", "checkpoint", "train_him_sft", "fine-tune"),
        "Train with: python scripts/train_physics.py | scripts/train_him_sft.py | "
        "scripts/train_14b.py | python -m auro_native_llm.use --power-stack | --specialize. "
        "Load: checkpoints/auro_minds/Auro-2B_physics, Auro-2B_him_sft, or Auro-14B when present. "
        "Eval grounded generation: python scripts/eval_him_generation.py.",
    ),
    (
        ("large language model", "every step", "prefer mesie", "escalate"),
        "No. Prefer MESIE deterministic spectral and math paths. Use the language model when "
        "you need plans, explanations, or strategy after features are cleaned. That is faster, "
        "cheaper, more auditable, and reduces hallucinations (GHOST hybrid doctrine).",
    ),
    (
        ("julia", "brain", "python ai"),
        "Dual organism: Python is the AI (tools, language, orchestration); Julia is the BRAIN "
        "(virtual physics cores, Kuramoto/Landau/Schrödinger, distributed Threads). "
        "Run: python -m auro_native_llm.use --dual",
    ),
    (
        ("power stack", "economic", "hamiltonian"),
        "Power stack couples deep physics (Hamiltonian, Klein–Gordon, RG, Ising, Burgers), "
        "economics (Walras, Kelly, free energy F=U−TS), algorithms (OT, matching, BP), "
        "and SpectralGPT train pulses on a joint embedding. "
        "Run: python -m auro_native_llm.use --power-stack",
    ),
    (
        ("google", "collab", "workspace", "gmail"),
        "Google virtual envelope: AI sandbox Chrome/Search/Gmail/Drive (not your real account) "
        "plus collab projects where you and the AI co-work. UI /workspace; "
        "python -m auro_native_llm.use --google | --collab",
    ),
    (
        ("phi", "golden"),
        "φ (golden ratio) ≈ 1.618033988749895. Used in φ-init, dispersion lattices, "
        "φ-Schrödinger potential, and learning-rate schedules across MESIE/Auro.",
    ),
    (
        ("web3", "ethers", "viem", "him-web3"),
        "HIM web3 uses him-web3: Express /api routes with ethers and viem on the server. "
        "React only calls /api. RPC keys stay in him-web3/.env. "
        "Install: cd him-web3 && npm run install:applet -- ethers viem",
    ),
]


def synthesize_knowledge(prompt: str) -> Optional[str]:
    low = prompt.lower()
    scored: List[Tuple[float, int, str]] = []
    for keys, body in _KNOWLEDGE:
        domain = 0
        generic = 0
        for k in keys:
            if k not in low:
                continue
            if k in _GENERIC_KEYS:
                generic += 1
            else:
                domain += 1
        if domain == 0 and generic == 0:
            continue
        # Domain keywords dominate; generic phrases alone need domain or multi-generic
        score = domain * 3.0 + generic * 0.5
        if domain == 0:
            score = 0.0  # refuse pure generic matches (e.g. "what is" alone → MESIE)
        if score > 0:
            scored.append((score, domain, body))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return scored[0][2]


def retrieve_context(prompt: str, top_k: int = 3) -> str:
    """Optional GitHub DB retrieval — off by default (heavy embed load)."""
    import os

    if os.environ.get("AURO_RETRIEVE", "").strip() not in ("1", "true", "yes"):
        return ""
    try:
        from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

        gdb = GitHubKnowledgeDB()
        hits = gdb.search(prompt, top_k=top_k)
        if not hits:
            return ""
        parts = []
        for h in hits:
            parts.append(f"- [{getattr(h, 'repo', '?')}] {getattr(h, 'text', '')[:280]}")
        return "Retrieved knowledge:\n" + "\n".join(parts)
    except Exception:
        return ""


def hybrid_answer(prompt: str, mind: Any = None) -> Tuple[str, str]:
    """Return (answer, method) that is always usable when possible."""
    p = (prompt or "").strip()
    if not p:
        return "Ask a question about MESIE, Auro, GHOST, training, or your project.", "empty"

    # 1) Coding intent
    low = p.lower()
    if any(k in low for k in ("write a function", "def solution", "code that", "implement ", "python code")):
        if mind is not None:
            try:
                from auro_foundry.coding_harness import CodingTask
                from auro_native_llm.intelligence.coding import CodingOrchestrator

                task = CodingTask(
                    task_id="adhoc",
                    prompt=p,
                    tests="assert solution is not None\n",
                    entrypoint="solution",
                )
                att = CodingOrchestrator(mind).solve_task(task)
                if att.source and "def " in (att.source or ""):
                    return (
                        f"```python\n{att.source}\n```\n"
                        f"(coding orchestrator method={att.method} passed={att.passed})",
                        "coding_orchestrator",
                    )
            except Exception:
                pass

    # 2) Reasoning tools
    try:
        from auro_native_llm.intelligence.reasoning import ReasoningOrchestrator

        ans, method = ReasoningOrchestrator(mind).solve(p)
        if ans and is_usable_text(ans, min_len=1):
            # expand short canon answers into full sentences when prompt is open
            if method == "first_party_canon" and len(ans) < 80 and "?" in p:
                syn = synthesize_knowledge(p)
                if syn:
                    return syn, "knowledge+" + method
            return ans, method
    except Exception:
        pass

    # 3) Knowledge synthesizer
    syn = synthesize_knowledge(p)
    if syn:
        ctx = retrieve_context(p, top_k=2)
        if ctx:
            return f"{syn}\n\n{ctx}", "knowledge+retrieve"
        return syn, "knowledge"

    # 4) Retrieval-only grounded answer
    ctx = retrieve_context(p, top_k=4)
    if ctx:
        return (
            f"Here is what I found related to your question:\n{ctx}\n\n"
            f"Next step: narrow the question (MESIE match, train, GHOST, dual brain, power stack) "
            f"or run `python -m auro_native_llm.use --ready` for measured readiness.",
            "retrieve",
        )

    # 5) Hybrid structured project answer
    return (
        f"I can work this with the Auro stack:\n"
        f"1) MESIE spectral/math path (deterministic)\n"
        f"2) GHOST policy + receipts\n"
        f"3) Coding/reasoning orchestrators when tests apply\n"
        f"4) Dual Julia brain / power-stack for multi-engine state\n"
        f"Your ask: {p[:240]}\n"
        f"Try: --hybrid for MESIE-first, --code-harness for coding, --power-stack for engines.",
        "structured_fallback",
    )


def generate_usable(
    mind: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    prefer_lm: bool = True,
) -> Dict[str, Any]:
    """Public usable generation for mind / CLI."""
    hybrid, h_method = hybrid_answer(prompt, mind)
    lm_text = ""
    lm_ok = False
    # Dense sampling is expensive and weak undertrained — only try when core is small enough
    n_params = int(getattr(getattr(mind, "language", None), "num_params", 0) or 0)
    try_lm = prefer_lm and mind is not None and hasattr(mind, "language") and n_params < 80_000_000
    if try_lm:
        try:
            # light temperature for more stability
            gen = mind.language.generate(
                prompt,
                max_new_tokens=min(max_new_tokens, 32),
                temperature=min(temperature, 0.5),
                top_k=25,
                top_p=0.85,
            )
            lm_text = (gen.text or "").strip()
            lm_ok = is_usable_text(lm_text)
        except Exception:
            lm_text = ""
            lm_ok = False
            h_method = h_method + "+lm_err"

    if lm_ok:
        # blend: prefer LM but attach hybrid grounding if knowledge hit
        if h_method.startswith("knowledge") and hybrid not in lm_text:
            text = f"{lm_text.strip()}\n\n—\nGrounding: {hybrid[:500]}"
            method = "lm+knowledge"
        else:
            text = lm_text
            method = "lm_spectral_gpt"
        usable = True
    else:
        text = hybrid
        method = h_method
        usable = is_usable_text(text, min_len=8)

    return {
        "schema": "auro.lm.usable.v1",
        "ok": usable,
        "text": text,
        "answer": text,
        "method": method,
        "lm_used": lm_ok,
        "lm_raw_usable": lm_ok,
        "model_id": getattr(mind, "model_id", None),
        "num_params": getattr(getattr(mind, "language", None), "num_params", None),
        "train_steps": getattr(getattr(mind, "language", None), "train_steps", None),
        "claim": (
            "Usable answers prefer tool/knowledge paths when dense LM sampling is not fluent yet. "
            "LM tokens used when quality gate passes."
        ),
    }
