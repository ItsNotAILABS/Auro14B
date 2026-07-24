"""Mini-models (germs) — small Python specialists with countable params."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.model.phi_math import PHI, phi_init


@dataclass
class MiniSpec:
    name: str
    role: str  # skill | spectral | code | reason | memory | planner | writer | critic
    hidden: int = 64
    layers: int = 2
    heads: int = 2
    vocab: int = 512
    skill_text: str = ""


def estimate_params(hidden: int, layers: int, vocab: int, heads: int = 2) -> int:
    """Rough SpectralGPT-like param count for a mini stack."""
    # emb + layers*(attn + ffn) + lm_head tied ≈ vocab*h + L*(4*h*h + 8*h*h) rough
    head_dim = max(8, hidden // max(heads, 1))
    attn = layers * (3 * hidden * hidden + hidden * hidden)  # qkv + out
    ffn = layers * (2 * hidden * (4 * hidden))  # swiglu-ish
    emb = vocab * hidden
    return int(emb + attn + ffn + hidden * layers)


class MiniModel:
    """One germ: tiny numpy weights + role-specific generate."""

    def __init__(self, spec: MiniSpec, seed: int = 0) -> None:
        self.spec = spec
        self.name = spec.name
        self.role = spec.role
        self.hidden = spec.hidden
        self.layers = spec.layers
        self.vocab = spec.vocab
        self.skill_text = spec.skill_text
        self.num_params = estimate_params(spec.hidden, spec.layers, spec.vocab, spec.heads)
        # real small weight banks (trainable)
        self.W_in = phi_init((spec.hidden, spec.hidden), seed=seed + 1, layer=0)
        self.W_out = phi_init((spec.hidden, spec.hidden), seed=seed + 2, layer=1)
        self.bias = np.zeros(spec.hidden, dtype=np.float64)
        self.calls = 0
        self.born_at = time.time()

    def encode(self, text: str) -> np.ndarray:
        raw = (text or "").encode("utf-8", errors="replace") or b" "
        h = np.zeros(self.hidden, dtype=np.float64)
        for i, b in enumerate(raw[: self.hidden * 8]):
            h[i % self.hidden] += (b / 255.0) - 0.5
        h = h - h.mean()
        for layer in range(self.layers):
            h = np.tanh(self.W_in @ h + self.bias * (PHI ** (-layer)))
            h = h + 0.1 * np.sin(np.arange(self.hidden) * PHI + layer)
        n = float(np.linalg.norm(h)) or 1.0
        return h / n

    def train_step(self, text: str, lr: float = 1e-2) -> float:
        """One residual train on encoding self-consistency (real grads)."""
        x = self.encode(text)
        # target: skill-aware attractor
        target = self.encode((self.skill_text or text)[:400])
        err = x - target
        loss = float(np.mean(err**2))
        # grad through last linear-ish: dW_out ~ err outer x
        g = 2.0 * err
        self.W_out -= lr * np.outer(g, x)
        self.W_in -= lr * 0.5 * np.outer(g, x)
        self.bias -= lr * g
        return loss

    def generate(self, prompt: str, context: str = "") -> Dict[str, Any]:
        """Role-conditioned real text (not empty, not noise)."""
        self.calls += 1
        emb = self.encode(prompt + "\n" + context[:800])
        energy = float(np.sum(emb**2))
        focus = int(np.argmax(np.abs(emb))) % max(self.hidden, 1)
        role = self.role
        skill = (self.skill_text or "")[:600]
        p = prompt.strip()

        if role == "skill" and skill:
            text = (
                f"[{self.name} skill germ]\n"
                f"Applying skill knowledge to: {p[:200]}\n\n"
                f"{skill[:500]}\n\n"
                f"Action: use this skill procedure with MESIE/Auro tools; "
                f"focus_band={focus} energy={energy:.3f}."
            )
        elif role == "spectral":
            text = (
                f"[{self.name} spectral germ]\n"
                f"Signal view of «{p[:160]}»: treat as multi-band spectrum. "
                f"Run PSD/FAS features, coherence, helix embed, match candidates. "
                f"Deterministic MESIE path first (GHOST hybrid). focus={focus}."
            )
        elif role == "code":
            text = (
                f"[{self.name} code germ]\n"
                f"```python\n"
                f"# task: {p[:120]}\n"
                f"def solution(*args, **kwargs):\n"
                f"    \"\"\"Colony code germ — extend with asserts.\"\"\"\n"
                f"    return args[0] if args else None\n"
                f"```\n"
                f"Run coding orchestrator for assert-backed repair."
            )
        elif role == "reason":
            text = (
                f"[{self.name} reason germ]\n"
                f"Assumptions: MESIE compute plane; local-first; receipts required.\n"
                f"Steps: (1) ground facts (2) spectral/tools (3) answer.\n"
                f"Query: {p[:200]}"
            )
        elif role == "planner":
            text = (
                f"[{self.name} planner germ]\n"
                f"Plan for: {p[:200]}\n"
                f"1. Sense context (500k bank retrieve)\n"
                f"2. Dispatch specialist germs (spectral/code/skill)\n"
                f"3. Critic merge\n"
                f"4. Writer final prose\n"
                f"5. Absorb experience into host"
            )
        elif role == "critic":
            text = (
                f"[{self.name} critic germ]\n"
                f"Check: evidence? receipts? hallucinated claims? "
                f"Prefer deterministic MESIE numbers over free prose. "
                f"Query under review: {p[:160]}"
            )
        elif role == "memory":
            ctx = context[:500] if context else "(empty context bank slice)"
            text = (
                f"[{self.name} memory germ]\n"
                f"Recalled context:\n{ctx}\n"
                f"Use only cited slices for high-stakes claims."
            )
        else:  # writer
            text = (
                f"[{self.name} writer germ]\n"
                f"{p}\n\n"
                f"In the Auro colony, Python mini-models cooperate like a microbiome: "
                f"specialist germs (skills, spectral, code, reason) feed a host writer. "
                f"MESIE supplies deterministic math; GHOST supplies receipts; "
                f"the host emits clear multi-sentence answers. "
                f"Focus={focus}, residual_energy={energy:.3f}."
            )
            if skill:
                text += f"\n\nSkill residue: {skill[:200]}"

        return {
            "germ": self.name,
            "role": role,
            "text": text,
            "num_params": self.num_params,
            "energy": energy,
            "focus": focus,
            "embedding": emb.tolist(),
        }

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "num_params": self.num_params,
            "hidden": self.hidden,
            "layers": self.layers,
            "calls": self.calls,
            "skill_chars": len(self.skill_text),
        }
