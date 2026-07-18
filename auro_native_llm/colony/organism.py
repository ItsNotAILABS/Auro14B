"""Colony LLM host — human-scale organism made of mini-model germs.

Python does all the work: spawn specialists, 500k context, train germs,
compose real multi-paragraph generation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.colony.context import Context500k, estimate_tokens
from auro_native_llm.colony.mini import MiniModel, MiniSpec, estimate_params
from auro_native_llm.model.usable import hybrid_answer, is_usable_text


def _load_skills(max_skills: int = 48) -> List[Dict[str, str]]:
    roots = [
        Path(__file__).resolve().parents[2] / ".grok" / "skills",
        Path.home() / ".grok" / "skills",
        Path.home() / ".grok" / "bundled" / "skills",
    ]
    out: List[Dict[str, str]] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for skill_md in sorted(root.rglob("SKILL.md")):
            name = skill_md.parent.name
            if name in seen:
                continue
            try:
                text = skill_md.read_text(encoding="utf-8", errors="ignore")[:2500]
            except Exception:
                continue
            if len(text) < 40:
                continue
            seen.add(name)
            out.append({"name": name, "text": text})
            if len(out) >= max_skills:
                return out
    return out


class ColonyLLM:
    """Model-of-models: host + germs. More params = sum of mini stacks."""

    def __init__(
        self,
        *,
        n_extra_germs: int = 12,
        context_tokens: int = 500_000,
        mind: Any = None,
        train_on_build: bool = True,
    ) -> None:
        self.mind = mind
        self.context = Context500k(token_budget=context_tokens)
        self.germs: Dict[str, MiniModel] = {}
        self.born_at = time.time()
        self.gen_count = 0
        self._build_colony(n_extra_germs=n_extra_germs, train_on_build=train_on_build)

    def _build_colony(self, n_extra_germs: int, train_on_build: bool) -> None:
        # Core organs (always)
        # Larger host organs → more live params (Python mini stacks)
        core = [
            MiniSpec("host.writer", "writer", hidden=192, layers=4, vocab=2048),
            MiniSpec("host.planner", "planner", hidden=160, layers=3, vocab=1536),
            MiniSpec("host.critic", "critic", hidden=128, layers=3, vocab=1024),
            MiniSpec("host.reason", "reason", hidden=160, layers=4, vocab=1536),
            MiniSpec("host.code", "code", hidden=160, layers=3, vocab=1536),
            MiniSpec("host.spectral", "spectral", hidden=192, layers=4, vocab=2048),
            MiniSpec("host.memory", "memory", hidden=128, layers=3, vocab=1024),
        ]
        for i, sp in enumerate(core):
            self.germs[sp.name] = MiniModel(sp, seed=10 + i)

        # Skill germs from SKILL.md (microbiome of skills)
        skills = _load_skills(max_skills=max(8, n_extra_germs))
        for i, sk in enumerate(skills):
            sp = MiniSpec(
                name=f"skill.{sk['name']}",
                role="skill",
                hidden=96 + (i % 5) * 16,
                layers=3,
                vocab=768,
                skill_text=sk["text"],
            )
            g = MiniModel(sp, seed=100 + i)
            self.germs[sp.name] = g
            # ingest skill into 500k context
            self.context.ingest(sk["text"], kind="skill", meta={"skill": sk["name"]})

        # Extra free germs (diversity / capacity)
        for i in range(max(0, n_extra_germs - len(skills))):
            sp = MiniSpec(
                name=f"germ.free.{i}",
                role="writer" if i % 2 == 0 else "reason",
                hidden=56,
                layers=2,
                vocab=400,
            )
            self.germs[sp.name] = MiniModel(sp, seed=200 + i)

        # Doctrine + system cards into context
        doctrine = (
            "Colony doctrine: Python mini-models do the work. "
            "MESIE is deterministic math. GHOST receipts for audit. "
            "Host writer synthesizes germ outputs into real prose. "
            "500k logical context via hierarchical bank."
        )
        self.context.ingest(doctrine, kind="system")

        if train_on_build:
            self.train_germs(steps=2)

    @property
    def num_params_live(self) -> int:
        return int(sum(g.num_params for g in self.germs.values()))

    @property
    def n_germs(self) -> int:
        return len(self.germs)

    def train_germs(self, steps: int = 3, texts: Optional[List[str]] = None) -> Dict[str, Any]:
        corpus = texts or [
            "MESIE SpectralGPT multi-element spectral intelligence",
            "GHOST hybrid deterministic MESIE then LLM escalate",
            "Auro colony mini models skills python host writer",
            "500k context hierarchical retrieval scale summaries",
            "coding solution asserts power stack physics economy",
        ]
        # fold context maps into train text
        corpus = list(corpus) + [self.context.stats().__repr__()[:400]]
        losses = []
        for g in self.germs.values():
            for s in range(steps):
                t = corpus[(hash(g.name) + s) % len(corpus)]
                if g.skill_text:
                    t = g.skill_text[:300] + "\n" + t
                losses.append(g.train_step(t, lr=0.02))
        return {
            "ok": True,
            "steps_per_germ": steps,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "n_germs": self.n_germs,
            "num_params_live": self.num_params_live,
        }

    def _dispatch(self, prompt: str) -> List[str]:
        """Pick germs like immune recruitment."""
        low = prompt.lower()
        chosen = ["host.planner", "host.writer", "host.critic"]
        if any(k in low for k in ("code", "function", "python", "assert", "implement")):
            chosen.append("host.code")
        if any(k in low for k in ("spectral", "mesie", "psd", "fft", "embed", "match")):
            chosen.append("host.spectral")
        if any(k in low for k in ("why", "logic", "reason", "phi", "plan")):
            chosen.append("host.reason")
        chosen.append("host.memory")
        # skill germs by name token overlap
        for name, g in self.germs.items():
            if not name.startswith("skill."):
                continue
            key = name.replace("skill.", "").replace("-", " ").lower()
            if any(tok in low for tok in key.split() if len(tok) > 3):
                chosen.append(name)
        # unique preserve order
        seen = set()
        out = []
        for c in chosen:
            if c in self.germs and c not in seen:
                seen.add(c)
                out.append(c)
        return out[:14]

    def generate(self, prompt: str, *, max_sections: int = 6) -> Dict[str, Any]:
        """Real multi-germ text generation with 500k-aware context."""
        t0 = time.time()
        self.gen_count += 1
        # ingest user turn into long context
        self.context.ingest(prompt, kind="chat", meta={"turn": self.gen_count})
        ctx_slice = self.context.retrieve(prompt, top_k=10, token_cap=6000)

        # host mind hybrid if present (usable intelligence)
        host_ans, host_method = hybrid_answer(prompt, self.mind)

        selected = self._dispatch(prompt)
        sections: List[Dict[str, Any]] = []
        embeddings = []
        for name in selected:
            g = self.germs[name]
            out = g.generate(prompt, context=ctx_slice)
            sections.append(out)
            embeddings.append(np.asarray(out["embedding"], dtype=np.float64))

        # critic merge → final prose
        critic = self.germs["host.critic"].generate(
            prompt,
            context="\n".join(s["text"][:280] for s in sections[:8]),
        )
        writer = self.germs["host.writer"].generate(
            prompt,
            context=ctx_slice[:1500] + "\n" + host_ans[:800],
        )

        # Compose real multi-paragraph answer
        body_parts = [
            writer["text"].split("\n", 1)[-1].strip()
            if writer["text"].startswith("[")
            else writer["text"],
        ]
        # inject best specialist paragraphs
        for s in sections:
            if s["role"] in ("spectral", "skill", "code", "reason", "planner"):
                # strip germ header line
                lines = s["text"].split("\n")
                para = "\n".join(lines[1:]).strip() if len(lines) > 1 else s["text"]
                if para and para not in body_parts[0]:
                    body_parts.append(para)
        if host_ans and is_usable_text(host_ans):
            body_parts.insert(0, host_ans)
        # critic footnote
        crit_lines = critic["text"].split("\n")
        body_parts.append(
            "Quality check: " + (" ".join(crit_lines[1:]).strip() if len(crit_lines) > 1 else critic["text"][:200])
        )

        # unique-ish join
        final_chunks = []
        seen_txt = set()
        for p in body_parts:
            key = p[:80]
            if key in seen_txt:
                continue
            seen_txt.add(key)
            final_chunks.append(p.strip())
            if len(final_chunks) >= max_sections:
                break
        text = "\n\n".join(final_chunks)
        if not is_usable_text(text):
            text = host_ans if is_usable_text(host_ans) else text

        # train germs lightly on success
        train = self.train_germs(steps=1, texts=[prompt, text[:500]])

        # fused embedding
        if embeddings:
            # pad to same
            d = max(e.size for e in embeddings)
            mat = np.zeros((len(embeddings), d))
            for i, e in enumerate(embeddings):
                mat[i, : e.size] = e
            fused = mat.mean(axis=0)
            fused = fused / (np.linalg.norm(fused) + 1e-12)
        else:
            fused = np.zeros(32)

        return {
            "schema": "auro.colony.generate.v1",
            "ok": is_usable_text(text),
            "text": text,
            "answer": text,
            "method": f"colony+{host_method}",
            "n_germs_total": self.n_germs,
            "n_germs_active": len(selected),
            "active_germs": selected,
            "num_params_live": self.num_params_live,
            "context": self.context.stats(),
            "train": train,
            "embedding_dim": int(fused.size),
            "latency_ms": (time.time() - t0) * 1000.0,
            "architecture": "python_colony_of_mini_models",
            "metaphor": "host+germs (human+microbiome)",
            "context_window_tokens": self.context.token_budget,
            "claim_boundary": (
                f"Live params = sum of {self.n_germs} mini Python models "
                f"({self.num_params_live:,}). "
                f"Context window = {self.context.token_budget:,} tokens logical "
                f"(hierarchical bank, not single Softmax). "
                f"Real multi-paragraph generation from germ composition + host knowledge."
            ),
        }

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.colony.info.v1",
            "n_germs": self.n_germs,
            "num_params_live": self.num_params_live,
            "context": self.context.stats(),
            "germs": [g.info() for g in list(self.germs.values())[:40]],
            "gen_count": self.gen_count,
            "context_window_tokens": self.context.token_budget,
        }


def build_colony(
    mind: Any = None,
    *,
    n_extra_germs: int = 16,
    context_tokens: int = 500_000,
) -> ColonyLLM:
    return ColonyLLM(
        mind=mind,
        n_extra_germs=n_extra_germs,
        context_tokens=context_tokens,
        train_on_build=True,
    )
