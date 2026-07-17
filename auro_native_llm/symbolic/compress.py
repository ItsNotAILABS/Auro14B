"""Real symbolic compression for the 2–4B future.

Thesis (Novel Chaos Labs): a 2–4B *parameter* model + dense *symbolic*
program/rule/doctrine store can beat 70B+ stale dense models on sovereign
tasks — code apps, research, math — when compute is spent on retrieval,
gates, and polyglot engines rather than dead weight.

Compression stack:
  1. Neural core (SpectralGPT) — small dense mass
  2. Symbolic program table — AST/templates/rules (not gradient params)
  3. Doctrine/canon gates — hard symbolic constraints
  4. GitHub retrieval — external knowledge not memorized in weights
  5. Multi-embed codes — compressed continuous memory
  6. Expert routing symbols — MoE + symbolic tags

Effective intelligence ≈ f(params_live, symbols, retrieval, tools, doctrine)
not ≈ params alone.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auro_native_llm.model.phi_math import PHI


@dataclass
class SymbolicProgram:
    """Compressed capability unit — not a neural weight."""

    prog_id: str
    domain: str  # code | research | math | web | doctrine
    template: str
    symbols: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)  # tools/engines to call
    priority: float = 1.0

    def fingerprint(self) -> str:
        h = hashlib.sha256(
            f"{self.domain}|{self.template}|{','.join(self.symbols)}".encode()
        ).hexdigest()
        return h[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def expand(self, **slots: str) -> str:
        out = self.template
        for k, v in slots.items():
            out = out.replace("{" + k + "}", v)
        return out

    @property
    def symbolic_bits(self) -> int:
        # rough information content of symbols+template (not param count)
        return 8 * (len(self.template) + sum(len(s) for s in self.symbols))


@dataclass
class FutureCoreSpec:
    """2B / 4B future architecture with symbolic overlay."""

    name: str
    parameter_target: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    num_experts: int
    top_k_experts: int
    vocab_size: int
    max_seq_len: int
    symbolic_program_budget: int  # max programs in store
    retrieval_dim: int
    thesis: str

    def neural_overrides(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ffn_dim": self.ffn_dim,
            "num_experts": self.num_experts,
            "top_k_experts": self.top_k_experts,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "use_moe": True,
            "use_cross_modal": True,
            "use_spectral_encoder": True,
            "use_spectral_fusion": True,
            "positional_encoding": "rotary",
            "normalization": "rms_norm",
            "activation": "swiglu",
            "qk_norm": True,
            "num_modalities": 8,
            "parameter_target": self.parameter_target,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def future_2b_spec() -> FutureCoreSpec:
    # Live-leaning 2B-class MoE (executable when RAM allows; smaller than full dense 2B)
    return FutureCoreSpec(
        name="Auro-2B-Symbolic",
        parameter_target=2_000_000_000,
        hidden_dim=1536,
        num_layers=20,
        num_heads=16,
        head_dim=96,
        ffn_dim=4096,
        num_experts=16,
        top_k_experts=2,
        vocab_size=32000,
        max_seq_len=4096,
        symbolic_program_budget=50_000,
        retrieval_dim=512,
        thesis=(
            "2B neural + 50k symbolic programs + multi-repo retrieval + "
            "polyglot teachers beat stale 70B chat weights on sovereign work."
        ),
    )


def future_4b_spec() -> FutureCoreSpec:
    return FutureCoreSpec(
        name="Auro-4B-Symbolic",
        parameter_target=4_000_000_000,
        hidden_dim=2048,
        num_layers=24,
        num_heads=16,
        head_dim=128,
        ffn_dim=5504,
        num_experts=16,
        top_k_experts=2,
        vocab_size=48000,
        max_seq_len=8192,
        symbolic_program_budget=100_000,
        retrieval_dim=768,
        thesis=(
            "4B neural + 100k symbols + BRAIN-AI heart + ChaosCUDA + multi-site "
            "agents = society-advancing stack, not param vanity."
        ),
    )


# Seed programs — real capability compression
_SEED_PROGRAMS: List[SymbolicProgram] = [
    SymbolicProgram(
        "sym.code.api",
        "code",
        "APP:REST path={path} method={method} validate={schema} persist={store}",
        ["REST", "validate", "sqlite", "auth"],
        ["python.run", "monaco.create"],
        1.2,
    ),
    SymbolicProgram(
        "sym.code.spa",
        "code",
        "APP:SPA routes={routes} state={store} fetch={api}",
        ["React", "router", "state"],
        ["monaco.create"],
        1.1,
    ),
    SymbolicProgram(
        "sym.math.dft",
        "math",
        "MATH:spectral_energy(x)=sum|DFT(x)| verify_parity(py,jl,hs)",
        ["DFT", "parity", "phi"],
        ["polyglot.suite", "eng.jl.spectral"],
        1.3,
    ),
    SymbolicProgram(
        "sym.research.survey",
        "research",
        "RESEARCH:query={q} retrieve=github_db top_k={k} synthesize=doctrine",
        ["retrieve", "doctrine", "cite"],
        ["sites.work", "search", "brains.teach"],
        1.2,
    ),
    SymbolicProgram(
        "sym.web.multisite",
        "web",
        "WEB:open_many({urls}) || read_all || digest || mind.reason",
        ["parallel", "dom", "fleet"],
        ["sites.open_many", "sites.work", "chrome.navigate"],
        1.4,
    ),
    SymbolicProgram(
        "sym.doctrine.gate",
        "doctrine",
        "GATE:intent={intent} deny_if∈denied_intents allow_ops∈canon",
        ["PL-001", "PL-004", "governance"],
        ["mind.info"],
        1.5,
    ),
    SymbolicProgram(
        "sym.compress.embed",
        "math",
        "COMPRESS:multi_embed(text)→SVD_topk→code dim={d} store=bank",
        ["SVD", "topk", "phi", "1334→256"],
        ["train.entangled"],
        1.2,
    ),
    SymbolicProgram(
        "sym.train.entangled",
        "math",
        "TRAIN:student CE + council(jl,hs,py,chaos) residual→emb",
        ["teacher", "engine", "orchestrator"],
        ["train.entangled", "polyglot.suite"],
        1.3,
    ),
]


class SymbolicCompressor:
    """Program store + neural mass accounting + retrieval expand."""

    def __init__(self, budget: int = 50_000) -> None:
        self.budget = budget
        self.programs: Dict[str, SymbolicProgram] = {}
        for p in _SEED_PROGRAMS:
            self.programs[p.prog_id] = p
        self.hits: Dict[str, int] = {}

    def add(self, prog: SymbolicProgram) -> None:
        if len(self.programs) >= self.budget:
            # drop lowest priority lowest hit
            victim = min(
                self.programs.values(),
                key=lambda p: (p.priority, self.hits.get(p.prog_id, 0)),
            )
            self.programs.pop(victim.prog_id, None)
        self.programs[prog.prog_id] = prog

    def match(self, query: str, top_k: int = 5) -> List[SymbolicProgram]:
        q = query.lower()
        scored: List[Tuple[float, SymbolicProgram]] = []
        for p in self.programs.values():
            s = 0.0
            blob = (p.template + " " + " ".join(p.symbols) + " " + p.domain).lower()
            for tok in re.findall(r"[a-z0-9_]+", q):
                if len(tok) > 2 and tok in blob:
                    s += 1.0
            s *= p.priority
            if s > 0:
                scored.append((s, p))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, p in scored[:top_k]:
            self.hits[p.prog_id] = self.hits.get(p.prog_id, 0) + 1
            out.append(p)
        return out

    def expand_context(self, query: str, top_k: int = 5) -> str:
        progs = self.match(query, top_k=top_k)
        lines = ["[SYMBOLIC_COMPRESS programs]"]
        for p in progs:
            lines.append(
                f"- {p.prog_id} ({p.domain}) {p.template} hooks={p.hooks} bits≈{p.symbolic_bits}"
            )
        lines.append("[/SYMBOLIC_COMPRESS]")
        return "\n".join(lines)

    def effective_intelligence(
        self,
        neural_params: int,
        *,
        retrieval_docs: int = 0,
        tools: int = 0,
    ) -> Dict[str, Any]:
        """Society-advancing metric: not raw params."""
        sym_bits = sum(p.symbolic_bits for p in self.programs.values())
        # crude "effective" units: neural + symbolic*scale + retrieval
        eff = neural_params + sym_bits * 50 + retrieval_docs * 10_000 + tools * 1_000_000
        return {
            "neural_params": neural_params,
            "symbolic_programs": len(self.programs),
            "symbolic_bits": sym_bits,
            "retrieval_docs": retrieval_docs,
            "tools": tools,
            "effective_units": eff,
            "ratio_symbolic_to_neural": (sym_bits * 50) / max(neural_params, 1),
            "thesis": (
                "Effective intelligence grows with symbols+tools+retrieval, "
                "not only dense parameters. 2–4B + symbols can beat stale 70B."
            ),
        }

    def compress_vector(self, vec: np.ndarray, code_dim: int = 64) -> Dict[str, Any]:
        """Symbolic tags + numeric code for a continuous vector."""
        v = np.asarray(vec, dtype=np.float64).ravel()
        if v.size == 0:
            return {"code": [], "tags": [], "energy": 0.0}
        energy = float(np.linalg.norm(v))
        # top-k indices as symbols
        k = min(code_dim, v.size)
        idx = np.argpartition(np.abs(v), -k)[-k:]
        idx = np.sort(idx)
        code = v[idx].tolist()
        tags = [f"s{int(i)}" for i in idx[:16]]
        # match domain by energy bands
        if energy > 1.0:
            tags.append("high_energy")
        tags.append(f"phi_bin_{int((energy * PHI) % 8)}")
        return {
            "code": code[:code_dim],
            "indices": idx.tolist(),
            "tags": tags,
            "energy": energy,
            "original_dim": int(v.size),
            "code_dim": k,
            "compression_ratio": k / max(v.size, 1),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "schema": "auro.symbolic.compress.v1",
            "programs": len(self.programs),
            "budget": self.budget,
            "total_symbolic_bits": sum(p.symbolic_bits for p in self.programs.values()),
            "top_hits": sorted(self.hits.items(), key=lambda x: -x[1])[:10],
            "seed_ids": list(self.programs.keys())[:20],
            "future": {
                "2b": future_2b_spec().to_dict(),
                "4b": future_4b_spec().to_dict(),
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "programs": [p.to_dict() for p in self.programs.values()],
            "hits": self.hits,
            "budget": self.budget,
            "ts": time.time(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SymbolicCompressor":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(budget=int(raw.get("budget", 50_000)))
        c.programs.clear()
        for p in raw.get("programs") or []:
            c.programs[p["prog_id"]] = SymbolicProgram(
                prog_id=p["prog_id"],
                domain=p["domain"],
                template=p["template"],
                symbols=list(p.get("symbols") or []),
                hooks=list(p.get("hooks") or []),
                priority=float(p.get("priority", 1.0)),
            )
        c.hits = dict(raw.get("hits") or {})
        return c
