"""Entangled multi-language engines, transformers, orchestrators, teachers.

Not side tools — these roles sit *inside* the training/inference loop:

  ENGINE        — runs spectral / embed / micro-train kernels (Py / Julia / HS)
  TRANSFORMER   — maps text → multi-view residual for the MESIE hidden stream
  ORCHESTRATOR  — routes work across engines by capability + latency
  TEACHER       — produces teaching signals (labels, soft targets, doctrine)

The main SpectralGPT remains the student; polyglot roles accelerate and teach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auro_native_llm.polyglot.cuda_plane import CudaPlane, get_cuda_plane
from auro_native_llm.polyglot.organ import PolyglotOrgan
from auro_native_llm.polyglot.runtimes import PolyglotRuntime


class RoleKind(str, Enum):
    ENGINE = "engine"
    TRANSFORMER = "transformer"
    ORCHESTRATOR = "orchestrator"
    TEACHER = "teacher"


@dataclass
class LangRole:
    role_id: str
    kind: RoleKind
    lang: str  # python | julia | haskell | cuda
    capability: str
    description: str
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "kind": self.kind.value,
            "lang": self.lang,
            "capability": self.capability,
            "description": self.description,
            "weight": self.weight,
        }


# Canonical entanglement map
DEFAULT_ROLES: List[LangRole] = [
    # ENGINES
    LangRole("eng.py.spectral", RoleKind.ENGINE, "python", "spectral_energy", "NumPy DFT energy"),
    LangRole("eng.jl.spectral", RoleKind.ENGINE, "julia", "spectral_energy", "Julia multi-thread DFT"),
    LangRole("eng.hs.spectral", RoleKind.ENGINE, "haskell", "spectral_energy", "Haskell pure DFT"),
    LangRole("eng.cuda.matmul", RoleKind.ENGINE, "cuda", "matmul", "CUDA/torch/numpy matmul plane"),
    LangRole("eng.jl.phi", RoleKind.ENGINE, "julia", "phi_powers", "Julia φ lattice"),
    LangRole("eng.hs.phi", RoleKind.ENGINE, "haskell", "phi_powers", "Haskell φ lattice"),
    # TRANSFORMERS
    LangRole("xf.py.multi", RoleKind.TRANSFORMER, "python", "multi_embed", "MESIE 9-view multi-embed"),
    LangRole("xf.jl.fft", RoleKind.TRANSFORMER, "julia", "multi_fft_embed", "Julia multi-scale FFT embed"),
    LangRole("xf.hs.fft", RoleKind.TRANSFORMER, "haskell", "multi_fft_embed", "Haskell multi-scale embed"),
    # ORCHESTRATORS
    LangRole("orc.py.route", RoleKind.ORCHESTRATOR, "python", "route", "Route engines by latency+parity"),
    LangRole("orc.py.council", RoleKind.ORCHESTRATOR, "python", "council", "Blend teacher signals into CE"),
    # TEACHERS
    LangRole("tch.py.doctrine", RoleKind.TEACHER, "python", "doctrine", "Python organ + PL laws"),
    LangRole("tch.jl.spectral", RoleKind.TEACHER, "julia", "spectral_teacher", "Julia soft spectral targets"),
    LangRole("tch.hs.structure", RoleKind.TEACHER, "haskell", "structure_teacher", "Haskell structure/φ targets"),
    LangRole("tch.cuda.fast", RoleKind.TEACHER, "cuda", "fast_ce", "Accelerated CE / linear head step"),
]


@dataclass
class CouncilSignal:
    """Teaching residual for the student LM."""

    source: str
    kind: str
    vector: Optional[np.ndarray] = None
    scalar: float = 0.0
    text: str = ""
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "scalar": self.scalar,
            "text": self.text[:500],
            "weight": self.weight,
            "dim": 0 if self.vector is None else int(self.vector.size),
            "meta": self.meta,
        }


class PolyglotOrchestrator:
    """Routes engines and assembles a teaching council each train step."""

    def __init__(self) -> None:
        self.roles = list(DEFAULT_ROLES)
        self.rt = PolyglotRuntime()
        self.cuda = get_cuda_plane()
        self.organ = PolyglotOrgan()
        self.latency_ema: Dict[str, float] = {}
        self.calls = 0

    def roster(self) -> Dict[str, Any]:
        st = self.rt.status()
        return {
            "schema": "auro.polyglot.roster.v1",
            "roles": [r.to_dict() for r in self.roles],
            "by_kind": {
                k.value: [r.to_dict() for r in self.roles if r.kind == k]
                for k in RoleKind
            },
            "runtimes": st,
            "cuda": self.cuda.info(),
            "entangled": True,
            "note": (
                "Languages are engines/transformers/orchestrators/teachers "
                "inside the MESIE student loop — not external chat tools."
            ),
        }

    def route_engine(self, capability: str) -> LangRole:
        """Pick engine for capability: prefer lowest EMA latency among healthy."""
        cands = [r for r in self.roles if r.kind == RoleKind.ENGINE and r.capability == capability]
        if not cands:
            cands = [r for r in self.roles if r.kind == RoleKind.ENGINE]
        # prefer julia for spectral when available, cuda for matmul
        if capability == "matmul":
            for r in cands:
                if r.lang == "cuda":
                    return r
        if capability == "spectral_energy" and self.rt.julia.available:
            for r in cands:
                if r.lang == "julia":
                    return r
        # latency
        cands = sorted(cands, key=lambda r: self.latency_ema.get(r.role_id, 1.0))
        return cands[0]

    def run_engine(self, capability: str, **payload: Any) -> Dict[str, Any]:
        role = self.route_engine(capability)
        t0 = time.time()
        out: Dict[str, Any]
        if capability == "spectral_energy":
            x = payload.get("x") or [1.0, 0, -1, 0, 1, 0, -1]
            if role.lang == "julia":
                out = self.rt.julia_call("spectral_energy", {"x": x})
            elif role.lang == "haskell":
                out = self.rt.haskell_call(
                    "spectral_energy", ",".join(str(float(v)) for v in x)
                )
            else:
                out = self.organ.spectral_energy_all(x).results["python"]
            out["routed_to"] = role.role_id
        elif capability == "matmul" or capability == "fast_ce":
            W = payload.get("W")
            X = payload.get("X")
            Y = payload.get("Y")
            if W is None:
                rng = np.random.default_rng(0)
                d = int(payload.get("dim", 64))
                b = int(payload.get("batch", 4))
                W = rng.standard_normal((d, d)) * 0.01
                X = rng.standard_normal((d, b))
                Y = rng.standard_normal((d, b))
            out = self.cuda.train_step_linear(
                np.asarray(W), np.asarray(X), np.asarray(Y), lr=float(payload.get("lr", 1e-3))
            )
            out["routed_to"] = role.role_id
        elif capability == "phi_powers":
            n = int(payload.get("n", 12))
            if role.lang == "julia" or self.rt.julia.available:
                out = self.rt.julia_call("phi_powers", {"n": n})
                out["routed_to"] = "eng.jl.phi"
            else:
                out = self.rt.haskell_call("phi_powers", str(n))
                out["routed_to"] = "eng.hs.phi"
        else:
            out = {"ok": False, "error": f"unknown capability {capability}"}
        dt = time.time() - t0
        rid = out.get("routed_to") or role.role_id
        prev = self.latency_ema.get(rid, dt)
        self.latency_ema[rid] = 0.8 * prev + 0.2 * dt
        out["latency_s"] = dt
        return out

    def transform_text(self, text: str) -> Dict[str, Any]:
        """All transformer langs produce embeddings; fuse into residual."""
        t0 = time.time()
        views: Dict[str, Any] = {}
        # Python multi-MESIE
        try:
            from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder

            v = MultiMesieEmbedder().embed_text(text)
            views["python_multi"] = v
        except Exception as exc:
            views["python_multi_error"] = str(exc)
        jl = self.rt.julia_call("multi_fft_embed", {"text": text[:500]})
        if jl.get("ok") and jl.get("embedding"):
            views["julia_fft"] = np.asarray(jl["embedding"], dtype=np.float64)
        hs = self.rt.haskell_call("multi_fft_embed", text[:200])
        if hs.get("ok") and hs.get("embedding"):
            views["haskell_fft"] = np.asarray(hs["embedding"], dtype=np.float64)

        # Fuse: pad to max dim, weighted mean
        mats = []
        weights = []
        for key, w in (("python_multi", 1.0), ("julia_fft", 0.7), ("haskell_fft", 0.5)):
            if key in views and isinstance(views[key], np.ndarray):
                mats.append(views[key].ravel())
                weights.append(w)
        if not mats:
            fused = np.zeros(32, dtype=np.float64)
        else:
            d = max(m.size for m in mats)
            acc = np.zeros(d, dtype=np.float64)
            tw = 0.0
            for m, w in zip(mats, weights):
                if m.size < d:
                    m = np.pad(m, (0, d - m.size))
                acc += w * m[:d]
                tw += w
            fused = acc / max(tw, 1e-12)
            n = float(np.linalg.norm(fused) + 1e-12)
            fused = fused / n
        return {
            "ok": True,
            "fused": fused,
            "views": {k: (int(v.size) if isinstance(v, np.ndarray) else v) for k, v in views.items()},
            "latency_s": time.time() - t0,
            "transformers": ["xf.py.multi", "xf.jl.fft", "xf.hs.fft"],
        }

    def teach(self, text: str, *, student_ce: Optional[float] = None) -> List[CouncilSignal]:
        """Teachers emit signals the student train_step can absorb."""
        signals: List[CouncilSignal] = []
        # Doctrine teacher (python)
        try:
            from auro_native_llm.embedded.python_organ import load_python_doctrine

            doc = load_python_doctrine()
            signals.append(
                CouncilSignal(
                    source="tch.py.doctrine",
                    kind="doctrine",
                    text=doc.get("principle", ""),
                    scalar=0.95,
                    weight=1.2,
                    meta={"doctrine_id": doc.get("doctrine_id")},
                )
            )
        except Exception:
            pass
        # Julia spectral teacher
        xs = [float((ord(c) % 17) - 8) for c in (text[:32] or "MESIE")]
        jl = self.rt.julia_call("spectral_energy", {"x": xs})
        if jl.get("ok"):
            e = float(jl["energy"])
            signals.append(
                CouncilSignal(
                    source="tch.jl.spectral",
                    kind="spectral_target",
                    scalar=e,
                    weight=0.8,
                    text=f"julia_spectral_energy={e:.6f}",
                    meta={"lang": "julia"},
                )
            )
        # Haskell structure teacher
        hs = self.rt.haskell_call("phi_powers", "8")
        if hs.get("ok"):
            signals.append(
                CouncilSignal(
                    source="tch.hs.structure",
                    kind="phi_structure",
                    scalar=float(hs.get("sum", 0)),
                    weight=0.6,
                    text=f"haskell_phi_sum={hs.get('sum')}",
                    meta={"lang": "haskell", "native": hs.get("native_ghc")},
                )
            )
        # Transform residual as vector teacher
        xf = self.transform_text(text[:800])
        if xf.get("ok"):
            signals.append(
                CouncilSignal(
                    source="orc.py.council",
                    kind="multi_lang_residual",
                    vector=xf["fused"],
                    weight=1.0,
                    text="fused polyglot transformer residual",
                    meta=xf.get("views"),
                )
            )
        # Student CE context
        if student_ce is not None:
            signals.append(
                CouncilSignal(
                    source="tch.cuda.fast",
                    kind="ce_context",
                    scalar=float(student_ce),
                    weight=0.5,
                    text=f"student_ce={student_ce:.4f} backend={self.cuda.backend}",
                    meta=self.cuda.info(),
                )
            )
        self.calls += 1
        return signals

    def council_train_step(
        self,
        language: Any,
        input_ids: np.ndarray,
        label_ids: np.ndarray,
        *,
        lr: float = 2e-3,
        text_for_meaning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One student train_step with polyglot council entangled.

        1. Orchestrator transforms text (Py/Jl/Hs)
        2. Teachers emit signals
        3. Student CE train_step (MESIE SpectralGPT)
        4. Fuse residual into last-token embedding path via meaning blend
        5. CUDA plane micro-step on small head probe (acceleration signal)
        """
        t0 = time.time()
        text = text_for_meaning or ""
        # pre-council
        signals = self.teach(text or "MESIE Auro polyglot council")
        residual = None
        for s in signals:
            if s.vector is not None:
                residual = s.vector
                break

        # inject residual into meaning path for this step
        if residual is not None and language.meaning is not None and text:
            # temporarily bias meaning embed
            try:
                mvec = language.meaning.embed(text)
                d = min(mvec.size, residual.size)
                blend = 0.12
                # store side channel for fuse
                language._polyglot_residual = residual[:d]  # type: ignore[attr-defined]
                language._polyglot_blend = blend  # type: ignore[attr-defined]
            except Exception:
                language._polyglot_residual = None  # type: ignore[attr-defined]

        metrics = language.train_step(
            input_ids, label_ids, lr=lr, text_for_meaning=text_for_meaning
        )

        # apply residual nudge on last embedding row used (light)
        if residual is not None:
            try:
                emb = language.core.embedding.token_embeddings
                d = min(emb.shape[1], residual.size)
                # small structured noise-free nudge on mean row direction
                emb[:, :d] += (lr * 0.05) * residual[:d]
                language.core.embedding.token_embeddings = emb
            except Exception:
                pass

        # CUDA plane teacher: independent linear probe train (keeps accel path hot)
        d = 32
        rng = np.random.default_rng(int(metrics.get("ce", 1) * 1000) % 2**31)
        probe = self.cuda.train_step_linear(
            rng.standard_normal((d, d)) * 0.01,
            rng.standard_normal((d, 4)),
            rng.standard_normal((d, 4)),
            lr=1e-3,
        )

        # engine spectral for receipt
        eng = self.run_engine("spectral_energy", x=[1.0, 0, -1, 0, 1])

        out = {
            "ok": True,
            "student": metrics,
            "council": [s.to_dict() for s in signals],
            "cuda_probe": {
                "backend": probe.get("backend"),
                "loss": probe.get("loss"),
                "sec": probe.get("sec"),
            },
            "engine_spectral": eng,
            "orchestrator": "orc.py.council",
            "entangled_langs": ["python", "julia", "haskell", "cuda"],
            "sec": time.time() - t0,
        }
        # cleanup
        if hasattr(language, "_polyglot_residual"):
            language._polyglot_residual = None  # type: ignore[attr-defined]
        return out


_ORCH: Optional[PolyglotOrchestrator] = None


def get_orchestrator() -> PolyglotOrchestrator:
    global _ORCH
    if _ORCH is None:
        _ORCH = PolyglotOrchestrator()
    return _ORCH
