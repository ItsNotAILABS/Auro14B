"""Polyglot organ — Python + Julia + Haskell + CUDA in the mind.

Increases computation by farming spectral / train microkernels across languages
and the CUDA plane (torch/cupy when present).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.polyglot.cuda_plane import CudaPlane, get_cuda_plane
from auro_native_llm.polyglot.runtimes import PolyglotRuntime


@dataclass
class PolyglotResult:
    ok: bool
    action: str
    results: Dict[str, Any] = field(default_factory=dict)
    training_text: str = ""
    elapsed_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.polyglot.result.v1",
            "ok": self.ok,
            "action": self.action,
            "results": self.results,
            "training_text": self.training_text[:4000],
            "elapsed_s": self.elapsed_s,
            "meta": self.meta,
        }


class PolyglotOrgan:
    """Mind organ: multi-language compute + CUDA plane."""

    def __init__(self) -> None:
        self.rt = PolyglotRuntime()
        self.cuda = get_cuda_plane()
        self.history: List[Dict[str, Any]] = []
        self.total_runs = 0
        self.total_ok = 0

    def info(self) -> Dict[str, Any]:
        st = self.rt.status()
        return {
            "organ": "polyglot",
            "runtimes": st,
            "cuda": self.cuda.info(),
            "total_runs": self.total_runs,
            "total_ok": self.total_ok,
            "ok_rate": (self.total_ok / self.total_runs) if self.total_runs else None,
            "langs": {
                "python": True,
                "julia": st["julia"]["available"],
                # native GHC if on PATH; semantics port always available
                "haskell_native": st["haskell"]["available"],
                "haskell_semantics": True,
                "haskell": True,
            },
        }

    def health_all(self) -> PolyglotResult:
        t0 = time.time()
        py = {
            "ok": True,
            "lang": "python",
            "numpy": np.__version__,
            "cuda_plane": self.cuda.info(),
        }
        jl = self.rt.julia_call("health")
        hs = self.rt.haskell_call("health")
        # CUDA microbench
        a = np.random.randn(256, 256).astype(np.float32)
        b = np.random.randn(256, 256).astype(np.float32)
        ts = time.time()
        _ = self.cuda.matmul(a, b)
        cuda_sec = time.time() - ts
        results = {
            "python": py,
            "julia": jl,
            "haskell": hs,
            "cuda_matmul_256_sec": cuda_sec,
            "cuda": self.cuda.info(),
        }
        ok = bool(py.get("ok")) and bool(jl.get("ok")) and bool(hs.get("ok"))
        train = (
            f"[POLYGLOT_HEALTH ok={ok}]\n"
            f"python={py}\njulia={jl}\nhaskell={hs}\n"
            f"cuda={self.cuda.info()}\n[/POLYGLOT_HEALTH]"
        )
        return self._finish("health_all", ok, results, train, t0)

    def spectral_energy_all(self, x: Optional[List[float]] = None) -> PolyglotResult:
        t0 = time.time()
        x = x or [1.0, 0, -1, 0, 1, 0, -1]
        # Python
        n = len(x)
        e_py = 0.0
        for k in range(n // 2 + 1):
            re = im = 0.0
            for t in range(n):
                ang = 2 * np.pi * k * t / n
                re += x[t] * np.cos(ang)
                im -= x[t] * np.sin(ang)
            e_py += float(np.hypot(re, im))
        # CUDA/numpy rfft batch
        e_cuda = float(self.cuda.spectral_batch_energy(np.array([x], dtype=np.float32))[0])
        jl = self.rt.julia_call("spectral_energy", {"x": x})
        hs = self.rt.haskell_call(
            "spectral_energy", ",".join(str(float(v)) for v in x)
        )
        results = {
            "python": {"ok": True, "energy": e_py, "lang": "python"},
            "julia": jl,
            "haskell": hs,
            "cuda_plane": {"ok": True, "energy": e_cuda, "backend": self.cuda.backend},
        }
        # parity: energies should be close
        vals = [e_py, e_cuda]
        if jl.get("ok"):
            vals.append(float(jl["energy"]))
        if hs.get("ok"):
            vals.append(float(hs["energy"]))
        spread = max(vals) - min(vals)
        ok = all(r.get("ok") for r in (results["python"], jl, hs)) and spread < 1e-3
        train = (
            f"[POLYGLOT_SPECTRAL ok={ok} spread={spread}]\n"
            f"py={e_py} jl={jl.get('energy')} hs={hs.get('energy')} cuda={e_cuda}\n"
            f"[/POLYGLOT_SPECTRAL]"
        )
        return self._finish(
            "spectral_energy_all",
            ok,
            {**results, "parity_spread": spread},
            train,
            t0,
        )

    def phi_powers_all(self, n: int = 12) -> PolyglotResult:
        t0 = time.time()
        PHI = (1 + 5 ** 0.5) / 2
        p_py = [PHI ** i for i in range(1, n + 1)]
        jl = self.rt.julia_call("phi_powers", {"n": n})
        hs = self.rt.haskell_call("phi_powers", str(n))
        results = {
            "python": {"ok": True, "sum": sum(p_py), "powers": p_py, "lang": "python"},
            "julia": jl,
            "haskell": hs,
        }
        sums = [sum(p_py)]
        if jl.get("ok"):
            sums.append(float(jl["sum"]))
        if hs.get("ok"):
            sums.append(float(hs["sum"]))
        ok = max(sums) - min(sums) < 1e-6
        train = f"[POLYGLOT_PHI ok={ok} sums={sums}][/POLYGLOT_PHI]"
        return self._finish("phi_powers_all", ok, results, train, t0)

    def multi_embed_all(self, text: str = "MESIE SpectralGPT Auro") -> PolyglotResult:
        t0 = time.time()
        jl = self.rt.julia_call("multi_fft_embed", {"text": text})
        hs = self.rt.haskell_call("multi_fft_embed", text)
        # python multi via mesie power if present
        py_dim = 0
        try:
            from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder

            v = MultiMesieEmbedder().embed_text(text)
            py = {"ok": True, "dim": int(v.size), "lang": "python", "norm": float(np.linalg.norm(v))}
            py_dim = int(v.size)
        except Exception as exc:
            py = {"ok": False, "error": str(exc), "lang": "python"}
        results = {"python": py, "julia": jl, "haskell": hs}
        ok = bool(py.get("ok")) and bool(jl.get("ok")) and bool(hs.get("ok"))
        train = (
            f"[POLYGLOT_EMBED ok={ok} text={text[:80]} "
            f"py_dim={py_dim} jl_dim={jl.get('dim')} hs_dim={hs.get('dim')}]"
            f"[/POLYGLOT_EMBED]"
        )
        return self._finish("multi_embed_all", ok, results, train, t0)

    def accelerated_train_step(self, dim: int = 64, batch: int = 8) -> PolyglotResult:
        t0 = time.time()
        rng = np.random.default_rng(0)
        W = rng.standard_normal((dim, dim)).astype(np.float64) * 0.01
        X = rng.standard_normal((dim, batch)).astype(np.float64)
        Y = rng.standard_normal((dim, batch)).astype(np.float64)
        cuda_r = self.cuda.train_step_linear(W, X, Y, lr=1e-3)
        # Julia health still proves runtime during train
        jl = self.rt.julia_call("health")
        hs = self.rt.haskell_call("dot_train")
        ok = bool(cuda_r.get("ok")) and bool(jl.get("ok")) and bool(hs.get("ok"))
        results = {
            "cuda_plane": cuda_r,
            "julia": jl,
            "haskell": hs,
        }
        train = (
            f"[POLYGLOT_TRAIN ok={ok} backend={self.cuda.backend} "
            f"loss={cuda_r.get('loss')} jl_threads={jl.get('threads')} "
            f"hs_loss={hs.get('loss')}][/POLYGLOT_TRAIN]"
        )
        return self._finish("accelerated_train_step", ok, results, train, t0)

    def suite(self) -> PolyglotResult:
        """Full 100% polyglot suite."""
        t0 = time.time()
        parts = [
            self.health_all(),
            self.spectral_energy_all(),
            self.phi_powers_all(8),
            self.multi_embed_all(),
            self.accelerated_train_step(32, 4),
        ]
        ok = all(p.ok for p in parts)
        results = {p.action: p.to_dict() for p in parts}
        train = "\n".join(p.training_text for p in parts)
        return self._finish("suite", ok, results, train, t0)

    def _finish(
        self,
        action: str,
        ok: bool,
        results: Dict[str, Any],
        train: str,
        t0: float,
    ) -> PolyglotResult:
        self.total_runs += 1
        if ok:
            self.total_ok += 1
        r = PolyglotResult(
            ok=ok,
            action=action,
            results=results,
            training_text=train,
            elapsed_s=time.time() - t0,
            meta={"cuda_backend": self.cuda.backend},
        )
        self.history.append(r.to_dict())
        if len(self.history) > 50:
            self.history = self.history[-50:]
        return r
