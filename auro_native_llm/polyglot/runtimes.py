"""Detect and invoke Python / Julia / Haskell compute runtimes."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_JULIA_CLI = _REPO / "bindings" / "julia" / "AuroCompute" / "cli.jl"
_JULIA_MOD = _REPO / "bindings" / "julia" / "AuroCompute" / "src" / "AuroCompute.jl"
_HS_MAIN = _REPO / "bindings" / "haskell" / "AuroCompute.hs"


@dataclass
class RuntimeReport:
    lang: str
    available: bool
    path: str = ""
    version: str = ""
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "available": self.available,
            "path": self.path,
            "version": self.version,
            "error": self.error,
            "meta": self.meta,
        }


class PolyglotRuntime:
    def __init__(self) -> None:
        self.python = RuntimeReport(lang="python", available=True, path="builtin", version="3")
        self.julia = self._detect_julia()
        self.haskell = self._detect_haskell()

    def status(self) -> Dict[str, Any]:
        return {
            "schema": "auro.polyglot.runtimes.v1",
            "python": self.python.to_dict(),
            "julia": self.julia.to_dict(),
            "haskell": self.haskell.to_dict(),
            "all_available": self.python.available
            and self.julia.available
            and self.haskell.available,
        }

    def _detect_julia(self) -> RuntimeReport:
        exe = shutil.which("julia")
        if not exe:
            return RuntimeReport(lang="julia", available=False, error="julia not on PATH")
        try:
            r = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, timeout=15
            )
            ver = (r.stdout or r.stderr or "").strip()
            ok = _JULIA_MOD.exists() or _JULIA_CLI.exists()
            return RuntimeReport(
                lang="julia",
                available=ok and r.returncode == 0,
                path=exe,
                version=ver,
                meta={"cli": str(_JULIA_CLI), "module": str(_JULIA_MOD)},
            )
        except Exception as exc:
            return RuntimeReport(lang="julia", available=False, error=str(exc), path=exe or "")

    def _detect_haskell(self) -> RuntimeReport:
        for name in ("runghc", "ghc", "runhaskell"):
            exe = shutil.which(name)
            if exe:
                try:
                    r = subprocess.run(
                        [exe, "--version"] if name == "ghc" else [exe, "--help"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    ver = (r.stdout or r.stderr or "").splitlines()
                    ver_s = ver[0] if ver else name
                    return RuntimeReport(
                        lang="haskell",
                        available=_HS_MAIN.exists(),
                        path=exe,
                        version=ver_s[:120],
                        meta={"main": str(_HS_MAIN), "runner": name},
                    )
                except Exception as exc:
                    return RuntimeReport(
                        lang="haskell", available=False, error=str(exc), path=exe
                    )
        return RuntimeReport(
            lang="haskell",
            available=False,
            error="ghc/runghc not on PATH — install GHC (ghcup) for native Haskell",
            meta={"main": str(_HS_MAIN), "install": "https://www.haskell.org/ghcup/"},
        )

    # ---- Julia ----
    def julia_call(self, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.julia.available:
            return {"ok": False, "lang": "julia", "error": self.julia.error or "unavailable"}
        payload = dict(payload or {})
        payload["action"] = action
        # Prefer -e include without JSON.jl dependency for reliability
        if action in ("health", "spectral_energy", "phi_powers", "multi_fft_embed"):
            return self._julia_eval(action, payload)
        return self._julia_eval(action, payload)

    def _julia_eval(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run Julia without external JSON package (print simple JSON)."""
        exe = self.julia.path
        mod = str(_JULIA_MOD).replace("\\", "/")
        try:
            if action == "health":
                code = f'''
include("{mod}")
using .AuroCompute
h = health()
println("{{\\"ok\\":true,\\"lang\\":\\"julia\\",\\"version\\":\\"", VERSION, "\\",\\"threads\\":", Threads.nthreads(), ",\\"cpu_threads\\":", Sys.CPU_THREADS, ",\\"phi\\":", h["phi"], "}}")
'''
            elif action == "spectral_energy":
                xs = payload.get("x") or [1.0, 0, -1, 0, 1, 0, -1]
                xs_lit = "[" + ",".join(str(float(x)) for x in xs) + "]"
                code = f'''
include("{mod}")
using .AuroCompute
e = spectral_energy({xs_lit})
println("{{\\"ok\\":true,\\"lang\\":\\"julia\\",\\"energy\\":", e, "}}")
'''
            elif action == "phi_powers":
                n = int(payload.get("n") or 12)
                code = f'''
include("{mod}")
using .AuroCompute
p = phi_powers({n})
print("{{\\"ok\\":true,\\"lang\\":\\"julia\\",\\"sum\\":", sum(p), ",\\"powers\\":[")
print(join(string.(p), ","))
println("]}}")
'''
            elif action == "multi_fft_embed":
                text = str(payload.get("text") or "MESIE").replace("\\", "\\\\").replace('"', '\\"')
                code = f'''
include("{mod}")
using .AuroCompute
v = multi_fft_embed(Vector{{UInt8}}(codeunits("{text}")))
print("{{\\"ok\\":true,\\"lang\\":\\"julia\\",\\"dim\\":", length(v), ",\\"embedding\\":[")
print(join(string.(v), ","))
println("]}}")
'''
            elif action == "matmul_train_step":
                # use numpy-side for complex payloads; julia still available for simple ops
                return {"ok": False, "lang": "julia", "error": "use polyglot organ matmul via cuda/numpy for matrices"}
            else:
                return {"ok": False, "lang": "julia", "error": f"unknown {action}"}
            t0 = time.time()
            r = subprocess.run(
                [exe, f"--project={_JULIA_CLI.parent}", "-e", code],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode != 0:
                # retry without project (no JSON dep)
                r = subprocess.run(
                    [exe, "-e", code],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            line = (r.stdout or "").strip().splitlines()
            raw = line[-1] if line else ""
            if not raw:
                return {
                    "ok": False,
                    "lang": "julia",
                    "error": (r.stderr or "no output")[:500],
                    "sec": time.time() - t0,
                }
            out = json.loads(raw)
            out["sec"] = time.time() - t0
            return out
        except Exception as exc:
            return {"ok": False, "lang": "julia", "error": str(exc)[:500]}

    # ---- Haskell ----
    def haskell_call(self, action: str, *args: str) -> Dict[str, Any]:
        if not self.haskell.available:
            # pure semantics fallback implementing same API in-process
            # (only if GHC missing) — still labels as haskell_semantics
            return self._haskell_semantics(action, *args)
        runner = self.haskell.path
        cmd = [runner, str(_HS_MAIN), action, *args]
        # runghc takes the .hs file as script
        name = Path(runner).name.lower()
        if "ghc" == name and "run" not in name:
            # ghc needs compile — use interpreter style
            return {"ok": False, "lang": "haskell", "error": "use runghc"}
        try:
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            raw = (r.stdout or "").strip().splitlines()
            if r.returncode != 0 or not raw:
                # fallback semantics if runner fails
                fb = self._haskell_semantics(action, *args)
                fb["fallback"] = True
                fb["runner_error"] = (r.stderr or "")[:300]
                return fb
            out = json.loads(raw[-1])
            out["sec"] = time.time() - t0
            out["native_ghc"] = True
            return out
        except Exception as exc:
            fb = self._haskell_semantics(action, *args)
            fb["fallback"] = True
            fb["error_native"] = str(exc)[:200]
            return fb

    def _haskell_semantics(self, action: str, *args: str) -> Dict[str, Any]:
        """Faithful port of AuroCompute.hs algorithms when GHC is absent.

        Marked haskell_semantics so status stays honest until GHC is installed.
        """
        import math

        PHI = (1 + 5 ** 0.5) / 2

        def spectral_energy(xs: List[float]) -> float:
            n = len(xs)
            if n == 0:
                return 0.0
            e = 0.0
            for k in range(n // 2 + 1):
                re = im = 0.0
                for t in range(n):
                    ang = 2 * math.pi * k * t / n
                    re += xs[t] * math.cos(ang)
                    im -= xs[t] * math.sin(ang)
                e += (re * re + im * im) ** 0.5
            return e

        if action == "health":
            return {
                "ok": True,
                "lang": "haskell_semantics",
                "module": "AuroCompute",
                "phi": PHI,
                "native_ghc": False,
                "note": "Install GHC/runghc for native Haskell binary path",
                "actions": [
                    "health",
                    "spectral_energy",
                    "phi_powers",
                    "multi_fft_embed",
                    "dot_train",
                ],
            }
        if action == "spectral_energy":
            if args:
                xs = [float(x) for x in args[0].split(",") if x.strip()]
            else:
                xs = [1.0, 0, -1, 0, 1, 0, -1]
            return {
                "ok": True,
                "lang": "haskell_semantics",
                "energy": spectral_energy(xs),
                "native_ghc": False,
            }
        if action == "phi_powers":
            n = int(args[0]) if args else 12
            p = [PHI ** i for i in range(1, n + 1)]
            return {
                "ok": True,
                "lang": "haskell_semantics",
                "powers": p,
                "sum": sum(p),
                "native_ghc": False,
            }
        if action == "multi_fft_embed":
            text = args[0] if args else "MESIE spectral"
            raw = [float(ord(c)) for c in text] or [0.0] * 16
            if len(raw) < 16:
                raw = raw + [0.0] * (16 - len(raw))
            flat = []
            for i in range(4):
                win = max(16, int(round(len(raw) / (PHI ** i))))
                seg = (raw + [0.0] * win)[:win]
                e = spectral_energy(seg)
                mean = sum(seg) / len(seg)
                var = sum((x - mean) ** 2 for x in seg) / len(seg)
                peak = max(abs(x) for x in seg)
                flat.extend([e, mean, var ** 0.5, peak])
            nrm = sum(x * x for x in flat) ** 0.5 + 1e-12
            v = [x / nrm for x in flat]
            return {
                "ok": True,
                "lang": "haskell_semantics",
                "dim": len(v),
                "embedding": v,
                "native_ghc": False,
            }
        if action == "dot_train":
            w = [0.1, 0.2, 0.3, 0.4]
            x = [1, 0.5, 0.25, 0.125]
            pred = sum(a * b for a, b in zip(w, x))
            err = pred - 1.0
            loss = err * err
            w2 = [wi - 0.01 * 2 * err * xi for wi, xi in zip(w, x)]
            return {
                "ok": True,
                "lang": "haskell_semantics",
                "loss": loss,
                "w": w2,
                "native_ghc": False,
            }
        return {"ok": False, "lang": "haskell_semantics", "error": f"unknown {action}"}
