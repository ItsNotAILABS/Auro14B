"""Dual organism — stop letting the environment define the ceiling.

Architecture
------------
  JULIA  = BRAIN
    Virtual Physics Cores (Threads.@threads):
      dispersion ω(k), Kuramoto sync, Landau order, φ-Schrödinger residual,
      mean-field coupling across cores → distributed thought.

  PYTHON = AI
    Intent, tools, language, GHOST policy, Google envelope, training absorb.
    Never blocks on missing torch — AI routes work to the brain.

Novelty
-------
Distributed multi-core physics consensus is the cognition substrate.
Python does not simulate the brain in numpy first — it *asks Julia*.
"""

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
_BRAIN_JL = _REPO / "bindings" / "julia" / "AuroCompute" / "src" / "AuroBrain.jl"
_COMPUTE_JL = _REPO / "bindings" / "julia" / "AuroCompute" / "src" / "AuroCompute.jl"


@dataclass
class DualResult:
    ok: bool
    intent: str
    brain: Dict[str, Any]
    ai: Dict[str, Any]
    fused_embedding: List[float]
    latency_ms: float
    doctrine: str = "python=AI julia=BRAIN virtual_physics_cores=distributed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.dual.organism.v1",
            "ok": self.ok,
            "intent": self.intent,
            "brain": self.brain,
            "ai": self.ai,
            "fused_embedding_dim": len(self.fused_embedding),
            "fused_embedding_head": self.fused_embedding[:12],
            "latency_ms": self.latency_ms,
            "doctrine": self.doctrine,
            "environment_is_not_ceiling": True,
        }


class DualOrganism:
    """Python AI organ driving a Julia virtual-physics brain."""

    def __init__(
        self,
        mind: Any = None,
        *,
        n_cores: int = 0,
        dim: int = 64,
        julia_threads: Optional[int] = None,
    ) -> None:
        self.mind = mind
        self.n_cores = n_cores
        self.dim = dim
        self.julia_exe = shutil.which("julia") or ""
        self.julia_threads = julia_threads or max(4, (os_cpu_count() or 4))
        self.history: List[Dict[str, Any]] = []
        self.born_at = time.time()

    # ---- Julia BRAIN ----
    def brain_call(self, action: str, intent: str = "", steps: int = 3) -> Dict[str, Any]:
        if not self.julia_exe:
            return self._python_brain_fallback(action, intent, steps, reason="julia not on PATH")
        if not _BRAIN_JL.exists():
            return self._python_brain_fallback(action, intent, steps, reason="AuroBrain.jl missing")

        mod = str(_BRAIN_JL).replace("\\", "/")
        intent_esc = (
            intent.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", " ")
            .replace("\r", " ")
        )[:400]
        n = self.n_cores if self.n_cores > 0 else 0
        # Julia code: load brain, run cycle, print JSON-ish
        code = f'''
include("{mod}")
using .AuroBrain
br = get_brain({n}; dim={self.dim})
'''
        if action in ("health", "brain_health"):
            code += '''
h = brain_health(br)
print("{\\"ok\\":true,\\"role\\":\\"BRAIN\\",\\"lang\\":\\"julia\\",\\"version\\":\\"", VERSION,
  "\\",\\"threads\\":", Threads.nthreads(),
  ",\\"cpu_threads\\":", Sys.CPU_THREADS,
  ",\\"n_cores\\":", br.n_cores,
  ",\\"tick\\":", br.tick,
  ",\\"phi\\":", h["phi"],
  ",\\"distributed\\":", h["distributed"], "}")
'''
        else:
            code += f'''
c = brain_cycle!(br, "{intent_esc}"; steps={int(steps)})
t = c["thought"]
be = c["band_energy"]
print("{{\\"ok\\":true,\\"role\\":\\"BRAIN\\",\\"lang\\":\\"julia\\",")
print("\\"focus_band\\":", c["focus_band"], ",")
print("\\"n_cores\\":", t["n_cores"], ",")
print("\\"threads\\":", t["threads"], ",")
print("\\"tick\\":", t["tick"], ",")
print("\\"mean_kuramoto_r\\":", t["mean_kuramoto_r"], ",")
print("\\"mean_coherence\\":", t["mean_coherence"], ",")
print("\\"mean_resonance\\":", t["mean_resonance"], ",")
print("\\"mean_energy\\":", t["mean_energy"], ",")
print("\\"global_field_norm\\":", t["global_field_norm"], ",")
print("\\"advice_to_ai\\":\\"", c["advice_to_ai"], "\\",")
print("\\"band_energy\\":[")
print(join(string.(be), ","))
print("],")
print("\\"distributed\\":true,\\"novel\\":true,\\"physics_cores\\":true}}")
'''
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["JULIA_NUM_THREADS"] = str(self.julia_threads)
        t0 = time.time()
        try:
            r = subprocess.run(
                [self.julia_exe, f"--threads={self.julia_threads}", "-e", code],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            lines = (r.stdout or "").strip().splitlines()
            raw = lines[-1] if lines else ""
            if r.returncode != 0 or not raw:
                return self._python_brain_fallback(
                    action,
                    intent,
                    steps,
                    reason=(r.stderr or "julia empty")[:400],
                    sec=time.time() - t0,
                )
            out = json.loads(raw)
            out["sec"] = time.time() - t0
            out["julia_exe"] = self.julia_exe
            out["JULIA_NUM_THREADS"] = self.julia_threads
            return out
        except Exception as exc:
            return self._python_brain_fallback(
                action, intent, steps, reason=str(exc)[:400], sec=time.time() - t0
            )

    def _python_brain_fallback(
        self,
        action: str,
        intent: str,
        steps: int,
        *,
        reason: str,
        sec: float = 0.0,
    ) -> Dict[str, Any]:
        """If Julia process fails, still run multi-core physics in Python threads.

        Environment does NOT stop us — brain logic runs distributed in Python
        Threads as isomorphic virtual cores (same equations).
        """
        from concurrent.futures import ThreadPoolExecutor

        from auro_native_llm.physics.formulas import (
            PHI,
            dispersion_omega,
            kuramoto_step,
            kuramoto_order,
            landau_field_force,
            landau_free_energy,
            phi_schrodinger_step,
            resonance_score,
            spectral_action_density,
            text_to_physical_signal,
            wiener_coherence,
            spectrum_from_signal,
        )

        n = self.n_cores if self.n_cores > 0 else max(4, os_cpu_count() or 4)
        dim = self.dim
        drive0 = text_to_physical_signal(intent or "think", dim)

        def core_run(i: int) -> Dict[str, Any]:
            drive = np.roll(drive0, int(i * dim / n))
            drive = drive + 0.05 * np.sin(np.arange(dim) * (i + 1) * 0.17)
            psi = drive.copy()
            k = np.linspace(0.0, float(PHI), dim)
            omega = dispersion_omega(k)
            phase = np.random.default_rng(i + 1).random(dim) * 2 * np.pi
            m = np.random.default_rng(i + 7).standard_normal(dim)
            m = m / (np.linalg.norm(m) + 1e-12)
            h = drive / (np.linalg.norm(drive) + 1e-12)
            r = 0.0
            for _ in range(max(1, steps)):
                psi = psi + 0.15 * h
                psi = phi_schrodinger_step(psi, dt=0.05)
                phase = kuramoto_step(phase, omega, K=float(PHI), dt=0.05)
                r, _ = kuramoto_order(phase)
                force = landau_field_force(m, h)
                m = m + 0.12 * force
                m = m / (np.linalg.norm(m) + 1e-12)
            S, _ = spectral_action_density(np.abs(psi))
            coh, _ = wiener_coherence(psi, h)
            R = resonance_score(np.abs(psi), np.abs(h))
            F = landau_free_energy(m, h)
            return {
                "id": i + 1,
                "kuramoto_r": r,
                "coherence": coh,
                "resonance": R,
                "energy": float(S / (dim + 1)),
                "landau_F": F,
                "psi": psi,
            }

        with ThreadPoolExecutor(max_workers=n) as pool:
            cores = list(pool.map(core_run, range(n)))
        g = np.mean([c["psi"] for c in cores], axis=0)
        mean_r = float(np.mean([c["kuramoto_r"] for c in cores]))
        mean_coh = float(np.mean([c["coherence"] for c in cores]))
        mean_R = float(np.mean([c["resonance"] for c in cores]))
        mean_E = float(np.mean([c["energy"] for c in cores]))
        bands = 8
        band_e = []
        for b in range(bands):
            lo = int(b * dim / bands)
            hi = int((b + 1) * dim / bands)
            band_e.append(float(np.sum(g[lo:hi] ** 2)))
        focus = int(np.argmax(band_e)) + 1
        return {
            "ok": True,
            "role": "BRAIN",
            "lang": "python_isomorphic_cores",
            "fallback": True,
            "fallback_reason": reason,
            "n_cores": n,
            "threads": n,
            "tick": 1,
            "mean_kuramoto_r": mean_r,
            "mean_coherence": mean_coh,
            "mean_resonance": mean_R,
            "mean_energy": mean_E,
            "global_field_norm": float(np.linalg.norm(g)),
            "focus_band": focus,
            "band_energy": band_e,
            "distributed": True,
            "novel": True,
            "physics_cores": True,
            "advice_to_ai": f"Python AI: act on focus_band={focus}; resonance={mean_R:.3f}; lock_r={mean_r:.3f}",
            "sec": sec,
            "note": "Environment did not stop us — isomorphic physics cores ran in Python threads.",
        }

    # ---- Python AI ----
    def ai_act(self, intent: str, brain_out: Dict[str, Any]) -> Dict[str, Any]:
        """AI layer: interpret brain advice, optional mind generate, absorb train."""
        advice = str(brain_out.get("advice_to_ai") or "")
        focus = brain_out.get("focus_band")
        r = float(brain_out.get("mean_kuramoto_r") or 0)
        R = float(brain_out.get("mean_resonance") or 0)
        # policy: if high lock, AI commits; if low, explore
        mode = "commit" if r > 0.25 and R > 0.35 else "explore"
        plan = {
            "mode": mode,
            "focus_band": focus,
            "actions": [],
        }
        if mode == "commit":
            plan["actions"] = [
                "spectral_match_on_focus_band",
                "write_collab_note",
                "physics_train_pulse",
            ]
        else:
            plan["actions"] = [
                "spawn_more_physics_steps",
                "broaden_retrieval",
                "re_think_distributed",
            ]

        gen_text = ""
        if self.mind is not None and hasattr(self.mind, "language"):
            try:
                prompt = (
                    f"[PYTHON AI | JULIA BRAIN lock_r={r:.3f} R={R:.3f} focus={focus}]\n"
                    f"Intent: {intent}\nBrain: {advice}\n"
                    f"Produce a concrete next action for mode={mode}."
                )
                # light path
                lang = self.mind.language
                prev = getattr(self.mind, "train_every_act", False)
                if hasattr(self.mind, "train_every_act"):
                    self.mind.train_every_act = False
                try:
                    if hasattr(lang, "generate"):
                        g = lang.generate(prompt, max_new_tokens=32)
                        gen_text = str(getattr(g, "text", g))[:800]
                finally:
                    if hasattr(self.mind, "train_every_act"):
                        self.mind.train_every_act = prev
            except Exception as exc:
                gen_text = f"(ai light-path: {exc})"

        # absorb into mind if present
        absorbed = False
        if self.mind is not None and getattr(self.mind, "organs", None):
            try:
                from auro_native_llm.organism.self_train import Experience

                if self.mind.organs.trainer:
                    self.mind.organs.trainer.absorb(
                        Experience(
                            text=f"DUAL brain={brain_out.get('lang')} r={r:.3f} R={R:.3f} {intent}",
                            kind="dual_brain",
                            model_id=getattr(self.mind, "model_id", "Auro"),
                            reward=0.5 + 0.5 * min(1.0, R),
                            meta={"focus": focus, "mode": mode, "brain": brain_out.get("lang")},
                        )
                    )
                    absorbed = True
            except Exception:
                pass

        return {
            "role": "AI",
            "lang": "python",
            "mode": mode,
            "plan": plan,
            "advice_heard": advice,
            "generation": gen_text,
            "absorbed": absorbed,
            "physics_guided": True,
        }

    def fuse_embedding(self, intent: str, brain_out: Dict[str, Any]) -> List[float]:
        """Fuse brain band energies + physics embed + optional multi-embed."""
        from auro_native_llm.physics import get_physics_engine

        eng = get_physics_engine()
        phys = eng.embed_physics(intent, self.dim)
        bands = np.asarray(brain_out.get("band_energy") or [0.0] * 8, dtype=np.float64)
        if bands.size < self.dim:
            pad = np.zeros(self.dim)
            pad[: bands.size] = bands
            bands = pad
        else:
            bands = bands[: self.dim]
        bn = float(np.linalg.norm(bands)) or 1.0
        bands = bands / bn
        fused = 0.55 * phys + 0.45 * bands
        # scale by brain resonance
        R = float(brain_out.get("mean_resonance") or 0.5)
        fused = fused * (0.7 + 0.6 * R)
        n = float(np.linalg.norm(fused)) or 1.0
        return (fused / n).tolist()

    def think(self, intent: str, *, steps: int = 3) -> DualResult:
        t0 = time.perf_counter()
        brain = self.brain_call("brain_cycle", intent=intent, steps=steps)
        ai = self.ai_act(intent, brain)
        emb = self.fuse_embedding(intent, brain)
        res = DualResult(
            ok=bool(brain.get("ok", True)),
            intent=intent,
            brain=brain,
            ai=ai,
            fused_embedding=emb,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        self.history.append(res.to_dict())
        return res

    def health(self) -> Dict[str, Any]:
        b = self.brain_call("health")
        return {
            "schema": "auro.dual.health.v1",
            "python_ai": True,
            "julia_exe": self.julia_exe or None,
            "julia_threads": self.julia_threads,
            "brain": b,
            "doctrine": "python=AI julia=BRAIN",
            "environment_is_not_ceiling": True,
        }


def os_cpu_count() -> Optional[int]:
    import os

    return os.cpu_count()


def run_dual_think(
    intent: str,
    mind: Any = None,
    *,
    steps: int = 3,
    n_cores: int = 0,
) -> Dict[str, Any]:
    org = DualOrganism(mind, n_cores=n_cores)
    return org.think(intent, steps=steps).to_dict()
