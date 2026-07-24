"""Physics AI engine — applies real formulas inside LM train/forward."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from auro_native_llm.physics.formulas import (
    EQUATIONS,
    PhysicsReport,
    PhysicsState,
    fisher_natural_grad,
    physics_lr_schedule,
    physics_regularized_loss,
    phi_schrodinger_step,
    resonance_score,
    spectral_action_density,
    spectral_force_on_hidden,
    spectrum_from_signal,
    text_to_physical_signal,
    wiener_coherence,
    kuramoto_order,
    kuramoto_step,
    dispersion_omega,
    PHI,
    landau_field_force,
)

_ENGINE: Optional["PhysicsAIEngine"] = None


class PhysicsAIEngine:
    """Stateful physics layer attached to Auro LM training."""

    def __init__(self) -> None:
        self.last = PhysicsState()
        self.steps = 0
        self.history: list[Dict[str, float]] = []

    def signal_and_spectrum(self, text: str, length: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        sig = text_to_physical_signal(text or "φ", length=length)
        spec = spectrum_from_signal(sig)
        return sig, spec

    def fuse_hidden(
        self,
        hidden: np.ndarray,
        text: str,
        *,
        strength: float = 0.1,
    ) -> np.ndarray:
        """Physics residual fuse into [B,T,D] hidden."""
        h = np.asarray(hidden, dtype=np.float64)
        if h.ndim != 3:
            return h
        B, T, D = h.shape
        sig, spec = self.signal_and_spectrum(text, length=max(D, 64))
        out = h.copy()
        for b in range(B):
            tail = out[b, -1, :]
            forced = spectral_force_on_hidden(tail, spec, sig, strength=strength)
            # also mild Schrödinger on full sequence mean residual
            mean_tok = out[b].mean(axis=0)
            psi = phi_schrodinger_step(mean_tok, dt=0.03)
            out[b, -1, :] = forced
            out[b] += 0.02 * (psi - mean_tok)
        return out

    def loss_metrics(
        self,
        ce: float,
        text: str,
        hidden: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        h = np.asarray(hidden, dtype=np.float64)
        if h.ndim == 3:
            tail = h[:, -1, :].mean(axis=0)
        else:
            tail = h.ravel()
        sig, spec = self.signal_and_spectrum(text, length=max(tail.size, 64))
        L, m = physics_regularized_loss(ce, spec, sig, tail)
        self.last = PhysicsState(
            coherence=m["coherence"],
            kuramoto_r=m["kuramoto_r"],
            resonance=m["resonance"],
            spectral_action=m["spectral_action"],
            landau_F=m["landau_F"],
            physics_loss=m["physics_loss"],
            ce=m["ce"],
        )
        self.history.append(self.last.to_dict())
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return L, m

    def natural_grad_update(
        self,
        grad: np.ndarray,
        field: np.ndarray,
    ) -> np.ndarray:
        """Diagonal Fisher metric from field energy."""
        G = field * field + 1e-4
        return fisher_natural_grad(grad, G)

    def scheduled_lr(self, base_lr: float, step: int) -> float:
        return physics_lr_schedule(
            step,
            base_lr,
            coherence=self.last.coherence,
            kuramoto_r=self.last.kuramoto_r,
            resonance=self.last.resonance,
        )

    def embed_physics(self, text: str, dim: int) -> np.ndarray:
        """Physics embedding: spectrum + Schrödinger + Landau fixed point."""
        sig, spec = self.signal_and_spectrum(text, length=max(dim * 2, 64))
        # pad/trim spectrum to dim
        if spec.size < dim:
            ext = np.zeros(dim)
            ext[: spec.size] = spec
        else:
            idx = np.linspace(0, spec.size, dim, endpoint=False).astype(int)
            ext = spec[idx]
        ext = ext / (float(np.linalg.norm(ext)) + 1e-12)
        m = ext.copy()
        # iterate Landau dynamics toward fixed point with h=ext
        for _ in range(6):
            f = landau_field_force(m, ext, a=-0.5, b=1.0)
            m = m + 0.15 * f
            n = float(np.linalg.norm(m)) or 1.0
            m = m / n
        # Schrödinger smooth
        m = phi_schrodinger_step(m, dt=0.05)
        n = float(np.linalg.norm(m)) or 1.0
        return m / n

    def report(self) -> PhysicsReport:
        return PhysicsReport(
            ok=True,
            formulas=list(EQUATIONS.keys()),
            last_state=self.last,
            equations=dict(EQUATIONS),
        )


def get_physics_engine() -> PhysicsAIEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PhysicsAIEngine()
    return _ENGINE
