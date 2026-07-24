"""Deep physics engines — real dynamics, not labels.

Engines
-------
1. Hamiltonian spectral oscillator lattice
2. Nonlinear Klein–Gordon-like field on φ-lattice
3. Renormalization-group scale flow for spectral couplings
4. Ising-Glauber spin field (phase transition / memory)
5. Burgers / viscous transport (1D fluid) for signal advection
6. Coupled multi-engine step → joint state vector
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = PHI - 1.0
GOLDEN = 2.0 * math.pi / (PHI * PHI)


@dataclass
class PhysicsEngineState:
    q: np.ndarray  # positions / field
    p: np.ndarray  # momenta
    spins: np.ndarray  # ±1 Ising
    fluid: np.ndarray  # Burgers velocity
    coupling: float  # RG running coupling
    energy: float = 0.0
    temperature: float = 1.0
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": int(self.q.size),
            "energy": self.energy,
            "temperature": self.temperature,
            "coupling": self.coupling,
            "step": self.step,
            "metrics": self.metrics,
            "q_norm": float(np.linalg.norm(self.q)),
            "p_norm": float(np.linalg.norm(self.p)),
            "magnetization": float(self.spins.mean()) if self.spins.size else 0.0,
            "fluid_energy": float(np.mean(self.fluid**2)),
        }


class DeepPhysicsEngine:
    """Coupled lattice Hamiltonian + field + RG + Ising + Burgers."""

    def __init__(self, dim: int = 64, seed: int = 7) -> None:
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.state = PhysicsEngineState(
            q=rng.standard_normal(dim) * 0.1,
            p=rng.standard_normal(dim) * 0.05,
            spins=rng.choice([-1.0, 1.0], size=dim),
            fluid=rng.standard_normal(dim) * 0.02,
            coupling=0.3,
            temperature=1.0,
        )

    # ---- 1) Hamiltonian lattice oscillator ----
    def hamiltonian_step(self, dt: float = 0.05, omega0: float = 1.0, k_spring: float = 0.4) -> float:
        """H = Σ p²/2 + (ω0²/2) q² + (k/2)(q_i - q_{i+1})² + φ-potential."""
        q, p = self.state.q, self.state.p
        n = q.size
        # forces from nearest-neighbor springs
        force = -omega0**2 * q
        force += k_spring * (np.roll(q, 1) - 2 * q + np.roll(q, -1))
        # φ soft well
        idx = np.arange(n) / max(n - 1, 1)
        force += -PHI_INV * q * np.cos(GOLDEN * idx * n)
        # leapfrog
        p_half = p + 0.5 * dt * force
        q_new = q + dt * p_half
        force2 = -omega0**2 * q_new
        force2 += k_spring * (np.roll(q_new, 1) - 2 * q_new + np.roll(q_new, -1))
        force2 += -PHI_INV * q_new * np.cos(GOLDEN * idx * n)
        p_new = p_half + 0.5 * dt * force2
        self.state.q, self.state.p = q_new, p_new
        # energy
        kinetic = 0.5 * float(np.sum(p_new**2))
        pot = 0.5 * omega0**2 * float(np.sum(q_new**2))
        pot += 0.25 * k_spring * float(np.sum((q_new - np.roll(q_new, -1)) ** 2))
        return kinetic + pot

    # ---- 2) Nonlinear Klein–Gordon-like field ----
    def klein_gordon_step(self, dt: float = 0.04, c: float = PHI, mass: float = 0.5, lam: float = 0.1) -> float:
        """□φ + m²φ + λφ³ = 0 on 1D lattice; use q as field, p as ∂tφ."""
        phi, pi = self.state.q, self.state.p
        lap = np.roll(phi, 1) - 2 * phi + np.roll(phi, -1)
        dV = mass**2 * phi + lam * (phi**3)
        pi_new = pi + dt * (c**2 * lap - dV)
        phi_new = phi + dt * pi_new
        self.state.q, self.state.p = phi_new, pi_new
        return float(0.5 * np.sum(pi_new**2) + 0.5 * mass**2 * np.sum(phi_new**2) + 0.25 * lam * np.sum(phi_new**4))

    # ---- 3) RG flow for coupling ----
    def rg_flow_step(self, d_log_mu: float = 0.05) -> float:
        """β(g) = -ε g + b g² - c g³  (Wilson-like toy β-function with φ coeffs)."""
        g = self.state.coupling
        eps = 0.1
        b = PHI_INV
        c = PHI_INV**2
        beta = -eps * g + b * g**2 - c * g**3
        g_new = float(np.clip(g + d_log_mu * beta, 1e-4, 2.0))
        self.state.coupling = g_new
        return g_new

    # ---- 4) Ising–Glauber ----
    def ising_glauber_step(self, J: float = 0.8) -> float:
        """Single sweep Metropolis/Glauber at temperature T; returns magnetization."""
        s = self.state.spins
        n = s.size
        T = max(self.state.temperature, 1e-3)
        # couple J to RG coupling
        J_eff = J * (0.5 + self.state.coupling)
        for i in range(n):
            # local field from neighbors + continuous field bias
            h = J_eff * (s[(i - 1) % n] + s[(i + 1) % n]) + 0.15 * self.state.q[i]
            # flip probability
            dE = 2.0 * s[i] * h
            if dE <= 0 or np.random.random() < math.exp(-dE / T):
                s[i] = -s[i]
        self.state.spins = s
        return float(s.mean())

    # ---- 5) Burgers viscous transport ----
    def burgers_step(self, dt: float = 0.02, nu: float = 0.05) -> float:
        """∂t u + u ∂x u = ν ∂xx u  (1D)."""
        u = self.state.fluid
        # upwind-ish advection
        du = 0.5 * (np.roll(u, -1) - np.roll(u, 1))
        lap = np.roll(u, 1) - 2 * u + np.roll(u, -1)
        u_new = u - dt * u * du + nu * dt * lap
        # inject spectral drive from q
        u_new += 0.02 * dt * self.state.q
        self.state.fluid = u_new
        return float(np.mean(u_new**2))

    # ---- drive from external signal ----
    def inject_signal(self, signal: np.ndarray, strength: float = 0.1) -> None:
        s = np.asarray(signal, dtype=np.float64).ravel()
        d = self.dim
        if s.size < d:
            pad = np.zeros(d)
            pad[: s.size] = s
            s = pad
        else:
            s = s[:d]
        n = float(np.linalg.norm(s)) or 1.0
        s = s / n
        self.state.q += strength * s
        self.state.fluid += 0.5 * strength * s

    def step(self, n: int = 1, signal: Optional[np.ndarray] = None) -> PhysicsEngineState:
        if signal is not None:
            self.inject_signal(signal)
        for _ in range(max(1, n)):
            E_h = self.hamiltonian_step()
            E_kg = self.klein_gordon_step()
            g = self.rg_flow_step()
            m = self.ising_glauber_step()
            e_f = self.burgers_step()
            # temperature anneal toward order when energy drops
            self.state.temperature = float(
                np.clip(0.92 * self.state.temperature + 0.08 * (0.5 + abs(E_h) * 0.01), 0.05, 3.0)
            )
            self.state.energy = 0.4 * E_h + 0.3 * E_kg + 0.3 * e_f
            self.state.step += 1
            self.state.metrics = {
                "E_hamiltonian": E_h,
                "E_klein_gordon": E_kg,
                "rg_coupling": g,
                "magnetization": m,
                "fluid_energy": e_f,
                "phi": PHI,
            }
        return self.state

    def feature_vector(self, out_dim: int = 64) -> np.ndarray:
        """Pack engine state into a fixed embedding for transformers."""
        st = self.state
        parts = [
            st.q,
            st.p,
            st.spins,
            st.fluid,
            np.array(
                [
                    st.energy,
                    st.temperature,
                    st.coupling,
                    st.metrics.get("magnetization", 0.0),
                    st.metrics.get("E_hamiltonian", 0.0),
                    st.metrics.get("E_klein_gordon", 0.0),
                    float(st.step),
                    PHI,
                ]
            ),
        ]
        v = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        if v.size < out_dim:
            v = np.pad(v, (0, out_dim - v.size))
        else:
            # structured pool
            idx = np.linspace(0, v.size, out_dim, endpoint=False).astype(int)
            v = v[idx]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float64)
