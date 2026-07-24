"""Algorithms that bind physics + economy + spectral transformers.

1. Spectral Hungarian-lite matching (greedy optimal assignment on cost matrix)
2. Sinkhorn optimal transport between physics & economic measures
3. Message-passing loopy belief on engine graph
4. Online gradient on joint latent (multi-view)
5. Routing: choose MESIE vs escalate from joint scores
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def greedy_match(cost: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """Greedy bipartite matching (assignment) minimizing cost."""
    C = np.asarray(cost, dtype=np.float64).copy()
    n, m = C.shape
    used_r = set()
    used_c = set()
    pairs: List[Tuple[int, int]] = []
    total = 0.0
    flat = [(C[i, j], i, j) for i in range(n) for j in range(m)]
    flat.sort(key=lambda t: t[0])
    for c, i, j in flat:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
        total += float(c)
        if len(pairs) >= min(n, m):
            break
    return pairs, total


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float = 0.1,
    n_iter: int = 40,
) -> Tuple[np.ndarray, float]:
    """Entropic OT: min <P,C> + eps KL(P||ab^T). Returns (P, transport_cost)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    C = np.asarray(C, dtype=np.float64)
    K = np.exp(-C / max(eps, 1e-6))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)
    cost = float(np.sum(P * C))
    return P, cost


def belief_propagate_ring(
    potentials: np.ndarray,
    compatibility: float = 0.5,
    n_iter: int = 8,
) -> np.ndarray:
    """Simple loopy BP on a ring of nodes with 2-states (up/down from Ising)."""
    # potentials: [n, 2] log-potentials
    phi = np.asarray(potentials, dtype=np.float64)
    n = phi.shape[0]
    # messages m_{i→i+1} shape [n, 2]
    msg = np.zeros((n, 2))
    for _ in range(n_iter):
        new = np.zeros_like(msg)
        for i in range(n):
            j = (i + 1) % n
            # incoming from i-1
            prev = msg[(i - 1) % n]
            # belief at i
            b = phi[i] + prev
            # send to j with compatibility matrix [[c,1-c],[1-c,c]]
            for sj in range(2):
                terms = []
                for si in range(2):
                    comp = compatibility if si == sj else (1 - compatibility)
                    terms.append(b[si] + math.log(comp + 1e-12))
                new[i, sj] = float(np.logaddexp(terms[0], terms[1]))
            # normalize
            new[i] -= np.logaddexp(new[i, 0], new[i, 1])
        msg = new
    beliefs = phi + msg[np.arange(n) - 1] + np.roll(msg, 1, axis=0)
    # softmax
    beliefs = beliefs - beliefs.max(axis=1, keepdims=True)
    expb = np.exp(beliefs)
    return expb / (expb.sum(axis=1, keepdims=True) + 1e-12)


def joint_latent_step(
    z: np.ndarray,
    views: List[np.ndarray],
    lr: float = 0.05,
) -> np.ndarray:
    """Online: z ← z - lr Σ (z - V_i)  multi-view consensus."""
    z = np.asarray(z, dtype=np.float64).copy()
    g = np.zeros_like(z)
    for v in views:
        vv = np.asarray(v, dtype=np.float64).ravel()
        if vv.size < z.size:
            pad = np.zeros(z.size)
            pad[: vv.size] = vv
            vv = pad
        else:
            vv = vv[: z.size]
        g += z - vv
    z = z - lr * g / max(len(views), 1)
    n = float(np.linalg.norm(z)) or 1.0
    return z / n


def route_decision(
    resonance: float,
    kuramoto_r: float,
    excess_demand: float,
    free_energy: float,
    entropy: float,
) -> Dict[str, Any]:
    """Joint physics+econ score → MESIE vs LLM route."""
    # high physical lock + settled market → stay deterministic
    lock = 0.5 * resonance + 0.5 * kuramoto_r
    market_calm = 1.0 / (1.0 + excess_demand)
    # free energy peak with low entropy → sharp decision, no LLM
    sharp = math.tanh(abs(free_energy)) * (1.0 / (1.0 + entropy))
    score = 0.45 * lock + 0.35 * market_calm + 0.20 * sharp
    if score >= 0.55:
        route = "mesie_only"
        reason = "physics_econ_lock_sufficient"
    elif score >= 0.35:
        route = "mesie_then_optional_llm"
        reason = "borderline_spectral_market"
    else:
        route = "escalate_llm"
        reason = "low_lock_or_market_stress"
    return {
        "route": route,
        "reason": reason,
        "score": score,
        "lock": lock,
        "market_calm": market_calm,
        "sharp": sharp,
    }


class AlgorithmSuite:
    """Run all coupling algorithms on engine outputs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.z = np.zeros(dim)
        self.last: Dict[str, Any] = {}

    def couple(
        self,
        physics_feat: np.ndarray,
        econ_feat: np.ndarray,
        spectral_feat: np.ndarray,
        *,
        resonance: float,
        kuramoto_r: float,
        excess_demand: float,
        free_energy: float,
        entropy: float,
    ) -> Dict[str, Any]:
        pf = np.asarray(physics_feat, dtype=np.float64).ravel()
        ef = np.asarray(econ_feat, dtype=np.float64).ravel()
        sf = np.asarray(spectral_feat, dtype=np.float64).ravel()
        d = self.dim

        def fit(v: np.ndarray) -> np.ndarray:
            if v.size < d:
                o = np.zeros(d)
                o[: v.size] = v
                return o
            return v[:d]

        pf, ef, sf = fit(pf), fit(ef), fit(sf)
        # cost matrix between physics bins and econ bins (top k)
        k = 8
        p_m = np.abs(pf[:k]) + 1e-6
        e_m = np.abs(ef[:k]) + 1e-6
        C = np.abs(p_m[:, None] - e_m[None, :])
        pairs, match_cost = greedy_match(C)
        P, ot_cost = sinkhorn(p_m / p_m.sum(), e_m / e_m.sum(), C, eps=0.08)
        # BP on Ising-like potentials from spectral sign
        pot = np.stack([sf[:d], -sf[:d]], axis=1) if sf.size >= d else np.zeros((d, 2))
        if pot.shape[0] > 32:
            pot = pot[:32]
        beliefs = belief_propagate_ring(pot, compatibility=0.55 + 0.2 * kuramoto_r)
        self.z = joint_latent_step(self.z if self.z.any() else (pf + ef + sf) / 3.0, [pf, ef, sf])
        route = route_decision(resonance, kuramoto_r, excess_demand, free_energy, entropy)
        self.last = {
            "match_pairs": pairs,
            "match_cost": match_cost,
            "ot_cost": ot_cost,
            "ot_mass": float(P.sum()),
            "belief_entropy": float(-(beliefs * np.log(beliefs + 1e-12)).sum() / beliefs.shape[0]),
            "latent_norm": float(np.linalg.norm(self.z)),
            "route": route,
        }
        return self.last

    def feature_vector(self, out_dim: int = 64) -> np.ndarray:
        z = self.z
        if z.size < out_dim:
            z = np.pad(z, (0, out_dim - z.size))
        else:
            z = z[:out_dim]
        meta = np.array(
            [
                self.last.get("match_cost", 0.0),
                self.last.get("ot_cost", 0.0),
                self.last.get("belief_entropy", 0.0),
                float((self.last.get("route") or {}).get("score", 0.0)),
            ]
        )
        v = np.concatenate([z, meta])
        if v.size < out_dim:
            v = np.pad(v, (0, out_dim - v.size))
        else:
            v = v[:out_dim]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float64)
