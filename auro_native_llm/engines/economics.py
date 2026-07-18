"""Economic engines — real optimization / market dynamics coupled to physics state.

Engines
-------
1. Cobb–Douglas / CES utility + expenditure shares
2. Walrasian tatonnement (price adjustment toward equilibrium)
3. Spectral market: assets = spectral buckets; returns from physics energy
4. Kelly criterion portfolio on spectral "odds"
5. Entropy economics: free energy F = U - T S as value functional
6. Agent multi-market mean-field game step
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = PHI - 1.0


@dataclass
class EconomicState:
    prices: np.ndarray
    supplies: np.ndarray
    demands: np.ndarray
    portfolio: np.ndarray
    wealth: float
    utility: float
    free_energy: float
    entropy: float
    step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_assets": int(self.prices.size),
            "wealth": self.wealth,
            "utility": self.utility,
            "free_energy": self.free_energy,
            "entropy": self.entropy,
            "step": self.step,
            "price_mean": float(self.prices.mean()),
            "price_std": float(self.prices.std()),
            "excess_demand_l2": float(np.linalg.norm(self.demands - self.supplies)),
            "portfolio_sum": float(self.portfolio.sum()),
            "metrics": self.metrics,
        }


class EconomicEngine:
    """Multi-engine economy driven by spectral/physics signals."""

    def __init__(self, n_assets: int = 16, seed: int = 11) -> None:
        self.n = n_assets
        rng = np.random.default_rng(seed)
        prices = np.clip(rng.random(n_assets) + 0.5, 0.1, None)
        self.state = EconomicState(
            prices=prices,
            supplies=np.ones(n_assets),
            demands=np.ones(n_assets),
            portfolio=np.ones(n_assets) / n_assets,
            wealth=1.0,
            utility=0.0,
            free_energy=0.0,
            entropy=0.0,
        )
        # preference weights (Cobb–Douglas exponents) — φ-normalized
        w = np.array([PHI_INV ** (i % 5) for i in range(n_assets)], dtype=np.float64)
        self.alpha = w / w.sum()
        self.elasticity = 0.7  # CES rho related

    # ---- utility ----
    def cobb_douglas_utility(self, basket: np.ndarray) -> float:
        x = np.clip(np.asarray(basket, dtype=np.float64), 1e-12, None)
        return float(np.exp(np.sum(self.alpha * np.log(x))))

    def ces_utility(self, basket: np.ndarray, rho: float = -0.5) -> float:
        """CES U = (Σ α x^ρ)^{1/ρ}."""
        x = np.clip(np.asarray(basket, dtype=np.float64), 1e-12, None)
        rho = rho if abs(rho) > 1e-6 else -1e-3
        return float(np.sum(self.alpha * (x**rho)) ** (1.0 / rho))

    def marshallian_demand(self, wealth: float, prices: np.ndarray) -> np.ndarray:
        """Cobb–Douglas: x_i = α_i W / p_i."""
        p = np.clip(prices, 1e-9, None)
        return self.alpha * wealth / p

    # ---- Walras tatonnement ----
    def tatonnement_step(self, eta: float = 0.15) -> float:
        """ṗ = η z(p), z = demand - supply; returns ||z||."""
        st = self.state
        d = self.marshallian_demand(st.wealth, st.prices)
        st.demands = d
        z = d - st.supplies
        st.prices = np.clip(st.prices + eta * z, 1e-4, None)
        # normalize price level (numeraire)
        st.prices = st.prices / (st.prices.mean() + 1e-12)
        return float(np.linalg.norm(z))

    # ---- spectral market ----
    def spectral_returns(self, spectral_buckets: np.ndarray, physics_energy: float) -> np.ndarray:
        """Map spectral energy shares → asset returns (mean-reverting)."""
        b = np.asarray(spectral_buckets, dtype=np.float64).ravel()
        if b.size != self.n:
            if b.size < self.n:
                bb = np.zeros(self.n)
                bb[: b.size] = b
                b = bb
            else:
                # pool
                edges = np.linspace(0, b.size, self.n + 1).astype(int)
                b = np.array([b[edges[i] : max(edges[i + 1], edges[i] + 1)].mean() for i in range(self.n)])
        b = b / (b.sum() + 1e-12)
        # return ~ excess bucket energy vs equal weight + physics boost
        eq = 1.0 / self.n
        r = 0.05 * (b - eq) + 0.01 * math.tanh(physics_energy) * b
        return r

    def update_supplies_from_physics(self, magnetization: float, fluid_energy: float) -> None:
        """Physical order parameters shift production."""
        base = np.ones(self.n)
        # mag → tilt even/odd assets; fluid → overall capacity
        tilt = 1.0 + 0.2 * magnetization * np.array([1.0 if i % 2 == 0 else -1.0 for i in range(self.n)])
        cap = 1.0 + 0.3 * math.tanh(fluid_energy)
        self.state.supplies = np.clip(base * tilt * cap, 0.05, None)

    # ---- Kelly ----
    def kelly_portfolio(self, returns: np.ndarray, odds_scale: float = 1.0) -> np.ndarray:
        """Proportional betting: f_i ∝ max(edge_i, 0); simplex project."""
        edge = np.asarray(returns, dtype=np.float64) * odds_scale
        f = np.clip(edge, 0.0, None)
        if f.sum() <= 1e-12:
            f = np.ones(self.n) / self.n
        else:
            f = f / f.sum()
        # mix with prior portfolio (sticky capital)
        self.state.portfolio = 0.65 * f + 0.35 * self.state.portfolio
        self.state.portfolio = self.state.portfolio / (self.state.portfolio.sum() + 1e-12)
        return self.state.portfolio

    def wealth_update(self, returns: np.ndarray) -> float:
        r = np.asarray(returns, dtype=np.float64)
        growth = float(np.dot(self.state.portfolio, r))
        self.state.wealth = float(max(1e-6, self.state.wealth * (1.0 + growth)))
        return self.state.wealth

    # ---- entropy economics ----
    def free_energy(self, utility: float, T: float = 1.0) -> tuple[float, float]:
        """F = U - T S with S from portfolio Shannon entropy."""
        p = np.clip(self.state.portfolio, 1e-12, None)
        p = p / p.sum()
        S = float(-(p * np.log(p)).sum())
        U = utility
        F = U - T * S
        self.state.entropy = S
        self.state.free_energy = F
        self.state.utility = U
        return F, S

    # ---- mean-field agents ----
    def mean_field_step(self, n_agents: int = 8) -> float:
        """Simple replicator: strategies = assets; fitness = -price * excess."""
        st = self.state
        fitness = -st.prices * (st.demands - st.supplies)
        fitness = fitness - fitness.mean()
        # replicator on portfolio as population share
        x = st.portfolio
        x_dot = x * (fitness - float(np.dot(x, fitness)))
        st.portfolio = np.clip(x + 0.1 * x_dot, 1e-6, None)
        st.portfolio /= st.portfolio.sum()
        return float(np.dot(st.portfolio, fitness))

    def step(
        self,
        *,
        spectral_buckets: Optional[np.ndarray] = None,
        physics_energy: float = 0.0,
        magnetization: float = 0.0,
        fluid_energy: float = 0.0,
        temperature: float = 1.0,
    ) -> EconomicState:
        self.update_supplies_from_physics(magnetization, fluid_energy)
        z_norm = self.tatonnement_step()
        buckets = spectral_buckets if spectral_buckets is not None else np.ones(self.n) / self.n
        rets = self.spectral_returns(buckets, physics_energy)
        self.kelly_portfolio(rets)
        wealth = self.wealth_update(rets)
        demand = self.marshallian_demand(wealth, self.state.prices)
        util = self.cobb_douglas_utility(demand)
        F, S = self.free_energy(util, T=max(temperature, 0.05))
        fit = self.mean_field_step()
        self.state.step += 1
        self.state.metrics = {
            "excess_demand_l2": z_norm,
            "mean_return": float(rets.mean()),
            "max_return": float(rets.max()),
            "kelly_hhi": float(np.sum(self.state.portfolio**2)),
            "mean_field_fitness": fit,
            "wealth": wealth,
            "utility": util,
            "free_energy": F,
            "entropy": S,
        }
        return self.state

    def feature_vector(self, out_dim: int = 64) -> np.ndarray:
        st = self.state
        parts = [
            st.prices,
            st.supplies,
            st.demands,
            st.portfolio,
            np.array(
                [
                    st.wealth,
                    st.utility,
                    st.free_energy,
                    st.entropy,
                    st.metrics.get("excess_demand_l2", 0.0),
                    st.metrics.get("mean_return", 0.0),
                    st.metrics.get("kelly_hhi", 0.0),
                    float(st.step),
                ]
            ),
        ]
        v = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        if v.size < out_dim:
            v = np.pad(v, (0, out_dim - v.size))
        else:
            idx = np.linspace(0, v.size, out_dim, endpoint=False).astype(int)
            v = v[idx]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).astype(np.float64)
