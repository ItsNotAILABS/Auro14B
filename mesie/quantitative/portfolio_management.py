"""Portfolio Management and Optimization.

Modern Portfolio Theory, Black-Litterman model, and asset allocation.
Real mathematical formulations with numerical optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize, linprog


@dataclass
class PortfolioMetrics:
    """Performance metrics for a portfolio.
    
    Args:
        expected_return: Expected portfolio return
        volatility: Standard deviation of returns
        sharpe_ratio: Risk-adjusted return metric
        var_95: Value at Risk at 95% confidence
        cvar_95: Conditional Value at Risk at 95%
    """
    
    expected_return: float
    volatility: float
    sharpe_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    sortino_ratio: float = 0.0
    
    def __post_init__(self):
        if self.volatility > 0 and self.sharpe_ratio == 0:
            # Assume 2% risk-free rate
            self.sharpe_ratio = (self.expected_return - 0.02) / self.volatility


@dataclass
class AllocationStrategy:
    """Asset allocation strategy result.
    
    Args:
        weights: Portfolio weights (sum to 1)
        assets: Asset names
        expected_return: Expected portfolio return
        expected_volatility: Expected portfolio volatility
        constraints_satisfied: Whether constraints were met
    """
    
    weights: np.ndarray
    assets: List[str]
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    constraints_satisfied: bool = True
    metadata: Dict = field(default_factory=dict)


class ModernPortfolioTheory:
    """Markowitz Modern Portfolio Theory implementation.
    
    Solves: minimize w'Σw - λ*μ'w
    subject to: sum(w) = 1, 0 ≤ w ≤ 1
    
    where Σ = covariance matrix, μ = expected returns, λ = risk aversion
    """
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.02
    ):
        """Initialize with asset parameters.
        
        Args:
            expected_returns: Vector of expected returns (n_assets,)
            covariance_matrix: Covariance matrix (n_assets, n_assets)
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
    
    def optimize_min_variance(
        self,
        min_return: Optional[float] = None,
        max_weights: Optional[np.ndarray] = None
    ) -> AllocationStrategy:
        """Optimize for minimum variance.
        
        Args:
            min_return: Minimum required return
            max_weights: Maximum weight per asset
        
        Returns:
            Optimal allocation
        """
        
        def objective(w):
            """Portfolio variance: w'Σw"""
            return w @ self.covariance_matrix @ w
        
        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        if min_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: w @ self.expected_returns - min_return
            })
        
        # Bounds
        bounds = [(0, max_weights[i] if max_weights is not None else 1.0)
                  for i in range(self.n_assets)]
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP"
        )
        
        weights = result.x
        expected_return = weights @ self.expected_returns
        expected_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
        
        return AllocationStrategy(
            weights=weights,
            assets=list(range(self.n_assets)),
            expected_return=expected_return,
            expected_volatility=expected_vol,
            constraints_satisfied=result.success
        )
    
    def optimize_max_sharpe(
        self,
        max_weights: Optional[np.ndarray] = None
    ) -> AllocationStrategy:
        """Optimize for maximum Sharpe ratio.
        
        Equivalent to: minimize -λ*μ'w + w'Σw (with λ=1 normalized)
        
        Args:
            max_weights: Maximum weight per asset
        
        Returns:
            Optimal allocation
        """
        
        def objective(w):
            """Negative Sharpe ratio (to minimize)"""
            port_return = w @ self.expected_returns
            port_vol = np.sqrt(w @ self.covariance_matrix @ w)
            
            if port_vol < 1e-10:
                return 1e10
            
            sharpe = (port_return - self.risk_free_rate) / port_vol
            return -sharpe
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weights[i] if max_weights is not None else 1.0)
                  for i in range(self.n_assets)]
        
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP"
        )
        
        weights = result.x
        expected_return = weights @ self.expected_returns
        expected_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
        sharpe = (expected_return - self.risk_free_rate) / (expected_vol + 1e-10)
        
        return AllocationStrategy(
            weights=weights,
            assets=list(range(self.n_assets)),
            expected_return=expected_return,
            expected_volatility=expected_vol,
            constraints_satisfied=result.success,
            metadata={"sharpe_ratio": sharpe}
        )
    
    def compute_efficient_frontier(
        self,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute efficient frontier.
        
        Args:
            n_points: Number of points on frontier
        
        Returns:
            (volatilities, returns) arrays
        """
        
        min_return = np.min(self.expected_returns)
        max_return = np.max(self.expected_returns)
        
        returns_range = np.linspace(min_return, max_return, n_points)
        volatilities = []
        
        for target_return in returns_range:
            alloc = self.optimize_min_variance(min_return=target_return)
            vol = np.sqrt(alloc.weights @ self.covariance_matrix @ alloc.weights)
            volatilities.append(vol)
        
        return np.array(volatilities), returns_range


class BlackLittermanModel:
    """Black-Litterman model for portfolio optimization.
    
    Combines market equilibrium implied returns with investor views.
    
    Formula:
        μ_BL = μ_market + τΣP'(PτΣP' + Ω)^(-1)(Q - Pμ_market)
    
    where:
        μ_market = market equilibrium returns
        τ = scale factor (uncertainty in market equilibrium)
        P = view picking matrix
        Q = view returns
        Ω = view uncertainty diagonal matrix
    """
    
    def __init__(
        self,
        market_weights: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 2.5,
        risk_free_rate: float = 0.02,
        tau: float = 0.05
    ):
        """Initialize Black-Litterman model.
        
        Args:
            market_weights: Market capitalization weights
            covariance_matrix: Covariance of assets
            risk_aversion: Market risk aversion coefficient
            risk_free_rate: Risk-free rate
            tau: Scaling parameter
        """
        self.market_weights = market_weights
        self.covariance_matrix = covariance_matrix
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.n_assets = len(market_weights)
    
    def compute_market_implied_returns(self) -> np.ndarray:
        """Compute implied returns from market equilibrium.
        
        Formula: μ_market = r_f + λ * Σ * w_market
        
        Returns:
            Implied return vector
        """
        return (
            self.risk_free_rate +
            self.risk_aversion * (self.covariance_matrix @ self.market_weights)
        )
    
    def incorporate_views(
        self,
        views: Dict[str, Tuple[np.ndarray, float, float]]
    ) -> np.ndarray:
        """Incorporate investor views into return estimates.
        
        Args:
            views: Dict mapping view_name -> (picking_vector, view_return, confidence)
        
        Returns:
            Updated return estimates
        """
        
        implied_returns = self.compute_market_implied_returns()
        
        if not views:
            return implied_returns
        
        # Build view matrices
        P_list = []
        Q_list = []
        view_confidences = []
        
        for view_name, (picking, view_ret, confidence) in views.items():
            P_list.append(picking)
            Q_list.append(view_ret)
            view_confidences.append(confidence)
        
        P = np.array(P_list)
        Q = np.array(Q_list)
        
        # Construct view uncertainty matrix Ω
        # Ω = diag(1/confidence_i * variance_i)
        omega_diag = [
            (1.0 / (confidence + 1e-10)) * (P[i] @ self.covariance_matrix @ P[i])
            for i, confidence in enumerate(view_confidences)
        ]
        Omega = np.diag(omega_diag)
        
        # Compute BL posterior returns
        try:
            # μ_BL = μ + τΣP'(PτΣP' + Ω)^(-1)(Q - Pμ)
            term1 = P @ self.covariance_matrix @ P.T
            term2_inv = np.linalg.inv(self.tau * term1 + Omega + np.eye(len(Q)) * 1e-8)
            
            bl_adjustment = (
                self.tau * self.covariance_matrix @ P.T @ term2_inv @
                (Q - P @ implied_returns)
            )
            
            bl_returns = implied_returns + bl_adjustment
            return bl_returns
            
        except np.linalg.LinAlgError:
            # Singular matrix - return original implied returns
            return implied_returns


class PortfolioOptimizer:
    """High-level portfolio optimization interface."""
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        asset_names: List[str],
        risk_free_rate: float = 0.02
    ):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.asset_names = asset_names
        self.risk_free_rate = risk_free_rate
        
        self.mpt = ModernPortfolioTheory(
            expected_returns,
            covariance_matrix,
            risk_free_rate
        )
    
    def optimize(
        self,
        strategy: str = "max_sharpe",
        constraints: Optional[Dict] = None
    ) -> AllocationStrategy:
        """Optimize portfolio.
        
        Args:
            strategy: 'max_sharpe', 'min_variance', 'equal_weight'
            constraints: Optional constraint specifications
        
        Returns:
            Optimal allocation
        """
        
        if strategy == "max_sharpe":
            alloc = self.mpt.optimize_max_sharpe()
        elif strategy == "min_variance":
            alloc = self.mpt.optimize_min_variance()
        elif strategy == "equal_weight":
            weights = np.ones(len(self.expected_returns)) / len(self.expected_returns)
            alloc = AllocationStrategy(
                weights=weights,
                assets=self.asset_names,
                expected_return=weights @ self.expected_returns,
                expected_volatility=np.sqrt(weights @ self.covariance_matrix @ weights)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        alloc.assets = self.asset_names
        return alloc
