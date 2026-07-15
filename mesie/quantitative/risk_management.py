"""Risk Management and Measurement.

Value at Risk, Conditional Value at Risk, stress testing, and correlation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class RiskReport:
    """Comprehensive risk analysis report.
    
    Args:
        var_95: Value at Risk at 95% confidence
        cvar_95: Conditional Value at Risk at 95%
        expected_shortfall: Average loss beyond VaR
        max_drawdown: Maximum drawdown in returns
        volatility: Return volatility
        skewness: Distribution skewness
        kurtosis: Distribution kurtosis
        correlation_matrix: Asset correlations
        stress_scenarios: Stress test results
    """
    
    var_95: float
    cvar_95: float
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None
    stress_scenarios: Dict[str, float] = field(default_factory=dict)


class CorrelationMatrix:
    """Asset correlation analysis and monitoring.
    
    Tracks correlation breakdowns during stress periods.
    """
    
    def __init__(
        self,
        asset_names: List[str],
        return_series: np.ndarray
    ):
        """Initialize correlation matrix.
        
        Args:
            asset_names: Names of assets
            return_series: Return data (n_assets x n_observations)
        """
        self.asset_names = asset_names
        self.return_series = return_series
        self.n_assets = len(asset_names)
        self.correlation = np.corrcoef(return_series)
    
    def get_correlation(self, asset1: int, asset2: int) -> float:
        """Get correlation between two assets."""
        return self.correlation[asset1, asset2]
    
    def detect_correlation_breakdown(
        self,
        window_size: int = 60
    ) -> Dict[str, Tuple[float, float]]:
        """Detect periods of correlation breakdown.
        
        Args:
            window_size: Rolling window size
        
        Returns:
            Dict mapping asset pair -> (normal_corr, stress_corr)
        """
        
        breakdowns = {}
        n_obs = self.return_series.shape[1]
        
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                # Overall correlation
                overall_corr = self.correlation[i, j]
                
                # Find periods of high volatility (stress)
                combined_vol = np.abs(self.return_series[i]) + np.abs(self.return_series[j])
                stress_threshold = np.percentile(combined_vol, 75)
                
                stress_periods = combined_vol > stress_threshold
                
                if np.sum(stress_periods) > window_size:
                    stress_corr = np.corrcoef(
                        self.return_series[i][stress_periods],
                        self.return_series[j][stress_periods]
                    )[0, 1]
                    
                    if stress_corr > overall_corr + 0.1:  # Significant breakdown
                        pair_name = f"{self.asset_names[i]}-{self.asset_names[j]}"
                        breakdowns[pair_name] = (overall_corr, stress_corr)
        
        return breakdowns


class ValueAtRisk:
    """Value at Risk calculation - multiple methods.
    
    VaR quantifies potential loss at specified confidence level.
    """
    
    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        position_size: float = 1.0
    ) -> float:
        """Calculate VaR using historical simulation.
        
        Args:
            returns: Historical return series
            confidence_level: Confidence level (default 95%)
            position_size: Portfolio size
        
        Returns:
            VaR (loss amount)
        """
        
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(returns, percentile)
        return var * position_size
    
    @staticmethod
    def parametric_var(
        expected_return: float,
        volatility: float,
        confidence_level: float = 0.95,
        position_size: float = 1.0,
        time_horizon: float = 1.0
    ) -> float:
        """Calculate VaR assuming normal distribution.
        
        Formula: VaR = -(μ + σ*Φ^(-1)(1-confidence)) * √T
        
        Args:
            expected_return: Expected return
            volatility: Return volatility
            confidence_level: Confidence level
            position_size: Portfolio size
            time_horizon: Time horizon in years
        
        Returns:
            VaR (loss amount)
        """
        
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence_level)
        horizon_volatility = volatility * np.sqrt(time_horizon)
        
        var = -(expected_return * time_horizon + z_score * horizon_volatility)
        return var * position_size
    
    @staticmethod
    def monte_carlo_var(
        expected_return: float,
        volatility: float,
        confidence_level: float = 0.95,
        position_size: float = 1.0,
        n_simulations: int = 10000,
        time_horizon: float = 1.0
    ) -> float:
        """Calculate VaR using Monte Carlo simulation.
        
        Args:
            expected_return: Expected return
            volatility: Volatility
            confidence_level: Confidence level
            position_size: Position size
            n_simulations: Number of simulations
            time_horizon: Time horizon in years
        
        Returns:
            VaR (loss amount)
        """
        
        # Simulate returns
        dt = time_horizon
        z = np.random.standard_normal(n_simulations)
        returns = expected_return * dt + volatility * np.sqrt(dt) * z
        
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(returns, percentile)
        
        return var * position_size


class ConditionalValueAtRisk:
    """Conditional Value at Risk (Expected Shortfall).
    
    Average loss given loss exceeds VaR.
    """
    
    @staticmethod
    def cvar(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        position_size: float = 1.0
    ) -> float:
        """Calculate CVaR (Expected Shortfall).
        
        Formula: CVaR = E[R | R ≤ VaR]
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            position_size: Position size
        
        Returns:
            CVaR (expected loss)
        """
        
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        
        # Average of returns worse than VaR
        cvar = np.mean(returns[returns <= var])
        
        return -cvar * position_size


class StressTestScenario:
    """Predefined stress test scenarios.
    
    Tests portfolio resilience under extreme conditions.
    """
    
    # Market scenarios
    MARKET_CRASH_2008 = {
        "name": "2008 Financial Crisis",
        "equity_shock": -0.50,  # 50% equities down
        "credit_shock": 0.05,    # Credit spreads up 5%
        "vol_increase": 3.0,     # VIX multiplier
        "description": "2008 financial crisis"
    }
    
    RISING_RATES = {
        "name": "Rising Rate Environment",
        "rate_change": 0.02,     # 2% rate increase
        "duration": 7.0,         # Years
        "equity_beta": -0.3,     # 30% negative beta to rates
        "description": "Rising interest rate scenario"
    }
    
    CREDIT_EVENT = {
        "name": "Credit Event",
        "credit_spread": 0.03,   # 300 bps spread widening
        "equity_correlation": 0.9,  # High correlation spike
        "liquidity_factor": 0.5,    # 50% liquidity discount
        "description": "Credit stress event"
    }
    
    def __init__(
        self,
        portfolio_weights: np.ndarray,
        asset_names: List[str],
        baseline_returns: np.ndarray,
        baseline_volatilities: np.ndarray
    ):
        """Initialize stress testing.
        
        Args:
            portfolio_weights: Asset weights
            asset_names: Asset names
            baseline_returns: Baseline returns
            baseline_volatilities: Baseline volatilities
        """
        
        self.weights = portfolio_weights
        self.asset_names = asset_names
        self.baseline_returns = baseline_returns
        self.baseline_vols = baseline_volatilities
    
    def apply_scenario(
        self,
        scenario: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Apply stress scenario to portfolio.
        
        Args:
            scenario: Scenario specification
        
        Returns:
            (portfolio_loss, asset_impacts)
        """
        
        asset_impacts = {}
        stressed_returns = self.baseline_returns.copy()
        
        for asset_idx, asset_name in enumerate(self.asset_names):
            impact = 0.0
            
            # Apply equity shock
            if "equity_shock" in scenario and "equity" in asset_name.lower():
                impact += scenario["equity_shock"]
            
            # Apply rate shock
            if "rate_change" in scenario:
                duration = scenario.get("duration", 7.0)
                impact -= duration * scenario["rate_change"]
            
            # Apply credit shock
            if "credit_shock" in scenario and "credit" in asset_name.lower():
                impact += scenario["credit_shock"]
            
            # Apply volatility increase
            if "vol_increase" in scenario:
                impact -= self.baseline_vols[asset_idx] * scenario["vol_increase"]
            
            stressed_returns[asset_idx] = self.baseline_returns[asset_idx] + impact
            asset_impacts[asset_name] = impact
        
        portfolio_loss = np.dot(self.weights, -stressed_returns)
        
        return portfolio_loss, asset_impacts


def compute_drawdown(returns: np.ndarray) -> Tuple[float, int]:
    """Compute maximum drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        (max_drawdown, duration_in_periods)
    """
    
    cumulative_returns = np.cumprod(1 + returns) - 1
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / (1 + running_max)
    
    max_drawdown_idx = np.argmin(cumulative_returns)
    max_drawdown = np.min(drawdown)
    
    return max_drawdown, max_drawdown_idx


def compute_risk_metrics(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    position_size: float = 1.0
) -> RiskReport:
    """Compute comprehensive risk report.
    
    Args:
        returns: Return series
        confidence_level: VaR confidence level
        position_size: Portfolio size
    
    Returns:
        Risk report with all metrics
    """
    
    var_95 = ValueAtRisk.historical_var(returns, confidence_level, position_size)
    cvar_95 = ConditionalValueAtRisk.cvar(returns, confidence_level, position_size)
    max_dd, _ = compute_drawdown(returns)
    
    return RiskReport(
        var_95=var_95,
        cvar_95=cvar_95,
        expected_shortfall=cvar_95,
        max_drawdown=max_dd,
        volatility=np.std(returns),
        skewness=stats.skew(returns),
        kurtosis=stats.kurtosis(returns)
    )
