"""Asset Pricing Models.

CAPM, Factor models, APT, and derivative pricing with full mathematical foundations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm


@dataclass
class OptionGreeks:
    """Greek derivatives for option pricing sensitivity.
    
    Args:
        delta: Change in option price per unit change in asset price
        gamma: Second derivative of option price w.r.t. asset price
        vega: Change in option price per 1% change in volatility
        theta: Change in option price per day (time decay)
        rho: Change in option price per 1% change in interest rate
    """
    
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class CAPMModel:
    """Capital Asset Pricing Model.
    
    Formula: E[R_i] = r_f + β_i(E[R_m] - r_f)
    
    where:
        β_i = Cov(R_i, R_m) / Var(R_m)
    """
    
    def __init__(
        self,
        risk_free_rate: float,
        market_return: float,
        market_variance: float
    ):
        """Initialize CAPM model.
        
        Args:
            risk_free_rate: Risk-free rate (r_f)
            market_return: Expected market return (E[R_m])
            market_variance: Market return variance
        """
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.market_variance = market_variance
        self.market_risk_premium = market_return - risk_free_rate
    
    def compute_beta(
        self,
        asset_return_series: np.ndarray,
        market_return_series: np.ndarray
    ) -> float:
        """Compute beta for an asset.
        
        β = Cov(R_asset, R_market) / Var(R_market)
        
        Args:
            asset_return_series: Asset returns time series
            market_return_series: Market returns time series
        
        Returns:
            Beta coefficient
        """
        
        covariance = np.cov(asset_return_series, market_return_series)[0, 1]
        beta = covariance / self.market_variance
        return beta
    
    def expected_return(self, beta: float) -> float:
        """Compute expected return using CAPM.
        
        E[R_i] = r_f + β(E[R_m] - r_f)
        
        Args:
            beta: Asset beta
        
        Returns:
            Expected return
        """
        return self.risk_free_rate + beta * self.market_risk_premium


class FactorModel:
    """Multi-factor asset pricing model.
    
    Formula: E[R_i] = r_f + Σ β_ij * λ_j
    
    where:
        β_ij = factor loading
        λ_j = factor risk premium
    """
    
    def __init__(
        self,
        risk_free_rate: float,
        factor_names: List[str],
        factor_premiums: np.ndarray
    ):
        """Initialize factor model.
        
        Args:
            risk_free_rate: Risk-free rate
            factor_names: Names of risk factors
            factor_premiums: Risk premiums for each factor (λ_j)
        """
        self.risk_free_rate = risk_free_rate
        self.factor_names = factor_names
        self.factor_premiums = factor_premiums
        self.n_factors = len(factor_names)
    
    def expected_return(self, factor_loadings: np.ndarray) -> float:
        """Compute expected return.
        
        E[R_i] = r_f + Σ β_ij * λ_j
        
        Args:
            factor_loadings: Beta coefficients for each factor (β_ij)
        
        Returns:
            Expected return
        """
        return self.risk_free_rate + np.dot(factor_loadings, self.factor_premiums)
    
    def extract_factor_loadings(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray
    ) -> np.ndarray:
        """Extract factor loadings using regression.
        
        Solves: R_i - r_f = β_i1*F_1 + ... + β_in*F_n + ε_i
        
        Args:
            asset_returns: Asset return series
            factor_returns: Factor return series (n_factors x n_observations)
        
        Returns:
            Factor loadings vector
        """
        
        # Standardize
        asset_centered = asset_returns - self.risk_free_rate
        
        # Regression: (F'F)^(-1) F'R
        try:
            loadings = np.linalg.lstsq(
                factor_returns.T,
                asset_centered,
                rcond=None
            )[0]
            return loadings
        except np.linalg.LinAlgError:
            return np.zeros(self.n_factors)


class APTModel:
    """Arbitrage Pricing Theory model.
    
    Similar to factor models but derived from no-arbitrage assumptions.
    """
    
    def __init__(
        self,
        risk_free_rate: float,
        factor_names: List[str],
        factor_premiums: np.ndarray
    ):
        self.risk_free_rate = risk_free_rate
        self.factor_names = factor_names
        self.factor_premiums = factor_premiums
        self.n_factors = len(factor_names)
    
    def expected_return(self, factor_loadings: np.ndarray) -> float:
        """Expected return under APT.
        
        E[R] = r_f + Σ β_i * (E[R_factor_i] - r_f)
        """
        return self.risk_free_rate + np.dot(factor_loadings, self.factor_premiums)
    
    def detect_arbitrage(
        self,
        expected_returns: np.ndarray,
        factor_loadings: np.ndarray
    ) -> Optional[float]:
        """Detect arbitrage opportunity.
        
        Returns arbitrage profit if found, None otherwise.
        """
        
        theoretical_returns = np.array([
            self.expected_return(fl) for fl in factor_loadings
        ])
        
        arbitrage = expected_returns - theoretical_returns
        
        max_arb = np.max(np.abs(arbitrage))
        return max_arb if max_arb > 1e-6 else None


class BinomialPricingTree:
    """Binomial tree for European option pricing.
    
    Recursively builds tree of possible price paths.
    """
    
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        n_steps: int = 50,
        is_call: bool = True
    ):
        """Initialize binomial tree.
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_maturity: Time to expiration (years)
            risk_free_rate: Risk-free rate
            volatility: Stock volatility (annualized)
            n_steps: Number of steps in tree
            is_call: True for call, False for put
        """
        
        self.S0 = spot_price
        self.K = strike_price
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.sigma = volatility
        self.n_steps = n_steps
        self.is_call = is_call
        
        self.dt = time_to_maturity / n_steps
        self.u = np.exp(volatility * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor
        self.p = (np.exp(risk_free_rate * self.dt) - self.d) / (self.u - self.d)
    
    def price(self) -> float:
        """Compute option price using binomial tree.
        
        Returns:
            Option price
        """
        
        # Initialize terminal payoffs
        payoffs = np.zeros(self.n_steps + 1)
        
        for j in range(self.n_steps + 1):
            stock_price = self.S0 * (self.u ** (self.n_steps - j)) * (self.d ** j)
            
            if self.is_call:
                payoffs[j] = max(stock_price - self.K, 0)
            else:
                payoffs[j] = max(self.K - stock_price, 0)
        
        # Backward induction
        for i in range(self.n_steps - 1, -1, -1):
            for j in range(i + 1):
                payoffs[j] = (
                    (self.p * payoffs[j] + (1 - self.p) * payoffs[j + 1]) *
                    np.exp(-self.r * self.dt)
                )
        
        return payoffs[0]


class DerivativePricer:
    """Black-Scholes option pricing model.
    
    Closed-form solution for European options.
    """
    
    @staticmethod
    def black_scholes(
        spot_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        is_call: bool = True
    ) -> Tuple[float, OptionGreeks]:
        """Price European option and compute Greeks.
        
        Formula:
            C = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
            P = K*e^(-r*T)*N(-d2) - S*e^(-q*T)*N(-d1)
        
        where:
            d1 = (ln(S/K) + (r-q+σ²/2)*T) / (σ*√T)
            d2 = d1 - σ*√T
        
        Args:
            spot_price: Current stock price (S)
            strike_price: Strike price (K)
            time_to_maturity: Time to expiration in years (T)
            risk_free_rate: Risk-free rate (r)
            volatility: Annualized volatility (σ)
            dividend_yield: Dividend yield (q)
            is_call: True for call, False for put
        
        Returns:
            (option_price, greeks)
        """
        
        S = spot_price
        K = strike_price
        T = time_to_maturity
        r = risk_free_rate
        sigma = volatility
        q = dividend_yield
        
        # Avoid division by zero
        if T < 1e-10:
            if is_call:
                return max(S - K, 0), OptionGreeks(0, 0, 0, 0, 0)
            else:
                return max(K - S, 0), OptionGreeks(0, 0, 0, 0, 0)
        
        # Compute d1 and d2
        d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Option price
        if is_call:
            price = (S * np.exp(-q * T) * norm.cdf(d1) -
                    K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                    S * np.exp(-q * T) * norm.cdf(-d1))
        
        # Greeks
        sqrt_T = np.sqrt(T)
        
        # Delta
        if is_call:
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        
        # Gamma
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * sqrt_T)
        
        # Vega (per 1% change in volatility)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100
        
        # Theta (per day)
        if is_call:
            theta = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T) -
                r * K * np.exp(-r * T) * norm.cdf(d2) +
                q * S * np.exp(-q * T) * norm.cdf(d1)
            ) / 365
        else:
            theta = (
                -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T) +
                r * K * np.exp(-r * T) * norm.cdf(-d2) -
                q * S * np.exp(-q * T) * norm.cdf(-d1)
            ) / 365
        
        # Rho (per 1% change in rate)
        if is_call:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        greeks = OptionGreeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
        
        return price, greeks
