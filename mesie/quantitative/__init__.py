"""MESIE Quantitative Intelligence Layer.

Production-grade quantitative finance, game theory, and multi-agent strategic models.
All mathematics implemented with real formulas — no placeholders.
"""

from mesie.quantitative.game_theory import (
    GamePayoffMatrix,
    NashEquilibriumSolver,
    CooperativeGame,
    ShapleyValueCalculator,
    StrategicAgent,
    GameTheoryResult,
)
from mesie.quantitative.portfolio_management import (
    PortfolioOptimizer,
    ModernPortfolioTheory,
    BlackLittermanModel,
    PortfolioMetrics,
    AllocationStrategy,
)
from mesie.quantitative.pricing_models import (
    CAPMModel,
    FactorModel,
    APTModel,
    BinomialPricingTree,
    DerivativePricer,
    OptionGreeks,
)
from mesie.quantitative.risk_management import (
    ValueAtRisk,
    ConditionalValueAtRisk,
    StressTestScenario,
    CorrelationMatrix,
    RiskReport,
)
from mesie.quantitative.market_dynamics import (
    FundFlowModel,
    CapitalAllocationEngine,
    MarketMicrostructure,
    ExecutionAlgorithm,
)
from mesie.quantitative.agent_strategies import (
    StrategicAgent,
    AgentPortfolio,
    MultiAgentSystem,
    StrategyOptimizer,
)
from mesie.quantitative.invariants import (
    MathematicalInvariant,
    InvariantEnforcer,
    ConstraintViolationException,
    GovernanceConstraint,
    RuntimeGuardian,
)

__all__ = [
    "GamePayoffMatrix",
    "NashEquilibriumSolver",
    "CooperativeGame",
    "ShapleyValueCalculator",
    "StrategicAgent",
    "GameTheoryResult",
    "PortfolioOptimizer",
    "ModernPortfolioTheory",
    "BlackLittermanModel",
    "PortfolioMetrics",
    "AllocationStrategy",
    "CAPMModel",
    "FactorModel",
    "APTModel",
    "BinomialPricingTree",
    "DerivativePricer",
    "OptionGreeks",
    "ValueAtRisk",
    "ConditionalValueAtRisk",
    "StressTestScenario",
    "CorrelationMatrix",
    "RiskReport",
    "FundFlowModel",
    "CapitalAllocationEngine",
    "MarketMicrostructure",
    "ExecutionAlgorithm",
    "MultiAgentSystem",
    "StrategyOptimizer",
    "MathematicalInvariant",
    "InvariantEnforcer",
    "ConstraintViolationException",
    "GovernanceConstraint",
    "RuntimeGuardian",
]
