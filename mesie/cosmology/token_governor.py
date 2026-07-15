"""Calendrical Token Governor — bounded compute governance for spectral operations.

Implements the Mesoamerican calendrical model as a resource governance system:
- Each compute cycle is a "Tun" (360-unit period) with bounded token budgets
- Operations consume tokens (sacrifice) to produce results (renewal)
- If the budget is exhausted before convergence, the cycle ends with
  partial results (the sun stops moving without sacrifice)

This enforces proof-bounded cognition: spectral matching, embedding, and
generation must converge within allocated token-cost constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class TokenExpenditure:
    """Record of a single token expenditure (sacrifice).

    Attributes:
        operation: Name of the operation that consumed tokens.
        tokens_spent: Number of tokens consumed.
        timestamp: When the expenditure occurred.
        result_quality: Quality metric of the result (0-1).
        metadata: Additional context.
    """

    operation: str
    tokens_spent: float
    timestamp: float = field(default_factory=time.time)
    result_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenBudget:
    """A bounded compute budget for a calendrical cycle.

    Attributes:
        cycle_name: Name of this compute cycle (e.g., "Tun-47").
        total_tokens: Total token budget for this cycle.
        spent_tokens: Tokens already consumed.
        expenditures: History of individual expenditures.
        cycle_start: When this cycle began.
        cycle_deadline: Optional hard deadline (unix timestamp).
        convergence_threshold: Quality threshold for early termination.
    """

    cycle_name: str
    total_tokens: float
    spent_tokens: float = 0.0
    expenditures: List[TokenExpenditure] = field(default_factory=list)
    cycle_start: float = field(default_factory=time.time)
    cycle_deadline: Optional[float] = None
    convergence_threshold: float = 0.95

    @property
    def remaining_tokens(self) -> float:
        """Tokens remaining in this cycle."""
        return max(self.total_tokens - self.spent_tokens, 0.0)

    @property
    def utilization(self) -> float:
        """Fraction of budget consumed."""
        if self.total_tokens <= 0:
            return 1.0
        return self.spent_tokens / self.total_tokens

    @property
    def is_exhausted(self) -> bool:
        """Whether the budget is fully consumed."""
        return self.remaining_tokens <= 0.0

    @property
    def is_expired(self) -> bool:
        """Whether the cycle has exceeded its deadline."""
        if self.cycle_deadline is None:
            return False
        return time.time() > self.cycle_deadline

    @property
    def is_active(self) -> bool:
        """Whether the budget can still be spent."""
        return not self.is_exhausted and not self.is_expired

    def spend(self, operation: str, tokens: float, quality: float = 0.0) -> TokenExpenditure:
        """Record a token expenditure.

        Args:
            operation: Operation name.
            tokens: Tokens to spend.
            quality: Result quality metric.

        Returns:
            The expenditure record.

        Raises:
            RuntimeError: If budget is exhausted or expired.
        """
        if not self.is_active:
            raise RuntimeError(
                f"Cycle '{self.cycle_name}' is no longer active: "
                f"{'exhausted' if self.is_exhausted else 'expired'}"
            )
        expenditure = TokenExpenditure(
            operation=operation,
            tokens_spent=tokens,
            result_quality=quality,
        )
        self.spent_tokens += tokens
        self.expenditures.append(expenditure)
        return expenditure

    @property
    def mean_quality(self) -> float:
        """Average result quality across all expenditures."""
        if not self.expenditures:
            return 0.0
        return float(np.mean([e.result_quality for e in self.expenditures]))

    @property
    def has_converged(self) -> bool:
        """Whether mean quality has exceeded the convergence threshold."""
        return self.mean_quality >= self.convergence_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget state."""
        return {
            "cycle_name": self.cycle_name,
            "total_tokens": self.total_tokens,
            "spent_tokens": self.spent_tokens,
            "remaining_tokens": self.remaining_tokens,
            "utilization": self.utilization,
            "mean_quality": self.mean_quality,
            "has_converged": self.has_converged,
            "is_active": self.is_active,
            "n_expenditures": len(self.expenditures),
        }


class CalendricalTokenGovernor:
    """Governs compute allocation using calendrical cycles.

    Models the Mesoamerican Tun calendar as bounded compute windows:
    - Each cycle (Tun) has a fixed token budget
    - Operations must sacrifice tokens to produce results
    - Cycles end when budget is exhausted, deadline passes, or convergence achieved
    - New cycles begin with renewed budgets (regeneration principle)

    The governor tracks efficiency across cycles and adaptively adjusts
    budgets based on historical convergence patterns.

    Args:
        default_budget: Default token budget per cycle.
        cycle_duration: Optional default cycle duration in seconds.
        convergence_threshold: Quality threshold for convergence.
        adaptive: Whether to adaptively adjust budgets based on history.
    """

    # Maya calendar constants
    TUN = 360  # 360-day cycle (base unit)
    KATUN = 7200  # 20 Tuns
    BAKTUN = 144000  # 20 Katuns

    def __init__(
        self,
        default_budget: float = 360.0,
        cycle_duration: Optional[float] = None,
        convergence_threshold: float = 0.95,
        adaptive: bool = True,
    ) -> None:
        self.default_budget = default_budget
        self.cycle_duration = cycle_duration
        self.convergence_threshold = convergence_threshold
        self.adaptive = adaptive
        self._cycle_count = 0
        self._active_budget: Optional[TokenBudget] = None
        self._history: List[TokenBudget] = []

    @property
    def current_cycle(self) -> Optional[TokenBudget]:
        """The currently active budget cycle."""
        return self._active_budget

    @property
    def cycle_count(self) -> int:
        """Total number of cycles initiated."""
        return self._cycle_count

    @property
    def history(self) -> List[TokenBudget]:
        """Completed cycle history."""
        return list(self._history)

    def begin_cycle(self, name: Optional[str] = None, budget: Optional[float] = None) -> TokenBudget:
        """Begin a new calendrical compute cycle.

        Ends any active cycle and starts a fresh budget. If adaptive mode
        is enabled, the budget may be adjusted based on historical efficiency.

        Args:
            name: Optional cycle name. Defaults to "Tun-{count}".
            budget: Optional custom budget. Defaults to adaptive/default budget.

        Returns:
            The new active TokenBudget.
        """
        # Archive any active cycle
        if self._active_budget is not None:
            self._history.append(self._active_budget)

        self._cycle_count += 1
        cycle_name = name or f"Tun-{self._cycle_count}"

        # Adaptive budget: learn from history
        if budget is None:
            budget = self._compute_adaptive_budget()

        deadline = None
        if self.cycle_duration is not None:
            deadline = time.time() + self.cycle_duration

        self._active_budget = TokenBudget(
            cycle_name=cycle_name,
            total_tokens=budget,
            cycle_start=time.time(),
            cycle_deadline=deadline,
            convergence_threshold=self.convergence_threshold,
        )
        return self._active_budget

    def spend(self, operation: str, tokens: float, quality: float = 0.0) -> TokenExpenditure:
        """Spend tokens from the active cycle.

        Args:
            operation: Operation consuming tokens.
            tokens: Token cost.
            quality: Result quality (0-1).

        Returns:
            Expenditure record.

        Raises:
            RuntimeError: If no active cycle or budget exhausted.
        """
        if self._active_budget is None:
            raise RuntimeError("No active cycle. Call begin_cycle() first.")
        return self._active_budget.spend(operation, tokens, quality)

    def end_cycle(self) -> Optional[TokenBudget]:
        """End the current cycle and archive it.

        Returns:
            The completed cycle budget, or None if no active cycle.
        """
        if self._active_budget is None:
            return None
        completed = self._active_budget
        self._history.append(completed)
        self._active_budget = None
        return completed

    def governed_operation(
        self,
        operation_name: str,
        cost: float,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[T]:
        """Execute an operation under token governance.

        If the budget allows, executes the function and records the expenditure.
        Returns None if the budget is exhausted (sacrifice refused).

        Args:
            operation_name: Name for tracking.
            cost: Token cost of this operation.
            func: Callable to execute.
            *args: Positional arguments to func.
            **kwargs: Keyword arguments to func.

        Returns:
            Function result, or None if budget exhausted.
        """
        if self._active_budget is None or not self._active_budget.is_active:
            return None

        if self._active_budget.remaining_tokens < cost:
            return None

        result = func(*args, **kwargs)

        # Estimate quality from result if it's a numeric score
        quality = 0.5  # default
        if isinstance(result, (int, float)):
            quality = min(max(float(result), 0.0), 1.0)

        self._active_budget.spend(operation_name, cost, quality)
        return result

    def _compute_adaptive_budget(self) -> float:
        """Compute adaptive budget based on historical cycles."""
        if not self.adaptive or not self._history:
            return self.default_budget

        # If recent cycles converged with leftover budget, reduce
        # If recent cycles exhausted without converging, increase
        recent = self._history[-5:]  # last 5 cycles
        avg_utilization = float(np.mean([c.utilization for c in recent]))
        avg_converged = float(np.mean([1.0 if c.has_converged else 0.0 for c in recent]))

        if avg_converged > 0.8 and avg_utilization < 0.7:
            # Converging efficiently — can reduce budget
            return self.default_budget * 0.85
        elif avg_converged < 0.3:
            # Struggling to converge — increase budget
            return self.default_budget * 1.3
        else:
            return self.default_budget

    def summary(self) -> Dict[str, Any]:
        """Get governor summary statistics."""
        total_spent = sum(c.spent_tokens for c in self._history)
        total_allocated = sum(c.total_tokens for c in self._history)
        convergence_rate = (
            float(np.mean([1.0 if c.has_converged else 0.0 for c in self._history]))
            if self._history else 0.0
        )
        return {
            "cycles_completed": len(self._history),
            "active_cycle": self._active_budget.to_dict() if self._active_budget else None,
            "total_tokens_spent": total_spent,
            "total_tokens_allocated": total_allocated,
            "overall_efficiency": total_spent / max(total_allocated, 1e-12),
            "convergence_rate": convergence_rate,
        }
