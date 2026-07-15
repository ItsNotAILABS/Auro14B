"""Mathematical Invariants and Runtime Governance Enforcement.

Encodes unbreakable mathematical laws into the system runtime.
Governs state, transactions, and decision-making through enforced constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import numpy as np


class InvariantType(Enum):
    """Types of invariants MESIE enforces."""
    
    CONSERVATION = "conservation"      # Conservation laws (mass, energy, capital)
    EQUILIBRIUM = "equilibrium"        # System equilibrium constraints
    CAUSALITY = "causality"            # Temporal causality
    CAPITAL_PRESERVATION = "capital_preservation"  # No capital creation
    DECISION_MONOTONICITY = "monotonicity"  # Monotonic decision paths
    CONSISTENCY = "consistency"        # State consistency
    ENTROPY = "entropy"                # Thermodynamic limits


@dataclass
class MathematicalInvariant:
    """Defines a mathematical invariant that cannot be violated.
    
    Args:
        name: Invariant identifier
        invariant_type: Category of invariant
        condition: Function returning True if invariant holds
        formula: Mathematical formula as string for documentation
        penalty: Consequence of violation (exception, logging, correction)
        metadata: Additional constraint metadata
    """
    
    name: str
    invariant_type: InvariantType
    condition: Callable[[Dict[str, Any]], bool]
    formula: str
    penalty: str = "exception"  # exception, warn, correct, audit
    metadata: Dict[str, Any] = field(default_factory=dict)
    violation_count: int = 0
    
    def validate(self, state: Dict[str, Any]) -> bool:
        """Check if invariant holds on given state."""
        try:
            return self.condition(state)
        except Exception as e:
            self.violation_count += 1
            return False


@dataclass
class GovernanceConstraint:
    """Governance rule enforced at runtime.
    
    Args:
        policy_id: References mesie.governance.policies
        constraint: Enforcing function
        enforcement_level: hard (exception), soft (warning), adaptive (adjust)
        applies_to: Which components this constraint applies to
    """
    
    policy_id: str
    constraint: Callable[[Dict[str, Any]], bool]
    enforcement_level: str = "hard"  # hard, soft, adaptive
    applies_to: List[str] = field(default_factory=list)
    violation_log: List[str] = field(default_factory=list)
    
    def enforce(self, state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Enforce constraint. Returns (is_valid, error_message)."""
        try:
            valid = self.constraint(state)
            if not valid and self.enforcement_level == "hard":
                msg = f"Governance violation: {self.policy_id}"
                self.violation_log.append(msg)
                return False, msg
            return True, None
        except Exception as e:
            return False, str(e)


class ConstraintViolationException(Exception):
    """Raised when unbreakable invariant is violated."""
    pass


class RuntimeGuardian:
    """Enforces mathematical invariants at runtime.
    
    Acts as gatekeeper for all state mutations and decisions.
    Ensures system operates only within legal mathematical bounds.
    """
    
    def __init__(self, name: str = "MESIE Guardian"):
        self.name = name
        self.invariants: Dict[str, MathematicalInvariant] = {}
        self.constraints: Dict[str, GovernanceConstraint] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        self._violation_counts = {}
    
    def register_invariant(self, invariant: MathematicalInvariant) -> None:
        """Register a mathematical invariant."""
        self.invariants[invariant.name] = invariant
        self._violation_counts[invariant.name] = 0
    
    def register_constraint(self, constraint: GovernanceConstraint) -> None:
        """Register a governance constraint."""
        self.constraints[constraint.policy_id] = constraint
    
    def validate_state(self, state: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check all invariants against state.
        
        Returns:
            (all_valid, list_of_violations)
        """
        violations = []
        
        for inv_name, invariant in self.invariants.items():
            if not invariant.validate(state):
                msg = f"Invariant violated: {inv_name}"
                violations.append(msg)
                self._violation_counts[inv_name] = self._violation_counts.get(inv_name, 0) + 1
                
                if invariant.penalty == "exception":
                    self._log_violation(msg, state)
                    raise ConstraintViolationException(msg)
        
        return len(violations) == 0, violations
    
    def enforce_constraints(self, state: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check all governance constraints.
        
        Returns:
            (all_valid, list_of_violations)
        """
        violations = []
        
        for constraint_id, constraint in self.constraints.items():
            valid, error = constraint.enforce(state)
            if not valid:
                violations.append(error or constraint_id)
        
        return len(violations) == 0, violations
    
    def authorize_action(
        self, 
        action: str, 
        parameters: Dict[str, Any],
        actor: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Authorize action under current state and constraints.
        
        Returns:
            (authorized, reason_if_denied)
        """
        state = {
            "action": action,
            "parameters": parameters,
            "actor": actor,
        }
        
        # Check invariants
        try:
            inv_valid, _ = self.validate_state(state)
        except ConstraintViolationException as e:
            return False, str(e)
        
        # Check constraints
        const_valid, violations = self.enforce_constraints(state)
        
        self._log_action(action, parameters, actor, inv_valid and const_valid)
        
        if not (inv_valid and const_valid):
            return False, "Action violates invariants or constraints"
        
        return True, None
    
    def _log_violation(self, message: str, state: Dict[str, Any]) -> None:
        """Log invariant violation."""
        entry = {
            "timestamp": np.datetime64("now"),
            "type": "violation",
            "message": message,
            "state_keys": list(state.keys()) if isinstance(state, dict) else [],
        }
        self.audit_trail.append(entry)
    
    def _log_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        actor: Optional[str],
        authorized: bool
    ) -> None:
        """Log action authorization."""
        entry = {
            "timestamp": np.datetime64("now"),
            "type": "action",
            "action": action,
            "actor": actor,
            "authorized": authorized,
            "param_keys": list(parameters.keys()) if parameters else [],
        }
        self.audit_trail.append(entry)
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get immutable audit trail of all decisions."""
        return list(self.audit_trail)  # Return copy
    
    def get_violation_summary(self) -> Dict[str, int]:
        """Get summary of invariant violations."""
        return dict(self._violation_counts)
    
    def reset_violation_counts(self) -> None:
        """Reset violation counters."""
        for key in self._violation_counts:
            self._violation_counts[key] = 0


class InvariantEnforcer:
    """Higher-level enforcer integrating with MESIE governance."""
    
    # Standard mathematical invariants
    CAPITAL_CONSERVATION = MathematicalInvariant(
        name="capital_conservation",
        invariant_type=InvariantType.CAPITAL_PRESERVATION,
        condition=lambda state: (
            "total_capital_in" in state and "total_capital_out" in state and
            abs(state["total_capital_in"] - state["total_capital_out"]) < 1e-9
        ),
        formula="Σ(capital_in) = Σ(capital_out)",
        penalty="exception"
    )
    
    PORTFOLIO_WEIGHTS = MathematicalInvariant(
        name="portfolio_weights",
        invariant_type=InvariantType.CONSISTENCY,
        condition=lambda state: (
            "weights" in state and
            abs(np.sum(state["weights"]) - 1.0) < 1e-9 and
            np.all(state["weights"] >= -1e-9)  # Allow numerical error
        ),
        formula="Σ(w_i) = 1, w_i ≥ 0",
        penalty="exception"
    )
    
    PRICE_MONOTONICITY = MathematicalInvariant(
        name="price_monotonicity",
        invariant_type=InvariantType.CAUSALITY,
        condition=lambda state: (
            "prices" in state and np.all(np.diff(state.get("prices", [1])) >= 0)
        ),
        formula="∀t: price(t) ≥ price(t-1)",
        penalty="warn"
    )
    
    def __init__(self):
        self.guardian = RuntimeGuardian("MESIE Governance Guardian")
        self._setup_standard_invariants()
    
    def _setup_standard_invariants(self) -> None:
        """Register all standard mathematical invariants."""
        self.guardian.register_invariant(self.CAPITAL_CONSERVATION)
        self.guardian.register_invariant(self.PORTFOLIO_WEIGHTS)
        self.guardian.register_invariant(self.PRICE_MONOTONICITY)
    
    def add_custom_invariant(self, invariant: MathematicalInvariant) -> None:
        """Add domain-specific invariant."""
        self.guardian.register_invariant(invariant)
    
    def add_governance_constraint(self, constraint: GovernanceConstraint) -> None:
        """Add governance constraint from policies."""
        self.guardian.register_constraint(constraint)
    
    def validate(self, state: Dict[str, Any]) -> bool:
        """Validate state against all invariants."""
        try:
            valid, violations = self.guardian.validate_state(state)
            return valid
        except ConstraintViolationException:
            return False
    
    def authorize(self, action: str, params: Dict[str, Any]) -> bool:
        """Check if action is authorized."""
        authorized, _ = self.guardian.authorize_action(action, params)
        return authorized
    
    def audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail."""
        return self.guardian.get_audit_trail()


# Global enforcer instance
_enforcer = InvariantEnforcer()


def get_enforcer() -> InvariantEnforcer:
    """Get the global invariant enforcer."""
    return _enforcer
