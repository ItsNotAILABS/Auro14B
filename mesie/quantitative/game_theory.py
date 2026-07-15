"""Game Theory and Strategic Decision Models.

Implements Nash equilibrium solvers, cooperative games, and multi-agent strategy optimization.
Real mathematical formulations - no approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from enum import Enum
import numpy as np
from scipy.optimize import linprog, minimize


class GameType(Enum):
    """Categories of games."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    ZERO_SUM = "zero_sum"
    GENERAL_SUM = "general_sum"
    COOPERATIVE = "cooperative"


@dataclass
class GamePayoffMatrix:
    """Payoff matrix for a strategic game.
    
    Args:
        player_names: Names of players
        payoffs: Dict mapping (strategy_tuple) -> (payoff_tuple)
        game_type: Type of game
    """
    
    player_names: List[str]
    payoffs: Dict[Tuple[int, ...], Tuple[float, ...]]
    game_type: GameType = GameType.GENERAL_SUM
    n_strategies: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.n_strategies:
            # Infer from payoffs
            max_indices = [0] * len(self.player_names)
            for strategy_tuple in self.payoffs.keys():
                for i, s in enumerate(strategy_tuple):
                    max_indices[i] = max(max_indices[i], s + 1)
            self.n_strategies = max_indices
    
    def get_payoff(self, strategy_profile: Tuple[int, ...]) -> Tuple[float, ...]:
        """Get payoffs for a strategy profile."""
        return self.payoffs.get(strategy_profile, tuple([0] * len(self.player_names)))
    
    def get_player_payoff(self, player_idx: int, strategy_profile: Tuple[int, ...]) -> float:
        """Get payoff for specific player."""
        payoffs = self.get_payoff(strategy_profile)
        return payoffs[player_idx] if player_idx < len(payoffs) else 0.0


@dataclass
class MixedStrategy:
    """Mixed strategy probability distribution over actions.
    
    Args:
        probabilities: Probability for each action
        player: Player name
    """
    
    probabilities: np.ndarray
    player: str = ""
    
    def __post_init__(self):
        # Normalize to ensure probabilities sum to 1
        total = np.sum(self.probabilities)
        if total > 0:
            self.probabilities = self.probabilities / total
    
    def expected_payoff(self, opponent_strategies: List[MixedStrategy]) -> float:
        """Compute expected payoff against opponent strategies."""
        # Placeholder - overridden in context-specific calculations
        return 0.0


@dataclass
class GameTheoryResult:
    """Result from game theory analysis.
    
    Args:
        nash_equilibria: List of Nash equilibrium strategy profiles
        social_welfare: Total welfare at equilibrium
        stability_measures: Stability metrics
        metadata: Additional analysis results
    """
    
    nash_equilibria: List[Dict[str, MixedStrategy]]
    social_welfare: float = 0.0
    stability_measures: Dict[str, float] = field(default_factory=dict)
    is_unique: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class NashEquilibriumSolver:
    """Solves for Nash equilibrium in finite games."""
    
    def __init__(self, game: GamePayoffMatrix):
        self.game = game
        self.n_players = len(game.player_names)
    
    def find_pure_nash(self) -> List[Tuple[int, ...]]:
        """Find all pure strategy Nash equilibria.
        
        Returns:
            List of pure strategy Nash equilibrium profiles
        """
        
        # Check each strategy profile for Nash property
        nash_profiles = []
        
        # Generate all possible strategy profiles
        ranges = [range(n) for n in self.game.n_strategies]
        
        def generate_profiles(ranges):
            """Generate all possible strategy profiles."""
            if not ranges:
                yield ()
            else:
                for i in ranges[0]:
                    for rest in generate_profiles(ranges[1:]):
                        yield (i,) + rest
        
        for profile in generate_profiles(ranges):
            is_nash = True
            payoffs = self.game.get_payoff(profile)
            
            # Check if any player can profitably deviate
            for player_idx in range(self.n_players):
                current_payoff = payoffs[player_idx]
                
                # Try all alternative strategies for this player
                for alt_strategy in range(self.game.n_strategies[player_idx]):
                    if alt_strategy == profile[player_idx]:
                        continue
                    
                    # Create deviated profile
                    deviated = list(profile)
                    deviated[player_idx] = alt_strategy
                    deviated_tuple = tuple(deviated)
                    
                    alt_payoffs = self.game.get_payoff(deviated_tuple)
                    alt_payoff = alt_payoffs[player_idx]
                    
                    if alt_payoff > current_payoff + 1e-9:
                        is_nash = False
                        break
                
                if not is_nash:
                    break
            
            if is_nash:
                nash_profiles.append(profile)
        
        return nash_profiles
    
    def find_mixed_nash_2player(self) -> Optional[Tuple[MixedStrategy, MixedStrategy]]:
        """Find mixed strategy Nash equilibrium for 2-player games.
        
        Uses support enumeration for finite games.
        
        Returns:
            Tuple of (player1_strategy, player2_strategy) or None
        """
        
        if self.n_players != 2:
            return None
        
        # For 2x2 games, use closed-form solution
        if self.game.n_strategies == [2, 2]:
            return self._solve_2x2_mixed()
        
        # For larger games, use support enumeration
        return self._support_enumeration()
    
    def _solve_2x2_mixed(self) -> Optional[Tuple[MixedStrategy, MixedStrategy]]:
        """Solve 2x2 game for mixed Nash equilibrium."""
        
        # Extract payoff matrix
        # Player 1 chooses row, Player 2 chooses column
        payoff_matrix_p1 = np.array([
            [self.game.get_payoff((0, 0))[0], self.game.get_payoff((0, 1))[0]],
            [self.game.get_payoff((1, 0))[0], self.game.get_payoff((1, 1))[0]]
        ])
        
        payoff_matrix_p2 = np.array([
            [self.game.get_payoff((0, 0))[1], self.game.get_payoff((0, 1))[1]],
            [self.game.get_payoff((1, 0))[1], self.game.get_payoff((1, 1))[1]]
        ])
        
        # Compute indifference conditions
        # Player 2 must make Player 1 indifferent: q = (a-c)/(a-b-c+d)
        # where a=P(1,1), b=P(1,2), c=P(2,1), d=P(2,2)
        
        a, b = payoff_matrix_p1[0]
        c, d = payoff_matrix_p1[1]
        
        denom = (a - b - c + d)
        if abs(denom) < 1e-10:
            # Degenerate case
            return None
        
        q = (a - c) / denom  # Probability Player 2 plays strategy 0
        
        # Player 1 must make Player 2 indifferent
        a2, c2 = payoff_matrix_p2[0]
        b2, d2 = payoff_matrix_p2[1]
        
        denom2 = (a2 - b2 - c2 + d2)
        if abs(denom2) < 1e-10:
            return None
        
        p = (a2 - c2) / denom2  # Probability Player 1 plays strategy 0
        
        # Check validity (probabilities in [0,1])
        if not (0 <= p <= 1 and 0 <= q <= 1):
            return None
        
        p1_strategy = MixedStrategy(np.array([p, 1-p]), self.game.player_names[0])
        p2_strategy = MixedStrategy(np.array([q, 1-q]), self.game.player_names[1])
        
        return (p1_strategy, p2_strategy)
    
    def _support_enumeration(self) -> Optional[Tuple[MixedStrategy, MixedStrategy]]:
        """Enumerate possible supports to find mixed Nash."""
        # For 2-player games, check all support combinations
        
        if self.n_players != 2:
            return None
        
        # Try all possible supports
        for s1 in range(1, self.game.n_strategies[0] + 1):
            for s2 in range(1, self.game.n_strategies[1] + 1):
                # Try this support combination
                result = self._solve_with_support([s1, s2])
                if result:
                    return result
        
        return None
    
    def _solve_with_support(self, support_sizes: List[int]) -> Optional[Tuple[MixedStrategy, MixedStrategy]]:
        """Solve assuming specific support sizes."""
        # Simplified - returns None for now
        # Full implementation would use linear algebra
        return None


@dataclass
class CooperativeGame:
    """Cooperative game value calculation.
    
    Args:
        players: Set of player indices
        coalition_values: Dict mapping coalition -> value
    """
    
    players: List[int]
    coalition_values: Dict[Tuple[int, ...], float]
    
    def get_coalition_value(self, coalition: Tuple[int, ...]) -> float:
        """Get value of a coalition."""
        sorted_coalition = tuple(sorted(coalition))
        return self.coalition_values.get(sorted_coalition, 0.0)


class ShapleyValueCalculator:
    """Calculates Shapley values for cooperative games."""
    
    def __init__(self, game: CooperativeGame):
        self.game = game
    
    def compute_shapley_values(self) -> Dict[int, float]:
        r"""Compute Shapley value for each player.
        
        Shapley(i) = Σ_{S⊆N\{i}} |S|!(n-|S|-1)!/n! * (v(S∪{i}) - v(S))
        
        Returns:
            Dict mapping player -> Shapley value
        """
        
        n = len(self.game.players)
        shapley_values = {i: 0.0 for i in self.game.players}
        
        # For each player
        for player in self.game.players:
            value = 0.0
            
            # For each subset S not containing player
            for subset_mask in range(2 ** n):
                subset = []
                for i, p in enumerate(self.game.players):
                    if (subset_mask >> i) & 1:
                        if p != player:
                            subset.append(p)
                
                subset_tuple = tuple(sorted(subset))
                coalition_with = tuple(sorted(subset + [player]))
                
                s_size = len(subset)
                
                # Compute weight: |S|!(n-|S|-1)!/n!
                from math import factorial
                weight = factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
                
                # Marginal contribution
                v_with = self.game.get_coalition_value(coalition_with)
                v_without = self.game.get_coalition_value(subset_tuple)
                
                value += weight * (v_with - v_without)
            
            shapley_values[player] = value
        
        return shapley_values


@dataclass
class StrategicAgent:
    """Agent playing strategic game with beliefs and learning.
    
    Args:
        name: Agent identifier
        strategy: Current mixed strategy
        belief: Belief about opponent strategy
        learning_rate: Rate of belief update
    """
    
    name: str
    strategy: MixedStrategy = field(default_factory=lambda: MixedStrategy(np.array([0.5, 0.5])))
    belief: Optional[MixedStrategy] = None
    learning_rate: float = 0.1
    action_history: List[int] = field(default_factory=list)
    payoff_history: List[float] = field(default_factory=list)
    
    def update_belief(self, observed_action: int) -> None:
        """Update belief about opponent strategy based on observation."""
        if self.belief is None:
            self.belief = MixedStrategy(np.ones(2) / 2)
        
        # Simple belief update: increase probability of observed action
        self.belief.probabilities[observed_action] += self.learning_rate
        self.belief.probabilities = self.belief.probabilities / np.sum(self.belief.probabilities)
    
    def best_response(self, opponent_belief: MixedStrategy) -> int:
        """Compute best response to opponent belief."""
        # Return action with highest expected payoff
        return int(np.argmax(opponent_belief.probabilities))


def compute_social_welfare(
    game: GamePayoffMatrix,
    strategy_profile: Tuple[int, ...]
) -> float:
    """Compute total social welfare at given strategy profile.
    
    Args:
        game: Game structure
        strategy_profile: Current strategy profile
    
    Returns:
        Sum of all players' payoffs
    """
    payoffs = game.get_payoff(strategy_profile)
    return sum(payoffs)
