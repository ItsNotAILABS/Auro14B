"""Reinforcement learning for autonomous spectral sensing.

Provides RL agents for adaptive parameter tuning, active learning,
and autonomous spectral anomaly detection strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AgentType(Enum):
    """Types of RL agents available."""

    Q_LEARNING = "q_learning"
    POLICY_GRADIENT = "policy_gradient"
    BANDIT = "bandit"
    ACTOR_CRITIC = "actor_critic"


@dataclass
class RLConfig:
    """Configuration for RL agents."""

    state_dim: int = 16
    n_actions: int = 5
    learning_rate: float = 0.01
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    buffer_size: int = 10000
    batch_size: int = 32


@dataclass
class Experience:
    """Single RL experience tuple."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class AgentMetrics:
    """Training metrics for RL agents."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    epsilon_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards[-100:]))

    @property
    def total_episodes(self) -> int:
        return len(self.episode_rewards)


class QLearningAgent:
    """Q-Learning agent for discrete spectral decision making.

    Learns value function for spectral parameter selection
    using tabular Q-learning with epsilon-greedy exploration.
    """

    def __init__(self, config: Optional[RLConfig] = None) -> None:
        self.config = config or RLConfig()
        self._q_table: dict[tuple, np.ndarray] = {}
        self._metrics = AgentMetrics()
        self._current_epsilon = self.config.epsilon
        self._total_steps = 0

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for tabular Q-learning."""
        return tuple(np.round(state, 1))

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state."""
        key = self._discretize_state(state)
        if key not in self._q_table:
            self._q_table[key] = np.zeros(self.config.n_actions)
        return self._q_table[key]

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self._current_epsilon:
            return np.random.randint(self.config.n_actions)
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, experience: Experience) -> float:
        """Update Q-value from experience."""
        state_key = self._discretize_state(experience.state)
        if state_key not in self._q_table:
            self._q_table[state_key] = np.zeros(self.config.n_actions)

        current_q = self._q_table[state_key][experience.action]

        if experience.done:
            target = experience.reward
        else:
            next_q = self._get_q_values(experience.next_state)
            target = experience.reward + self.config.gamma * np.max(next_q)

        td_error = target - current_q
        self._q_table[state_key][experience.action] += self.config.learning_rate * td_error

        # Decay epsilon
        self._current_epsilon = max(
            self.config.min_epsilon,
            self._current_epsilon * self.config.epsilon_decay,
        )
        self._total_steps += 1

        return float(abs(td_error))

    def train_episode(
        self, states: np.ndarray, rewards: np.ndarray, actions: np.ndarray
    ) -> float:
        """Train on a complete episode."""
        total_reward = 0.0
        for i in range(len(states) - 1):
            exp = Experience(
                state=states[i],
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_state=states[i + 1],
                done=(i == len(states) - 2),
            )
            self.update(exp)
            total_reward += rewards[i]

        self._metrics.episode_rewards.append(total_reward)
        self._metrics.epsilon_history.append(self._current_epsilon)
        return total_reward

    @property
    def metrics(self) -> AgentMetrics:
        return self._metrics


class PolicyGradientAgent:
    """Policy gradient agent for continuous spectral optimization.

    Uses REINFORCE algorithm for learning spectral sensing policies
    in continuous action spaces.
    """

    def __init__(self, config: Optional[RLConfig] = None) -> None:
        self.config = config or RLConfig()
        self._policy_weights = np.random.randn(
            self.config.state_dim, self.config.n_actions
        ) * 0.01
        self._baseline = 0.0
        self._metrics = AgentMetrics()
        self._episode_buffer: list[Experience] = []

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax computation."""
        exp_x = np.exp(x - x.max())
        return exp_x / exp_x.sum()

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities from policy."""
        logits = state @ self._policy_weights
        return self._softmax(logits)

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy."""
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.config.n_actions, p=probs))

    def store_experience(self, experience: Experience) -> None:
        """Store experience for episode-end update."""
        self._episode_buffer.append(experience)

    def update_policy(self) -> float:
        """REINFORCE update at end of episode."""
        if not self._episode_buffer:
            return 0.0

        # Compute returns
        returns = []
        G = 0.0
        for exp in reversed(self._episode_buffer):
            G = exp.reward + self.config.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        total_loss = 0.0
        for exp, ret in zip(self._episode_buffer, returns):
            probs = self.get_action_probs(exp.state)
            grad = np.outer(exp.state, np.eye(self.config.n_actions)[exp.action] - probs)
            self._policy_weights += self.config.learning_rate * ret * grad
            total_loss += abs(ret)

        episode_reward = sum(exp.reward for exp in self._episode_buffer)
        self._metrics.episode_rewards.append(episode_reward)
        self._episode_buffer = []

        return float(total_loss / len(returns))

    @property
    def metrics(self) -> AgentMetrics:
        return self._metrics


class MultiArmedBandit:
    """Contextual bandit for spectral parameter selection.

    Selects optimal spectral analysis parameters based on
    contextual features using Thompson Sampling.
    """

    def __init__(self, n_arms: int = 5, context_dim: int = 8) -> None:
        self.n_arms = n_arms
        self.context_dim = context_dim
        self._alpha = np.ones(n_arms)  # Success counts
        self._beta = np.ones(n_arms)  # Failure counts
        self._context_weights = np.random.randn(context_dim, n_arms) * 0.01
        self._total_pulls = 0
        self._rewards_history: list[float] = []

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """Thompson Sampling arm selection."""
        if context is not None:
            # Contextual: use context to adjust sampling parameters
            adjustments = context @ self._context_weights
            adjusted_alpha = self._alpha + np.maximum(0, adjustments.flatten()[:self.n_arms])
            adjusted_beta = self._beta + np.maximum(0, -adjustments.flatten()[:self.n_arms])
            samples = np.random.beta(adjusted_alpha, adjusted_beta)
        else:
            samples = np.random.beta(self._alpha, self._beta)

        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update arm statistics with observed reward."""
        if reward > 0.5:
            self._alpha[arm] += 1
        else:
            self._beta[arm] += 1
        self._total_pulls += 1
        self._rewards_history.append(reward)

    @property
    def arm_estimates(self) -> np.ndarray:
        """Current estimated value of each arm."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def cumulative_regret(self) -> float:
        """Estimate cumulative regret."""
        if not self._rewards_history:
            return 0.0
        best_arm_value = self.arm_estimates.max()
        return float(best_arm_value * self._total_pulls - sum(self._rewards_history))


class SpectralEnvironment:
    """Simulated environment for spectral RL experiments."""

    def __init__(self, state_dim: int = 16, n_actions: int = 5) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self._state = np.random.randn(state_dim)
        self._step_count = 0
        self._max_steps = 100

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self._state = np.random.randn(self.state_dim)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Execute action and return (next_state, reward, done)."""
        # Simple dynamics: action modifies state components
        noise = np.random.randn(self.state_dim) * 0.1
        action_effect = np.zeros(self.state_dim)
        idx = action % self.state_dim
        action_effect[idx] = 0.5

        self._state = self._state * 0.95 + action_effect + noise

        # Reward based on state properties
        reward = -float(np.abs(self._state).mean()) + 1.0
        self._step_count += 1
        done = self._step_count >= self._max_steps

        return self._state.copy(), reward, done

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()
