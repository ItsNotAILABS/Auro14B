"""Mixture of Experts (MoE) layers for SpectralGPT.

Implements modality-aware expert routing that allows the model to
specialize different expert networks for different spectral domains
while sharing common representational structure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ExpertLayer:
    """A single expert network in the MoE layer.

    Each expert is a feed-forward network that specializes in
    processing certain types of spectral patterns.

    Attributes:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        activation: Activation function.
        expert_id: Unique expert identifier.
        specialization: Optional specialization label.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 4096,
        output_dim: int = 1024,
        activation: str = "swiglu",
        expert_id: int = 0,
        specialization: Optional[str] = None,
    ):
        """Initialize expert layer.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output feature dimension.
            activation: Activation function name.
            expert_id: Unique expert ID.
            specialization: Optional domain specialization.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.expert_id = expert_id
        self.specialization = specialization

        # Gated FFN weights
        self.gate_proj = np.random.randn(input_dim, hidden_dim) * 0.02
        self.up_proj = np.random.randn(input_dim, hidden_dim) * 0.02
        self.down_proj = np.random.randn(hidden_dim, output_dim) * 0.02

        # Statistics tracking
        self.total_tokens_processed = 0
        self.load_history: List[float] = []

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation in ("swiglu", "silu"):
            return x * (1.0 / (1.0 + np.exp(-x)))
        elif self.activation in ("geglu", "gelu"):
            return 0.5 * x * (1.0 + np.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through expert.

        Args:
            x: Input [..., input_dim].

        Returns:
            Output [..., output_dim].
        """
        # Gated architecture
        gate = np.einsum("...d,dh->...h", x, self.gate_proj)
        gate = self._activate(gate)

        up = np.einsum("...d,dh->...h", x, self.up_proj)
        hidden = gate * up

        output = np.einsum("...h,ho->...o", hidden, self.down_proj)

        self.total_tokens_processed += np.prod(x.shape[:-1])
        return output

    def get_statistics(self) -> Dict[str, Any]:
        """Get expert utilization statistics."""
        return {
            "expert_id": self.expert_id,
            "specialization": self.specialization,
            "total_tokens": self.total_tokens_processed,
            "param_count": (
                self.input_dim * self.hidden_dim * 2 + self.hidden_dim * self.output_dim
            ),
        }


class TopKRouter:
    """Top-K expert routing mechanism.

    Routes each token to the top-K experts based on learned routing
    scores, with optional load balancing and capacity constraints.

    Attributes:
        input_dim: Input dimension for routing.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        capacity_factor: Capacity factor for expert buffers.
        noise_std: Standard deviation of routing noise.
        use_aux_loss: Whether to compute auxiliary load balancing loss.
        jitter_noise: Whether to add jitter noise during routing.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noise_std: float = 0.1,
        use_aux_loss: bool = True,
        jitter_noise: bool = True,
    ):
        """Initialize Top-K router.

        Args:
            input_dim: Input dimension.
            num_experts: Number of available experts.
            top_k: Number of experts activated per token.
            capacity_factor: Expert capacity factor.
            noise_std: Noise for exploration.
            use_aux_loss: Whether to use load balancing loss.
            jitter_noise: Whether to add jitter noise.
        """
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.use_aux_loss = use_aux_loss
        self.jitter_noise = jitter_noise

        # Router weights
        self.router_weights = np.random.randn(input_dim, num_experts) * 0.02
        self.router_bias = np.zeros(num_experts)

        # Load balancing tracking
        self.expert_counts = np.zeros(num_experts)
        self.total_routed = 0

    def route(
        self, x: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Route tokens to experts.

        Args:
            x: Input tokens [..., input_dim].
            training: Whether in training mode (affects noise).

        Returns:
            Tuple of:
            - expert_indices: [..., top_k] expert assignments
            - expert_weights: [..., top_k] routing weights
            - routing_info: Dictionary with routing statistics
        """
        # Compute router logits
        logits = np.einsum("...d,de->...e", x, self.router_weights) + self.router_bias

        # Add noise during training
        if training and self.jitter_noise:
            noise = np.random.randn(*logits.shape) * self.noise_std
            logits = logits + noise

        # Softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        # Top-K selection
        flat_probs = probs.reshape(-1, self.num_experts) if probs.ndim > 1 else probs.reshape(1, -1)
        num_tokens = flat_probs.shape[0]

        top_k_indices = np.zeros((num_tokens, self.top_k), dtype=np.int64)
        top_k_weights = np.zeros((num_tokens, self.top_k))

        for t in range(num_tokens):
            indices = np.argsort(flat_probs[t])[::-1][:self.top_k]
            top_k_indices[t] = indices
            top_k_weights[t] = flat_probs[t][indices]

        # Normalize weights
        weight_sum = np.sum(top_k_weights, axis=-1, keepdims=True) + 1e-10
        top_k_weights = top_k_weights / weight_sum

        # Reshape back
        original_shape = x.shape[:-1]
        expert_indices = top_k_indices.reshape(*original_shape, self.top_k)
        expert_weights = top_k_weights.reshape(*original_shape, self.top_k)

        # Update statistics
        for idx in top_k_indices.flatten():
            self.expert_counts[idx] += 1
        self.total_routed += num_tokens

        # Compute routing info
        routing_info: Dict[str, Any] = {
            "expert_utilization": self.expert_counts / (self.total_routed + 1e-10),
            "load_balance_loss": self._compute_load_balance_loss(probs),
            "router_entropy": self._compute_entropy(probs),
        }

        return expert_indices, expert_weights, routing_info

    def _compute_load_balance_loss(self, probs: np.ndarray) -> float:
        """Compute load balancing auxiliary loss.

        Args:
            probs: Routing probabilities.

        Returns:
            Load balance loss value.
        """
        if not self.use_aux_loss:
            return 0.0

        # Fraction of tokens routed to each expert
        flat_probs = probs.reshape(-1, self.num_experts)
        num_tokens = flat_probs.shape[0]

        # Expert fraction (how much probability mass goes to each expert)
        expert_fraction = np.mean(flat_probs, axis=0)

        # Ideal uniform distribution
        uniform = np.ones(self.num_experts) / self.num_experts

        # Loss encourages uniform distribution
        loss = float(self.num_experts * np.sum(expert_fraction * expert_fraction))
        return loss

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute routing entropy (higher = more diverse routing)."""
        flat_probs = probs.reshape(-1, self.num_experts)
        entropy = -np.sum(flat_probs * np.log(flat_probs + 1e-10), axis=-1)
        return float(np.mean(entropy))

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self.expert_counts = np.zeros(self.num_experts)
        self.total_routed = 0


class ModalityAwareRouter(TopKRouter):
    """Modality-aware expert router.

    Extends TopKRouter with modality-specific routing that can
    bias certain experts toward specific spectral domains while
    still allowing cross-modality sharing.

    Attributes:
        modality_embeddings: Learned embeddings for each modality.
        modality_bias: Per-modality bias for expert selection.
        cross_modality_sharing: Degree of cross-modality expert sharing.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        num_modalities: int = 7,
        modality_names: Optional[List[str]] = None,
        cross_modality_sharing: float = 0.3,
        **kwargs,
    ):
        """Initialize modality-aware router.

        Args:
            input_dim: Input dimension.
            num_experts: Number of experts.
            top_k: Experts per token.
            num_modalities: Number of modalities.
            modality_names: Names for each modality.
            cross_modality_sharing: Sharing ratio (0=no sharing, 1=full sharing).
            **kwargs: Additional TopKRouter arguments.
        """
        super().__init__(input_dim, num_experts, top_k, **kwargs)

        self.num_modalities = num_modalities
        self.modality_names = modality_names or [
            "seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"
        ]
        self.cross_modality_sharing = cross_modality_sharing

        # Modality embeddings
        self.modality_embeddings = np.random.randn(
            num_modalities, input_dim
        ) * 0.02

        # Modality-expert affinity matrix
        self.modality_expert_affinity = self._initialize_affinity()

        # Modality-specific biases
        self.modality_bias = np.zeros((num_modalities, num_experts))

    def _initialize_affinity(self) -> np.ndarray:
        """Initialize modality-expert affinity matrix.

        Creates an initial affinity that assigns primary experts to
        each modality while allowing some sharing.

        Returns:
            Affinity matrix [num_modalities, num_experts].
        """
        affinity = np.ones((self.num_modalities, self.num_experts)) * self.cross_modality_sharing

        # Assign primary experts to modalities
        experts_per_modality = max(1, self.num_experts // self.num_modalities)
        for m in range(self.num_modalities):
            start_expert = (m * experts_per_modality) % self.num_experts
            for e in range(experts_per_modality):
                expert_idx = (start_expert + e) % self.num_experts
                affinity[m, expert_idx] = 1.0

        # Normalize
        affinity = affinity / affinity.sum(axis=-1, keepdims=True)
        return affinity

    def route_with_modality(
        self,
        x: np.ndarray,
        modality_id: int,
        training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Route tokens with modality awareness.

        Args:
            x: Input tokens [..., input_dim].
            modality_id: Index of the current modality.
            training: Whether in training mode.

        Returns:
            Same as TopKRouter.route() plus modality info.
        """
        # Add modality embedding
        modality_emb = self.modality_embeddings[modality_id]
        x_modality = x + modality_emb

        # Compute base routing logits
        logits = np.einsum("...d,de->...e", x_modality, self.router_weights) + self.router_bias

        # Apply modality bias
        logits = logits + self.modality_bias[modality_id]

        # Apply modality-expert affinity
        affinity = self.modality_expert_affinity[modality_id]
        logits = logits * affinity

        # Add noise
        if training and self.jitter_noise:
            noise = np.random.randn(*logits.shape) * self.noise_std
            logits = logits + noise

        # Softmax and top-K (same as parent)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        flat_probs = probs.reshape(-1, self.num_experts) if probs.ndim > 1 else probs.reshape(1, -1)
        num_tokens = flat_probs.shape[0]

        top_k_indices = np.zeros((num_tokens, self.top_k), dtype=np.int64)
        top_k_weights = np.zeros((num_tokens, self.top_k))

        for t in range(num_tokens):
            indices = np.argsort(flat_probs[t])[::-1][:self.top_k]
            top_k_indices[t] = indices
            top_k_weights[t] = flat_probs[t][indices]

        weight_sum = np.sum(top_k_weights, axis=-1, keepdims=True) + 1e-10
        top_k_weights = top_k_weights / weight_sum

        original_shape = x.shape[:-1]
        expert_indices = top_k_indices.reshape(*original_shape, self.top_k)
        expert_weights = top_k_weights.reshape(*original_shape, self.top_k)

        routing_info: Dict[str, Any] = {
            "modality": self.modality_names[modality_id],
            "modality_id": modality_id,
            "expert_utilization": self.expert_counts / (self.total_routed + 1e-10),
            "modality_affinity": affinity.tolist(),
            "load_balance_loss": self._compute_load_balance_loss(probs),
        }

        return expert_indices, expert_weights, routing_info


class MixtureOfExperts:
    """Full Mixture of Experts layer for SpectralGPT.

    Combines multiple expert networks with a routing mechanism to
    create a sparse, modality-aware processing layer.

    Attributes:
        hidden_dim: Model hidden dimension.
        num_experts: Number of expert networks.
        top_k: Number of active experts per token.
        expert_dim: Hidden dimension within each expert.
        modality_aware: Whether to use modality-aware routing.
        capacity_factor: Capacity factor for load balancing.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dim: int = 4096,
        modality_aware: bool = True,
        num_modalities: int = 7,
        capacity_factor: float = 1.25,
        activation: str = "swiglu",
        noise_std: float = 0.1,
    ):
        """Initialize MoE layer.

        Args:
            hidden_dim: Model hidden dimension.
            num_experts: Number of experts.
            top_k: Active experts per token.
            expert_dim: Expert hidden dimension.
            modality_aware: Whether to use modality routing.
            num_modalities: Number of modalities.
            capacity_factor: Expert capacity factor.
            activation: Expert activation function.
            noise_std: Router noise.
        """
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim
        self.modality_aware = modality_aware

        # Modality labels for experts
        modality_names = [
            "seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"
        ]

        # Create experts
        self.experts = []
        for i in range(num_experts):
            specialization = modality_names[i % len(modality_names)] if i < len(modality_names) else None
            expert = ExpertLayer(
                input_dim=hidden_dim,
                hidden_dim=expert_dim,
                output_dim=hidden_dim,
                activation=activation,
                expert_id=i,
                specialization=specialization,
            )
            self.experts.append(expert)

        # Router
        if modality_aware:
            self.router = ModalityAwareRouter(
                input_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_modalities=num_modalities,
                capacity_factor=capacity_factor,
                noise_std=noise_std,
            )
        else:
            self.router = TopKRouter(
                input_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                noise_std=noise_std,
            )

        # Shared expert (always active)
        self.shared_expert = ExpertLayer(
            input_dim=hidden_dim,
            hidden_dim=expert_dim // 2,
            output_dim=hidden_dim,
            activation=activation,
            expert_id=-1,
            specialization="shared",
        )
        self.shared_weight = 0.1

    def forward(
        self,
        x: np.ndarray,
        modality_id: Optional[int] = None,
        training: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass through MoE layer.

        Args:
            x: Input [..., hidden_dim].
            modality_id: Optional modality index for routing.
            training: Whether in training mode.

        Returns:
            Tuple of (output, moe_info).
        """
        original_shape = x.shape

        # Route tokens
        if self.modality_aware and modality_id is not None:
            expert_indices, expert_weights, routing_info = self.router.route_with_modality(
                x, modality_id, training
            )
        else:
            expert_indices, expert_weights, routing_info = self.router.route(x, training)

        # Flatten for processing
        flat_x = x.reshape(-1, self.hidden_dim)
        flat_indices = expert_indices.reshape(-1, self.top_k)
        flat_weights = expert_weights.reshape(-1, self.top_k)
        num_tokens = flat_x.shape[0]

        # Process through experts
        output = np.zeros_like(flat_x)

        for k in range(self.top_k):
            for e in range(self.num_experts):
                # Find tokens routed to this expert
                mask = flat_indices[:, k] == e
                if not np.any(mask):
                    continue

                # Get tokens for this expert
                expert_input = flat_x[mask]
                expert_output = self.experts[e].forward(expert_input)

                # Weight and accumulate
                weights = flat_weights[mask, k:k + 1]
                output[mask] += expert_output * weights

        # Add shared expert contribution
        shared_output = self.shared_expert.forward(flat_x)
        output = output + self.shared_weight * shared_output

        # Reshape back
        output = output.reshape(original_shape)

        # Compute MoE info
        moe_info: Dict[str, Any] = {
            "routing_info": routing_info,
            "expert_load": {
                f"expert_{i}": int(self.experts[i].total_tokens_processed)
                for i in range(self.num_experts)
            },
            "load_balance_loss": routing_info.get("load_balance_loss", 0.0),
        }

        return output, moe_info

    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get statistics for all experts.

        Returns:
            Dictionary with per-expert statistics.
        """
        stats = {}
        for expert in self.experts:
            stats[f"expert_{expert.expert_id}"] = expert.get_statistics()
        stats["shared_expert"] = self.shared_expert.get_statistics()
        stats["router_entropy"] = float(
            -np.sum(
                (self.router.expert_counts / (self.router.total_routed + 1e-10))
                * np.log(self.router.expert_counts / (self.router.total_routed + 1e-10) + 1e-10)
            )
        )
        return stats

    def reset_statistics(self) -> None:
        """Reset all tracking statistics."""
        for expert in self.experts:
            expert.total_tokens_processed = 0
            expert.load_history = []
        self.shared_expert.total_tokens_processed = 0
        self.router.reset_statistics()
