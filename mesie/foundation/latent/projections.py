"""Modality-specific projection heads.

Implements various projection architectures for mapping
modality features into the universal latent space.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LinearProjection:
    """Simple linear projection.

    Maps input features to latent space with a single linear layer
    followed by optional normalization.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output latent dimension.
        use_bias: Whether to use bias.
        normalize: Whether to L2-normalize output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_bias: bool = True,
        normalize: bool = True,
    ):
        """Initialize linear projection.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            use_bias: Include bias term.
            normalize: L2 normalize output.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.normalize = normalize

        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weight = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros(output_dim) if use_bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project input.

        Args:
            x: Input [B, ..., input_dim].

        Returns:
            Projected output [B, ..., output_dim].
        """
        out = np.dot(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.normalize:
            out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)
        return out

    def get_params(self) -> List[np.ndarray]:
        """Get parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class MLPProjection:
    """Multi-layer perceptron projection.

    Standard MLP with configurable depth, width, activations,
    and normalization for projecting into latent space.

    Attributes:
        layers: List of layer configurations.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        activation: str = "gelu",
        dropout: float = 0.1,
        normalize_output: bool = True,
        use_residual: bool = False,
    ):
        """Initialize MLP projection.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            hidden_dim: Hidden layer dimension (default: 4x output_dim).
            num_layers: Number of layers.
            activation: Activation function.
            dropout: Dropout rate.
            normalize_output: Whether to normalize output.
            use_residual: Whether to use residual connections.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or (output_dim * 4)
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        self.normalize_output = normalize_output
        self.use_residual = use_residual

        # Build layers
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.layer_norms: List[Dict[str, np.ndarray]] = []

        dims = [input_dim] + [self.hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))
            if i < len(dims) - 2:  # No norm on last layer
                self.layer_norms.append({
                    "gamma": np.ones(fan_out),
                    "beta": np.zeros(fan_out),
                })

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through MLP.

        Args:
            x: Input [B, ..., input_dim].
            training: Training mode.

        Returns:
            Output [B, ..., output_dim].
        """
        h = x
        for i in range(len(self.weights)):
            # Store for residual
            residual = h if self.use_residual and h.shape[-1] == self.weights[i].shape[1] else None

            # Linear
            h = np.dot(h, self.weights[i]) + self.biases[i]

            # Layer norm + activation for all but last
            if i < len(self.weights) - 1:
                # Layer norm
                if i < len(self.layer_norms):
                    h = self._layer_norm(h, self.layer_norms[i])

                # Activation
                h = self._activate(h)

                # Dropout
                if training and self.dropout > 0:
                    mask = np.random.binomial(1, 1 - self.dropout, h.shape)
                    h = h * mask / (1 - self.dropout)

                # Residual
                if residual is not None:
                    h = h + residual

        if self.normalize_output:
            h = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)

        return h

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "gelu":
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "silu":
            return x * (1 / (1 + np.exp(-x)))
        elif self.activation == "tanh":
            return np.tanh(x)
        return x

    def _layer_norm(self, x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return params["gamma"] * x_norm + params["beta"]

    def get_params(self) -> List[np.ndarray]:
        """Get all parameters."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        for ln in self.layer_norms:
            params.extend([ln["gamma"], ln["beta"]])
        return params


class GatedProjection:
    """Gated projection with learned information filtering.

    Uses a gating mechanism to selectively pass information
    from the input to the output, allowing the network to
    learn which features are most relevant for each modality.

    Attributes:
        input_dim: Input dimension.
        output_dim: Output dimension.
        gate_type: Type of gating mechanism.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        gate_type: str = "sigmoid",
        num_gates: int = 1,
    ):
        """Initialize gated projection.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            hidden_dim: Hidden dimension.
            gate_type: Gating type ('sigmoid', 'softmax', 'sparsemax').
            num_gates: Number of gate layers.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        self.gate_type = gate_type
        self.num_gates = num_gates

        # Value projection
        scale = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        self.value_weight = np.random.randn(input_dim, self.hidden_dim) * scale
        self.value_bias = np.zeros(self.hidden_dim)

        # Gate projection
        self.gate_weight = np.random.randn(input_dim, self.hidden_dim) * scale
        self.gate_bias = np.zeros(self.hidden_dim)

        # Output projection
        scale_out = np.sqrt(2.0 / (self.hidden_dim + output_dim))
        self.output_weight = np.random.randn(self.hidden_dim, output_dim) * scale_out
        self.output_bias = np.zeros(output_dim)

        # Additional gates for multi-gate version
        self.extra_gates: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(num_gates - 1):
            self.extra_gates.append((
                np.random.randn(self.hidden_dim, self.hidden_dim) * scale,
                np.zeros(self.hidden_dim),
            ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with gating.

        Args:
            x: Input [B, ..., input_dim].

        Returns:
            Gated output [B, ..., output_dim].
        """
        # Compute value and gate
        value = np.dot(x, self.value_weight) + self.value_bias
        gate = np.dot(x, self.gate_weight) + self.gate_bias

        # Apply gate activation
        if self.gate_type == "sigmoid":
            gate = 1 / (1 + np.exp(-gate))
        elif self.gate_type == "softmax":
            max_g = np.max(gate, axis=-1, keepdims=True)
            exp_g = np.exp(gate - max_g)
            gate = exp_g / (np.sum(exp_g, axis=-1, keepdims=True) + 1e-10)
        elif self.gate_type == "sparsemax":
            gate = self._sparsemax(gate)

        # Apply gating
        hidden = value * gate

        # Additional gate layers
        for gate_w, gate_b in self.extra_gates:
            extra_gate = 1 / (1 + np.exp(-(np.dot(hidden, gate_w) + gate_b)))
            hidden = hidden * extra_gate

        # Activation
        hidden = hidden * np.tanh(hidden)  # Squared tanh gating

        # Output projection
        output = np.dot(hidden, self.output_weight) + self.output_bias

        # Normalize
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-8)

        return output

    def _sparsemax(self, x: np.ndarray) -> np.ndarray:
        """Sparsemax activation (sparse version of softmax)."""
        sorted_x = np.sort(x, axis=-1)[:, ::-1]
        cumsum = np.cumsum(sorted_x, axis=-1)
        k = np.arange(1, x.shape[-1] + 1)
        threshold = (cumsum - 1) / k
        mask = sorted_x > threshold
        # Find support size
        support = np.sum(mask, axis=-1, keepdims=True)
        tau_idx = np.clip(support - 1, 0, x.shape[-1] - 1)
        tau = np.take_along_axis(threshold, tau_idx.astype(int), axis=-1)
        return np.maximum(x - tau, 0)

    def get_params(self) -> List[np.ndarray]:
        """Get all parameters."""
        params = [
            self.value_weight, self.value_bias,
            self.gate_weight, self.gate_bias,
            self.output_weight, self.output_bias,
        ]
        for gw, gb in self.extra_gates:
            params.extend([gw, gb])
        return params


class ModalityAdaptiveProjection:
    """Modality-adaptive projection with dynamic routing.

    Uses a routing mechanism to adapt the projection based on
    the detected modality characteristics. Combines multiple
    expert projections with learned routing weights.

    Attributes:
        num_experts: Number of expert projections.
        input_dim: Input dimension.
        output_dim: Output dimension.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        hidden_dim: Optional[int] = None,
        top_k: int = 2,
    ):
        """Initialize adaptive projection.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            num_experts: Number of expert projections.
            hidden_dim: Hidden dimension per expert.
            top_k: Number of experts to activate per sample.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim or output_dim
        self.top_k = top_k

        # Router
        self.router_weight = np.random.randn(input_dim, num_experts) * 0.01

        # Expert projections
        self.experts: List[Dict[str, np.ndarray]] = []
        for _ in range(num_experts):
            scale_in = np.sqrt(2.0 / (input_dim + self.hidden_dim))
            scale_out = np.sqrt(2.0 / (self.hidden_dim + output_dim))
            self.experts.append({
                "w1": np.random.randn(input_dim, self.hidden_dim) * scale_in,
                "b1": np.zeros(self.hidden_dim),
                "w2": np.random.randn(self.hidden_dim, output_dim) * scale_out,
                "b2": np.zeros(output_dim),
            })

        # Shared projection (always active)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.shared_weight = np.random.randn(input_dim, output_dim) * scale
        self.shared_bias = np.zeros(output_dim)
        self.shared_ratio = 0.2  # Weight for shared projection

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward with adaptive routing.

        Args:
            x: Input [B, ..., input_dim].

        Returns:
            Adaptively projected output [B, ..., output_dim].
        """
        original_shape = x.shape
        # Flatten to 2D for routing
        x_flat = x.reshape(-1, self.input_dim)
        batch_size = x_flat.shape[0]

        # Compute routing weights
        router_logits = np.dot(x_flat, self.router_weight)  # [B, num_experts]

        # Top-k selection
        top_k_indices = np.argsort(router_logits, axis=-1)[:, -self.top_k:]
        top_k_mask = np.zeros_like(router_logits)
        for i in range(batch_size):
            top_k_mask[i, top_k_indices[i]] = 1.0

        # Softmax over selected experts
        masked_logits = router_logits * top_k_mask + (1 - top_k_mask) * (-1e9)
        max_logits = np.max(masked_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(masked_logits - max_logits) * top_k_mask
        routing_weights = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        # Compute expert outputs
        output = np.zeros((batch_size, self.output_dim))

        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            # Two-layer MLP per expert
            h = np.dot(x_flat, expert["w1"]) + expert["b1"]
            h = h * (1 / (1 + np.exp(-h)))  # SiLU activation
            expert_out = np.dot(h, expert["w2"]) + expert["b2"]

            # Weight by routing
            output += expert_out * routing_weights[:, expert_idx:expert_idx+1]

        # Add shared projection
        shared_out = np.dot(x_flat, self.shared_weight) + self.shared_bias
        output = (1 - self.shared_ratio) * output + self.shared_ratio * shared_out

        # Normalize
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-8)

        # Reshape back
        output_shape = list(original_shape[:-1]) + [self.output_dim]
        return output.reshape(output_shape)

    def get_load_balance_loss(self, x: np.ndarray) -> float:
        """Compute load balance loss for expert utilization.

        Encourages uniform usage of experts.

        Args:
            x: Input batch.

        Returns:
            Load balance loss.
        """
        x_flat = x.reshape(-1, self.input_dim)
        router_logits = np.dot(x_flat, self.router_weight)

        # Compute routing probabilities
        max_logits = np.max(router_logits, axis=-1, keepdims=True)
        probs = np.exp(router_logits - max_logits)
        probs = probs / (np.sum(probs, axis=-1, keepdims=True) + 1e-10)

        # Average routing probability per expert
        avg_prob = np.mean(probs, axis=0)

        # Fraction of samples routed to each expert
        top_expert = np.argmax(router_logits, axis=-1)
        expert_counts = np.bincount(top_expert, minlength=self.num_experts)
        fraction = expert_counts / (len(top_expert) + 1e-10)

        # Load balance loss (product of avg_prob and fraction)
        balance_loss = float(
            self.num_experts * np.sum(avg_prob * fraction)
        )

        return balance_loss

    def get_params(self) -> List[np.ndarray]:
        """Get all parameters."""
        params = [self.router_weight, self.shared_weight, self.shared_bias]
        for expert in self.experts:
            params.extend([expert["w1"], expert["b1"], expert["w2"], expert["b2"]])
        return params


class ProjectionFactory:
    """Factory for creating projection heads."""

    _registry = {
        "linear": LinearProjection,
        "mlp": MLPProjection,
        "gated": GatedProjection,
        "adaptive": ModalityAdaptiveProjection,
    }

    @classmethod
    def create(cls, projection_type: str, **kwargs) -> Any:
        """Create a projection head.

        Args:
            projection_type: Type of projection.
            **kwargs: Projection parameters.

        Returns:
            Initialized projection.
        """
        if projection_type not in cls._registry:
            raise ValueError(
                f"Unknown projection: {projection_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[projection_type](**kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Any:
        """Create from config dict."""
        config = dict(config)
        proj_type = config.pop("type", "mlp")
        return cls.create(proj_type, **config)

    @classmethod
    def available(cls) -> List[str]:
        """List available projections."""
        return list(cls._registry.keys())
