"""Optimizers for spectral foundation model pretraining.

Implements pure-numpy optimizer variants that simulate gradient-based
optimization for the spectral pretraining framework.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseOptimizer:
    """Base optimizer class.

    Manages parameter groups and provides common optimizer functionality.

    Attributes:
        params: List of parameter arrays.
        lr: Learning rate.
        weight_decay: L2 regularization factor.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        """Initialize optimizer.

        Args:
            params: List of parameter arrays to optimize.
            lr: Learning rate.
            weight_decay: Weight decay factor.
        """
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_count = 0
        self._state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Zero out gradients.

        Args:
            gradients: List of gradient arrays.

        Returns:
            List of zeroed gradient arrays.
        """
        return [np.zeros_like(g) for g in gradients]

    def step(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Perform optimization step.

        Args:
            gradients: List of gradient arrays.

        Returns:
            Updated parameters.
        """
        self.step_count += 1
        return self._update(gradients)

    def _update(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Update parameters (to be overridden)."""
        raise NotImplementedError

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set learning rate."""
        self.lr = lr

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "state": {k: {sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                         for sk, sv in v.items()}
                     for k, v in self._state.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.step_count = state["step_count"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]


class AdamW(BaseOptimizer):
    """AdamW optimizer (Loshchilov & Hutter, 2019).

    Decoupled weight decay regularization with Adam.

    Attributes:
        beta1: First moment decay rate.
        beta2: Second moment decay rate.
        eps: Numerical stability epsilon.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        gradient_clip: float = 1.0,
    ):
        """Initialize AdamW.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            beta1: First moment exponential decay.
            beta2: Second moment exponential decay.
            eps: Numerical stability.
            weight_decay: Decoupled weight decay.
            amsgrad: Whether to use AMSGrad variant.
            gradient_clip: Maximum gradient norm.
        """
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        self.gradient_clip = gradient_clip

        # Initialize state for each parameter
        for i, p in enumerate(self.params):
            self._state[i] = {
                "m": np.zeros_like(p),  # First moment
                "v": np.zeros_like(p),  # Second moment
            }
            if self.amsgrad:
                self._state[i]["v_max"] = np.zeros_like(p)

    def _clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradients by global norm.

        Args:
            gradients: List of gradient arrays.

        Returns:
            Clipped gradients.
        """
        total_norm = 0.0
        for g in gradients:
            total_norm += np.sum(g ** 2)
        total_norm = math.sqrt(total_norm)

        if total_norm > self.gradient_clip:
            clip_coef = self.gradient_clip / (total_norm + 1e-6)
            gradients = [g * clip_coef for g in gradients]

        return gradients

    def _update(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Perform AdamW update.

        Args:
            gradients: Parameter gradients.

        Returns:
            Updated parameters.
        """
        # Clip gradients
        if self.gradient_clip > 0:
            gradients = self._clip_gradients(gradients)

        t = self.step_count
        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t

        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            state = self._state[i]

            # Update moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = state["m"] / bias_correction1
            v_hat = state["v"] / bias_correction2

            if self.amsgrad:
                state["v_max"] = np.maximum(state["v_max"], v_hat)
                denom = np.sqrt(state["v_max"]) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps

            # Adam update
            update = m_hat / denom

            # Decoupled weight decay
            if self.weight_decay > 0:
                update = update + self.weight_decay * param

            # Apply update
            param = param - self.lr * update
            updated_params.append(param)

        self.params = updated_params
        return updated_params


class LAMB(BaseOptimizer):
    """LAMB optimizer (You et al., 2019).

    Layer-wise Adaptive Moments optimizer for large batch training.
    Particularly effective for pretraining with large batch sizes.

    Attributes:
        beta1: First moment decay.
        beta2: Second moment decay.
        eps: Numerical stability.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        trust_clip: float = 10.0,
    ):
        """Initialize LAMB optimizer.

        Args:
            params: Parameters to optimize.
            lr: Learning rate.
            beta1: First moment decay.
            beta2: Second moment decay.
            eps: Numerical stability.
            weight_decay: Weight decay factor.
            gradient_clip: Gradient clipping norm.
            trust_clip: Maximum trust ratio.
        """
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.gradient_clip = gradient_clip
        self.trust_clip = trust_clip

        for i, p in enumerate(self.params):
            self._state[i] = {
                "m": np.zeros_like(p),
                "v": np.zeros_like(p),
            }

    def _update(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Perform LAMB update with trust ratio.

        Args:
            gradients: Parameter gradients.

        Returns:
            Updated parameters.
        """
        t = self.step_count
        bias_correction1 = 1 - self.beta1 ** t
        bias_correction2 = 1 - self.beta2 ** t

        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            state = self._state[i]

            # Update moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = state["m"] / bias_correction1
            v_hat = state["v"] / bias_correction2

            # Compute Adam-like update
            update = m_hat / (np.sqrt(v_hat) + self.eps)

            # Add weight decay
            if self.weight_decay > 0:
                update = update + self.weight_decay * param

            # Compute trust ratio (layer-wise scaling)
            param_norm = np.sqrt(np.sum(param ** 2))
            update_norm = np.sqrt(np.sum(update ** 2))

            if param_norm > 0 and update_norm > 0:
                trust_ratio = param_norm / update_norm
                trust_ratio = min(trust_ratio, self.trust_clip)
            else:
                trust_ratio = 1.0

            # Apply update with trust ratio
            param = param - self.lr * trust_ratio * update
            updated_params.append(param)

        self.params = updated_params
        return updated_params


class SGDMomentum(BaseOptimizer):
    """SGD with Nesterov momentum.

    Standard SGD with optional Nesterov momentum and weight decay.

    Attributes:
        momentum: Momentum factor.
        nesterov: Whether to use Nesterov momentum.
        dampening: Dampening for momentum.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = True,
        dampening: float = 0.0,
    ):
        """Initialize SGD with momentum.

        Args:
            params: Parameters.
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: Weight decay.
            nesterov: Use Nesterov acceleration.
            dampening: Dampening for momentum.
        """
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening

        for i, p in enumerate(self.params):
            self._state[i] = {
                "velocity": np.zeros_like(p),
            }

    def _update(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Perform SGD + momentum update.

        Args:
            gradients: Parameter gradients.

        Returns:
            Updated parameters.
        """
        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            state = self._state[i]

            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update velocity
            if self.momentum > 0:
                velocity = state["velocity"]
                velocity = self.momentum * velocity + (1 - self.dampening) * grad
                state["velocity"] = velocity

                if self.nesterov:
                    update = grad + self.momentum * velocity
                else:
                    update = velocity
            else:
                update = grad

            param = param - self.lr * update
            updated_params.append(param)

        self.params = updated_params
        return updated_params


class Adafactor(BaseOptimizer):
    """Adafactor optimizer (Shazeer & Stern, 2018).

    Memory-efficient optimizer that uses factored second moments.
    Suitable for very large models where Adam's memory is prohibitive.

    Attributes:
        eps: Numerical stability.
        clip_threshold: Gradient clipping threshold.
        decay_rate: Second moment decay rate.
        beta1: First moment decay (None = no momentum).
        relative_step: Whether to use relative step sizes.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        """Initialize Adafactor.

        Args:
            params: Parameters.
            lr: Learning rate (None for relative step).
            eps: (eps1, eps2) for stability.
            clip_threshold: RMS gradient clipping.
            decay_rate: Second moment decay.
            beta1: Momentum (None = disabled).
            weight_decay: Weight decay.
            relative_step: Use relative step size.
            warmup_init: Whether to warmup.
        """
        _lr = lr if lr is not None else 1e-3
        super().__init__(params, _lr, weight_decay)
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.relative_step = relative_step
        self.warmup_init = warmup_init

        for i, p in enumerate(self.params):
            state: Dict[str, Any] = {}
            if len(p.shape) >= 2:
                # Factored second moment for matrices
                state["v_row"] = np.zeros(p.shape[:-1])
                state["v_col"] = np.zeros(p.shape[:-2] + (p.shape[-1],))
            else:
                state["v"] = np.zeros_like(p)
            if self.beta1 is not None:
                state["m"] = np.zeros_like(p)
            self._state[i] = state

    def _rms(self, x: np.ndarray) -> float:
        """Compute RMS of array."""
        return float(np.sqrt(np.mean(x ** 2)))

    def _get_rho(self) -> float:
        """Get second moment decay rate."""
        t = self.step_count
        return min(1.0, 1.0 - math.exp(self.decay_rate * math.log(t + 1)))

    def _get_lr(self) -> float:
        """Compute adaptive learning rate."""
        if not self.relative_step:
            return self.lr

        t = self.step_count
        rel_step = max(1e-6, min(1.0 / math.sqrt(t + 1), 1e-2))

        if self.warmup_init:
            rel_step = min(rel_step, 1e-2 * t)

        return rel_step

    def _update(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Perform Adafactor update.

        Args:
            gradients: Parameter gradients.

        Returns:
            Updated parameters.
        """
        rho = self._get_rho()
        lr = self._get_lr()

        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            state = self._state[i]

            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update second moment
            if len(param.shape) >= 2:
                # Factored update for matrices
                row_factor = np.mean(grad ** 2, axis=-1)
                col_factor = np.mean(grad ** 2, axis=tuple(range(len(grad.shape) - 1)))

                state["v_row"] = rho * state["v_row"] + (1 - rho) * row_factor
                state["v_col"] = rho * state["v_col"] + (1 - rho) * col_factor

                # Reconstruct second moment
                row_mean = np.mean(state["v_row"])
                if row_mean > 0:
                    v = np.outer(
                        state["v_row"].flatten(),
                        state["v_col"].flatten()
                    ).reshape(param.shape) / (row_mean + self.eps[0])
                else:
                    v = np.ones_like(param) * self.eps[1]
            else:
                state["v"] = rho * state["v"] + (1 - rho) * (grad ** 2)
                v = state["v"]

            # Compute update
            update = grad / (np.sqrt(v) + self.eps[0])

            # RMS clipping
            update_rms = self._rms(update)
            if update_rms > self.clip_threshold:
                update = update * (self.clip_threshold / (update_rms + 1e-10))

            # Apply momentum if configured
            if self.beta1 is not None:
                state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * update
                update = state["m"]

            # Scale by parameter RMS
            param_rms = max(self._rms(param), self.eps[1])
            update = update * param_rms

            param = param - lr * update
            updated_params.append(param)

        self.params = updated_params
        return updated_params


class Lion(BaseOptimizer):
    """Lion optimizer (Chen et al., 2023).

    Evolved Sign Momentum - uses sign of momentum for updates.
    More memory efficient than Adam, often better performance.

    Attributes:
        beta1: Momentum decay for update computation.
        beta2: Momentum decay for state.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.01,
    ):
        """Initialize Lion optimizer.

        Args:
            params: Parameters.
            lr: Learning rate (typically 3-10x smaller than Adam).
            beta1: Momentum for update direction.
            beta2: Momentum for state update.
            weight_decay: Weight decay.
        """
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2

        for i, p in enumerate(self.params):
            self._state[i] = {"m": np.zeros_like(p)}

    def _update(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Perform Lion update.

        Args:
            gradients: Parameter gradients.

        Returns:
            Updated parameters.
        """
        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            state = self._state[i]
            m = state["m"]

            # Compute update direction using interpolation
            update = self.beta1 * m + (1 - self.beta1) * grad

            # Use sign of update (key difference from Adam)
            update = np.sign(update)

            # Add weight decay
            if self.weight_decay > 0:
                update = update + self.weight_decay * param

            # Update momentum state
            state["m"] = self.beta2 * m + (1 - self.beta2) * grad

            # Apply update
            param = param - self.lr * update
            updated_params.append(param)

        self.params = updated_params
        return updated_params


class OptimizerFactory:
    """Factory for creating optimizers from configuration."""

    _registry = {
        "adamw": AdamW,
        "lamb": LAMB,
        "sgd": SGDMomentum,
        "adafactor": Adafactor,
        "lion": Lion,
    }

    @classmethod
    def create(cls, optimizer_type: str, params: List[np.ndarray], **kwargs) -> BaseOptimizer:
        """Create optimizer by type.

        Args:
            optimizer_type: Type string.
            params: Parameters to optimize.
            **kwargs: Optimizer hyperparameters.

        Returns:
            Initialized optimizer.
        """
        if optimizer_type not in cls._registry:
            raise ValueError(
                f"Unknown optimizer: {optimizer_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[optimizer_type](params, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any], params: List[np.ndarray]) -> BaseOptimizer:
        """Create optimizer from config dict.

        Args:
            config: Must contain 'type' and optimizer params.
            params: Parameters.

        Returns:
            Initialized optimizer.
        """
        config = dict(config)
        opt_type = config.pop("type", "adamw")
        return cls.create(opt_type, params, **config)

    @classmethod
    def available(cls) -> List[str]:
        """List available optimizers."""
        return list(cls._registry.keys())
