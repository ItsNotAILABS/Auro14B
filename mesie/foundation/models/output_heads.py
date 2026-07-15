"""Output heads for SpectralGPT multi-task pretraining.

Implements various output heads for different pretraining objectives
including spectral reconstruction, next-window prediction, contrastive
learning, and downstream classification.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SpectralReconstructionHead:
    """Output head for spectral reconstruction objectives.

    Reconstructs masked or corrupted spectral tokens back to their
    original representation. Supports both magnitude and phase
    reconstruction with optional uncertainty estimation.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output spectral dimension.
        num_layers: Number of projection layers.
        use_uncertainty: Whether to predict reconstruction uncertainty.
        predict_phase: Whether to reconstruct phase.
        use_spectral_loss: Whether to use spectral-aware loss weighting.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 1024,
        num_layers: int = 2,
        hidden_dim: int = 2048,
        use_uncertainty: bool = True,
        predict_phase: bool = True,
        use_spectral_loss: bool = True,
        activation: str = "gelu",
    ):
        """Initialize reconstruction head.

        Args:
            input_dim: Input dimension from transformer.
            output_dim: Output spectral dimension.
            num_layers: Number of hidden layers.
            hidden_dim: Hidden layer dimension.
            use_uncertainty: Whether to estimate uncertainty.
            predict_phase: Whether to predict phase.
            use_spectral_loss: Whether to weight loss spectrally.
            activation: Activation function.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_uncertainty = use_uncertainty
        self.predict_phase = predict_phase
        self.use_spectral_loss = use_spectral_loss
        self.activation = activation

        # Build projection layers
        self.layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append({
                "weight": np.random.randn(dims[i], dims[i + 1]) * 0.02,
                "bias": np.zeros(dims[i + 1]),
            })

        # Magnitude output
        self.magnitude_proj = np.random.randn(hidden_dim, output_dim) * 0.02
        self.magnitude_bias = np.zeros(output_dim)

        # Phase output (if enabled)
        if predict_phase:
            self.phase_proj = np.random.randn(hidden_dim, output_dim) * 0.02
            self.phase_bias = np.zeros(output_dim)

        # Uncertainty output (if enabled)
        if use_uncertainty:
            self.uncertainty_proj = np.random.randn(hidden_dim, output_dim) * 0.02
            self.uncertainty_bias = np.zeros(output_dim)

        # Spectral weighting (frequency-dependent loss weights)
        if use_spectral_loss:
            # Higher weight for low frequencies (more perceptually important)
            freq_axis = np.linspace(0, 1, output_dim)
            self.spectral_weights = 1.0 / (1.0 + freq_axis * 5.0)
            self.spectral_weights = self.spectral_weights / np.sum(self.spectral_weights)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "gelu":
            return 0.5 * x * (1.0 + np.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "silu":
            return x * (1.0 / (1.0 + np.exp(-x)))
        return x

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass through reconstruction head.

        Args:
            x: Input features [..., input_dim].

        Returns:
            Dictionary with:
            - 'magnitude': Reconstructed magnitude [..., output_dim]
            - 'phase': Reconstructed phase [..., output_dim] (if enabled)
            - 'uncertainty': Uncertainty estimate [..., output_dim] (if enabled)
        """
        # Hidden layers
        hidden = x
        for layer in self.layers:
            hidden = np.einsum("...d,do->...o", hidden, layer["weight"]) + layer["bias"]
            hidden = self._activate(hidden)

        outputs: Dict[str, np.ndarray] = {}

        # Magnitude prediction
        magnitude = np.einsum("...d,do->...o", hidden, self.magnitude_proj) + self.magnitude_bias
        # Ensure non-negative magnitudes
        magnitude = np.abs(magnitude)
        outputs["magnitude"] = magnitude

        # Phase prediction (circular output in [-pi, pi])
        if self.predict_phase:
            phase_raw = np.einsum("...d,do->...o", hidden, self.phase_proj) + self.phase_bias
            phase = np.tanh(phase_raw) * np.pi  # Bound to [-pi, pi]
            outputs["phase"] = phase

        # Uncertainty (log-variance)
        if self.use_uncertainty:
            log_var = np.einsum("...d,do->...o", hidden, self.uncertainty_proj) + self.uncertainty_bias
            outputs["uncertainty"] = log_var

        return outputs

    def compute_loss(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute reconstruction loss.

        Args:
            predictions: Predicted values from forward().
            targets: Ground truth values.

        Returns:
            Dictionary of loss components.
        """
        losses: Dict[str, float] = {}

        # Magnitude loss
        if "magnitude" in targets:
            mag_diff = predictions["magnitude"] - targets["magnitude"]
            if self.use_uncertainty and "uncertainty" in predictions:
                # Heteroscedastic loss
                log_var = predictions["uncertainty"]
                mag_loss = 0.5 * np.mean(
                    np.exp(-log_var) * mag_diff ** 2 + log_var
                )
            else:
                if self.use_spectral_loss:
                    mag_loss = float(np.mean(
                        mag_diff ** 2 * self.spectral_weights
                    ))
                else:
                    mag_loss = float(np.mean(mag_diff ** 2))
            losses["magnitude_loss"] = mag_loss

        # Phase loss (circular)
        if self.predict_phase and "phase" in targets:
            phase_diff = predictions["phase"] - targets["phase"]
            # Circular loss (handle wraparound)
            phase_loss = float(np.mean(1.0 - np.cos(phase_diff)))
            losses["phase_loss"] = phase_loss

        # Total loss
        losses["total_loss"] = sum(losses.values())

        return losses


class NextWindowPredictionHead:
    """Output head for next-window (autoregressive) prediction.

    Predicts the next spectral window in a sequence, enabling the
    model to learn temporal dynamics of spectral evolution.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output prediction dimension.
        prediction_steps: Number of future steps to predict.
        use_autoregressive: Whether to use AR decoding.
        prediction_type: What to predict ('tokens', 'embeddings', 'spectral').
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 1024,
        prediction_steps: int = 4,
        hidden_dim: int = 2048,
        use_autoregressive: bool = True,
        prediction_type: str = "embeddings",
        num_layers: int = 3,
    ):
        """Initialize next-window prediction head.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension per step.
            prediction_steps: Number of future steps.
            hidden_dim: Hidden dimension.
            use_autoregressive: Whether to use AR.
            prediction_type: Prediction target type.
            num_layers: Number of prediction layers.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prediction_steps = prediction_steps
        self.hidden_dim = hidden_dim
        self.use_autoregressive = use_autoregressive
        self.prediction_type = prediction_type

        # Context aggregation
        self.context_proj = np.random.randn(input_dim, hidden_dim) * 0.02
        self.context_bias = np.zeros(hidden_dim)

        # Per-step prediction heads
        self.step_heads = []
        for step in range(prediction_steps):
            head = {
                "layers": [],
                "output_weight": np.random.randn(hidden_dim, output_dim) * 0.02,
                "output_bias": np.zeros(output_dim),
            }
            # Build layers for this step
            dims = [hidden_dim] * (num_layers + 1)
            for i in range(num_layers):
                head["layers"].append({
                    "weight": np.random.randn(dims[i], dims[i + 1]) * 0.02,
                    "bias": np.zeros(dims[i + 1]),
                })
            self.step_heads.append(head)

        # Temporal decay (predictions further in future are less certain)
        self.temporal_weights = np.exp(-0.1 * np.arange(prediction_steps))
        self.temporal_weights = self.temporal_weights / np.sum(self.temporal_weights)

        # Autoregressive state
        if use_autoregressive:
            self.ar_proj = np.random.randn(output_dim, hidden_dim) * 0.02

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        ))

    def forward(
        self, x: np.ndarray, context: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Predict next spectral windows.

        Args:
            x: Input features [..., input_dim] or [..., seq_len, input_dim].
            context: Optional additional context.

        Returns:
            Dictionary with predictions for each step.
        """
        # Aggregate context
        if x.ndim > 2:
            # Use last position as primary context
            primary = x[..., -1, :]
        else:
            primary = x

        hidden = np.einsum("...d,do->...o", primary, self.context_proj) + self.context_bias
        hidden = self._gelu(hidden)

        predictions: Dict[str, Any] = {"steps": [], "weights": []}
        ar_state = hidden

        for step in range(self.prediction_steps):
            step_hidden = ar_state

            # Process through step-specific layers
            for layer in self.step_heads[step]["layers"]:
                step_hidden = np.einsum(
                    "...d,do->...o", step_hidden, layer["weight"]
                ) + layer["bias"]
                step_hidden = self._gelu(step_hidden)

            # Output prediction
            step_pred = np.einsum(
                "...d,do->...o", step_hidden,
                self.step_heads[step]["output_weight"]
            ) + self.step_heads[step]["output_bias"]

            predictions["steps"].append(step_pred)
            predictions["weights"].append(float(self.temporal_weights[step]))

            # Update AR state for next step
            if self.use_autoregressive:
                ar_contribution = np.einsum("...d,do->...o", step_pred, self.ar_proj)
                ar_state = ar_state + ar_contribution
                ar_state = self._gelu(ar_state)

        return predictions

    def compute_loss(
        self,
        predictions: Dict[str, Any],
        targets: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute next-window prediction loss.

        Args:
            predictions: Predictions from forward().
            targets: List of target arrays for each step.

        Returns:
            Dictionary of loss components.
        """
        losses: Dict[str, float] = {}
        total_loss = 0.0

        for step in range(min(self.prediction_steps, len(targets))):
            pred = predictions["steps"][step]
            target = targets[step]
            step_loss = float(np.mean((pred - target) ** 2))

            # Weight by temporal importance
            weighted_loss = step_loss * self.temporal_weights[step]
            losses[f"step_{step}_loss"] = step_loss
            total_loss += weighted_loss

        losses["total_loss"] = total_loss
        return losses


class ContrastiveProjectionHead:
    """Projection head for contrastive learning objectives.

    Projects transformer representations into a normalized embedding
    space suitable for contrastive learning (InfoNCE, NT-Xent, etc.).

    Attributes:
        input_dim: Input feature dimension.
        projection_dim: Final projection dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of projection layers.
        normalize: Whether to L2-normalize output.
        temperature: Temperature for contrastive loss.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        normalize: bool = True,
        temperature: float = 0.07,
        use_batch_norm: bool = True,
    ):
        """Initialize contrastive projection head.

        Args:
            input_dim: Input dimension.
            projection_dim: Output projection dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            normalize: Whether to normalize output.
            temperature: Contrastive temperature.
            use_batch_norm: Whether to use batch normalization.
        """
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.normalize = normalize
        self.temperature = temperature
        self.use_batch_norm = use_batch_norm

        # Build layers
        self.layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [projection_dim]
        for i in range(num_layers):
            layer = {
                "weight": np.random.randn(dims[i], dims[i + 1]) * 0.02,
                "bias": np.zeros(dims[i + 1]),
                "is_last": i == num_layers - 1,
            }
            if use_batch_norm and not layer["is_last"]:
                layer["bn_gamma"] = np.ones(dims[i + 1])
                layer["bn_beta"] = np.zeros(dims[i + 1])
            self.layers.append(layer)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        ))

    def _batch_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Apply batch normalization."""
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-5)
        return gamma * normalized + beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project features for contrastive learning.

        Args:
            x: Input features [..., input_dim].

        Returns:
            Projected (and optionally normalized) features [..., projection_dim].
        """
        hidden = x
        for layer in self.layers:
            hidden = np.einsum("...d,do->...o", hidden, layer["weight"]) + layer["bias"]
            if not layer["is_last"]:
                if self.use_batch_norm and "bn_gamma" in layer:
                    if hidden.ndim >= 2:
                        hidden = self._batch_norm(
                            hidden, layer["bn_gamma"], layer["bn_beta"]
                        )
                hidden = self._gelu(hidden)

        if self.normalize:
            norm = np.sqrt(np.sum(hidden ** 2, axis=-1, keepdims=True) + 1e-10)
            hidden = hidden / norm

        return hidden

    def compute_info_nce_loss(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute InfoNCE contrastive loss.

        Args:
            anchors: Anchor embeddings [batch, projection_dim].
            positives: Positive pair embeddings [batch, projection_dim].
            negatives: Optional negative embeddings [num_neg, projection_dim].

        Returns:
            Dictionary with loss and metrics.
        """
        # Compute similarities
        if anchors.ndim == 1:
            anchors = anchors[np.newaxis, :]
        if positives.ndim == 1:
            positives = positives[np.newaxis, :]

        batch_size = anchors.shape[0]

        # Positive similarities
        pos_sim = np.sum(anchors * positives, axis=-1) / self.temperature

        # Negative similarities (use other positives in batch as negatives)
        if negatives is not None:
            neg_sim = np.einsum("bd,nd->bn", anchors, negatives) / self.temperature
        else:
            # In-batch negatives
            neg_sim = np.einsum("bd,nd->bn", anchors, positives) / self.temperature
            # Mask out positive pair
            mask = np.eye(batch_size) * -1e9
            neg_sim = neg_sim + mask

        # InfoNCE loss
        all_sim = np.concatenate([pos_sim[:, np.newaxis], neg_sim], axis=-1)
        log_sum_exp = np.log(np.sum(np.exp(all_sim - np.max(all_sim, axis=-1, keepdims=True)), axis=-1) + 1e-10)
        log_sum_exp = log_sum_exp + np.max(all_sim, axis=-1)
        loss = float(np.mean(-pos_sim + log_sum_exp))

        # Accuracy (positive should be highest)
        correct = np.sum(pos_sim > np.max(neg_sim, axis=-1))
        accuracy = float(correct / batch_size)

        return {
            "contrastive_loss": loss,
            "contrastive_accuracy": accuracy,
            "avg_positive_sim": float(np.mean(pos_sim * self.temperature)),
            "avg_negative_sim": float(np.mean(neg_sim * self.temperature)),
        }


class ClassificationHead:
    """Classification head for downstream tasks.

    Provides a flexible classification head that can be attached
    to the foundation model for fine-tuning on specific tasks.

    Attributes:
        input_dim: Input feature dimension.
        num_classes: Number of output classes.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of layers.
        dropout: Dropout rate.
        pooling: Pooling strategy for sequence inputs.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 10,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
        use_layer_norm: bool = True,
    ):
        """Initialize classification head.

        Args:
            input_dim: Input dimension.
            num_classes: Number of classes.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            dropout: Dropout rate.
            pooling: Sequence pooling method.
            use_layer_norm: Whether to use layer normalization.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.use_layer_norm = use_layer_norm

        # Build layers
        self.layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers - 1):
            layer = {
                "weight": np.random.randn(dims[i], dims[i + 1]) * 0.02,
                "bias": np.zeros(dims[i + 1]),
            }
            if use_layer_norm:
                layer["ln_weight"] = np.ones(dims[i + 1])
                layer["ln_bias"] = np.zeros(dims[i + 1])
            self.layers.append(layer)

        # Final classification layer
        final_in = hidden_dim if num_layers > 1 else input_dim
        self.classifier_weight = np.random.randn(final_in, num_classes) * 0.02
        self.classifier_bias = np.zeros(num_classes)

    def _pool(self, x: np.ndarray) -> np.ndarray:
        """Pool sequence representations.

        Args:
            x: Input [..., seq_len, input_dim].

        Returns:
            Pooled output [..., input_dim].
        """
        if x.ndim <= 2:
            return x

        if self.pooling == "mean":
            return np.mean(x, axis=-2)
        elif self.pooling == "max":
            return np.max(x, axis=-2)
        elif self.pooling == "first":
            return x[..., 0, :]
        elif self.pooling == "last":
            return x[..., -1, :]
        elif self.pooling == "attention":
            # Simple attention pooling
            scores = np.sum(x ** 2, axis=-1, keepdims=True)
            weights = np.exp(scores) / (np.sum(np.exp(scores), axis=-2, keepdims=True) + 1e-10)
            return np.sum(x * weights, axis=-2)
        else:
            return np.mean(x, axis=-2)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through classification head.

        Args:
            x: Input features [..., (seq_len,) input_dim].

        Returns:
            Class logits [..., num_classes].
        """
        # Pool if sequence
        hidden = self._pool(x)

        # Hidden layers
        for layer in self.layers:
            hidden = np.einsum("...d,do->...o", hidden, layer["weight"]) + layer["bias"]
            if self.use_layer_norm and "ln_weight" in layer:
                mean = np.mean(hidden, axis=-1, keepdims=True)
                var = np.var(hidden, axis=-1, keepdims=True)
                hidden = (hidden - mean) / np.sqrt(var + 1e-5)
                hidden = hidden * layer["ln_weight"] + layer["ln_bias"]
            hidden = self._gelu(hidden)

        # Classification
        logits = np.einsum(
            "...d,do->...o", hidden, self.classifier_weight
        ) + self.classifier_bias

        return logits

    def compute_loss(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification loss (cross-entropy).

        Args:
            logits: Predicted logits [..., num_classes].
            targets: Target class indices [...].

        Returns:
            Dictionary with loss and accuracy.
        """
        # Softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

        # Cross-entropy
        flat_targets = targets.flatten().astype(int)
        flat_probs = probs.reshape(-1, self.num_classes)
        num_samples = len(flat_targets)

        ce_loss = 0.0
        for i in range(num_samples):
            ce_loss -= math.log(flat_probs[i, flat_targets[i]] + 1e-10)
        ce_loss /= num_samples

        # Accuracy
        predictions = np.argmax(logits, axis=-1).flatten()
        accuracy = float(np.mean(predictions == flat_targets))

        return {
            "cross_entropy_loss": ce_loss,
            "accuracy": accuracy,
        }


class MultiTaskHead:
    """Multi-task output head combining multiple objectives.

    Manages multiple output heads and combines their losses with
    learned or fixed weighting for multi-task pretraining.

    Attributes:
        heads: Dictionary of named output heads.
        task_weights: Per-task loss weights.
        use_uncertainty_weighting: Whether to learn weights via uncertainty.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        use_uncertainty_weighting: bool = True,
    ):
        """Initialize multi-task head.

        Args:
            input_dim: Input feature dimension.
            task_configs: Configuration for each task head.
            use_uncertainty_weighting: Whether to use learned task weights.
        """
        self.input_dim = input_dim
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Default task configurations
        if task_configs is None:
            task_configs = {
                "reconstruction": {"type": "reconstruction", "output_dim": 1024},
                "next_window": {"type": "next_window", "prediction_steps": 4},
                "contrastive": {"type": "contrastive", "projection_dim": 256},
                "classification": {"type": "classification", "num_classes": 10},
            }

        # Build heads
        self.heads: Dict[str, Any] = {}
        self.task_weights: Dict[str, float] = {}
        self.log_variances: Dict[str, float] = {}

        for name, config in task_configs.items():
            task_type = config.get("type", "classification")

            if task_type == "reconstruction":
                self.heads[name] = SpectralReconstructionHead(
                    input_dim=input_dim,
                    output_dim=config.get("output_dim", 1024),
                )
            elif task_type == "next_window":
                self.heads[name] = NextWindowPredictionHead(
                    input_dim=input_dim,
                    output_dim=config.get("output_dim", 1024),
                    prediction_steps=config.get("prediction_steps", 4),
                )
            elif task_type == "contrastive":
                self.heads[name] = ContrastiveProjectionHead(
                    input_dim=input_dim,
                    projection_dim=config.get("projection_dim", 256),
                )
            elif task_type == "classification":
                self.heads[name] = ClassificationHead(
                    input_dim=input_dim,
                    num_classes=config.get("num_classes", 10),
                )

            self.task_weights[name] = config.get("weight", 1.0)
            self.log_variances[name] = 0.0  # Learnable log-variance

    def forward(
        self, x: np.ndarray, active_tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Forward pass through all active task heads.

        Args:
            x: Input features [..., input_dim].
            active_tasks: Optional list of active tasks (default: all).

        Returns:
            Dictionary of task outputs.
        """
        tasks = active_tasks or list(self.heads.keys())
        outputs: Dict[str, Any] = {}

        for task_name in tasks:
            if task_name in self.heads:
                head = self.heads[task_name]
                outputs[task_name] = head.forward(x)

        return outputs

    def compute_total_loss(
        self,
        task_losses: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Combine task losses with learned or fixed weighting.

        Args:
            task_losses: Per-task loss values.

        Returns:
            Tuple of (total_loss, weighted_losses).
        """
        total_loss = 0.0
        weighted_losses: Dict[str, float] = {}

        for task_name, loss in task_losses.items():
            if self.use_uncertainty_weighting:
                # Uncertainty weighting: L_total = sum(1/(2*sigma^2) * L_i + log(sigma))
                log_var = self.log_variances.get(task_name, 0.0)
                precision = math.exp(-log_var)
                weighted_loss = precision * loss + log_var
            else:
                weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = weight * loss

            weighted_losses[task_name] = weighted_loss
            total_loss += weighted_loss

        weighted_losses["total"] = total_loss
        return total_loss, weighted_losses
