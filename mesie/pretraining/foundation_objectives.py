"""Foundation pretraining objectives for spectral reasoning.

Implements the core self-supervised objectives that turn raw spectra into
stable, rich embeddings:

- Masked Spectral Modeling: Reconstruct masked frequency bands from context.
- InfoNCE Contrastive Learning: Learn invariant representations via
  augmentation-based positive pairs and in-batch negatives.
- Temporal Prediction: Predict future spectral embeddings from past windows.

These objectives form Stage 1 of the training recipe and are designed to
be combined with the auxiliary world tasks (resonance, coherence, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Masked Spectral Modeling
# ---------------------------------------------------------------------------


@dataclass
class MaskConfig:
    """Configuration for spectral masking.

    Attributes
    ----------
    mask_ratio : float
        Fraction of frequency bins to mask (0, 1).
    mask_strategy : str
        Strategy for selecting masked bins: 'random', 'contiguous', 'band'.
    contiguous_max_width : int
        Maximum width of contiguous masked region (for 'contiguous' strategy).
    n_bands : int
        Number of frequency bands to potentially mask (for 'band' strategy).
    replace_value : float
        Value used to replace masked bins (0.0 = zero-masking).
    """

    mask_ratio: float = 0.15
    mask_strategy: str = "random"
    contiguous_max_width: int = 32
    n_bands: int = 8
    replace_value: float = 0.0


class MaskedSpectralModeling:
    """Masked spectral modeling pretraining objective.

    Corrupts input spectra by masking frequency bands and trains the model
    to reconstruct the original values at masked positions:

        L = E[ || x_masked - f_theta(x_corrupted) ||^2 ]

    Parameters
    ----------
    config : MaskConfig or None
        Masking configuration. Uses defaults if None.
    """

    def __init__(self, config: Optional[MaskConfig] = None) -> None:
        self.config = config or MaskConfig()

    def generate_mask(
        self, n_freq: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate a boolean mask indicating which bins to corrupt.

        Parameters
        ----------
        n_freq : int
            Number of frequency bins.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        ndarray, shape (n_freq,), dtype bool
            True where the spectrum should be masked.
        """
        rng = np.random.default_rng(seed)
        n_mask = max(1, int(n_freq * self.config.mask_ratio))
        mask = np.zeros(n_freq, dtype=bool)

        if self.config.mask_strategy == "random":
            indices = rng.choice(n_freq, size=n_mask, replace=False)
            mask[indices] = True

        elif self.config.mask_strategy == "contiguous":
            width = min(n_mask, self.config.contiguous_max_width)
            start = rng.integers(0, max(1, n_freq - width))
            mask[start: start + width] = True

        elif self.config.mask_strategy == "band":
            band_size = n_freq // self.config.n_bands
            n_bands_to_mask = max(1, int(self.config.n_bands * self.config.mask_ratio))
            bands = rng.choice(self.config.n_bands, size=n_bands_to_mask, replace=False)
            for b in bands:
                start = b * band_size
                end = min(start + band_size, n_freq)
                mask[start:end] = True

        else:
            raise ValueError(
                f"Unknown mask strategy: {self.config.mask_strategy}. "
                "Use 'random', 'contiguous', or 'band'."
            )

        return mask

    def corrupt(
        self,
        spectra: np.ndarray,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply masking corruption to spectra.

        Parameters
        ----------
        spectra : ndarray, shape (n_samples, n_freq) or (n_freq,)
            Input spectral data.
        mask : ndarray or None
            Pre-computed mask. Generated if None.
        seed : int or None
            Random seed.

        Returns
        -------
        corrupted : ndarray, same shape as spectra
            Spectra with masked bins replaced.
        mask : ndarray, shape (n_freq,)
            Boolean mask used.
        """
        spectra = np.atleast_2d(spectra)
        n_freq = spectra.shape[-1]

        if mask is None:
            mask = self.generate_mask(n_freq, seed=seed)

        corrupted = spectra.copy()
        corrupted[:, mask] = self.config.replace_value
        return corrupted, mask

    def compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute masked reconstruction loss (MSE over masked positions).

        Parameters
        ----------
        predictions : ndarray, shape (n_samples, n_freq)
            Model predictions for all frequency bins.
        targets : ndarray, shape (n_samples, n_freq)
            Original (uncorrupted) spectra.
        mask : ndarray, shape (n_freq,), dtype bool
            Mask indicating which bins were corrupted.

        Returns
        -------
        float
            Mean squared error over masked positions.
        """
        predictions = np.atleast_2d(predictions)
        targets = np.atleast_2d(targets)

        masked_pred = predictions[:, mask]
        masked_target = targets[:, mask]

        if masked_pred.size == 0:
            return 0.0

        return float(np.mean((masked_pred - masked_target) ** 2))

    def create_training_sample(
        self,
        spectra: np.ndarray,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Create a complete training sample (input, target, mask).

        Parameters
        ----------
        spectra : ndarray, shape (n_samples, n_freq) or (n_freq,)
            Raw spectral data.
        seed : int or None
            Random seed.

        Returns
        -------
        dict with keys:
            'corrupted': Masked input spectra.
            'targets': Original spectra.
            'mask': Boolean mask applied.
        """
        spectra = np.atleast_2d(spectra)
        corrupted, mask = self.corrupt(spectra, seed=seed)
        return {
            "corrupted": corrupted,
            "targets": spectra,
            "mask": mask,
        }


# ---------------------------------------------------------------------------
# InfoNCE Contrastive Learning
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for spectral augmentations used in contrastive learning.

    Attributes
    ----------
    noise_std : float
        Standard deviation of additive Gaussian noise.
    freq_mask_ratio : float
        Fraction of frequency bins to zero-mask as augmentation.
    amplitude_scale_range : tuple of float
        (min, max) random scale factor for amplitudes.
    time_shift_max : int
        Maximum circular shift in frequency bins.
    """

    noise_std: float = 0.01
    freq_mask_ratio: float = 0.1
    amplitude_scale_range: Tuple[float, float] = (0.8, 1.2)
    time_shift_max: int = 5


class InfoNCEContrastiveLoss:
    """InfoNCE contrastive learning for spectral embeddings.

    Creates positive pairs via spectral augmentations and uses in-batch
    negatives to learn invariant representations:

        L = -log( exp(sim(z_i, z_i+) / tau) / sum_j exp(sim(z_i, z_j) / tau) )

    Parameters
    ----------
    temperature : float
        Temperature parameter tau for scaling similarities.
    augmentation_config : AugmentationConfig or None
        Configuration for augmentation pipeline.
    similarity : str
        Similarity metric: 'cosine' or 'dot'.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        augmentation_config: Optional[AugmentationConfig] = None,
        similarity: str = "cosine",
    ) -> None:
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.similarity = similarity

    def augment(
        self,
        spectra: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Apply random augmentations to create a positive view.

        Parameters
        ----------
        spectra : ndarray, shape (n_samples, n_freq)
            Input spectra.
        seed : int or None
            Random seed.

        Returns
        -------
        ndarray, same shape as spectra
            Augmented version.
        """
        rng = np.random.default_rng(seed)
        cfg = self.augmentation_config
        augmented = spectra.copy()

        # Additive noise
        augmented += rng.normal(0, cfg.noise_std, size=augmented.shape)

        # Random frequency masking
        n_freq = augmented.shape[-1]
        n_mask = max(1, int(n_freq * cfg.freq_mask_ratio))
        for i in range(augmented.shape[0]):
            mask_idx = rng.choice(n_freq, size=n_mask, replace=False)
            augmented[i, mask_idx] = 0.0

        # Amplitude scaling
        scale_min, scale_max = cfg.amplitude_scale_range
        scales = rng.uniform(scale_min, scale_max, size=(augmented.shape[0], 1))
        augmented *= scales

        # Circular frequency shift
        if cfg.time_shift_max > 0:
            shift = rng.integers(-cfg.time_shift_max, cfg.time_shift_max + 1)
            augmented = np.roll(augmented, shift, axis=-1)

        return augmented

    def compute_similarity_matrix(
        self, embeddings_a: np.ndarray, embeddings_b: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between two sets of embeddings.

        Parameters
        ----------
        embeddings_a : ndarray, shape (n, d)
        embeddings_b : ndarray, shape (m, d)

        Returns
        -------
        ndarray, shape (n, m)
            Similarity scores.
        """
        if self.similarity == "cosine":
            norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)
            norm_a = np.maximum(norm_a, 1e-8)
            norm_b = np.maximum(norm_b, 1e-8)
            a_normed = embeddings_a / norm_a
            b_normed = embeddings_b / norm_b
            return a_normed @ b_normed.T
        elif self.similarity == "dot":
            return embeddings_a @ embeddings_b.T
        else:
            raise ValueError(
                f"Unknown similarity: {self.similarity}. Use 'cosine' or 'dot'."
            )

    def compute_loss(
        self,
        embeddings: np.ndarray,
        positive_embeddings: np.ndarray,
    ) -> float:
        """Compute InfoNCE loss.

        Parameters
        ----------
        embeddings : ndarray, shape (n, d)
            Embeddings of anchor samples.
        positive_embeddings : ndarray, shape (n, d)
            Embeddings of positive (augmented) counterparts.

        Returns
        -------
        float
            InfoNCE loss value.
        """
        n = embeddings.shape[0]
        if n < 2:
            return 0.0

        # Concatenate all embeddings for negatives: [anchors; positives]
        all_embeddings = np.concatenate([embeddings, positive_embeddings], axis=0)

        # Similarity of each anchor to all samples (2n total)
        sim_matrix = self.compute_similarity_matrix(
            embeddings, all_embeddings
        ) / self.temperature

        # Positive indices are at positions n, n+1, ..., 2n-1
        positive_indices = np.arange(n) + n

        # For numerical stability, subtract max
        sim_max = np.max(sim_matrix, axis=1, keepdims=True)
        sim_matrix_stable = sim_matrix - sim_max

        # Mask self-similarities (anchor index in all_embeddings)
        mask = np.ones((n, 2 * n), dtype=bool)
        for i in range(n):
            mask[i, i] = False  # exclude self

        # Log-sum-exp over all valid negatives + positive
        exp_sim = np.exp(sim_matrix_stable) * mask
        log_sum_exp = np.log(np.sum(exp_sim, axis=1) + 1e-8)

        # Positive similarities
        positive_sim = sim_matrix_stable[np.arange(n), positive_indices]

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        loss = -positive_sim + log_sum_exp
        return float(np.mean(loss))

    def create_training_batch(
        self,
        spectra: np.ndarray,
        encoder_fn,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Create a complete contrastive training batch.

        Parameters
        ----------
        spectra : ndarray, shape (n_samples, n_freq)
            Raw spectra.
        encoder_fn : callable
            Function mapping spectra -> embeddings, shape (n, d).
        seed : int or None
            Random seed.

        Returns
        -------
        dict with keys:
            'anchor_embeddings': Embeddings of original spectra.
            'positive_embeddings': Embeddings of augmented spectra.
            'augmented_spectra': The augmented versions.
            'loss': Computed InfoNCE loss.
        """
        spectra = np.atleast_2d(spectra)
        augmented = self.augment(spectra, seed=seed)

        anchor_emb = encoder_fn(spectra)
        positive_emb = encoder_fn(augmented)

        loss = self.compute_loss(anchor_emb, positive_emb)

        return {
            "anchor_embeddings": anchor_emb,
            "positive_embeddings": positive_emb,
            "augmented_spectra": augmented,
            "loss": loss,
        }


# ---------------------------------------------------------------------------
# Temporal Prediction
# ---------------------------------------------------------------------------


@dataclass
class TemporalPredictionConfig:
    """Configuration for temporal spectral prediction.

    Attributes
    ----------
    context_window : int
        Number of past time steps used as context.
    prediction_horizon : int
        Number of future time steps to predict.
    aggregation : str
        How to aggregate context: 'mean', 'last', 'weighted', 'concat'.
    decay_factor : float
        Exponential decay factor for 'weighted' aggregation (recent = higher).
    """

    context_window: int = 10
    prediction_horizon: int = 3
    aggregation: str = "weighted"
    decay_factor: float = 0.9


class TemporalPrediction:
    """Temporal prediction pretraining objective.

    Predicts future spectral embeddings or drift from past windows:

        Given z_{t-k:t}, predict z_{t+1:t+h}

    This forces the model to capture temporal dynamics and spectral evolution.

    Parameters
    ----------
    config : TemporalPredictionConfig or None
        Configuration. Uses defaults if None.
    """

    def __init__(self, config: Optional[TemporalPredictionConfig] = None) -> None:
        self.config = config or TemporalPredictionConfig()

    def aggregate_context(self, context: np.ndarray) -> np.ndarray:
        """Aggregate a context window into a single representation.

        Parameters
        ----------
        context : ndarray, shape (window_size, embed_dim)
            Sequence of past embeddings.

        Returns
        -------
        ndarray, shape (embed_dim,) or (window_size * embed_dim,)
            Aggregated context representation.
        """
        if context.shape[0] == 0:
            return np.zeros(context.shape[1] if context.ndim > 1 else 1)

        if self.config.aggregation == "mean":
            return np.mean(context, axis=0)

        elif self.config.aggregation == "last":
            return context[-1]

        elif self.config.aggregation == "weighted":
            n = context.shape[0]
            weights = np.array(
                [self.config.decay_factor ** (n - 1 - i) for i in range(n)]
            )
            weights /= weights.sum()
            return np.average(context, axis=0, weights=weights)

        elif self.config.aggregation == "concat":
            return context.flatten()

        else:
            raise ValueError(
                f"Unknown aggregation: {self.config.aggregation}. "
                "Use 'mean', 'last', 'weighted', or 'concat'."
            )

    def generate_samples(
        self, embedding_sequence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate (context, target) pairs from a temporal sequence.

        Parameters
        ----------
        embedding_sequence : ndarray, shape (n_timesteps, embed_dim)
            Temporal sequence of spectral embeddings.

        Returns
        -------
        contexts : ndarray, shape (n_valid, embed_dim) or (n_valid, window*d)
            Aggregated context for each valid time step.
        targets : ndarray, shape (n_valid, horizon * embed_dim)
            Future embedding targets (flattened over horizon).
        valid_indices : ndarray, shape (n_valid,)
            Time indices where prediction starts.
        """
        n_steps = embedding_sequence.shape[0]
        window = self.config.context_window
        horizon = self.config.prediction_horizon

        min_required = window + horizon
        if n_steps < min_required:
            embed_dim = embedding_sequence.shape[1] if embedding_sequence.ndim > 1 else 1
            return (
                np.empty((0, embed_dim)),
                np.empty((0, horizon * embed_dim)),
                np.empty((0,), dtype=int),
            )

        contexts = []
        targets = []
        valid_indices = []

        for t in range(window, n_steps - horizon + 1):
            ctx = embedding_sequence[t - window: t]
            agg_ctx = self.aggregate_context(ctx)
            contexts.append(agg_ctx)

            future = embedding_sequence[t: t + horizon]
            targets.append(future.flatten())
            valid_indices.append(t)

        return (
            np.array(contexts),
            np.array(targets),
            np.array(valid_indices, dtype=int),
        )

    def compute_loss(
        self,
        predicted_future: np.ndarray,
        target_future: np.ndarray,
    ) -> float:
        """Compute temporal prediction loss.

        Parameters
        ----------
        predicted_future : ndarray, shape (n_samples, horizon * embed_dim)
            Model predictions of future embeddings.
        target_future : ndarray, shape (n_samples, horizon * embed_dim)
            Actual future embeddings.

        Returns
        -------
        float
            Combined MSE + cosine distance loss.
        """
        if predicted_future.size == 0:
            return 0.0

        # MSE component
        mse = float(np.mean((predicted_future - target_future) ** 2))

        # Cosine similarity component (encourage directional alignment)
        norm_pred = np.linalg.norm(predicted_future, axis=1, keepdims=True)
        norm_target = np.linalg.norm(target_future, axis=1, keepdims=True)
        norm_pred = np.maximum(norm_pred, 1e-8)
        norm_target = np.maximum(norm_target, 1e-8)

        cos_sim = np.sum(
            (predicted_future / norm_pred) * (target_future / norm_target),
            axis=1,
        )
        cosine_loss = float(np.mean(1.0 - cos_sim))

        return mse + cosine_loss

    def compute_drift_prediction(
        self,
        embedding_sequence: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict spectral drift at each time step.

        Parameters
        ----------
        embedding_sequence : ndarray, shape (n_timesteps, embed_dim)
            Temporal embedding sequence.
        reference : ndarray or None, shape (embed_dim,)
            Reference embedding for drift computation.
            Defaults to the first embedding in the sequence.

        Returns
        -------
        ndarray, shape (n_timesteps,)
            Drift magnitude at each time step relative to reference.
        """
        if reference is None:
            reference = embedding_sequence[0]

        diffs = embedding_sequence - reference[np.newaxis, :]
        return np.linalg.norm(diffs, axis=1)


# ---------------------------------------------------------------------------
# Foundation Objective Suite
# ---------------------------------------------------------------------------


class FoundationObjectiveSuite:
    """Combined suite of all foundation pretraining objectives.

    Orchestrates masked modeling, contrastive learning, and temporal
    prediction into a unified training interface.

    Parameters
    ----------
    masked_config : MaskConfig or None
        Masked spectral modeling configuration.
    contrastive_temperature : float
        Temperature for InfoNCE loss.
    augmentation_config : AugmentationConfig or None
        Augmentation config for contrastive learning.
    temporal_config : TemporalPredictionConfig or None
        Temporal prediction configuration.
    weights : dict or None
        Loss weights: {'masked': w1, 'contrastive': w2, 'temporal': w3}.
    """

    def __init__(
        self,
        masked_config: Optional[MaskConfig] = None,
        contrastive_temperature: float = 0.07,
        augmentation_config: Optional[AugmentationConfig] = None,
        temporal_config: Optional[TemporalPredictionConfig] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.masked_modeling = MaskedSpectralModeling(masked_config)
        self.contrastive = InfoNCEContrastiveLoss(
            temperature=contrastive_temperature,
            augmentation_config=augmentation_config,
        )
        self.temporal = TemporalPrediction(temporal_config)

        self.weights = weights or {
            "masked": 1.0,
            "contrastive": 1.0,
            "temporal": 0.5,
        }

    def compute_total_loss(
        self,
        losses: Dict[str, float],
    ) -> float:
        """Compute weighted sum of objective losses.

        Parameters
        ----------
        losses : dict
            Mapping of objective name to loss value.

        Returns
        -------
        float
            Weighted total loss.
        """
        total = 0.0
        for name, loss in losses.items():
            weight = self.weights.get(name, 1.0)
            total += weight * loss
        return total
