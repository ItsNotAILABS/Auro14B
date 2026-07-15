"""Contrastive learning objectives for spectral pretraining.

Implements various contrastive and self-supervised objectives
that learn spectral representations without explicit labels.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SpectralInfoNCE:
    """InfoNCE contrastive loss adapted for spectral data.

    Creates positive pairs from augmented views of the same
    spectral window and uses other samples as negatives.

    Supports:
    - Standard InfoNCE
    - Debiased contrastive loss
    - Hard negative mining
    - Multi-crop strategy

    Attributes:
        temperature: Temperature for similarity scaling.
        num_negatives: Number of negative samples.
        debiased: Whether to use debiased estimator.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_negatives: int = -1,
        debiased: bool = False,
        hard_negative_weight: float = 0.0,
        multi_crop: bool = False,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
    ):
        """Initialize InfoNCE loss.

        Args:
            temperature: Softmax temperature.
            num_negatives: Number of negatives (-1 = all in batch).
            debiased: Use debiased estimator.
            hard_negative_weight: Extra weight for hard negatives.
            multi_crop: Use multi-crop strategy.
            num_global_crops: Number of global crops.
            num_local_crops: Number of local crops.
        """
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.debiased = debiased
        self.hard_negative_weight = hard_negative_weight
        self.multi_crop = multi_crop
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

    def compute(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negatives: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings [B, D].
            positive: Positive pair embeddings [B, D].
            negatives: Optional explicit negatives [B, K, D].

        Returns:
            Loss and metrics.
        """
        batch_size = anchor.shape[0]

        # Normalize
        anchor = anchor / (np.linalg.norm(anchor, axis=-1, keepdims=True) + 1e-8)
        positive = positive / (np.linalg.norm(positive, axis=-1, keepdims=True) + 1e-8)

        # Positive similarity
        pos_sim = np.sum(anchor * positive, axis=-1) / self.temperature  # [B]

        if negatives is not None:
            # Explicit negatives
            negatives = negatives / (np.linalg.norm(negatives, axis=-1, keepdims=True) + 1e-8)
            neg_sim = np.einsum("bd,bkd->bk", anchor, negatives) / self.temperature
        else:
            # Use other samples in batch as negatives
            neg_sim = np.dot(anchor, positive.T) / self.temperature  # [B, B]
            # Remove positive pairs from negatives (diagonal)
            mask = ~np.eye(batch_size, dtype=bool)
            neg_sim = neg_sim * mask + (~mask) * (-1e9)

        if self.debiased:
            loss = self._debiased_loss(pos_sim, neg_sim, batch_size)
        else:
            loss = self._standard_loss(pos_sim, neg_sim)

        # Metrics
        with np.errstate(invalid="ignore"):
            accuracy = float(np.mean(
                pos_sim > np.max(neg_sim if negatives is not None
                                 else neg_sim * mask, axis=-1)
            ))

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "avg_pos_sim": float(np.mean(pos_sim * self.temperature)),
            "avg_neg_sim": float(np.mean(neg_sim[neg_sim > -1e8] * self.temperature))
            if np.any(neg_sim > -1e8) else 0.0,
            "temperature": self.temperature,
        }

        return loss, metrics

    def _standard_loss(self, pos_sim: np.ndarray, neg_sim: np.ndarray) -> float:
        """Standard InfoNCE loss."""
        # Log-sum-exp of negatives
        all_sim = np.concatenate([pos_sim[:, np.newaxis], neg_sim], axis=-1)
        max_sim = np.max(all_sim, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(all_sim - max_sim), axis=-1) + 1e-10) + max_sim.flatten()

        loss = float(np.mean(-pos_sim + log_sum_exp))
        return loss

    def _debiased_loss(
        self, pos_sim: np.ndarray, neg_sim: np.ndarray, batch_size: int
    ) -> float:
        """Debiased contrastive loss (Chuang et al., 2020)."""
        tau_plus = 1.0 / batch_size  # Prior probability of positive

        # Reweight negatives
        neg_exp = np.exp(neg_sim)
        neg_mean = np.mean(neg_exp, axis=-1)

        # Debiased negative
        pos_exp = np.exp(pos_sim)
        N = batch_size - 1
        debiased_neg = (neg_mean * N - tau_plus * pos_exp) / (1 - tau_plus)
        debiased_neg = np.maximum(debiased_neg, N * np.exp(-1.0 / self.temperature))

        loss = float(np.mean(-pos_sim + np.log(pos_exp + debiased_neg + 1e-10)))
        return loss


class BarlowTwins:
    """Barlow Twins self-supervised loss (Zbontar et al., 2021).

    Reduces redundancy between embedding dimensions by making
    the cross-correlation matrix close to identity.

    Does NOT require negative samples or large batches.

    Attributes:
        lambda_off_diag: Weight for off-diagonal terms.
        latent_dim: Embedding dimension.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        lambda_off_diag: float = 0.0051,
        normalize: bool = True,
    ):
        """Initialize Barlow Twins.

        Args:
            latent_dim: Embedding dimension.
            lambda_off_diag: Off-diagonal loss weight.
            normalize: Whether to standardize embeddings.
        """
        self.latent_dim = latent_dim
        self.lambda_off_diag = lambda_off_diag
        self.normalize = normalize

    def compute(
        self,
        z_a: np.ndarray,
        z_b: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute Barlow Twins loss.

        Args:
            z_a: First view embeddings [B, D].
            z_b: Second view embeddings [B, D].

        Returns:
            Loss and metrics.
        """
        batch_size = z_a.shape[0]

        # Standardize along batch dimension
        if self.normalize:
            z_a = (z_a - np.mean(z_a, axis=0)) / (np.std(z_a, axis=0) + 1e-5)
            z_b = (z_b - np.mean(z_b, axis=0)) / (np.std(z_b, axis=0) + 1e-5)

        # Cross-correlation matrix [D, D]
        c = np.dot(z_a.T, z_b) / batch_size

        # Loss: diagonal should be 1, off-diagonal should be 0
        on_diag = np.sum((np.diag(c) - 1) ** 2)
        off_diag = np.sum(c ** 2) - np.sum(np.diag(c) ** 2)

        loss = float(on_diag + self.lambda_off_diag * off_diag)

        # Metrics
        metrics = {
            "loss": loss,
            "on_diag_loss": float(on_diag),
            "off_diag_loss": float(off_diag),
            "mean_diag": float(np.mean(np.diag(c))),
            "mean_off_diag": float(np.mean(np.abs(c - np.diag(np.diag(c))))),
            "correlation_rank": int(np.sum(np.abs(np.diag(c)) > 0.5)),
        }

        return loss, metrics


class VICReg:
    """VICReg loss (Bardes et al., 2022).

    Variance-Invariance-Covariance Regularization.
    Three terms ensure:
    - Variance: embeddings don't collapse
    - Invariance: positive pairs are similar
    - Covariance: dimensions are decorrelated

    Attributes:
        sim_weight: Invariance (similarity) weight.
        var_weight: Variance weight.
        cov_weight: Covariance weight.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        variance_target: float = 1.0,
    ):
        """Initialize VICReg.

        Args:
            latent_dim: Embedding dimension.
            sim_weight: Invariance loss weight.
            var_weight: Variance loss weight.
            cov_weight: Covariance loss weight.
            variance_target: Target standard deviation.
        """
        self.latent_dim = latent_dim
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.variance_target = variance_target

    def compute(
        self,
        z_a: np.ndarray,
        z_b: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute VICReg loss.

        Args:
            z_a: First view [B, D].
            z_b: Second view [B, D].

        Returns:
            Loss and metrics.
        """
        batch_size = z_a.shape[0]

        # 1. Invariance: MSE between representations
        sim_loss = float(np.mean((z_a - z_b) ** 2))

        # 2. Variance: std of each dimension should be >= target
        std_a = np.sqrt(np.var(z_a, axis=0) + 1e-4)
        std_b = np.sqrt(np.var(z_b, axis=0) + 1e-4)

        var_loss_a = float(np.mean(np.maximum(0, self.variance_target - std_a)))
        var_loss_b = float(np.mean(np.maximum(0, self.variance_target - std_b)))
        var_loss = (var_loss_a + var_loss_b) / 2

        # 3. Covariance: off-diagonal of covariance should be 0
        z_a_centered = z_a - np.mean(z_a, axis=0)
        z_b_centered = z_b - np.mean(z_b, axis=0)

        cov_a = np.dot(z_a_centered.T, z_a_centered) / (batch_size - 1)
        cov_b = np.dot(z_b_centered.T, z_b_centered) / (batch_size - 1)

        # Zero diagonal before computing off-diagonal loss
        np.fill_diagonal(cov_a, 0)
        np.fill_diagonal(cov_b, 0)

        cov_loss_a = float(np.sum(cov_a ** 2) / self.latent_dim)
        cov_loss_b = float(np.sum(cov_b ** 2) / self.latent_dim)
        cov_loss = (cov_loss_a + cov_loss_b) / 2

        # Total loss
        loss = self.sim_weight * sim_loss + \
            self.var_weight * var_loss + \
            self.cov_weight * cov_loss

        metrics = {
            "loss": loss,
            "invariance_loss": sim_loss,
            "variance_loss": var_loss,
            "covariance_loss": cov_loss,
            "mean_std_a": float(np.mean(std_a)),
            "mean_std_b": float(np.mean(std_b)),
            "collapsed_dims": int(np.sum(std_a < 0.01) + np.sum(std_b < 0.01)),
        }

        return loss, metrics


class DINO:
    """DINO self-distillation loss (Caron et al., 2021).

    Self-distillation with no labels using student-teacher
    framework with centering and sharpening.

    Attributes:
        out_dim: Output dimension (number of prototypes).
        student_temperature: Student softmax temperature.
        teacher_temperature: Teacher softmax temperature.
        center_momentum: EMA for centering.
    """

    def __init__(
        self,
        out_dim: int = 4096,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        center_momentum: float = 0.9,
    ):
        """Initialize DINO loss.

        Args:
            out_dim: Output dimension.
            student_temperature: Student temperature.
            teacher_temperature: Teacher temperature.
            center_momentum: Center EMA momentum.
        """
        self.out_dim = out_dim
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.center = np.zeros(out_dim)

    def compute(
        self,
        student_output: np.ndarray,
        teacher_output: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute DINO loss.

        Args:
            student_output: Student logits [B, K].
            teacher_output: Teacher logits [B, K].

        Returns:
            Loss and metrics.
        """
        # Teacher: centering and sharpening
        teacher_centered = teacher_output - self.center
        teacher_probs = self._softmax(teacher_centered / self.teacher_temperature)

        # Student: just softmax
        student_log_probs = self._log_softmax(student_output / self.student_temperature)

        # Cross-entropy loss
        loss = float(-np.mean(np.sum(teacher_probs * student_log_probs, axis=-1)))

        # Update center
        self.center = (
            self.center_momentum * self.center +
            (1 - self.center_momentum) * np.mean(teacher_output, axis=0)
        )

        # Metrics
        teacher_entropy = float(-np.mean(
            np.sum(teacher_probs * np.log(teacher_probs + 1e-10), axis=-1)
        ))
        student_probs = np.exp(student_log_probs)
        student_entropy = float(-np.mean(
            np.sum(student_probs * student_log_probs, axis=-1)
        ))

        metrics = {
            "loss": loss,
            "teacher_entropy": teacher_entropy,
            "student_entropy": student_entropy,
            "center_norm": float(np.linalg.norm(self.center)),
            "teacher_sharpness": float(np.mean(np.max(teacher_probs, axis=-1))),
        }

        return loss, metrics

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

    def _log_softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute log softmax."""
        max_x = np.max(x, axis=-1, keepdims=True)
        return x - max_x - np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True) + 1e-10)
