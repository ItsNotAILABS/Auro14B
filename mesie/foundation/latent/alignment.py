"""Cross-modal alignment methods for the universal latent space.

Implements various strategies for aligning representations from
different spectral modalities into a coherent shared space.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CrossModalAligner:
    """Base class for cross-modal alignment strategies.

    Aligns representations from different modalities so that
    semantically equivalent signals from different domains
    map to nearby points in latent space.

    Attributes:
        latent_dim: Dimension of latent space.
        temperature: Softmax temperature.
        num_modalities: Number of modalities.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        temperature: float = 0.07,
        num_modalities: int = 7,
    ):
        """Initialize aligner.

        Args:
            latent_dim: Latent dimension.
            temperature: Temperature for similarity.
            num_modalities: Number of modalities.
        """
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.num_modalities = num_modalities

    def compute_loss(
        self,
        embeddings: Dict[str, np.ndarray],
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute alignment loss across modality pairs.

        Args:
            embeddings: Dict mapping modality name to embeddings [B, D].
            pairs: Optional list of modality pairs to align.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        raise NotImplementedError


class ContrastiveAligner(CrossModalAligner):
    """Contrastive cross-modal alignment.

    Uses InfoNCE-style contrastive loss to pull together
    paired cross-modal representations and push apart
    non-paired ones.

    Supports:
    - Standard InfoNCE
    - Hard negative mining
    - Multi-positive extension
    - Temperature annealing
    """

    def __init__(
        self,
        latent_dim: int = 768,
        temperature: float = 0.07,
        num_modalities: int = 7,
        hard_negative_weight: float = 0.5,
        num_hard_negatives: int = 16,
        label_smoothing: float = 0.0,
    ):
        """Initialize contrastive aligner.

        Args:
            latent_dim: Latent dimension.
            temperature: Softmax temperature.
            num_modalities: Number of modalities.
            hard_negative_weight: Weight for hard negatives.
            num_hard_negatives: Number of hard negatives to use.
            label_smoothing: Label smoothing factor.
        """
        super().__init__(latent_dim, temperature, num_modalities)
        self.hard_negative_weight = hard_negative_weight
        self.num_hard_negatives = num_hard_negatives
        self.label_smoothing = label_smoothing

        # Queue for momentum contrast (MoCo-style)
        self.queue_size = 65536
        self.queues: Dict[str, np.ndarray] = {}
        self.queue_ptrs: Dict[str, int] = {}

    def compute_loss(
        self,
        embeddings: Dict[str, np.ndarray],
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute contrastive alignment loss.

        Args:
            embeddings: Modality embeddings {name: [B, D]}.
            pairs: Modality pairs to align.

        Returns:
            Loss and metrics.
        """
        if pairs is None:
            # Default: all pairs
            modalities = list(embeddings.keys())
            pairs = [(modalities[i], modalities[j])
                     for i in range(len(modalities))
                     for j in range(i + 1, len(modalities))]

        total_loss = 0.0
        pair_losses: Dict[str, float] = {}

        for mod_a, mod_b in pairs:
            if mod_a not in embeddings or mod_b not in embeddings:
                continue

            emb_a = embeddings[mod_a]
            emb_b = embeddings[mod_b]

            # Normalize
            emb_a = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
            emb_b = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)

            loss = self._infonce_loss(emb_a, emb_b)
            total_loss += loss
            pair_losses[f"{mod_a}-{mod_b}"] = loss

        avg_loss = total_loss / max(len(pairs), 1)

        metrics = {
            "total_alignment_loss": avg_loss,
            "pair_losses": pair_losses,
            "num_pairs": len(pairs),
        }

        return avg_loss, metrics

    def _infonce_loss(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> float:
        """Compute InfoNCE loss between two sets of embeddings.

        Args:
            emb_a: First embeddings [B, D].
            emb_b: Second embeddings [B, D].

        Returns:
            InfoNCE loss value.
        """
        batch_size = emb_a.shape[0]
        sim = np.dot(emb_a, emb_b.T) / self.temperature

        # Hard negative mining
        if self.hard_negative_weight > 0:
            sim = self._apply_hard_negatives(sim)

        # Labels with smoothing
        labels = np.arange(batch_size)
        if self.label_smoothing > 0:
            smooth_labels = np.ones((batch_size, batch_size)) * \
                self.label_smoothing / (batch_size - 1)
            np.fill_diagonal(smooth_labels, 1 - self.label_smoothing)
        else:
            smooth_labels = np.eye(batch_size)

        # Cross-entropy loss
        # A -> B
        max_sim = np.max(sim, axis=-1, keepdims=True)
        log_probs = sim - max_sim - np.log(
            np.sum(np.exp(sim - max_sim), axis=-1, keepdims=True) + 1e-10
        )
        loss_ab = -np.sum(smooth_labels * log_probs) / batch_size

        # B -> A
        sim_t = sim.T
        max_sim_t = np.max(sim_t, axis=-1, keepdims=True)
        log_probs_t = sim_t - max_sim_t - np.log(
            np.sum(np.exp(sim_t - max_sim_t), axis=-1, keepdims=True) + 1e-10
        )
        loss_ba = -np.sum(smooth_labels * log_probs_t) / batch_size

        return float((loss_ab + loss_ba) / 2)

    def _apply_hard_negatives(self, sim: np.ndarray) -> np.ndarray:
        """Apply hard negative weighting to similarity matrix.

        Args:
            sim: Similarity matrix [B, B].

        Returns:
            Modified similarity matrix.
        """
        batch_size = sim.shape[0]

        # Mask positives
        mask = ~np.eye(batch_size, dtype=bool)
        neg_sims = sim * mask

        # Find hardest negatives
        sorted_negs = np.sort(neg_sims, axis=-1)[:, ::-1]
        hard_neg_sims = sorted_negs[:, :self.num_hard_negatives]

        # Upweight hard negatives
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and sim[i, j] >= hard_neg_sims[i, -1]:
                    sim[i, j] *= (1 + self.hard_negative_weight)

        return sim

    def update_queue(self, modality: str, embeddings: np.ndarray) -> None:
        """Update momentum queue for a modality.

        Args:
            modality: Modality name.
            embeddings: New embeddings to enqueue.
        """
        if modality not in self.queues:
            self.queues[modality] = np.zeros((self.queue_size, self.latent_dim))
            self.queue_ptrs[modality] = 0

        batch_size = embeddings.shape[0]
        ptr = self.queue_ptrs[modality]

        if ptr + batch_size <= self.queue_size:
            self.queues[modality][ptr:ptr + batch_size] = embeddings
        else:
            overflow = (ptr + batch_size) - self.queue_size
            self.queues[modality][ptr:] = embeddings[:batch_size - overflow]
            self.queues[modality][:overflow] = embeddings[batch_size - overflow:]

        self.queue_ptrs[modality] = (ptr + batch_size) % self.queue_size


class DistillationAligner(CrossModalAligner):
    """Knowledge distillation-based alignment.

    Uses a teacher model (or stronger modality) to guide
    alignment of weaker modalities through soft targets.

    Attributes:
        teacher_temperature: Temperature for teacher softmax.
        student_temperature: Temperature for student softmax.
        alpha: Mixing weight between hard and soft targets.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        temperature: float = 0.07,
        num_modalities: int = 7,
        teacher_temperature: float = 4.0,
        student_temperature: float = 1.0,
        alpha: float = 0.5,
    ):
        """Initialize distillation aligner.

        Args:
            latent_dim: Latent dimension.
            temperature: Base temperature.
            num_modalities: Number of modalities.
            teacher_temperature: Teacher softmax temperature.
            student_temperature: Student softmax temperature.
            alpha: Soft target weight (1-alpha for hard targets).
        """
        super().__init__(latent_dim, temperature, num_modalities)
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.alpha = alpha

    def compute_loss(
        self,
        embeddings: Dict[str, np.ndarray],
        pairs: Optional[List[Tuple[str, str]]] = None,
        teacher_logits: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute distillation alignment loss.

        Args:
            embeddings: Modality embeddings.
            pairs: Teacher-student pairs (teacher first).
            teacher_logits: Pre-computed teacher logits.

        Returns:
            Loss and metrics.
        """
        if pairs is None:
            modalities = list(embeddings.keys())
            if len(modalities) < 2:
                return 0.0, {}
            pairs = [(modalities[0], m) for m in modalities[1:]]

        total_loss = 0.0
        metrics: Dict[str, Any] = {}

        for teacher_mod, student_mod in pairs:
            if teacher_mod not in embeddings or student_mod not in embeddings:
                continue

            teacher_emb = embeddings[teacher_mod]
            student_emb = embeddings[student_mod]

            # Teacher soft targets
            teacher_sim = np.dot(teacher_emb, teacher_emb.T) / self.teacher_temperature
            teacher_probs = self._softmax(teacher_sim)

            # Student predictions
            student_sim = np.dot(student_emb, teacher_emb.T) / self.student_temperature
            student_log_probs = self._log_softmax(student_sim)

            # KL divergence (soft targets)
            kl_loss = -np.sum(teacher_probs * student_log_probs) / teacher_emb.shape[0]

            # Hard target loss (contrastive)
            hard_loss = self._contrastive_loss(student_emb, teacher_emb)

            # Combined loss
            loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss
            total_loss += loss

            metrics[f"{teacher_mod}->{student_mod}"] = {
                "kl_loss": float(kl_loss),
                "hard_loss": float(hard_loss),
                "combined_loss": float(loss),
            }

        avg_loss = total_loss / max(len(pairs), 1)
        metrics["total_loss"] = avg_loss
        return avg_loss, metrics

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)

    def _log_softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute log softmax."""
        max_x = np.max(x, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True) + 1e-10)
        return x - max_x - log_sum_exp

    def _contrastive_loss(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Simple contrastive loss."""
        emb_a = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)
        sim = np.dot(emb_a, emb_b.T) / self.temperature
        labels = np.arange(emb_a.shape[0])
        log_probs = self._log_softmax(sim)
        return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


class OptimalTransportAligner(CrossModalAligner):
    """Optimal transport-based cross-modal alignment.

    Uses Sinkhorn divergence to compute optimal matching
    between modality distributions in latent space.

    This enables alignment without requiring paired data -
    aligns distributions rather than individual samples.

    Attributes:
        reg: Entropic regularization.
        num_iterations: Sinkhorn iterations.
        cost_type: Type of cost matrix.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        temperature: float = 0.07,
        num_modalities: int = 7,
        reg: float = 0.05,
        num_iterations: int = 50,
        cost_type: str = "cosine",
    ):
        """Initialize OT aligner.

        Args:
            latent_dim: Latent dimension.
            temperature: Temperature.
            num_modalities: Number of modalities.
            reg: Entropic regularization.
            num_iterations: Sinkhorn iterations.
            cost_type: Cost type ('cosine', 'euclidean', 'learned').
        """
        super().__init__(latent_dim, temperature, num_modalities)
        self.reg = reg
        self.num_iterations = num_iterations
        self.cost_type = cost_type

    def compute_loss(
        self,
        embeddings: Dict[str, np.ndarray],
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute optimal transport alignment loss.

        Args:
            embeddings: Modality embeddings.
            pairs: Pairs to align.

        Returns:
            Loss and metrics.
        """
        if pairs is None:
            modalities = list(embeddings.keys())
            pairs = [(modalities[i], modalities[j])
                     for i in range(len(modalities))
                     for j in range(i + 1, len(modalities))]

        total_loss = 0.0
        metrics: Dict[str, Any] = {}

        for mod_a, mod_b in pairs:
            if mod_a not in embeddings or mod_b not in embeddings:
                continue

            emb_a = embeddings[mod_a]
            emb_b = embeddings[mod_b]

            # Compute cost matrix
            cost = self._compute_cost(emb_a, emb_b)

            # Compute Sinkhorn divergence
            transport_plan, sinkhorn_cost = self._sinkhorn(cost)

            # Wasserstein distance approximation
            loss = float(np.sum(transport_plan * cost))
            total_loss += loss

            metrics[f"{mod_a}-{mod_b}"] = {
                "wasserstein": loss,
                "sinkhorn_cost": float(sinkhorn_cost),
                "transport_sparsity": float(np.mean(transport_plan > 1e-5)),
            }

        avg_loss = total_loss / max(len(pairs), 1)
        metrics["total_loss"] = avg_loss
        return avg_loss, metrics

    def _compute_cost(self, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """Compute cost matrix between two sets of embeddings.

        Args:
            emb_a: First embeddings [N, D].
            emb_b: Second embeddings [M, D].

        Returns:
            Cost matrix [N, M].
        """
        if self.cost_type == "cosine":
            # Cosine distance
            a_norm = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
            b_norm = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)
            sim = np.dot(a_norm, b_norm.T)
            return 1 - sim
        elif self.cost_type == "euclidean":
            # Squared Euclidean
            diff = emb_a[:, None] - emb_b[None, :]
            return np.sum(diff ** 2, axis=-1)
        else:
            # Default to cosine
            a_norm = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
            b_norm = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)
            return 1 - np.dot(a_norm, b_norm.T)

    def _sinkhorn(
        self, cost: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute Sinkhorn optimal transport plan.

        Args:
            cost: Cost matrix [N, M].

        Returns:
            Transport plan and Sinkhorn cost.
        """
        n, m = cost.shape

        # Uniform marginals
        a = np.ones(n) / n
        b = np.ones(m) / m

        # Gibbs kernel
        K = np.exp(-cost / self.reg)

        # Sinkhorn iterations
        u = np.ones(n)
        v = np.ones(m)

        for _ in range(self.num_iterations):
            u = a / (np.dot(K, v) + 1e-10)
            v = b / (np.dot(K.T, u) + 1e-10)

        # Transport plan
        transport = np.diag(u) @ K @ np.diag(v)

        # Sinkhorn cost
        sinkhorn_cost = float(np.sum(transport * cost))

        return transport, sinkhorn_cost

    def compute_barycenter(
        self,
        embeddings: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Compute Wasserstein barycenter of modality distributions.

        Finds the distribution that minimizes weighted sum of
        Wasserstein distances to all modality distributions.

        Args:
            embeddings: Modality embeddings.
            weights: Per-modality weights.

        Returns:
            Barycenter points [K, D].
        """
        modalities = list(embeddings.keys())
        if weights is None:
            weights = {m: 1.0 / len(modalities) for m in modalities}

        # Initialize barycenter from weighted average
        all_emb = np.concatenate(list(embeddings.values()), axis=0)
        k = min(100, len(all_emb))
        indices = np.random.choice(len(all_emb), k, replace=False)
        barycenter = all_emb[indices]

        # Iterative barycenter computation
        for iteration in range(20):
            weighted_update = np.zeros_like(barycenter)

            for modality in modalities:
                emb = embeddings[modality]
                w = weights.get(modality, 1.0 / len(modalities))

                # Compute transport from barycenter to modality
                cost = self._compute_cost(barycenter, emb)
                transport, _ = self._sinkhorn(cost)

                # Weighted displacement
                displacement = np.dot(transport, emb) - barycenter * np.sum(transport, axis=1, keepdims=True)
                weighted_update += w * displacement

            # Update barycenter
            barycenter = barycenter + 0.5 * weighted_update

        return barycenter
