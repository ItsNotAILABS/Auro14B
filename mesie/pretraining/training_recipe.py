"""Multi-stage training recipe for spectral pretraining.

Implements the three-stage training pipeline:
  Stage 1 — MESIE pretraining (offline):
      Self-supervised + contrastive + auxiliary spectral tasks.
  Stage 2 — Environment pretraining (sim / twins):
      Plug MESIE into agents as observation encoder.
      Train agents with RL/IL where spectral understanding matters.
  Stage 3 — Fine-tuning (real world):
      Use logged spectral data from real systems.
      Continue training MESIE + policies with conservative objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mesie.pretraining.world_tasks import WorldTaskSuite
from mesie.pretraining.digital_twin import DigitalTwinEnvironment, SpectralStream
from mesie.pretraining.spectral_memory import SpectralMemoryStore


class StageStatus(Enum):
    """Status of a training stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageMetrics:
    """Metrics collected during a training stage.

    Attributes
    ----------
    losses : list of float
        Loss values per step.
    rewards : list of float
        Reward values per step (for RL stages).
    metrics : dict
        Additional named metrics.
    best_loss : float
        Best (lowest) loss achieved.
    best_step : int
        Step at which best loss was achieved.
    """

    losses: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    best_loss: float = float("inf")
    best_step: int = 0

    def update(self, loss: float, step: int, **kwargs: float) -> None:
        """Record a new step's metrics."""
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = step
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    @property
    def mean_loss(self) -> float:
        """Mean loss across all steps."""
        return float(np.mean(self.losses)) if self.losses else float("inf")

    @property
    def mean_reward(self) -> float:
        """Mean reward across all steps."""
        return float(np.mean(self.rewards)) if self.rewards else 0.0


# ---------------------------------------------------------------------------
# Stage 1: Self-supervised pretraining
# ---------------------------------------------------------------------------


class PretrainingStage:
    """Stage 1: Self-supervised spectral pretraining.

    Runs the WorldTaskSuite on offline spectral data to learn spectral
    representations through auxiliary self-supervised objectives.

    Parameters
    ----------
    world_tasks : WorldTaskSuite or None
        Task suite configuration. Created with defaults if None.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Learning rate (conceptual, for integration with external optimizers).
    contrastive_weight : float
        Weight for contrastive learning objective.
    """

    def __init__(
        self,
        world_tasks: Optional[WorldTaskSuite] = None,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        contrastive_weight: float = 0.5,
    ):
        self.world_tasks = world_tasks or WorldTaskSuite()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contrastive_weight = contrastive_weight
        self.status = StageStatus.PENDING
        self.metrics = StageMetrics()

    def run(
        self,
        frequencies: np.ndarray,
        spectral_data: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        encoder_fn: Optional[Callable] = None,
    ) -> StageMetrics:
        """Execute Stage 1 pretraining.

        Parameters
        ----------
        frequencies : ndarray, shape (n_freq,)
            Frequency axis.
        spectral_data : ndarray, shape (n_samples, n_freq)
            Training spectral data.
        embeddings : ndarray or None, shape (n_samples, embed_dim)
            Pre-computed embeddings. If None and encoder_fn is provided,
            embeddings are computed on-the-fly.
        encoder_fn : callable or None
            Function mapping spectra -> embeddings.

        Returns
        -------
        StageMetrics
            Training metrics.
        """
        self.status = StageStatus.RUNNING
        spectral_data = np.atleast_2d(spectral_data)
        n_samples = spectral_data.shape[0]

        # Compute embeddings if needed
        if embeddings is None and encoder_fn is not None:
            embeddings = np.array([encoder_fn(s) for s in spectral_data])

        try:
            for epoch in range(self.n_epochs):
                epoch_losses = []

                # Mini-batch iteration
                indices = np.random.permutation(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    batch_idx = indices[start: start + self.batch_size]
                    batch_spectra = spectral_data[batch_idx]
                    batch_embeddings = (
                        embeddings[batch_idx] if embeddings is not None else None
                    )

                    # Generate labels from all world tasks
                    labels = self.world_tasks.evaluate_all(
                        frequencies=frequencies,
                        amplitudes=batch_spectra,
                        embeddings=batch_embeddings,
                    )

                    # Compute combined loss (simulated)
                    batch_loss = self._compute_batch_loss(labels, batch_spectra)
                    epoch_losses.append(batch_loss)

                # Record epoch metrics
                mean_epoch_loss = float(np.mean(epoch_losses))
                self.metrics.update(mean_epoch_loss, epoch)

            self.status = StageStatus.COMPLETED
        except Exception:
            self.status = StageStatus.FAILED
            raise

        return self.metrics

    def _compute_batch_loss(
        self, labels: Dict, batch_spectra: np.ndarray
    ) -> float:
        """Compute combined self-supervised loss for a batch."""
        task_losses = {}

        # Resonance task loss (using self-supervised labels as targets)
        if "resonance" in labels:
            scores = labels["resonance"]["resonance_score"]
            # Simulated prediction: perturbed score
            predictions = scores + np.random.randn(len(scores)) * 0.1
            task_losses["resonance"] = self.world_tasks.resonance_head.compute_loss(
                np.clip(predictions / 10.0, 0, 1),
                labels["resonance"]["classification"],
                task="classification",
            )

        # Harmonic task loss
        if "harmonic" in labels:
            target = labels["harmonic"]["harmonic_amplitudes"]
            predicted = target + np.random.randn(*target.shape) * 0.05
            task_losses["harmonic"] = self.world_tasks.harmonic_head.compute_loss(
                predicted, target, task="reconstruction"
            )

        # Drift task loss
        if "drift" in labels:
            target = labels["drift"]["drift_scores"]
            predicted = target + np.random.randn(*target.shape) * 0.1
            task_losses["drift"] = self.world_tasks.drift_head.compute_loss(
                predicted, target
            )

        # Lineage task loss
        if "lineage" in labels:
            targets = labels["lineage"]["targets"]
            if targets.shape[0] > 0:
                predicted = targets + np.random.randn(*targets.shape) * 0.1
                task_losses["lineage"] = self.world_tasks.lineage_head.compute_loss(
                    predicted, targets
                )

        # Contrastive loss (simplified: encourage embeddings of augmented
        # versions to be similar)
        contrastive_loss = self._contrastive_loss(batch_spectra)
        task_losses["contrastive"] = contrastive_loss * self.contrastive_weight

        return self.world_tasks.compute_total_loss(task_losses)

    def _contrastive_loss(self, spectra: np.ndarray) -> float:
        """Simplified contrastive loss via spectral augmentation."""
        if spectra.shape[0] < 2:
            return 0.0

        # Create augmented views (frequency masking + noise)
        augmented = spectra + np.random.randn(*spectra.shape) * 0.01

        # Positive pairs should be similar
        positive_dist = np.mean(np.sqrt(np.sum((spectra - augmented) ** 2, axis=1)))

        # Negative pairs (different samples) should be dissimilar
        rolled = np.roll(spectra, 1, axis=0)
        negative_dist = np.mean(np.sqrt(np.sum((spectra - rolled) ** 2, axis=1)))

        # Contrastive margin loss
        margin = 1.0
        loss = max(0.0, positive_dist - negative_dist + margin)
        return loss


# ---------------------------------------------------------------------------
# Stage 2: Environment pretraining
# ---------------------------------------------------------------------------


class EnvironmentStage:
    """Stage 2: Agent pretraining in digital twin environments.

    Plugs the MESIE backbone into agents as an observation encoder and
    trains with RL/IL using spectral reasoning reward signals.

    Parameters
    ----------
    environment : DigitalTwinEnvironment or None
        Simulation environment. Created with factory defaults if None.
    memory_store : SpectralMemoryStore or None
        Memory store for lineage-conditioned policies.
    n_episodes : int
        Number of training episodes.
    max_steps_per_episode : int
        Maximum steps per episode.
    """

    def __init__(
        self,
        environment: Optional[DigitalTwinEnvironment] = None,
        memory_store: Optional[SpectralMemoryStore] = None,
        n_episodes: int = 100,
        max_steps_per_episode: int = 500,
    ):
        self.environment = environment or DigitalTwinEnvironment.create_factory_environment()
        self.memory_store = memory_store or SpectralMemoryStore()
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.status = StageStatus.PENDING
        self.metrics = StageMetrics()

    def run(
        self,
        encoder_fn: Optional[Callable] = None,
        policy_fn: Optional[Callable] = None,
    ) -> StageMetrics:
        """Execute Stage 2 environment pretraining.

        Parameters
        ----------
        encoder_fn : callable or None
            Function mapping spectrum -> embedding.
            Defaults to simple normalization.
        policy_fn : callable or None
            Function mapping (embedding, context) -> action.
            Defaults to random exploration.

        Returns
        -------
        StageMetrics
            Training metrics.
        """
        self.status = StageStatus.RUNNING

        if encoder_fn is None:
            encoder_fn = self._default_encoder
        if policy_fn is None:
            policy_fn = self._default_policy

        try:
            for episode in range(self.n_episodes):
                episode_reward = 0.0
                observations = self.environment.reset()

                for step in range(self.max_steps_per_episode):
                    # Encode observations
                    entity_embeddings = {}
                    for entity_id, spectrum in observations.items():
                        embedding = encoder_fn(spectrum)
                        entity_embeddings[entity_id] = embedding

                        # Store in memory
                        event_type = self._detect_event(spectrum)
                        self.memory_store.store(
                            timestamp=float(step),
                            embedding=embedding,
                            event_type=event_type,
                            metadata={"entity_id": entity_id, "episode": episode},
                        )

                    # Get lineage-conditioned representations
                    actions = {}
                    for entity_id, embedding in entity_embeddings.items():
                        context = self.memory_store.get_lineage(embedding)
                        action = policy_fn(context)
                        actions[entity_id] = action

                    # Step environment
                    observations, rewards, done, info = self.environment.step(actions)
                    episode_reward += rewards.total

                    if done:
                        break

                # Record episode metrics
                self.metrics.rewards.append(episode_reward)
                self.metrics.update(
                    loss=-episode_reward,  # Use negative reward as loss proxy
                    step=episode,
                    episode_reward=episode_reward,
                )

                # Decay memory importance
                self.memory_store.decay_importance()

            self.status = StageStatus.COMPLETED
        except Exception:
            self.status = StageStatus.FAILED
            raise

        return self.metrics

    def _default_encoder(self, spectrum: np.ndarray) -> np.ndarray:
        """Default encoder: normalized spectrum features."""
        features = np.array([
            np.mean(spectrum),
            np.std(spectrum),
            np.max(spectrum),
            np.argmax(spectrum) / len(spectrum),
            np.sum(spectrum ** 2),
        ])
        norm = np.linalg.norm(features)
        return features / max(norm, 1e-8)

    def _default_policy(self, context: np.ndarray) -> np.ndarray:
        """Default policy: random exploration with small perturbations."""
        return np.random.randn(len(self.environment.frequencies)) * 0.01

    def _detect_event(self, spectrum: np.ndarray) -> str:
        """Simple event detection from a spectrum."""
        mean_amp = np.mean(spectrum)
        if mean_amp > 0:
            peak_ratio = np.max(spectrum) / mean_amp
            if peak_ratio > 8.0:
                return "resonance"
            elif peak_ratio > 5.0:
                return "near_resonance"
        return "normal"


# ---------------------------------------------------------------------------
# Stage 3: Fine-tuning
# ---------------------------------------------------------------------------


class FineTuningStage:
    """Stage 3: Fine-tuning on real-world spectral data.

    Continues training the MESIE backbone and policies using logged spectral
    data from real systems with conservative objectives to avoid catastrophic
    forgetting.

    Parameters
    ----------
    conservative_coeff : float
        Coefficient for conservative regularization (KL penalty from pretrained).
    replay_ratio : float
        Fraction of each batch from replay buffer (pretrained data).
    n_epochs : int
        Number of fine-tuning epochs.
    batch_size : int
        Batch size.
    """

    def __init__(
        self,
        conservative_coeff: float = 0.1,
        replay_ratio: float = 0.25,
        n_epochs: int = 50,
        batch_size: int = 16,
    ):
        self.conservative_coeff = conservative_coeff
        self.replay_ratio = replay_ratio
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.status = StageStatus.PENDING
        self.metrics = StageMetrics()

    def run(
        self,
        real_data: np.ndarray,
        frequencies: np.ndarray,
        pretrained_embeddings: Optional[np.ndarray] = None,
        encoder_fn: Optional[Callable] = None,
        world_tasks: Optional[WorldTaskSuite] = None,
    ) -> StageMetrics:
        """Execute Stage 3 fine-tuning.

        Parameters
        ----------
        real_data : ndarray, shape (n_samples, n_freq)
            Real-world spectral data.
        frequencies : ndarray, shape (n_freq,)
            Frequency axis.
        pretrained_embeddings : ndarray or None
            Embeddings from Stage 1 for conservative regularization.
        encoder_fn : callable or None
            Encoder function.
        world_tasks : WorldTaskSuite or None
            Task suite for auxiliary losses.

        Returns
        -------
        StageMetrics
            Fine-tuning metrics.
        """
        self.status = StageStatus.RUNNING
        real_data = np.atleast_2d(real_data)
        n_samples = real_data.shape[0]

        if world_tasks is None:
            world_tasks = WorldTaskSuite()

        try:
            for epoch in range(self.n_epochs):
                epoch_losses = []
                indices = np.random.permutation(n_samples)

                for start in range(0, n_samples, self.batch_size):
                    batch_idx = indices[start: start + self.batch_size]
                    batch = real_data[batch_idx]

                    # Task loss on real data
                    labels = world_tasks.evaluate_all(
                        frequencies=frequencies,
                        amplitudes=batch,
                    )

                    task_loss = self._compute_task_loss(labels, world_tasks)

                    # Conservative regularization
                    conservative_loss = 0.0
                    if pretrained_embeddings is not None and encoder_fn is not None:
                        current_emb = np.array([encoder_fn(s) for s in batch])
                        # Use subset of pretrained embeddings
                        replay_idx = np.random.choice(
                            len(pretrained_embeddings),
                            size=min(len(batch_idx), len(pretrained_embeddings)),
                            replace=False,
                        )
                        replay_emb = pretrained_embeddings[replay_idx]
                        # KL-like divergence penalty
                        conservative_loss = float(
                            np.mean((current_emb[:len(replay_emb)] - replay_emb) ** 2)
                        ) * self.conservative_coeff

                    total_loss = task_loss + conservative_loss
                    epoch_losses.append(total_loss)

                mean_loss = float(np.mean(epoch_losses))
                self.metrics.update(mean_loss, epoch)

            self.status = StageStatus.COMPLETED
        except Exception:
            self.status = StageStatus.FAILED
            raise

        return self.metrics

    def _compute_task_loss(
        self, labels: Dict, world_tasks: WorldTaskSuite
    ) -> float:
        """Compute task losses for fine-tuning."""
        task_losses = {}

        if "resonance" in labels:
            scores = labels["resonance"]["resonance_score"]
            predictions = scores + np.random.randn(len(scores)) * 0.05
            task_losses["resonance"] = world_tasks.resonance_head.compute_loss(
                np.clip(predictions / 10.0, 0, 1),
                labels["resonance"]["classification"],
                task="classification",
            )

        if "harmonic" in labels:
            target = labels["harmonic"]["harmonic_amplitudes"]
            predicted = target + np.random.randn(*target.shape) * 0.02
            task_losses["harmonic"] = world_tasks.harmonic_head.compute_loss(
                predicted, target, task="reconstruction"
            )

        return world_tasks.compute_total_loss(task_losses)


# ---------------------------------------------------------------------------
# Training Recipe Orchestrator
# ---------------------------------------------------------------------------


class TrainingRecipe:
    """Orchestrates the full three-stage training pipeline.

    Manages the lifecycle of:
      Stage 1: Self-supervised pretraining with world tasks
      Stage 2: Environment pretraining in digital twins
      Stage 3: Fine-tuning on real-world data

    Parameters
    ----------
    stage1 : PretrainingStage or None
        Stage 1 configuration.
    stage2 : EnvironmentStage or None
        Stage 2 configuration.
    stage3 : FineTuningStage or None
        Stage 3 configuration.
    """

    def __init__(
        self,
        stage1: Optional[PretrainingStage] = None,
        stage2: Optional[EnvironmentStage] = None,
        stage3: Optional[FineTuningStage] = None,
    ):
        self.stage1 = stage1 or PretrainingStage()
        self.stage2 = stage2 or EnvironmentStage()
        self.stage3 = stage3 or FineTuningStage()

    @property
    def stages(self) -> List:
        """Get all stages in order."""
        return [self.stage1, self.stage2, self.stage3]

    @property
    def status(self) -> Dict[str, StageStatus]:
        """Get status of all stages."""
        return {
            "stage1_pretraining": self.stage1.status,
            "stage2_environment": self.stage2.status,
            "stage3_finetuning": self.stage3.status,
        }

    def run_stage1(
        self,
        frequencies: np.ndarray,
        spectral_data: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        encoder_fn: Optional[Callable] = None,
    ) -> StageMetrics:
        """Run Stage 1: Self-supervised pretraining.

        Parameters
        ----------
        frequencies : ndarray
            Frequency axis.
        spectral_data : ndarray
            Training data.
        embeddings : ndarray or None
            Pre-computed embeddings.
        encoder_fn : callable or None
            Encoder function.

        Returns
        -------
        StageMetrics
        """
        return self.stage1.run(frequencies, spectral_data, embeddings, encoder_fn)

    def run_stage2(
        self,
        encoder_fn: Optional[Callable] = None,
        policy_fn: Optional[Callable] = None,
    ) -> StageMetrics:
        """Run Stage 2: Environment pretraining.

        Parameters
        ----------
        encoder_fn : callable or None
            Observation encoder.
        policy_fn : callable or None
            Agent policy.

        Returns
        -------
        StageMetrics
        """
        return self.stage2.run(encoder_fn, policy_fn)

    def run_stage3(
        self,
        real_data: np.ndarray,
        frequencies: np.ndarray,
        pretrained_embeddings: Optional[np.ndarray] = None,
        encoder_fn: Optional[Callable] = None,
    ) -> StageMetrics:
        """Run Stage 3: Fine-tuning.

        Parameters
        ----------
        real_data : ndarray
            Real-world spectral data.
        frequencies : ndarray
            Frequency axis.
        pretrained_embeddings : ndarray or None
            Pretrained embeddings for conservative objectives.
        encoder_fn : callable or None
            Encoder function.

        Returns
        -------
        StageMetrics
        """
        return self.stage3.run(real_data, frequencies, pretrained_embeddings, encoder_fn)

    def run_all(
        self,
        frequencies: np.ndarray,
        pretrain_data: np.ndarray,
        real_data: Optional[np.ndarray] = None,
        encoder_fn: Optional[Callable] = None,
        policy_fn: Optional[Callable] = None,
    ) -> Dict[str, StageMetrics]:
        """Run the full three-stage training pipeline.

        Parameters
        ----------
        frequencies : ndarray
            Shared frequency axis.
        pretrain_data : ndarray
            Data for Stage 1 pretraining.
        real_data : ndarray or None
            Real data for Stage 3. Defaults to pretrain_data.
        encoder_fn : callable or None
            Shared encoder function.
        policy_fn : callable or None
            Agent policy for Stage 2.

        Returns
        -------
        dict mapping stage name -> StageMetrics
        """
        results = {}

        # Stage 1
        results["stage1"] = self.run_stage1(
            frequencies, pretrain_data, encoder_fn=encoder_fn
        )

        # Stage 2
        results["stage2"] = self.run_stage2(encoder_fn, policy_fn)

        # Stage 3
        if real_data is None:
            real_data = pretrain_data
        results["stage3"] = self.run_stage3(
            real_data, frequencies, encoder_fn=encoder_fn
        )

        return results
