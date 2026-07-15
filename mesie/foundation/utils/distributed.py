"""Distributed training simulation for spectral pretraining.

Provides abstractions for data-parallel and model-parallel
training strategies with gradient synchronization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        world_size: Total number of processes.
        rank: Current process rank.
        local_rank: Local GPU rank.
        num_nodes: Number of nodes.
        gpus_per_node: GPUs per node.
        backend: Communication backend.
        gradient_compression: Whether to compress gradients.
        overlap_comm: Whether to overlap communication.
    """

    world_size: int = 8
    rank: int = 0
    local_rank: int = 0
    num_nodes: int = 1
    gpus_per_node: int = 8
    backend: str = "nccl"
    gradient_compression: bool = False
    overlap_comm: bool = True
    fp16_allreduce: bool = True
    bucket_size_mb: int = 25
    find_unused_parameters: bool = False

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size across all processes."""
        return self.world_size  # Multiplied by per-GPU batch size


class AllReduceSimulator:
    """Simulates all-reduce operations for gradient synchronization.

    In production, this would use NCCL/MPI. Here we simulate
    the behavior for correctness verification.

    Attributes:
        world_size: Number of simulated workers.
        algorithm: Reduction algorithm.
    """

    def __init__(
        self,
        world_size: int = 8,
        algorithm: str = "ring",
        compression: str = "none",
    ):
        """Initialize all-reduce simulator.

        Args:
            world_size: Number of workers.
            algorithm: Algorithm ('ring', 'tree', 'recursive_halving').
            compression: Gradient compression ('none', 'topk', 'random_k').
        """
        self.world_size = world_size
        self.algorithm = algorithm
        self.compression = compression

        # Statistics
        self.total_bytes_communicated = 0
        self.num_allreduce_ops = 0

    def all_reduce(
        self,
        gradients_per_worker: List[List[np.ndarray]],
        operation: str = "mean",
    ) -> List[np.ndarray]:
        """Simulate all-reduce across workers.

        Args:
            gradients_per_worker: Gradients from each worker.
            operation: Reduction operation ('sum', 'mean').

        Returns:
            Reduced gradients (same on all workers).
        """
        self.num_allreduce_ops += 1

        num_params = len(gradients_per_worker[0])
        reduced = []

        for param_idx in range(num_params):
            # Gather from all workers
            param_grads = [
                worker_grads[param_idx]
                for worker_grads in gradients_per_worker
            ]

            # Apply compression if configured
            if self.compression == "topk":
                param_grads = [self._topk_compress(g, k=0.1) for g in param_grads]
            elif self.compression == "random_k":
                param_grads = [self._random_compress(g, k=0.1) for g in param_grads]

            # Reduce
            stacked = np.stack(param_grads, axis=0)
            if operation == "sum":
                result = np.sum(stacked, axis=0)
            else:  # mean
                result = np.mean(stacked, axis=0)

            reduced.append(result)

            # Track communication
            self.total_bytes_communicated += result.nbytes * 2 * (self.world_size - 1)

        return reduced

    def _topk_compress(self, gradient: np.ndarray, k: float = 0.1) -> np.ndarray:
        """Top-k gradient compression.

        Args:
            gradient: Full gradient.
            k: Fraction of elements to keep.

        Returns:
            Compressed gradient (zeros except top-k values).
        """
        flat = gradient.flatten()
        num_keep = max(1, int(len(flat) * k))

        # Keep top-k by magnitude
        indices = np.argsort(np.abs(flat))[-num_keep:]
        compressed = np.zeros_like(flat)
        compressed[indices] = flat[indices]

        return compressed.reshape(gradient.shape)

    def _random_compress(self, gradient: np.ndarray, k: float = 0.1) -> np.ndarray:
        """Random-k gradient compression.

        Args:
            gradient: Full gradient.
            k: Fraction of elements to keep.

        Returns:
            Compressed gradient.
        """
        mask = np.random.random(gradient.shape) < k
        return gradient * mask / k  # Scale to maintain expected value

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            "total_bytes_communicated": self.total_bytes_communicated,
            "total_mb_communicated": self.total_bytes_communicated / (1024 * 1024),
            "num_allreduce_ops": self.num_allreduce_ops,
            "world_size": self.world_size,
            "algorithm": self.algorithm,
            "compression": self.compression,
        }


class DataParallelWrapper:
    """Simulates data-parallel training.

    Distributes batches across workers and synchronizes gradients.

    Attributes:
        world_size: Number of parallel workers.
        all_reduce: All-reduce simulator.
    """

    def __init__(
        self,
        world_size: int = 8,
        gradient_compression: str = "none",
    ):
        """Initialize data parallel wrapper.

        Args:
            world_size: Number of workers.
            gradient_compression: Compression strategy.
        """
        self.world_size = world_size
        self.all_reduce = AllReduceSimulator(
            world_size=world_size,
            compression=gradient_compression,
        )

    def distribute_batch(
        self, batch: Dict[str, np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """Split batch across workers.

        Args:
            batch: Full batch.

        Returns:
            Per-worker mini-batches.
        """
        per_worker = []
        for rank in range(self.world_size):
            worker_batch = {}
            for key, value in batch.items():
                batch_size = value.shape[0]
                per_worker_size = batch_size // self.world_size
                start = rank * per_worker_size
                end = start + per_worker_size
                worker_batch[key] = value[start:end]
            per_worker.append(worker_batch)
        return per_worker

    def synchronize_gradients(
        self,
        all_gradients: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Synchronize gradients across workers.

        Args:
            all_gradients: Per-worker gradient lists.

        Returns:
            Synchronized (averaged) gradients.
        """
        return self.all_reduce.all_reduce(all_gradients, operation="mean")


class DistributedTrainer:
    """Full distributed training coordinator.

    Manages the complete distributed training workflow including:
    - Data distribution
    - Forward pass coordination
    - Gradient synchronization
    - Parameter broadcasting
    - Checkpoint coordination

    Attributes:
        config: Distributed configuration.
        data_parallel: Data parallel wrapper.
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        """Initialize distributed trainer.

        Args:
            config: Distributed training configuration.
        """
        self.config = config or DistributedConfig()
        self.data_parallel = DataParallelWrapper(
            world_size=self.config.world_size,
            gradient_compression="topk" if self.config.gradient_compression else "none",
        )

        # Training state
        self._step = 0
        self._global_batch_count = 0

    def prepare_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare batch for distributed training.

        Extracts this rank's portion of the batch.

        Args:
            batch: Full global batch.

        Returns:
            Local batch for this rank.
        """
        per_worker_batches = self.data_parallel.distribute_batch(batch)
        return per_worker_batches[self.config.rank]

    def sync_gradients(
        self, local_gradients: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Synchronize local gradients with all workers.

        In simulation, we just return the local gradients divided by world_size.

        Args:
            local_gradients: This rank's gradients.

        Returns:
            Synchronized gradients.
        """
        # In simulation, assume all workers have same gradients
        all_worker_grads = [local_gradients] * self.config.world_size
        return self.data_parallel.synchronize_gradients(all_worker_grads)

    def broadcast_params(
        self, params: List[np.ndarray], src_rank: int = 0
    ) -> List[np.ndarray]:
        """Broadcast parameters from source rank.

        Args:
            params: Parameters to broadcast.
            src_rank: Source rank.

        Returns:
            Broadcasted parameters (same on all ranks).
        """
        # In simulation, just return params
        return params

    def train_step(
        self,
        batch: Dict[str, np.ndarray],
        compute_fn: Any,
        optimizer: Any,
    ) -> Dict[str, float]:
        """Execute one distributed training step.

        Args:
            batch: Global batch.
            compute_fn: Function that computes loss and gradients.
            optimizer: Optimizer to update parameters.

        Returns:
            Loss metrics.
        """
        self._step += 1

        # Distribute batch
        local_batch = self.prepare_batch(batch)

        # Compute local gradients and loss
        local_grads, loss_dict = compute_fn(local_batch)

        # Synchronize gradients
        synced_grads = self.sync_gradients(local_grads)

        # Optimizer step (only on main process in some implementations)
        optimizer.step(synced_grads)

        self._global_batch_count += self.config.world_size

        return loss_dict

    def get_statistics(self) -> Dict[str, Any]:
        """Get distributed training statistics."""
        return {
            "config": {
                "world_size": self.config.world_size,
                "num_nodes": self.config.num_nodes,
                "gpus_per_node": self.config.gpus_per_node,
                "backend": self.config.backend,
            },
            "training": {
                "step": self._step,
                "global_batch_count": self._global_batch_count,
            },
            "communication": self.data_parallel.all_reduce.get_statistics(),
        }
