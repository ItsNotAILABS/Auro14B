"""Mixture-of-Experts layers for SpectralGPT with adaptive token compute.

The router can vary the number of active specialist experts per token from
``min_k`` through ``top_k``. Difficulty is estimated from normalized routing
entropy and low router confidence, optionally blended with an external
``difficulty_signal`` supplied by a higher AURO recurrent/surprise controller.
Unused expert slots are emitted as index ``-1`` with zero weight and are never
executed, so adaptive routing changes real expert compute rather than telemetry.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ExpertLayer:
    """Single gated feed-forward expert."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 4096,
        output_dim: int = 1024,
        activation: str = "swiglu",
        expert_id: int = 0,
        specialization: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.expert_id = expert_id
        self.specialization = specialization
        self.gate_proj = np.random.randn(input_dim, hidden_dim) * 0.02
        self.up_proj = np.random.randn(input_dim, hidden_dim) * 0.02
        self.down_proj = np.random.randn(hidden_dim, output_dim) * 0.02
        self.total_tokens_processed = 0
        self.load_history: List[float] = []

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation in ("swiglu", "silu"):
            return x * (1.0 / (1.0 + np.exp(-x)))
        if self.activation in ("geglu", "gelu"):
            return 0.5 * x * (1.0 + np.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        if self.activation == "relu":
            return np.maximum(0, x)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = self._activate(np.einsum("...d,dh->...h", x, self.gate_proj))
        up = np.einsum("...d,dh->...h", x, self.up_proj)
        output = np.einsum("...h,ho->...o", gate * up, self.down_proj)
        self.total_tokens_processed += int(np.prod(x.shape[:-1]))
        return output

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "specialization": self.specialization,
            "total_tokens": self.total_tokens_processed,
            "param_count": self.input_dim * self.hidden_dim * 2 + self.hidden_dim * self.output_dim,
        }


class TopKRouter:
    """Top-K router with adaptive per-token specialist budget."""

    def __init__(
        self,
        input_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noise_std: float = 0.1,
        use_aux_loss: bool = True,
        jitter_noise: bool = True,
        adaptive_compute: bool = True,
        min_k: int = 1,
        external_difficulty_weight: float = 0.50,
    ):
        self.input_dim = input_dim
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.min_k = max(1, min(int(min_k), self.top_k))
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.use_aux_loss = use_aux_loss
        self.jitter_noise = jitter_noise
        self.adaptive_compute = adaptive_compute
        self.external_difficulty_weight = float(np.clip(external_difficulty_weight, 0.0, 1.0))
        self.router_weights = np.random.randn(input_dim, num_experts) * 0.02
        self.router_bias = np.zeros(num_experts)
        self.expert_counts = np.zeros(num_experts)
        self.total_routed = 0
        self.total_specialist_activations = 0
        self.last_active_k_histogram = {k: 0 for k in range(self.min_k, self.top_k + 1)}

    def _probs(self, logits: np.ndarray) -> np.ndarray:
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)

    def _difficulty(self, probs: np.ndarray, external: Optional[np.ndarray] = None) -> np.ndarray:
        flat = probs.reshape(-1, self.num_experts)
        entropy = -np.sum(flat * np.log(flat + 1e-10), axis=-1)
        entropy /= max(math.log(max(2, self.num_experts)), 1e-10)
        confidence_gap = 1.0 - np.max(flat, axis=-1)
        intrinsic = np.clip(0.65 * entropy + 0.35 * confidence_gap, 0.0, 1.0)
        if external is None:
            return intrinsic
        ext = np.asarray(external, dtype=float)
        if ext.ndim == 0:
            ext = np.full_like(intrinsic, float(ext))
        else:
            ext = np.broadcast_to(ext, probs.shape[:-1]).reshape(-1)
        ext = np.clip(ext, 0.0, 1.0)
        w = self.external_difficulty_weight
        return np.clip((1.0 - w) * intrinsic + w * ext, 0.0, 1.0)

    def _adaptive_select(
        self,
        probs: np.ndarray,
        difficulty_signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        flat = probs.reshape(-1, self.num_experts)
        difficulty = self._difficulty(probs, difficulty_signal)
        num_tokens = flat.shape[0]
        indices_out = np.full((num_tokens, self.top_k), -1, dtype=np.int64)
        weights_out = np.zeros((num_tokens, self.top_k), dtype=np.float64)
        histogram = {k: 0 for k in range(self.min_k, self.top_k + 1)}
        active_counts = np.zeros(num_tokens, dtype=np.int64)

        for token_idx in range(num_tokens):
            if self.adaptive_compute and self.top_k > self.min_k:
                span = self.top_k - self.min_k
                active_k = self.min_k + int(np.floor(difficulty[token_idx] * (span + 1)))
                active_k = min(self.top_k, max(self.min_k, active_k))
            else:
                active_k = self.top_k
            active_counts[token_idx] = active_k
            histogram[active_k] += 1
            chosen = np.argsort(flat[token_idx])[::-1][:active_k]
            chosen_weights = flat[token_idx][chosen]
            chosen_weights /= np.sum(chosen_weights) + 1e-10
            indices_out[token_idx, :active_k] = chosen
            weights_out[token_idx, :active_k] = chosen_weights

        self.last_active_k_histogram = histogram
        return indices_out, weights_out, {
            "token_difficulty_mean": float(np.mean(difficulty)) if len(difficulty) else 0.0,
            "token_difficulty_max": float(np.max(difficulty)) if len(difficulty) else 0.0,
            "active_k_mean": float(np.mean(active_counts)) if len(active_counts) else 0.0,
            "active_k_min": int(np.min(active_counts)) if len(active_counts) else 0,
            "active_k_max": int(np.max(active_counts)) if len(active_counts) else 0,
            "active_k_histogram": histogram,
            "specialist_compute_fraction": float(np.mean(active_counts) / self.top_k) if self.top_k else 0.0,
            "adaptive_compute": self.adaptive_compute,
            "min_k": self.min_k,
            "max_k": self.top_k,
        }

    def _finalize_route(
        self,
        x: np.ndarray,
        probs: np.ndarray,
        difficulty_signal: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        flat_indices, flat_weights, adaptive_info = self._adaptive_select(probs, difficulty_signal)
        for idx in flat_indices.ravel():
            if idx >= 0:
                self.expert_counts[idx] += 1
                self.total_specialist_activations += 1
        num_tokens = flat_indices.shape[0]
        self.total_routed += num_tokens
        shape = x.shape[:-1]
        info: Dict[str, Any] = {
            "expert_utilization": self.expert_counts / max(1, self.total_specialist_activations),
            "load_balance_loss": self._compute_load_balance_loss(probs),
            "router_entropy": self._compute_entropy(probs),
            **adaptive_info,
        }
        if extra_info:
            info.update(extra_info)
        return (
            flat_indices.reshape(*shape, self.top_k),
            flat_weights.reshape(*shape, self.top_k),
            info,
        )

    def route(
        self,
        x: np.ndarray,
        training: bool = True,
        difficulty_signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        logits = np.einsum("...d,de->...e", x, self.router_weights) + self.router_bias
        if training and self.jitter_noise:
            logits = logits + np.random.randn(*logits.shape) * self.noise_std
        probs = self._probs(logits)
        return self._finalize_route(x, probs, difficulty_signal)

    def _compute_load_balance_loss(self, probs: np.ndarray) -> float:
        if not self.use_aux_loss:
            return 0.0
        flat = probs.reshape(-1, self.num_experts)
        fraction = np.mean(flat, axis=0)
        return float(self.num_experts * np.sum(fraction * fraction))

    def _compute_entropy(self, probs: np.ndarray) -> float:
        flat = probs.reshape(-1, self.num_experts)
        entropy = -np.sum(flat * np.log(flat + 1e-10), axis=-1)
        return float(np.mean(entropy))

    def reset_statistics(self) -> None:
        self.expert_counts = np.zeros(self.num_experts)
        self.total_routed = 0
        self.total_specialist_activations = 0
        self.last_active_k_histogram = {k: 0 for k in range(self.min_k, self.top_k + 1)}


class ModalityAwareRouter(TopKRouter):
    """Adaptive router with modality-specific expert priors."""

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
        super().__init__(input_dim, num_experts, top_k, **kwargs)
        self.num_modalities = num_modalities
        self.modality_names = modality_names or [
            "seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"
        ]
        if len(self.modality_names) < num_modalities:
            self.modality_names = self.modality_names + [
                f"modality_{i}" for i in range(len(self.modality_names), num_modalities)
            ]
        self.cross_modality_sharing = cross_modality_sharing
        self.modality_embeddings = np.random.randn(num_modalities, input_dim) * 0.02
        self.modality_expert_affinity = self._initialize_affinity()
        self.modality_bias = np.zeros((num_modalities, num_experts))

    def _initialize_affinity(self) -> np.ndarray:
        affinity = np.ones((self.num_modalities, self.num_experts)) * self.cross_modality_sharing
        experts_per_modality = max(1, self.num_experts // self.num_modalities)
        for modality in range(self.num_modalities):
            start = (modality * experts_per_modality) % self.num_experts
            for offset in range(experts_per_modality):
                affinity[modality, (start + offset) % self.num_experts] = 1.0
        affinity /= affinity.sum(axis=-1, keepdims=True)
        return affinity

    def route_with_modality(
        self,
        x: np.ndarray,
        modality_id: int,
        training: bool = True,
        difficulty_signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if not 0 <= int(modality_id) < self.num_modalities:
            raise ValueError(f"invalid modality_id {modality_id}")
        modality_id = int(modality_id)
        x_modality = x + self.modality_embeddings[modality_id]
        logits = np.einsum("...d,de->...e", x_modality, self.router_weights) + self.router_bias
        logits = logits + self.modality_bias[modality_id]
        affinity = self.modality_expert_affinity[modality_id]
        # Log-prior rather than multiplication preserves logit ordering geometry.
        logits = logits + np.log(affinity + 1e-10)
        if training and self.jitter_noise:
            logits = logits + np.random.randn(*logits.shape) * self.noise_std
        probs = self._probs(logits)
        return self._finalize_route(
            x,
            probs,
            difficulty_signal,
            {
                "modality": self.modality_names[modality_id],
                "modality_id": modality_id,
                "modality_affinity": affinity.tolist(),
            },
        )


class MixtureOfExperts:
    """Sparse MoE with real token-dependent specialist compute."""

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
        adaptive_compute: bool = True,
        min_k: int = 1,
    ):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = max(1, min(int(top_k), int(num_experts)))
        self.expert_dim = expert_dim
        self.modality_aware = modality_aware
        modality_names = ["seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"]
        self.experts = [
            ExpertLayer(
                input_dim=hidden_dim,
                hidden_dim=expert_dim,
                output_dim=hidden_dim,
                activation=activation,
                expert_id=i,
                specialization=modality_names[i % len(modality_names)],
            )
            for i in range(num_experts)
        ]
        router_cls = ModalityAwareRouter if modality_aware else TopKRouter
        router_kwargs = dict(
            input_dim=hidden_dim,
            num_experts=num_experts,
            top_k=self.top_k,
            capacity_factor=capacity_factor,
            noise_std=noise_std,
            adaptive_compute=adaptive_compute,
            min_k=min_k,
        )
        if modality_aware:
            router_kwargs["num_modalities"] = num_modalities
        self.router = router_cls(**router_kwargs)
        self.shared_expert = ExpertLayer(
            input_dim=hidden_dim,
            hidden_dim=max(1, expert_dim // 2),
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
        difficulty_signal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        original_shape = x.shape
        if self.modality_aware and modality_id is not None:
            expert_indices, expert_weights, routing_info = self.router.route_with_modality(
                x, modality_id, training, difficulty_signal
            )
        else:
            expert_indices, expert_weights, routing_info = self.router.route(
                x, training, difficulty_signal
            )

        flat_x = x.reshape(-1, self.hidden_dim)
        flat_indices = expert_indices.reshape(-1, self.top_k)
        flat_weights = expert_weights.reshape(-1, self.top_k)
        output = np.zeros_like(flat_x)

        # Zero-weight / -1 adaptive slots never execute.
        for k in range(self.top_k):
            slot_indices = flat_indices[:, k]
            slot_weights = flat_weights[:, k]
            active_slot = (slot_indices >= 0) & (slot_weights > 0.0)
            if not np.any(active_slot):
                continue
            for expert_id in np.unique(slot_indices[active_slot]):
                expert_id = int(expert_id)
                mask = active_slot & (slot_indices == expert_id)
                expert_output = self.experts[expert_id].forward(flat_x[mask])
                output[mask] += expert_output * slot_weights[mask, None]

        shared_output = self.shared_expert.forward(flat_x)
        output = output + self.shared_weight * shared_output
        output = output.reshape(original_shape)
        moe_info: Dict[str, Any] = {
            "routing_info": routing_info,
            "expert_load": {
                f"expert_{i}": int(self.experts[i].total_tokens_processed)
                for i in range(self.num_experts)
            },
            "load_balance_loss": routing_info.get("load_balance_loss", 0.0),
            "adaptive_compute": routing_info.get("adaptive_compute", False),
            "specialist_compute_fraction": routing_info.get("specialist_compute_fraction", 1.0),
            "active_k_mean": routing_info.get("active_k_mean", float(self.top_k)),
        }
        return output, moe_info

    def get_expert_statistics(self) -> Dict[str, Any]:
        stats = {f"expert_{e.expert_id}": e.get_statistics() for e in self.experts}
        stats["shared_expert"] = self.shared_expert.get_statistics()
        stats["router_entropy"] = float(
            -np.sum(
                (self.router.expert_counts / max(1, self.router.total_specialist_activations))
                * np.log(
                    self.router.expert_counts / max(1, self.router.total_specialist_activations) + 1e-10
                )
            )
        )
        stats["adaptive_compute"] = {
            "enabled": self.router.adaptive_compute,
            "min_k": self.router.min_k,
            "max_k": self.router.top_k,
            "last_active_k_histogram": dict(self.router.last_active_k_histogram),
            "total_specialist_activations": int(self.router.total_specialist_activations),
        }
        return stats

    def reset_statistics(self) -> None:
        for expert in self.experts:
            expert.total_tokens_processed = 0
            expert.load_history = []
        self.shared_expert.total_tokens_processed = 0
        self.router.reset_statistics()
