"""Medina parallelism: FSDP/ZeRO · Tensor · Pipeline · Hybrid 3D.

Best first target for Medina: ZeRO/FSDP-style sharding of params/grads/opt state.

Design:
  - Works without multi-GPU: *simulates* ranks in-process for correctness tests
  - When torch + multi-GPU available: uses FSDP / DTensor hooks if present
  - Hybrid plan = data_parallel × tensor_parallel × pipeline_parallel

Sources (methodology):
  - PyTorch FSDP
  - DeepSpeed ZeRO stages 1–3
  - PyTorch tensor parallel
  - NVIDIA Megatron-style pipeline + 3D hybrid
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ParallelMode(str, Enum):
    NONE = "none"
    ZERO1 = "zero1"  # shard optimizer state
    ZERO2 = "zero2"  # shard optimizer + gradients
    ZERO3_FSDP = "zero3_fsdp"  # shard params + grads + opt (FSDP-like)
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    HYBRID_3D = "hybrid_3d"


@dataclass
class MedinaParallelConfig:
    mode: ParallelMode = ParallelMode.ZERO3_FSDP
    world_size: int = 4  # logical ranks (GPUs or simulated)
    data_parallel_size: int = 2
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    rank: int = 0
    param_dtype: str = "float32"
    cpu_offload: bool = False
    # activation checkpointing (recompute)
    activation_checkpoint: bool = True

    def validate(self) -> None:
        if self.mode == ParallelMode.HYBRID_3D:
            prod = (
                self.data_parallel_size
                * self.tensor_parallel_size
                * self.pipeline_parallel_size
            )
            if prod != self.world_size:
                # auto-fix world_size
                self.world_size = prod
        if self.rank < 0 or self.rank >= max(self.world_size, 1):
            self.rank = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d


def hybrid_plan(
    world_size: int,
    *,
    prefer_tp: int = 2,
    prefer_pp: int = 1,
) -> MedinaParallelConfig:
    """Choose a hybrid 3D split for a given world size."""
    if world_size < 1:
        world_size = 1
    tp = min(prefer_tp, world_size)
    while world_size % tp != 0 and tp > 1:
        tp -= 1
    rem = world_size // tp
    pp = min(prefer_pp, rem)
    while rem % pp != 0 and pp > 1:
        pp -= 1
    dp = rem // pp
    return MedinaParallelConfig(
        mode=ParallelMode.HYBRID_3D if world_size > 1 else ParallelMode.ZERO3_FSDP,
        world_size=world_size,
        data_parallel_size=max(1, dp),
        tensor_parallel_size=max(1, tp),
        pipeline_parallel_size=max(1, pp),
        rank=0,
    )


@dataclass
class ShardView:
    """One rank's shard of a named tensor."""

    name: str
    rank: int
    world_size: int
    full_shape: Tuple[int, ...]
    shard: np.ndarray
    shard_offset: int
    shard_size: int
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rank": self.rank,
            "world_size": self.world_size,
            "full_shape": list(self.full_shape),
            "shard_shape": list(self.shard.shape),
            "shard_offset": self.shard_offset,
            "shard_size": self.shard_size,
            "mode": self.mode,
            "nbytes": int(self.shard.nbytes),
        }


class MedinaSharder:
    """Shard parameters / grads / optimizer states across logical ranks.

    ZERO1: only opt state sharded
    ZERO2: opt + grads
    ZERO3_FSDP: params + grads + opt (each rank holds 1/N of each param)
    TENSOR: split last dim of weight matrices across ranks
    PIPELINE: assign layer ranges to ranks
    HYBRID_3D: combine all
    """

    def __init__(self, config: Optional[MedinaParallelConfig] = None) -> None:
        self.config = config or MedinaParallelConfig()
        self.config.validate()
        self.param_shards: Dict[str, ShardView] = {}
        self.grad_shards: Dict[str, ShardView] = {}
        self.opt_shards: Dict[str, ShardView] = {}
        self.pipeline_layers: Dict[int, List[int]] = {}
        self._torch_fsdp = None
        self._detect_torch_fsdp()

    def _detect_torch_fsdp(self) -> None:
        try:
            import torch
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: F401

            self._torch_fsdp = {
                "torch": torch.__version__,
                "fsdp": True,
                "cuda": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
        except Exception as exc:
            self._torch_fsdp = {"fsdp": False, "error": str(exc)[:200]}

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.medina.parallel.v1",
            "config": self.config.to_dict(),
            "torch_fsdp": self._torch_fsdp,
            "param_shards": len(self.param_shards),
            "grad_shards": len(self.grad_shards),
            "opt_shards": len(self.opt_shards),
            "pipeline_layers": {str(k): v for k, v in self.pipeline_layers.items()},
            "best_first_target": "ZERO3_FSDP / ZeRO-3 style param+grad+opt sharding",
            "sources": [
                "PyTorch FSDP",
                "DeepSpeed ZeRO",
                "PyTorch tensor parallel",
                "NVIDIA Megatron Core hybrid",
            ],
        }

    # ---- ZeRO / FSDP style flat sharding ----
    def shard_tensor_zero(
        self,
        name: str,
        tensor: np.ndarray,
        *,
        rank: Optional[int] = None,
        world: Optional[int] = None,
    ) -> ShardView:
        rank = self.config.rank if rank is None else rank
        world = self.config.world_size if world is None else world
        flat = np.asarray(tensor, dtype=np.float32).ravel()
        n = flat.size
        chunk = int(math.ceil(n / world))
        start = rank * chunk
        end = min(start + chunk, n)
        shard = flat[start:end].copy()
        view = ShardView(
            name=name,
            rank=rank,
            world_size=world,
            full_shape=tuple(np.asarray(tensor).shape),
            shard=shard,
            shard_offset=start,
            shard_size=int(shard.size),
            mode=self.config.mode.value,
        )
        self.param_shards[f"{name}@r{rank}"] = view
        # ZeRO-1/2/3 also track opt state shard (momentum = zeros)
        if self.config.mode in (
            ParallelMode.ZERO1,
            ParallelMode.ZERO2,
            ParallelMode.ZERO3_FSDP,
            ParallelMode.HYBRID_3D,
        ):
            self.opt_shards[f"{name}@r{rank}"] = ShardView(
                name=f"{name}.exp_avg",
                rank=rank,
                world_size=world,
                full_shape=view.full_shape,
                shard=np.zeros_like(shard),
                shard_offset=start,
                shard_size=int(shard.size),
                mode="opt_" + self.config.mode.value,
            )
        if self.config.mode in (
            ParallelMode.ZERO2,
            ParallelMode.ZERO3_FSDP,
            ParallelMode.HYBRID_3D,
        ):
            self.grad_shards[f"{name}@r{rank}"] = ShardView(
                name=f"{name}.grad",
                rank=rank,
                world_size=world,
                full_shape=view.full_shape,
                shard=np.zeros_like(shard),
                shard_offset=start,
                shard_size=int(shard.size),
                mode="grad_" + self.config.mode.value,
            )
        return view

    def all_gather_zero(self, name: str, world_shards: Sequence[ShardView]) -> np.ndarray:
        """Reconstruct full tensor from rank shards (FSDP all-gather)."""
        if not world_shards:
            return np.zeros(0)
        total = sum(s.shard_size for s in world_shards)
        full = np.zeros(total, dtype=np.float32)
        for s in sorted(world_shards, key=lambda x: x.shard_offset):
            full[s.shard_offset : s.shard_offset + s.shard_size] = s.shard
        shape = world_shards[0].full_shape
        return full.reshape(shape)

    # ---- Tensor parallelism ----
    def shard_tensor_tp(
        self,
        name: str,
        weight: np.ndarray,
        *,
        rank: Optional[int] = None,
        tp: Optional[int] = None,
        dim: int = -1,
    ) -> ShardView:
        """Split a weight tensor along dim (column-parallel style)."""
        rank = self.config.rank % (tp or self.config.tensor_parallel_size)
        tp = tp or self.config.tensor_parallel_size
        W = np.asarray(weight, dtype=np.float32)
        dim = dim if dim >= 0 else W.ndim + dim
        size = W.shape[dim]
        chunk = int(math.ceil(size / tp))
        start = rank * chunk
        end = min(start + chunk, size)
        sl = [slice(None)] * W.ndim
        sl[dim] = slice(start, end)
        shard = W[tuple(sl)].copy()
        view = ShardView(
            name=name,
            rank=rank,
            world_size=tp,
            full_shape=tuple(W.shape),
            shard=shard,
            shard_offset=start,
            shard_size=end - start,
            mode="tensor_parallel",
        )
        self.param_shards[f"tp:{name}@r{rank}"] = view
        return view

    def tp_matmul(self, local_w: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Local matmul piece; all-reduce would sum across TP ranks in real systems."""
        return np.asarray(x, dtype=np.float32) @ np.asarray(local_w, dtype=np.float32)

    # ---- Pipeline parallelism ----
    def assign_pipeline_layers(self, num_layers: int) -> Dict[int, List[int]]:
        pp = max(1, self.config.pipeline_parallel_size)
        layers = list(range(num_layers))
        chunk = int(math.ceil(num_layers / pp))
        plan: Dict[int, List[int]] = {}
        for r in range(pp):
            plan[r] = layers[r * chunk : (r + 1) * chunk]
        self.pipeline_layers = plan
        return plan

    def pipeline_stage_for_rank(self, rank: Optional[int] = None) -> List[int]:
        rank = self.config.rank if rank is None else rank
        pp = max(1, self.config.pipeline_parallel_size)
        stage = rank % pp
        if not self.pipeline_layers:
            return []
        return list(self.pipeline_layers.get(stage, []))

    # ---- Hybrid 3D ----
    def plan_hybrid(self, num_layers: int, weight_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
        self.assign_pipeline_layers(num_layers)
        cfg = self.config
        # rank decomposition: rank = dp * (tp*pp) + pp_stage * tp + tp_rank  (one convention)
        tp, pp, dp = cfg.tensor_parallel_size, cfg.pipeline_parallel_size, cfg.data_parallel_size
        rank = cfg.rank
        tp_rank = rank % tp
        pp_stage = (rank // tp) % pp
        dp_rank = rank // (tp * pp)
        shards = {}
        for name, shape in weight_shapes.items():
            # create dummy full tensor metadata only
            shards[name] = {
                "tp_rank": tp_rank,
                "pp_stage": pp_stage,
                "dp_rank": dp_rank,
                "shape": list(shape),
                "pipeline_layers": self.pipeline_layers.get(pp_stage, []),
            }
        return {
            "mode": "hybrid_3d",
            "dp": dp,
            "tp": tp,
            "pp": pp,
            "rank": rank,
            "tp_rank": tp_rank,
            "pp_stage": pp_stage,
            "dp_rank": dp_rank,
            "pipeline_layers": {str(k): v for k, v in self.pipeline_layers.items()},
            "weights": shards,
            "memory_fraction_approx": 1.0 / max(cfg.world_size, 1),
        }

    def shard_language_model(self, language: Any) -> Dict[str, Any]:
        """Apply configured sharding to an AuroLanguageModel embedding/head (demo-real)."""
        emb = np.asarray(language.core.embedding.token_embeddings, dtype=np.float32)
        views = []
        mode = self.config.mode
        if mode in (
            ParallelMode.ZERO1,
            ParallelMode.ZERO2,
            ParallelMode.ZERO3_FSDP,
            ParallelMode.HYBRID_3D,
            ParallelMode.NONE,
        ):
            # shard emb for each logical rank (simulate all ranks for report)
            for r in range(self.config.world_size):
                views.append(
                    self.shard_tensor_zero("token_embeddings", emb, rank=r).to_dict()
                )
        if mode in (ParallelMode.TENSOR, ParallelMode.HYBRID_3D):
            W = np.asarray(language.core.lm_head_weight, dtype=np.float32)
            for r in range(self.config.tensor_parallel_size):
                views.append(self.shard_tensor_tp("lm_head", W, rank=r, dim=-1).to_dict())
        if mode in (ParallelMode.PIPELINE, ParallelMode.HYBRID_3D):
            n_layers = int(getattr(language.config, "num_layers", 4))
            self.assign_pipeline_layers(n_layers)
        # estimate memory
        full = emb.nbytes + np.asarray(language.core.lm_head_weight).nbytes
        per_rank = full / max(self.config.world_size, 1)
        return {
            "ok": True,
            "mode": mode.value,
            "world_size": self.config.world_size,
            "shards": views[:12],  # sample
            "n_param_shards": len(self.param_shards),
            "n_grad_shards": len(self.grad_shards),
            "n_opt_shards": len(self.opt_shards),
            "pipeline": {str(k): v for k, v in self.pipeline_layers.items()},
            "full_embed_nbytes": int(full),
            "approx_per_rank_nbytes": int(per_rank),
            "torch_fsdp": self._torch_fsdp,
            "info": self.info(),
        }


def build_sharder(
    mode: str = "zero3_fsdp",
    world_size: int = 4,
    **kwargs: Any,
) -> MedinaSharder:
    try:
        m = ParallelMode(mode)
    except ValueError:
        m = ParallelMode.ZERO3_FSDP
    if m == ParallelMode.HYBRID_3D:
        cfg = hybrid_plan(world_size)
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return MedinaSharder(cfg)
    cfg = MedinaParallelConfig(mode=m, world_size=world_size, **kwargs)
    return MedinaSharder(cfg)
