"""Experimental structured transformer components for controlled A/B evaluation.

Nothing in this module is promoted into the production Auro model by default.
Promotion requires matched training/evaluation receipts.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def fwht_torch(values: Tensor, *, normalize: bool = True) -> Tensor:
    """Vectorized differentiable FWHT over the last dimension."""
    if values.ndim == 0:
        raise ValueError("FWHT requires at least one dimension")
    n = values.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"last dimension must be a power of two, got {n}")
    result = values
    h = 1
    while h < n:
        shaped = result.reshape(*result.shape[:-1], -1, 2, h)
        left, right = shaped.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        h *= 2
    return result * (n ** -0.5) if normalize else result


class FastfoodLinear(nn.Module):
    """Learnable structured linear map without a dense weight matrix.

    Each block computes S H G P H B x. Non-power-of-two dimensions are padded,
    and multiple blocks are concatenated then cropped to the output width.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, seed: int = 0):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.width = next_power_of_two(in_features)
        self.blocks = math.ceil(out_features / self.width)
        generator = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (self.blocks, self.width), generator=generator, dtype=torch.int64)
        self.register_buffer("b", signs.to(torch.float32).mul_(2).sub_(1))
        self.register_buffer("permutation", torch.stack([torch.randperm(self.width, generator=generator) for _ in range(self.blocks)]))
        self.g = nn.Parameter(torch.randn(self.blocks, self.width, generator=generator))
        self.s = nn.Parameter(torch.ones(self.blocks, self.width))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {values.shape[-1]}")
        padded = F.pad(values, (0, self.width - self.in_features))
        outputs = []
        for block in range(self.blocks):
            x = fwht_torch(padded * self.b[block])
            x = x[..., self.permutation[block]] * self.g[block]
            x = fwht_torch(x) * self.s[block]
            outputs.append(x)
        result = torch.cat(outputs, dim=-1)[..., : self.out_features]
        return result + self.bias if self.bias is not None else result

    @property
    def dense_weight_elements(self) -> int:
        return 0


class StructuredMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, dropout: float = 0.0, seed: int = 0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = FastfoodLinear(d_model, d_model, seed=seed + 1)
        self.k_proj = FastfoodLinear(d_model, d_model, seed=seed + 2)
        self.v_proj = FastfoodLinear(d_model, d_model, seed=seed + 3)
        self.out_proj = FastfoodLinear(d_model, d_model, seed=seed + 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, *, attention_mask: Optional[Tensor] = None, causal: bool = True) -> Tensor:
        batch, sequence, channels = values.shape
        q = self.q_proj(values).view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(values).view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(batch, sequence, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal:
            causal_mask = torch.ones(sequence, sequence, device=values.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            mask = attention_mask.to(torch.bool)
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(probabilities, v).transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.out_proj(output)


class StructuredTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, d_ff: Optional[int] = None, dropout: float = 0.0, seed: int = 0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = StructuredMultiHeadAttention(d_model, n_heads, dropout=dropout, seed=seed)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_in = FastfoodLinear(d_model, d_ff, seed=seed + 10)
        self.ff_out = FastfoodLinear(d_ff, d_model, seed=seed + 11)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, *, attention_mask: Optional[Tensor] = None) -> Tensor:
        values = values + self.dropout(self.attention(self.norm1(values), attention_mask=attention_mask))
        hidden = F.gelu(self.ff_in(self.norm2(values)))
        return values + self.dropout(self.ff_out(hidden))


class OrthogonalMemory(nn.Module):
    """Key-value memory with fixed orthogonal scaffold and learnable values."""

    def __init__(self, slots: int, d_model: int, *, temperature: float = 1.0):
        super().__init__()
        if slots <= 0 or d_model <= 0 or temperature <= 0:
            raise ValueError("slots, d_model, and temperature must be positive")
        width = next_power_of_two(max(slots, d_model))
        eye = torch.eye(width)
        keys = fwht_torch(eye)[:slots, :d_model]
        keys = F.normalize(keys, dim=-1)
        self.register_buffer("keys", keys)
        self.values = nn.Parameter(torch.zeros(slots, d_model))
        self.temperature = temperature

    def retrieve(self, query: Tensor) -> tuple[Tensor, Tensor]:
        if query.shape[-1] != self.keys.shape[-1]:
            raise ValueError("query width does not match memory width")
        scores = torch.matmul(F.normalize(query, dim=-1), self.keys.transpose(0, 1)) / self.temperature
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, self.values), weights


@dataclass(frozen=True)
class StructuredArchitectureReceipt:
    d_model: int
    n_heads: int
    d_ff: int
    dense_parameter_count: int
    structured_parameter_count: int
    structured_dense_weight_elements: int
    benchmark_superiority_claimed: bool = False
    hallucination_reduction_claimed: bool = False
    promoted_to_production: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
