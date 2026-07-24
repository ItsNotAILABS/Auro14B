from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        phase = torch.outer(positions, frequencies)
        self.register_buffer("cos", phase.cos(), persistent=False)
        self.register_buffer("sin", phase.sin(), persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        cos = self.cos[offset : offset + sequence_length].to(dtype=query.dtype, device=query.device)
        sin = self.sin[offset : offset + sequence_length].to(dtype=query.dtype, device=query.device)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


def _apply_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, heads, sequence, dimension = value.shape
    value = value[:, :, None, :, :].expand(batch, heads, repeats, sequence, dimension)
    return value.reshape(batch, heads * repeats, sequence, dimension)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeats = config.num_heads // config.num_kv_heads
        self.query = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.bias)
        self.key = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=config.bias)
        self.value = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=config.bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self.query(hidden).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, key = self.rope(query, key)
        key = _repeat_kv(key, self.kv_repeats)
        value = _repeat_kv(value, self.kv_repeats)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.residual_dropout(self.attention(self.attention_norm(hidden)))
        return hidden + self.residual_dropout(self.mlp(self.mlp_norm(hidden)))


class AuroForCausalLM(nn.Module):
    """Native decoder-only Auro text-generation model.

    This is the same implementation for micro, local, 14B, and 206.7B configs.
    Large lanes are instantiated only when the operator has registered enough
    capacity through the MESIE training fabric.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        self._scale_residual_projections()

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        standard_deviation = 0.02 / math.sqrt(2 * self.config.num_layers)
        for block in self.layers:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=standard_deviation)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=standard_deviation)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input sequence exceeds configured context window")
        hidden = self.dropout(self.token_embedding(input_ids))
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.05,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits = self(context).logits[:, -1, :]
            if repetition_penalty != 1.0:
                for row in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[row])
                    values = logits[row, seen]
                    logits[row, seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probabilities = F.softmax(sorted_logits, dim=-1)
                    cumulative = probabilities.cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token.eq(eos_token_id))):
                break
        return input_ids

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "model_id": self.config.model_id,
            "parameters": total,
            "trainable_parameters": trainable,
            "config_estimate": self.config.estimated_parameters(),
            "dtype": str(next(self.parameters()).dtype),
        }
