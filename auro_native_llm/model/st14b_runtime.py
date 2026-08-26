from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ST14BRuntimeConfig:
    vocab_size: int = 128_000
    hidden_size: int = 5_120
    num_layers: int = 40
    num_heads: int = 40
    num_kv_heads: int = 5
    intermediate_size: int = 18_432
    max_seq_len: int = 8_192
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    bias: bool = False

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.max_seq_len < 1 or self.num_layers < 1:
            raise ValueError("model depth and context must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class LayerKVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    @property
    def sequence_length(self) -> int:
        return 0 if self.key is None else int(self.key.size(-2))

    def append(self, key: torch.Tensor, value: torch.Tensor, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            next_key, next_value = key, value
        else:
            next_key = torch.cat((self.key, key), dim=-2)
            next_value = torch.cat((self.value, value), dim=-2)
        if next_key.size(-2) > max_seq_len:
            next_key = next_key[..., -max_seq_len:, :]
            next_value = next_value[..., -max_seq_len:, :]
        self.key, self.value = next_key, next_value
        return next_key, next_value


class ST14BKVCache:
    def __init__(self, num_layers: int) -> None:
        self.layers = [LayerKVCache() for _ in range(num_layers)]

    def clear(self) -> None:
        for layer in self.layers:
            layer.key = None
            layer.value = None

    @property
    def sequence_length(self) -> int:
        return self.layers[0].sequence_length if self.layers else 0


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        normalized = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        phase = torch.outer(positions, inv_freq)
        self.register_buffer("cos", phase.cos(), persistent=False)
        self.register_buffer("sin", phase.sin(), persistent=False)

    @staticmethod
    def _apply(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.size(-2)
        cos = self.cos[offset : offset + length].to(query.device, query.dtype)[None, None, :, :]
        sin = self.sin[offset : offset + length].to(query.device, query.dtype)[None, None, :, :]
        return self._apply(query, cos, sin), self._apply(key, cos, sin)


class CacheAwareGroupedQueryAttention(nn.Module):
    def __init__(self, config: ST14BRuntimeConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=config.bias)
        self.key = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=config.bias)
        self.value = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=config.bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.last_attention_backend = "uninitialized"

    def forward(self, hidden: torch.Tensor, cache: LayerKVCache | None = None) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        offset = cache.sequence_length if cache is not None else 0
        query = self.query(hidden).view(batch, sequence, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        key = self.key(hidden).view(batch, sequence, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)
        value = self.value(hidden).view(batch, sequence, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)
        query, key = self.rope(query, key, offset)
        if cache is not None:
            key, value = cache.append(key, value, self.config.max_seq_len)
        is_prefill = cache is None or sequence > 1

        try:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=is_prefill,
                dropout_p=0.0,
                enable_gqa=True,
            )
            self.last_attention_backend = "torch-sdpa-native-gqa"
        except TypeError:
            # Compatibility only for older PyTorch versions. Production promotion
            # requires the native-GQA path so KV heads are not materialized to Hq.
            repeats = self.config.kv_group_size
            expanded_key = key.repeat_interleave(repeats, dim=1)
            expanded_value = value.repeat_interleave(repeats, dim=1)
            attended = F.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                is_causal=is_prefill,
                dropout_p=0.0,
            )
            self.last_attention_backend = "compat-expanded-kv"
        return self.output(attended.transpose(1, 2).contiguous().view(batch, sequence, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: ST14BRuntimeConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class ST14BBlock(nn.Module):
    def __init__(self, config: ST14BRuntimeConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = CacheAwareGroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, hidden: torch.Tensor, cache: LayerKVCache | None = None) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), cache)
        return hidden + self.mlp(self.mlp_norm(hidden))


class AuroST14BForCausalLM(nn.Module):
    """Dense AURO ST-14B decoder with native GQA KV caching."""

    def __init__(self, config: ST14BRuntimeConfig | None = None) -> None:
        super().__init__()
        self.config = config or ST14BRuntimeConfig()
        self.config.validate()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(ST14BBlock(self.config) for _ in range(self.config.num_layers))
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def new_cache(self) -> ST14BKVCache:
        return ST14BKVCache(self.config.num_layers)

    def forward(self, input_ids: torch.Tensor, cache: ST14BKVCache | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) + (cache.sequence_length if cache else 0) > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured context window")
        hidden = self.embedding(input_ids)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cache.layers[index] if cache is not None else None)
        return self.lm_head(self.final_norm(hidden))

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, cache: ST14BKVCache) -> torch.Tensor:
        return self.forward(input_ids, cache=cache)

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, cache: ST14BKVCache) -> torch.Tensor:
        if token_ids.size(1) != 1:
            raise ValueError("decode_step accepts exactly one token per sequence")
        return self.forward(token_ids, cache=cache)[:, -1, :]

    @torch.no_grad()
    def generate_cached(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None) -> torch.Tensor:
        cache = self.new_cache()
        logits = self.prefill(input_ids, cache)[:, -1, :]
        output = input_ids
        for _ in range(max_new_tokens):
            next_token = logits.argmax(dim=-1, keepdim=True)
            output = torch.cat((output, next_token), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token.eq(eos_token_id))):
                break
            logits = self.decode_step(next_token, cache)
        return output

    def attention_backends(self) -> list[str]:
        return [layer.attention.last_attention_backend for layer in self.layers]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def kv_cache_bytes(config: ST14BRuntimeConfig, batch_size: int, sequence_length: int, bytes_per_element: int) -> int:
    return 2 * batch_size * config.num_layers * config.num_kv_heads * config.head_dim * sequence_length * bytes_per_element


def mha_equivalent_kv_cache_bytes(config: ST14BRuntimeConfig, batch_size: int, sequence_length: int, bytes_per_element: int) -> int:
    return 2 * batch_size * config.num_layers * config.num_heads * config.head_dim * sequence_length * bytes_per_element
