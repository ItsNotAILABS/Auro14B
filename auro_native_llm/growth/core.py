"""Trainable NumPy reference core for Auro growth mechanics.

A minimal autograd engine plus a SpectralGPT-core causal transformer:
RMSNorm, RoPE, GQA attention with per-head QK-norm, SwiGLU FFN, and
soft-routed MoE on selected layers (every other layer in production
configs). Tied input/output embeddings.

This is the *trainable core subset* used to verify growth mechanics.
The full MESIE model adds spectral/cross-modal modules; production
growth would extend these operators to those modules (see README.md).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_FLOAT = np.float32


# ----------------------------------------------------------------------------
# Minimal autograd
# ----------------------------------------------------------------------------

class Tensor:
    """Scalar-valued-function node over float32 ndarrays (micrograd-style)."""

    def __init__(self, data, requires_grad: bool = False, _prev=(), _op: str = ""):
        self.data: np.ndarray = np.asarray(data, dtype=_FLOAT)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev = tuple(_prev)
        self._op = _op
        self._backward = lambda: None

    # -- utilities ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # -- graph -------------------------------------------------------------
    def backward(self, grad: Optional[np.ndarray] = None):
        topo: List["Tensor"] = []
        visited = set()

        def build(v: "Tensor"):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)

        build(self)
        self.grad = (
            np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=_FLOAT)
        )
        for v in reversed(topo):
            v._backward()

    # -- elementwise ---------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _prev=(self, other), _op="add")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(out.grad, other.shape))

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) - self

    def __neg__(self):
        out = Tensor(-self.data, _prev=(self,), _op="neg")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, -out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _prev=(self, other), _op="mul")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                _acc(other, _unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _prev=(self, other), _op="div")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad / other.data, self.shape))
            if other.requires_grad:
                _acc(
                    other,
                    _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape),
                )

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _prev=(self, other), _op="matmul")

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                ga = g @ _swap_last2(other.data)
                _acc(self, _unbroadcast(ga, self.shape))
            if other.requires_grad:
                gb = _swap_last2(self.data) @ g
                _acc(other, _unbroadcast(gb, other.shape))

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, _prev=(self,), _op="exp")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, out.grad * out_data)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def log(self):
        out = Tensor(np.log(self.data), _prev=(self,), _op="log")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, out.grad / self.data)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    # -- reductions ----------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _prev=(self,), _op="sum")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    ax = axis if isinstance(axis, tuple) else (axis,)
                    for a in sorted(ax):
                        g = np.expand_dims(g, a if a >= 0 else a + len(self.shape))
                _acc(self, np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def mean(self, axis=None, keepdims: bool = False):
        s = self.sum(axis=axis, keepdims=keepdims)
        n = self.data.size if axis is None else int(np.prod([self.data.shape[a] for a in _as_axes(axis, self.data.ndim)]))
        return s * (1.0 / n)

    # -- shape ops -------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), _prev=(self,), _op="reshape")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, out.grad.reshape(self.shape))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(range(self.data.ndim - 1, -1, -1))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(np.transpose(self.data, axes), _prev=(self,), _op="transpose")
        inv = np.argsort(axes)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, np.transpose(out.grad, inv))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def broadcast_to(self, shape):
        out = Tensor(np.broadcast_to(self.data, shape), _prev=(self,), _op="broadcast_to")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                _acc(self, _unbroadcast(out.grad.copy(), self.shape))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], _prev=(self,), _op="getitem")

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = np.zeros_like(self.data)
                # fresh buffer: in-place scatter is safe (no repeated idx in our uses)
                g[idx] += out.grad
                _acc(self, g)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out


def _acc(t: Tensor, g: np.ndarray):
    t.grad = g.copy() if t.grad is None else t.grad + g


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _swap_last2(a: np.ndarray) -> np.ndarray:
    return np.swapaxes(a, -1, -2)


def _as_axes(axis, ndim):
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a if a >= 0 else a + ndim for a in axis)


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), _prev=tuple(tensors), _op="stack")

    def _backward():
        if out.grad is None:
            return
        parts = np.split(out.grad, len(tensors), axis=axis)
        for t, p in zip(tensors, parts):
            if t.requires_grad:
                _acc(t, np.squeeze(p, axis=axis))

    out._backward = _backward
    out.requires_grad = any(t.requires_grad for t in tensors)
    return out


def gather(params: Tensor, idx: np.ndarray) -> Tensor:
    """Embedding lookup: params (V, d), idx (...) -> (..., d)."""
    idx = np.asarray(idx)
    out = Tensor(params.data[idx], _prev=(params,), _op="gather")

    def _backward():
        if out.grad is None:
            return
        if params.requires_grad:
            g = np.zeros_like(params.data)
            np.add.at(g, idx, out.grad)
            _acc(params, g)

    out._backward = _backward
    out.requires_grad = params.requires_grad
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.data.max(axis=axis, keepdims=True)
    e = (x - Tensor(m)).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.data.max(axis=axis, keepdims=True)
    e = (x - Tensor(m)).exp()
    return x - Tensor(m) - e.sum(axis=axis, keepdims=True).log()


def sigmoid(x: Tensor) -> Tensor:
    return Tensor(1.0) / (Tensor(1.0) + (-x).exp())


def silu(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    var = (x * x).mean(axis=-1, keepdims=True)
    inv = Tensor(1.0) / ((var + Tensor(eps)).log() * 0.5).exp()
    return x * inv * weight


# ----------------------------------------------------------------------------
# Growth config + parameter counting (pure, no weights needed)
# ----------------------------------------------------------------------------

@dataclass
class GrowthConfig:
    """Architecture knobs for one rung of the ladder.

    Mirrors the Auro SpectralGPT core: RMSNorm, RoPE, GQA attention with
    per-head QK-norm, SwiGLU FFN, soft-routed MoE on ``moe_layers``.
    """

    vocab: int = 512
    d: int = 64
    layers: int = 2
    heads: int = 2
    kv_heads: int = 2          # kv_heads == heads -> MHA; < heads -> GQA
    head_dim: int = 32
    ffn: int = 128
    seq: int = 64
    moe_layers: Tuple[int, ...] = ()
    experts: int = 2           # experts per MoE layer (soft routing)
    seed: int = 0

    def __post_init__(self):
        # Note: heads*head_dim may exceed d after width growth (head
        # duplication at fixed model dim); only the GQA ratio is invariant.
        assert self.heads % self.kv_heads == 0, "heads must be a multiple of kv_heads"
        assert self.d > 0 and self.head_dim > 0 and self.layers > 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["moe_layers"] = list(self.moe_layers)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "GrowthConfig":
        d = dict(d)
        d["moe_layers"] = tuple(d.get("moe_layers", ()))
        return cls(**d)


def count_params(cfg: GrowthConfig) -> int:
    """Exact parameter count for a config (embeddings tied)."""
    d, h, kv, hd, f = cfg.d, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.ffn
    total = cfg.vocab * d            # token embeddings (tied: no separate head)
    total += d                       # final RMSNorm
    for layer in range(cfg.layers):
        total += d                   # attn input norm
        total += d * h * hd          # Wq
        total += d * kv * hd         # Wk
        total += d * kv * hd         # Wv
        total += h * hd * d          # Wo
        total += h * hd + kv * hd    # per-head QK-norm gains (q: h heads, k: kv heads)
        total += d                   # ffn input norm
        if layer in cfg.moe_layers:
            total += d * cfg.experts            # router
            for _ in range(cfg.experts):
                total += 2 * d * f + f * d      # expert SwiGLU
        else:
            total += 2 * d * f + f * d          # SwiGLU FFN
    return total


def _param_shapes(cfg: GrowthConfig) -> Dict[str, Tuple[int, ...]]:
    """Parameter key -> shape for a config (mirrors :func:`_init_params`).

    Used to validate external checkpoints without materializing weights.
    """
    shapes: Dict[str, Tuple[int, ...]] = {
        "tok_emb": (cfg.vocab, cfg.d),
        "final_w": (cfg.d,),
    }
    for layer in range(cfg.layers):
        hd_all, kv_all = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        shapes.update({
            f"b{layer}.an1": (cfg.d,),
            f"b{layer}.wq": (cfg.d, hd_all),
            f"b{layer}.wk": (cfg.d, kv_all),
            f"b{layer}.wv": (cfg.d, kv_all),
            f"b{layer}.wo": (hd_all, cfg.d),
            f"b{layer}.qnw": (hd_all,),
            f"b{layer}.knw": (kv_all,),
            f"b{layer}.fn1": (cfg.d,),
        })
        if layer in cfg.moe_layers:
            shapes[f"b{layer}.router"] = (cfg.d, cfg.experts)
            for e in range(cfg.experts):
                shapes[f"b{layer}.e{e}.g"] = (cfg.d, cfg.ffn)
                shapes[f"b{layer}.e{e}.u"] = (cfg.d, cfg.ffn)
                shapes[f"b{layer}.e{e}.d"] = (cfg.ffn, cfg.d)
        else:
            shapes[f"b{layer}.fg"] = (cfg.d, cfg.ffn)
            shapes[f"b{layer}.fu"] = (cfg.d, cfg.ffn)
            shapes[f"b{layer}.fd"] = (cfg.ffn, cfg.d)
    return shapes


# ----------------------------------------------------------------------------
# Trainable transformer
# ----------------------------------------------------------------------------

def _init_params(cfg: GrowthConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    p: Dict[str, np.ndarray] = {}
    s = 0.02
    p["tok_emb"] = (rng.standard_normal((cfg.vocab, cfg.d)) * s).astype(_FLOAT)
    p["final_w"] = np.ones(cfg.d, dtype=_FLOAT)
    for l in range(cfg.layers):
        hd_all, kv_all = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
        p[f"b{l}.an1"] = np.ones(cfg.d, dtype=_FLOAT)
        p[f"b{l}.wq"] = (rng.standard_normal((cfg.d, hd_all)) * s).astype(_FLOAT)
        p[f"b{l}.wk"] = (rng.standard_normal((cfg.d, kv_all)) * s).astype(_FLOAT)
        p[f"b{l}.wv"] = (rng.standard_normal((cfg.d, kv_all)) * s).astype(_FLOAT)
        p[f"b{l}.wo"] = (rng.standard_normal((hd_all, cfg.d)) * s).astype(_FLOAT)
        p[f"b{l}.qnw"] = np.ones(hd_all, dtype=_FLOAT)
        p[f"b{l}.knw"] = np.ones(kv_all, dtype=_FLOAT)
        p[f"b{l}.fn1"] = np.ones(cfg.d, dtype=_FLOAT)
        if l in cfg.moe_layers:
            # Router zero-init -> uniform routing at birth (function-preserving
            # when converting a dense FFN: mean of identical experts == FFN).
            p[f"b{l}.router"] = np.zeros((cfg.d, cfg.experts), dtype=_FLOAT)
            for e in range(cfg.experts):
                p[f"b{l}.e{e}.g"] = (rng.standard_normal((cfg.d, cfg.ffn)) * s).astype(_FLOAT)
                p[f"b{l}.e{e}.u"] = (rng.standard_normal((cfg.d, cfg.ffn)) * s).astype(_FLOAT)
                p[f"b{l}.e{e}.d"] = (rng.standard_normal((cfg.ffn, cfg.d)) * s).astype(_FLOAT)
        else:
            p[f"b{l}.fg"] = (rng.standard_normal((cfg.d, cfg.ffn)) * s).astype(_FLOAT)
            p[f"b{l}.fu"] = (rng.standard_normal((cfg.d, cfg.ffn)) * s).astype(_FLOAT)
            p[f"b{l}.fd"] = (rng.standard_normal((cfg.ffn, cfg.d)) * s).astype(_FLOAT)
    return p


def _rope_cache(seq: int, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Full-dim interleaved RoPE cos/sin caches: (seq, head_dim)."""
    inv = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.outer(np.arange(seq, dtype=np.float64), inv)  # (seq, hd/2)
    c, s = np.cos(ang).astype(_FLOAT), np.sin(ang).astype(_FLOAT)
    cos_full = np.empty((seq, head_dim), dtype=_FLOAT)
    sin_full = np.empty((seq, head_dim), dtype=_FLOAT)
    cos_full[:, 0::2] = c
    cos_full[:, 1::2] = c
    sin_full[:, 0::2] = s
    sin_full[:, 1::2] = s
    return cos_full, sin_full


def _rotate_half(x: Tensor) -> Tensor:
    hd = x.shape[-1]
    xr = x.reshape(*x.shape[:-1], hd // 2, 2)
    r = stack([-xr[..., 1], xr[..., 0]], axis=-1)
    return r.reshape(*x.shape[:-1], hd)


class GrowthTransformer:
    """Causal decoder: RMSNorm + RoPE + GQA/QK-norm attention + SwiGLU/MoE."""

    def __init__(self, cfg: GrowthConfig, params: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = cfg
        self.params: Dict[str, np.ndarray] = params if params is not None else _init_params(cfg)
        cos, sin = _rope_cache(cfg.seq, cfg.head_dim)
        self._cos, self._sin = cos, sin
        # causal additive mask (constant)
        mask = np.zeros((cfg.seq, cfg.seq), dtype=_FLOAT)
        mask[np.triu_indices(cfg.seq, k=1)] = -1e9
        self._mask = mask

    # -- introspection ------------------------------------------------------
    def param_count(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, sd: Dict[str, np.ndarray]):
        self.params = {k: np.asarray(v, dtype=_FLOAT) for k, v in sd.items()}

    def _t(self, name: str) -> Tensor:
        return Tensor(self.params[name], requires_grad=True)

    def forward_train(self, tokens: np.ndarray, targets: np.ndarray):
        """Forward + backward in one call.

        Returns ``(loss_float, grads)`` where grads maps param names to
        numpy arrays. The internal ``_t`` override is removed afterwards,
        so the model is immediately usable for eval again.
        """
        tmap = {k: Tensor(v, requires_grad=True) for k, v in self.params.items()}
        self._t = lambda name: tmap[name]  # noqa: E731 (instance shadow)
        try:
            loss = causal_lm_loss(self.forward(tokens), targets)
            loss.backward()
        finally:
            del self.__dict__["_t"]
        grads = {k: (t.grad.copy() if t.grad is not None else None)
                 for k, t in tmap.items()}
        return float(loss.data), grads

    # -- forward -------------------------------------------------------------
    def forward(self, tokens: np.ndarray) -> Tensor:
        cfg = self.cfg
        bsz, seqlen = tokens.shape
        assert seqlen <= cfg.seq, f"seq {seqlen} > max {cfg.seq}"
        cos = Tensor(self._cos[:seqlen])
        sin = Tensor(self._sin[:seqlen])
        mask = Tensor(self._mask[:seqlen, :seqlen])

        x = gather(self._t("tok_emb"), tokens)                       # (B, T, d)
        for l in range(cfg.layers):
            x = x + self._attn(x, l, cos, sin, mask, bsz, seqlen)
            h = rms_norm(x, self._t(f"b{l}.fn1"))
            x = x + self._ffn(h, l)
        x = rms_norm(x, self._t("final_w"))
        return x @ self._t("tok_emb").transpose()                    # tied head

    def _attn(self, x: Tensor, l: int, cos: Tensor, sin: Tensor,
              mask: Tensor, bsz: int, seqlen: int) -> Tensor:
        cfg = self.cfg
        h, kv, hd = cfg.heads, cfg.kv_heads, cfg.head_dim
        n_rep = h // kv
        q = rms_norm(x, self._t(f"b{l}.an1"))
        q = (q @ self._t(f"b{l}.wq")).reshape(bsz, seqlen, h, hd).transpose(0, 2, 1, 3)
        k = (x @ self._t(f"b{l}.wk")).reshape(bsz, seqlen, kv, hd).transpose(0, 2, 1, 3)
        v = (x @ self._t(f"b{l}.wv")).reshape(bsz, seqlen, kv, hd).transpose(0, 2, 1, 3)
        # per-head QK-norm
        q = rms_norm(q, self._t(f"b{l}.qnw").reshape(1, h, 1, hd).broadcast_to((bsz, h, seqlen, hd)))
        k = rms_norm(k, self._t(f"b{l}.knw").reshape(1, kv, 1, hd).broadcast_to((bsz, kv, seqlen, hd)))
        # RoPE
        q = q * cos.reshape(1, 1, seqlen, hd).broadcast_to(q.shape) + \
            _rotate_half(q) * sin.reshape(1, 1, seqlen, hd).broadcast_to(q.shape)
        k = k * cos.reshape(1, 1, seqlen, hd).broadcast_to(k.shape) + \
            _rotate_half(k) * sin.reshape(1, 1, seqlen, hd).broadcast_to(k.shape)
        # GQA repeat
        if n_rep > 1:
            k = k.reshape(bsz, kv, 1, seqlen, hd).broadcast_to((bsz, kv, n_rep, seqlen, hd)).reshape(bsz, h, seqlen, hd)
            v = v.reshape(bsz, kv, 1, seqlen, hd).broadcast_to((bsz, kv, n_rep, seqlen, hd)).reshape(bsz, h, seqlen, hd)
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hd))
        att = att + mask.reshape(1, 1, seqlen, seqlen).broadcast_to(att.shape)
        att = softmax(att, axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(bsz, seqlen, h * hd)
        return o @ self._t(f"b{l}.wo")

    def _swiglu(self, x: Tensor, g: Tensor, u: Tensor, d: Tensor) -> Tensor:
        return (silu(x @ g) * (x @ u)) @ d

    def _ffn(self, x: Tensor, l: int) -> Tensor:
        cfg = self.cfg
        if l in cfg.moe_layers:
            probs = softmax(x @ self._t(f"b{l}.router"), axis=-1)     # (B, T, E)
            out = None
            for e in range(cfg.experts):
                ye = self._swiglu(x, self._t(f"b{l}.e{e}.g"),
                                  self._t(f"b{l}.e{e}.u"), self._t(f"b{l}.e{e}.d"))
                mask = np.zeros(cfg.experts, dtype=_FLOAT)
                mask[e] = 1.0
                we = (probs * Tensor(mask)).sum(axis=-1).reshape(*x.shape[:-1], 1)
                term = ye * we
                out = term if out is None else out + term
            return out
        return self._swiglu(x, self._t(f"b{l}.fg"), self._t(f"b{l}.fu"), self._t(f"b{l}.fd"))


def causal_lm_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy over all non-ignored positions (all positions here)."""
    lp = log_softmax(logits, axis=-1)                                # (B, T, V)
    bsz, seqlen = lp.shape[0], lp.shape[1]
    oh = np.zeros_like(lp.data)
    bi, ti = np.meshgrid(np.arange(bsz), np.arange(seqlen), indexing="ij")
    oh[bi, ti, np.asarray(targets).reshape(bsz, seqlen)] = 1.0
    return -(lp * Tensor(oh)).sum(axis=-1).mean()


class Adam:
    """Adam over the model's numpy param dict (reads .grad from Tensors)."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 3e-4,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 max_grad_norm: float = 1.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: Dict[str, np.ndarray]):
        self.t += 1
        # global grad-norm clip
        total = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values() if g is not None))
        clip = 1.0 if total == 0 else min(1.0, self.max_grad_norm / (total + 1e-12))
        for k, p in self.params.items():
            g = grads.get(k)
            if g is None:
                continue
            g = g * clip
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
