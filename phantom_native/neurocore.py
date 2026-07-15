"""SovereignNeuroCore — Native MESIE NeuroCore with resonance attention.

A self-contained neural processing unit combining:
- Helix-encoded weight initialization
- Resonance-weighted attention kernels
- TAURUS memory integration (working + long-term)
- SIMD-style vectorized forward pass

Zero heavy dependencies — uses only stdlib + SovereignTensor + TaurusMemory.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from phantom_native.sovereign_tensor import SovereignTensor
from phantom_native.taurus import TaurusMemory


class SovereignNeuroCore:
    """Native MESIE NeuroCore — resonance + helix + TAURUS aware.

    Attributes:
        d_model: Internal representation dimension.
        n_heads: Number of attention heads.
        taurus: Native TAURUS memory instance.
        weights: Helix-encoded weight matrices (Q, K, V).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.d_model = config.get("d_model", 128)
        self.n_heads = config.get("n_heads", 8)
        self.head_dim = self.d_model // self.n_heads
        self.taurus = TaurusMemory(
            capacity=config.get("memory_capacity", 32),
            decay_rate=config.get("decay_rate", 0.95),
        )
        self.weights = self._init_helix_weights()
        self._attention_maps: List[List[float]] = []

    def _init_helix_weights(self) -> Dict[str, List[float]]:
        """Initialize weights using helix encoding from MESIE primitives.

        Helix initialization provides structured frequency-aware starting
        points that align with spectral data characteristics.
        """
        d = self.d_model
        return {
            "query": [math.sin(i * 0.1) * 0.5 for i in range(d)],
            "key": [math.cos(i * 0.1) * 0.5 for i in range(d)],
            "value": [1.0 / math.sqrt(d) for _ in range(d)],
            "output": [math.sin(i * 0.05 + 0.3) * 0.3 for i in range(d)],
        }

    def _resonance_attention(
        self, q: List[float], k: List[float], v: List[float]
    ) -> List[float]:
        """Resonance-weighted attention kernel.

        Computes attention scores with exponential resonance decay,
        then applies softmax and weighted sum over values.

        Args:
            q: Query vector.
            k: Key vector.
            v: Value vector.

        Returns:
            Attention-weighted output vector.
        """
        n = len(q)
        scores: List[float] = []

        # Compute resonance-weighted attention scores
        scale = 1.0 / math.sqrt(n) if n > 0 else 1.0
        for i in range(n):
            dot = q[i] * k[i] * scale
            resonance = math.exp(-abs(dot) * 0.5)  # resonance decay
            scores.append(dot * resonance)

        # Softmax (numerically stable)
        max_s = max(scores) if scores else 0.0
        exp_s = [math.exp(s - max_s) for s in scores]
        total = sum(exp_s) or 1.0
        attn_weights = [e / total for e in exp_s]

        # Store attention map for interpretability
        self._attention_maps.append(attn_weights[:])

        # Weighted sum over values
        output = [0.0] * n
        for i in range(n):
            output[i] = attn_weights[i] * v[i]

        return output

    def forward(self, tensor: SovereignTensor) -> SovereignTensor:
        """Full forward pass with resonance attention, helix projection, and TAURUS.

        Pipeline:
        1. Project input to Q, K, V via helix weights
        2. Apply resonance attention kernel
        3. Output projection
        4. Store result in TAURUS working memory
        5. Fuse with top-k context from memory

        Args:
            tensor: Input SovereignTensor (any 1D shape).

        Returns:
            Processed SovereignTensor with attention-weighted embedding.
        """
        self._attention_maps.clear()
        d = self.d_model
        n = len(tensor.data)

        # Project to QKV (cyclic indexing for arbitrary input sizes)
        q = [
            tensor.data[i % n] * self.weights["query"][i % d] for i in range(d)
        ]
        k = [
            tensor.data[i % n] * self.weights["key"][i % d] for i in range(d)
        ]
        v = [
            tensor.data[i % n] * self.weights["value"][i % d] for i in range(d)
        ]

        # Multi-head resonance attention
        head_outputs: List[List[float]] = []
        for h in range(self.n_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            head_q = q[start:end]
            head_k = k[start:end]
            head_v = v[start:end]
            head_out = self._resonance_attention(head_q, head_k, head_v)
            head_outputs.append(head_out)

        # Concatenate heads
        concat = []
        for head_out in head_outputs:
            concat.extend(head_out)

        # Output projection
        output_data = [
            concat[i] * self.weights["output"][i % d] for i in range(len(concat))
        ]

        # Pad or truncate to match original input shape
        if len(output_data) < n:
            output_data.extend([0.0] * (n - len(output_data)))
        elif len(output_data) > n:
            output_data = output_data[:n]

        out_tensor = SovereignTensor(output_data, tensor.shape, tensor.spectral_meta)

        # TAURUS memory update
        self.taurus.store(out_tensor, importance=tensor.resonance)

        # Fuse with top-k memory context (residual connection)
        context = self.taurus.recall_top_k(4)
        if context:
            # Average context and add as residual
            ctx_sum = [0.0] * n
            for ctx_tensor in context:
                for i in range(min(len(ctx_tensor.data), n)):
                    ctx_sum[i] += ctx_tensor.data[i]
            ctx_weight = 0.1 / len(context)
            fused = [
                output_data[i] + ctx_sum[i] * ctx_weight for i in range(n)
            ]
            out_tensor = SovereignTensor(fused, tensor.shape, tensor.spectral_meta)

        return out_tensor

    def get_attention_analysis(self) -> Dict[str, Any]:
        """Return interpretability metrics from last forward pass.

        Returns:
            Dictionary with attention entropy, max attention, and sparsity
            metrics per head.
        """
        if not self._attention_maps:
            return {"n_heads": 0, "head_analyses": []}

        analyses = []
        for i, attn in enumerate(self._attention_maps):
            # Entropy
            entropy = -sum(
                w * math.log(w + 1e-10) for w in attn if w > 0
            )
            max_attn = max(attn) if attn else 0.0
            sparsity = sum(1 for w in attn if w < 0.01) / max(len(attn), 1)
            analyses.append(
                {
                    "head": i,
                    "attention_entropy": entropy,
                    "max_attention": max_attn,
                    "attention_sparsity": sparsity,
                }
            )

        return {"n_heads": len(self._attention_maps), "head_analyses": analyses}

    def reset_memory(self) -> None:
        """Clear TAURUS working memory."""
        self.taurus.working_memory.clear()

    def __repr__(self) -> str:
        return (
            f"SovereignNeuroCore(d_model={self.d_model}, n_heads={self.n_heads}, "
            f"memory={self.taurus.size()})"
        )
