"""Deterministic structured pre-wiring for native Auro checkpoints.

A normalized Walsh-Hadamard basis creates exact, inexpensive orthogonal channels.
A smaller harmonic residual preserves MESIE's continuous spectral geometry.
Neither mechanism substitutes for corpus training or benchmark evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from auro_native_llm.model.walsh_hadamard import (
    WalshOrder,
    diagnose,
    hadamard_matrix,
    next_power_of_two,
    walsh_tensor,
)

CIVILIZATION_PRIMITIVES: Sequence[str] = (
    "matter", "force", "energy", "space", "time", "structure", "flow",
    "boundary", "feedback", "memory", "identity", "cause", "consequence",
    "resource", "constraint", "adaptation", "cooperation", "governance",
    "creation", "measurement", "evidence", "tool", "execution", "receipt",
)

CONTROL_FAMILIES: Dict[str, Sequence[str]] = {
    "identity": ("system", "user", "assistant", "operator", "organization"),
    "memory": ("memory", "retrieve", "recur", "consolidate", "forget"),
    "execution": ("plan", "approve", "execute", "validate", "receipt"),
    "spectral": ("mesie", "frequency", "phase", "coherence", "resonance"),
    "modalities": ("text", "code", "image", "audio", "video", "document"),
    "civilization": CIVILIZATION_PRIMITIVES,
}


@dataclass(frozen=True)
class PrewiringConfig:
    version: str = "auro.prewiring.v2"
    seed: int = 873539
    embedding_strength: float = 0.05
    spectral_strength: float = 0.015
    plastic_noise: float = 0.003
    invariant_fraction: float = 0.02
    anchored_fraction: float = 0.18
    walsh_ordering: WalshOrder = "sequency"
    external_model_fallback: bool = False


@dataclass
class PrewiringReceipt:
    schema: str
    seed: int
    model_id: str
    parameter_target: int
    live_parameter_count: int
    applied_components: List[str]
    tensor_shapes: Dict[str, List[int]]
    transform_diagnostics: Dict[str, Any]
    manifest_sha256: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _harmonic_lattice(rows: int, cols: int, seed: int) -> np.ndarray:
    r = np.arange(rows, dtype=np.float64)[:, None] + 1.0
    c = np.arange(cols, dtype=np.float64)[None, :] + 1.0
    phase = (seed % 997) / 997.0
    lattice = np.sin((r * c) / max(rows, cols) + phase)
    lattice += 0.5 * np.cos((r + c) * 0.6180339887498948 + phase)
    norm = np.linalg.norm(lattice, axis=-1, keepdims=True)
    return lattice / np.maximum(norm, 1e-12)


def _token_ids(tokenizer: Any, words: Iterable[str]) -> List[int]:
    found: List[int] = []
    for word in words:
        try:
            ids = tokenizer.encode(word, add_bos=False, add_eos=False, max_length=None)
        except TypeError:
            ids = tokenizer.encode(word)
        for token_id in ids:
            value = int(token_id)
            if value >= 0 and value not in found:
                found.append(value)
    return found


def _manifest(config: PrewiringConfig, model: Any, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": config.version,
        "seed": config.seed,
        "model_id": model.model_id,
        "parameter_target": int(model.config.parameter_target),
        "live_parameter_count": int(model.num_params),
        "orthogonal_transform": {
            "name": "normalized_walsh_hadamard",
            "ordering": config.walsh_ordering,
            "complexity": "O(n log n)",
            "multiply_free_butterfly": True,
            "diagnostics": diagnostics,
        },
        "control_families": {key: list(value) for key, value in CONTROL_FAMILIES.items()},
        "civilization_primitives": list(CIVILIZATION_PRIMITIVES),
        "external_model_fallback": False,
        "claims": {
            "structured_inductive_bias": True,
            "factual_knowledge_without_training": False,
            "requires_ab_evaluation": True,
        },
    }


def apply_structured_prewiring(model: Any, config: PrewiringConfig | None = None) -> PrewiringReceipt:
    """Apply deterministic Walsh-Hadamard and MESIE residual structure in-place."""
    cfg = config or PrewiringConfig(seed=int(model.config.seed))
    embedding = model.core.embedding.token_embeddings
    rows, cols = embedding.shape
    transform_order = next_power_of_two(cols)
    diagnostics = diagnose(transform_order, ordering=cfg.walsh_ordering).to_dict()

    structured = walsh_tensor(
        rows,
        cols,
        seed=cfg.seed,
        ordering=cfg.walsh_ordering,
    ).astype(embedding.dtype) * cfg.embedding_strength
    structured += _harmonic_lattice(rows, cols, cfg.seed + 17).astype(embedding.dtype) * cfg.spectral_strength

    family_basis = hadamard_matrix(
        transform_order,
        normalize=True,
        ordering=cfg.walsh_ordering,
    )
    family_rows: Dict[str, List[int]] = {}
    for family_index, (family, words) in enumerate(CONTROL_FAMILIES.items()):
        ids = [token_id for token_id in _token_ids(model.tokenizer, words) if token_id < rows]
        if not ids:
            continue
        anchor = family_basis[family_index % transform_order, :cols].astype(embedding.dtype)
        anchor /= max(float(np.linalg.norm(anchor)), 1e-12)
        for token_id in ids:
            structured[token_id] += anchor * cfg.embedding_strength
        family_rows[family] = ids

    rng = np.random.default_rng(cfg.seed + 29)
    plastic = rng.standard_normal(embedding.shape).astype(embedding.dtype) * cfg.plastic_noise
    model.core.embedding.token_embeddings = embedding + structured + plastic

    applied = [
        "normalized_walsh_hadamard_embedding_basis",
        "walsh_sequency_control_channels" if cfg.walsh_ordering == "sequency" else "walsh_natural_control_channels",
        "mesie_harmonic_residual",
        "bounded_plasticity",
    ]
    shapes: Dict[str, List[int]] = {"token_embeddings": list(embedding.shape)}

    if not getattr(model.core, "tie_embeddings", True):
        head = model.core.lm_head_weight
        head_basis = walsh_tensor(
            *head.shape,
            seed=cfg.seed + 41,
            ordering=cfg.walsh_ordering,
        ).astype(head.dtype)
        model.core.lm_head_weight = head + head_basis * cfg.embedding_strength
        applied.append("normalized_walsh_hadamard_lm_head")
        shapes["lm_head_weight"] = list(head.shape)

    manifest = _manifest(cfg, model, diagnostics)
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    model.prewiring_manifest = manifest
    model.prewiring_manifest_sha256 = digest

    return PrewiringReceipt(
        schema=cfg.version,
        seed=cfg.seed,
        model_id=model.model_id,
        parameter_target=int(model.config.parameter_target),
        live_parameter_count=int(model.num_params),
        applied_components=applied,
        tensor_shapes=shapes,
        transform_diagnostics=diagnostics,
        manifest_sha256=digest,
        notes=[
            "Walsh-Hadamard structure is an inductive bias, not a substitute for training.",
            "Civilization knowledge must enter through provenance-controlled corpora and evaluation.",
            f"control_token_rows_touched={sum(len(value) for value in family_rows.values())}",
            f"control_families_anchored={len(family_rows)}",
        ],
    )
