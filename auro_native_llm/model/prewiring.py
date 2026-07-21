"""Deterministic structured pre-wiring for native Auro checkpoints.

This module adds measurable inductive structure to the native MESIE model at
birth. It does not claim to encode factual world knowledge without training.
Instead it establishes stable control, memory, spectral, multimodal, and
civilization-primitive geometry that training can refine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


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
    version: str = "auro.prewiring.v1"
    seed: int = 873539
    embedding_strength: float = 0.05
    spectral_strength: float = 0.03
    plastic_noise: float = 0.005
    invariant_fraction: float = 0.02
    anchored_fraction: float = 0.18
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
    manifest_sha256: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _orthogonal_basis(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width = max(rows, cols)
    q, _ = np.linalg.qr(rng.standard_normal((width, width)))
    return q[:rows, :cols]


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


def _manifest(config: PrewiringConfig, model: Any) -> Dict[str, Any]:
    return {
        "schema": config.version,
        "seed": config.seed,
        "model_id": model.model_id,
        "parameter_target": int(model.config.parameter_target),
        "live_parameter_count": int(model.num_params),
        "control_families": {k: list(v) for k, v in CONTROL_FAMILIES.items()},
        "civilization_primitives": list(CIVILIZATION_PRIMITIVES),
        "external_model_fallback": False,
        "claims": {
            "structured_inductive_bias": True,
            "factual_knowledge_without_training": False,
            "requires_ab_evaluation": True,
        },
    }


def apply_structured_prewiring(model: Any, config: PrewiringConfig | None = None) -> PrewiringReceipt:
    """Apply deterministic structure to the native Auro model in-place.

    Current MESIE compatibility is deliberately conservative: the embedding and
    untied LM head are modified through stable public attributes. Additional core
    tensors can be added later only after their exact semantics are verified.
    """
    cfg = config or PrewiringConfig(seed=int(model.config.seed))
    applied: List[str] = []
    shapes: Dict[str, List[int]] = {}

    embedding = model.core.embedding.token_embeddings
    rows, cols = embedding.shape
    lattice = _harmonic_lattice(rows, cols, cfg.seed).astype(embedding.dtype)
    basis = _orthogonal_basis(min(rows, cols), cols, cfg.seed + 11).astype(embedding.dtype)

    structured = lattice * cfg.spectral_strength
    structured[: basis.shape[0], :] += basis * cfg.embedding_strength

    family_offset = 0
    for family_index, words in enumerate(CONTROL_FAMILIES.values()):
        ids = [i for i in _token_ids(model.tokenizer, words) if i < rows]
        if not ids:
            continue
        anchor = _harmonic_lattice(1, cols, cfg.seed + 101 * (family_index + 1))[0]
        anchor = anchor.astype(embedding.dtype)
        for token_id in ids:
            structured[token_id] += anchor * cfg.embedding_strength
        family_offset += len(ids)

    rng = np.random.default_rng(cfg.seed + 29)
    plastic = rng.standard_normal(embedding.shape).astype(embedding.dtype) * cfg.plastic_noise
    model.core.embedding.token_embeddings = embedding + structured + plastic
    applied.extend(["harmonic_embedding_lattice", "orthogonal_control_basis", "bounded_plasticity"])
    shapes["token_embeddings"] = list(embedding.shape)

    if not getattr(model.core, "tie_embeddings", True):
        head = model.core.lm_head_weight
        model.core.lm_head_weight = head + _harmonic_lattice(*head.shape, cfg.seed + 41).astype(head.dtype) * cfg.spectral_strength
        applied.append("harmonic_lm_head")
        shapes["lm_head_weight"] = list(head.shape)

    manifest = _manifest(cfg, model)
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
        manifest_sha256=digest,
        notes=[
            "Structure is an inductive bias, not a substitute for training.",
            "Civilization knowledge must enter through provenance-controlled corpora and evaluation.",
            f"control_token_rows_touched={family_offset}",
        ],
    )
