"""Save / load Auro LM checkpoints (weights + tokenizer + config)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.config import AuroLMConfig
from auro_native_llm.model.tokenizer import AuroTokenizer


def save_checkpoint(model: AuroLanguageModel, directory: str | Path) -> Dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Core arrays
    np.savez_compressed(
        directory / "weights.npz",
        token_embeddings=model.core.embedding.token_embeddings,
        lm_head_weight=model.core.lm_head_weight,
        meaning_inject=model.meaning.inject if model.meaning is not None else np.zeros((1, 1)),
        spectral_proj=model.spectral.proj if model.spectral is not None else np.zeros((1, 1)),
        spectral_bias=model.spectral.bias if model.spectral is not None else np.zeros(1),
    )

    # Layer weights (attention + FFN) for restore fidelity
    layer_blob: Dict[str, np.ndarray] = {}
    for i, layer in enumerate(model.core.layers):
        tr = layer["transformer"]
        attn = getattr(tr, "attention", None)
        if attn is not None:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(attn, name):
                    layer_blob[f"L{i}_attn_{name}"] = getattr(attn, name)
        ffn = getattr(tr, "ffn", None)
        if ffn is not None:
            for name in ("gate_proj", "up_proj", "down_proj"):
                val = getattr(ffn, name, None)
                if val is not None:
                    layer_blob[f"L{i}_ffn_{name}"] = val
    if layer_blob:
        np.savez_compressed(directory / "layers.npz", **layer_blob)

    model.tokenizer.save(directory / "tokenizer.json")
    (directory / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2), encoding="utf-8"
    )
    meta = {
        "schema": "auro.lm.checkpoint.v1",
        "model_id": model.model_id,
        "num_params": model.num_params,
        "train_steps": model.train_steps,
        "compute_plane": "MESIE",
        "native": True,
        "saved_at_unix": int(time.time()),
        "parameter_target": model.config.parameter_target,
        "mode": model.config.mode,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_checkpoint(directory: str | Path) -> AuroLanguageModel:
    directory = Path(directory)
    cfg = AuroLMConfig.from_dict(
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
    )
    tok = AuroTokenizer.load(directory / "tokenizer.json")
    model = AuroLanguageModel(cfg, tokenizer=tok)

    weights = np.load(directory / "weights.npz")
    model.core.embedding.token_embeddings = weights["token_embeddings"]
    model.core.lm_head_weight = weights["lm_head_weight"]
    if model.meaning is not None and weights["meaning_inject"].shape == model.meaning.inject.shape:
        model.meaning.inject = weights["meaning_inject"]
    if model.spectral is not None and weights["spectral_proj"].shape == model.spectral.proj.shape:
        model.spectral.proj = weights["spectral_proj"]
        model.spectral.bias = weights["spectral_bias"]

    layers_path = directory / "layers.npz"
    if layers_path.exists():
        blob = np.load(layers_path)
        for i, layer in enumerate(model.core.layers):
            tr = layer["transformer"]
            attn = getattr(tr, "attention", None)
            if attn is not None:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    key = f"L{i}_attn_{name}"
                    if key in blob.files:
                        setattr(attn, name, blob[key])
            ffn = getattr(tr, "ffn", None)
            if ffn is not None:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    key = f"L{i}_ffn_{name}"
                    if key in blob.files:
                        setattr(ffn, name, blob[key])

    meta_path = directory / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model.train_steps = int(meta.get("train_steps", 0))
    return model
