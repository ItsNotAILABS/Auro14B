"""Save and load AURO language checkpoints with signed constitutional authorization."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.config import AuroLMConfig
from auro_native_llm.model.tokenizer import AuroTokenizer
from auro_native_llm.substrate import build_constitutional_checkpoint, load_and_verify_constitutional_manifest, write_constitutional_manifest
from auro_native_llm.substrate.promotion_evidence import evidence_flags, load_signed_promotion_evidence


def _promotion_context() -> tuple[bool, Dict[str, Any], str | None, str | None]:
    requested = os.environ.get("AURO_PROMOTION_REQUESTED", "").strip().lower() in {"1", "true", "yes", "on"}
    signing_key = os.environ.get("AURO_CHECKPOINT_SIGNING_KEY") or None
    authorized_by = os.environ.get("AURO_CHECKPOINT_AUTHORIZER") or None
    if not requested:
        return False, {}, signing_key, authorized_by
    evidence_path = os.environ.get("AURO_PROMOTION_EVIDENCE")
    if not evidence_path:
        raise RuntimeError("AURO_PROMOTION_EVIDENCE is required for promotion")
    receipt = load_signed_promotion_evidence(evidence_path, signing_key or "")
    return True, {**evidence_flags(receipt), "promotion_receipt": receipt}, signing_key, authorized_by


def save_checkpoint(model: AuroLanguageModel, directory: str | Path) -> Dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "weights.npz", token_embeddings=model.core.embedding.token_embeddings,
        lm_head_weight=model.core.lm_head_weight,
        meaning_inject=model.meaning.inject if model.meaning is not None else np.zeros((1, 1)),
        spectral_proj=model.spectral.proj if model.spectral is not None else np.zeros((1, 1)),
        spectral_bias=model.spectral.bias if model.spectral is not None else np.zeros(1))
    layer_blob: Dict[str, np.ndarray] = {}
    for i, layer in enumerate(model.core.layers):
        tr = layer["transformer"]
        attn = getattr(tr, "attention", None)
        if attn is not None:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(attn, name): layer_blob[f"L{i}_attn_{name}"] = getattr(attn, name)
        ffn = getattr(tr, "ffn", None)
        if ffn is not None:
            for name in ("gate_proj", "up_proj", "down_proj"):
                value = getattr(ffn, name, None)
                if value is not None: layer_blob[f"L{i}_ffn_{name}"] = value
    if layer_blob: np.savez_compressed(directory / "layers.npz", **layer_blob)
    model.tokenizer.save(directory / "tokenizer.json")
    (directory / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")

    checkpoint_id = os.environ.get("AURO_CHECKPOINT_ID", f"{model.model_id}-{model.train_steps}-{int(time.time())}")
    parent_id = os.environ.get("AURO_PARENT_CHECKPOINT_ID") or None
    promotion_requested, signed_evidence, signing_key, authorized_by = _promotion_context()
    safe_id = signed_evidence.get("rollback_target") or os.environ.get("AURO_SAFE_CHECKPOINT_ID") or parent_id
    files = ["weights.npz", "tokenizer.json", "config.json"]
    if (directory / "layers.npz").exists(): files.append("layers.npz")
    constitutional = build_constitutional_checkpoint(root=directory, checkpoint_id=checkpoint_id,
        checkpoint_class="weights", model_id=model.model_id, files=files,
        parent_checkpoint_id=parent_id,
        optimization={"train_steps": model.train_steps, "parameter_target": model.config.parameter_target, "mode": model.config.mode},
        identity={"model_id": model.model_id, "compute_plane": "MESIE"},
        rollback={"safe_checkpoint_id": safe_id}, evidence={**signed_evidence, "rollback_target": safe_id},
        promotion_requested=promotion_requested, signing_key=signing_key, authorized_by=authorized_by)
    write_constitutional_manifest(directory, constitutional)
    meta = {"schema": "auro.lm.checkpoint.v3", "model_id": model.model_id, "checkpoint_id": checkpoint_id,
        "num_params": model.num_params, "train_steps": model.train_steps, "compute_plane": "MESIE", "native": True,
        "saved_at_unix": int(time.time()), "parameter_target": model.config.parameter_target, "mode": model.config.mode,
        "constitutional_manifest": "constitutional_manifest.json", "promotion_status": constitutional.promotion_status,
        "constitutional_sha256": constitutional.manifest_sha256, "authorized_by": constitutional.authorized_by}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_checkpoint(directory: str | Path, *, allow_quarantined: bool = False) -> AuroLanguageModel:
    directory = Path(directory)
    constitutional = None
    path = directory / "constitutional_manifest.json"
    if path.exists():
        constitutional = load_and_verify_constitutional_manifest(directory,
            signing_key=os.environ.get("AURO_CHECKPOINT_SIGNING_KEY"), require_promoted=not allow_quarantined)
    elif not allow_quarantined:
        raise RuntimeError("constitutional manifest is required for load")
    cfg = AuroLMConfig.from_dict(json.loads((directory / "config.json").read_text(encoding="utf-8")))
    tok = AuroTokenizer.load(directory / "tokenizer.json")
    model = AuroLanguageModel(cfg, tokenizer=tok)
    weights = np.load(directory / "weights.npz")
    model.core.embedding.token_embeddings = weights["token_embeddings"]
    model.core.lm_head_weight = weights["lm_head_weight"]
    if model.meaning is not None and weights["meaning_inject"].shape == model.meaning.inject.shape: model.meaning.inject = weights["meaning_inject"]
    if model.spectral is not None and weights["spectral_proj"].shape == model.spectral.proj.shape:
        model.spectral.proj = weights["spectral_proj"]; model.spectral.bias = weights["spectral_bias"]
    layers_path = directory / "layers.npz"
    if layers_path.exists():
        blob = np.load(layers_path)
        for i, layer in enumerate(model.core.layers):
            tr = layer["transformer"]
            attn = getattr(tr, "attention", None)
            if attn is not None:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    key = f"L{i}_attn_{name}"
                    if key in blob.files: setattr(attn, name, blob[key])
            ffn = getattr(tr, "ffn", None)
            if ffn is not None:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    key = f"L{i}_ffn_{name}"
                    if key in blob.files: setattr(ffn, name, blob[key])
    meta_path = directory / "meta.json"
    if meta_path.exists(): model.train_steps = int(json.loads(meta_path.read_text(encoding="utf-8")).get("train_steps", 0))
    if constitutional is not None: model.constitutional_checkpoint = constitutional  # type: ignore[attr-defined]
    return model
