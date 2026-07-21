"""Auro-4B native construction lane.

The factory keeps structured pre-wiring explicit and auditable. Existing Auro
family construction is unchanged unless callers select this lane.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.prewiring import PrewiringConfig, PrewiringReceipt, apply_structured_prewiring


def build_auro4b(
    *,
    mode: str = "dev",
    structured: bool = True,
    prewiring: Optional[PrewiringConfig] = None,
    **overrides: Any,
) -> tuple[AuroLanguageModel, Optional[PrewiringReceipt]]:
    """Build the repository-native Auro-4B MESIE model.

    ``structured=False`` produces the matched repository baseline. The returned
    receipt is present only for the structured candidate.
    """
    model = AuroLanguageModel.build("Auro-4B", mode=mode, **overrides)
    receipt = apply_structured_prewiring(model, prewiring) if structured else None
    return model, receipt


def write_birth_certificate(
    model: AuroLanguageModel,
    output: str | Path,
    receipt: Optional[PrewiringReceipt],
    *,
    checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the model-birth certificate beside a checkpoint or audit bundle."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema": "auro.model.birth.v1",
        "model_id": model.model_id,
        "parameter_target": int(model.config.parameter_target),
        "live_parameter_count": int(model.num_params),
        "compute_plane": "MESIE",
        "native": True,
        "external_model_fallback": False,
        "structured_prewiring": receipt.to_dict() if receipt else None,
        "checkpoint_sha256": checkpoint_sha256,
        "claim_boundary": {
            "structured_inductive_bias": receipt is not None,
            "trained_general_knowledge": False,
            "benchmark_superiority_claimed": False,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
