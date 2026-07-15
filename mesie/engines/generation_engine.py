"""Generation engine — synthesize PSD/FAS/RotDNN records."""

from __future__ import annotations

from typing import Optional

import numpy as np

from mesie.core.config import GenerationConfig
from mesie.engines.base import Engine
from mesie.generation.fas import generate_fas
from mesie.generation.psd import generate_psd
from mesie.generation.rotdnn import generate_rotdnn
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.validation.validators import validate_record


class GenerationEngine(Engine):
    name = "generation"
    capabilities = ["generate_psd", "generate_fas", "generate_rotdnn"]

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            seed = int(message.payload.get("seed", 42))
            n = int(message.payload.get("n_points", 64))
            cfg = GenerationConfig(seed=seed, target_frequency=np.linspace(0.2, 40, n))
            generators = {
                "generate_psd": generate_psd,
                "generate_fas": generate_fas,
                "generate_rotdnn": generate_rotdnn,
            }
            rec = generators[action](cfg)
            report = validate_record(rec)
            return EngineResponse(
                True,
                self.name,
                action,
                {
                    "record_id": rec.record_id,
                    "valid": report.is_valid,
                    "validation_level": report.level,
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))