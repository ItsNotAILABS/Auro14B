"""PSD-compatible spectral generation."""

from __future__ import annotations

from typing import Optional

from mesie.core.records import MultiElementRecord
from mesie.core.config import GenerationConfig
from mesie.generation.single_component import generate_single


def generate_psd(config: GenerationConfig, **kwargs) -> MultiElementRecord:
    """Generate a PSD-compatible spectral record.

    Generates a power spectral density compatible output with appropriate
    units and representation markers.

    Args:
        config: Generation configuration.
        **kwargs: Additional arguments (reserved for future use).

    Returns:
        MultiElementRecord with PSD representation.
    """
    # Override output format to PSD
    from dataclasses import replace
    cfg = replace(config, output_format="psd")
    record = generate_single(cfg)

    # Update representation and units
    for comp in record.components:
        comp.units = "psd"
    record.representation = "psd"
    record.lineage = ["generated", "synthetic", "psd"]

    return record
