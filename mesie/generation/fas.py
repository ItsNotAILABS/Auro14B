"""FAS-compatible spectral generation."""

from __future__ import annotations

from typing import Optional

from mesie.core.records import MultiElementRecord
from mesie.core.config import GenerationConfig
from mesie.generation.single_component import generate_single


def generate_fas(config: GenerationConfig, **kwargs) -> MultiElementRecord:
    """Generate an FAS-compatible spectral record.

    Generates a Fourier amplitude spectrum compatible output with appropriate
    units and representation markers.

    Args:
        config: Generation configuration.
        **kwargs: Additional arguments (reserved for future use).

    Returns:
        MultiElementRecord with FAS representation.
    """
    from dataclasses import replace
    cfg = replace(config, output_format="fas")
    record = generate_single(cfg)

    # Update representation and units
    for comp in record.components:
        comp.units = "fas"
    record.representation = "fas"
    record.lineage = ["generated", "synthetic", "fas"]

    return record
