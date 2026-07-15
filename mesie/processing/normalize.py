"""Spectral normalization operations."""

from __future__ import annotations

from typing import Union

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.io.loaders import RecordInput, load_record


def normalize_component(component: SpectralComponent, method: str = "max") -> SpectralComponent:
    """Normalize a spectral component amplitude.

    Args:
        component: Input spectral component.
        method: Normalization method ('max', 'l2', or 'zscore').

    Returns:
        New SpectralComponent with normalized amplitude.

    Raises:
        ValueError: If method is not supported.
    """
    amp = component.amplitude.astype(float)
    if method == "max":
        scale = max(float(np.max(np.abs(amp))), 1e-12)
        out = amp / scale
    elif method == "l2":
        scale = max(float(np.linalg.norm(amp)), 1e-12)
        out = amp / scale
    elif method == "zscore":
        std = max(float(np.std(amp)), 1e-12)
        out = (amp - float(np.mean(amp))) / std
    else:
        raise ValueError(f"Unsupported normalization method '{method}'. Use 'max', 'l2', or 'zscore'.")
    return SpectralComponent(**{**component.__dict__, "amplitude": out})


def normalize_record(record: RecordInput, method: str = "max") -> MultiElementRecord:
    """Normalize all components in a spectral record.

    Args:
        record: Input record (any supported format).
        method: Normalization method ('max', 'l2', or 'zscore').

    Returns:
        New MultiElementRecord with normalized components.
    """
    rec = load_record(record)
    return MultiElementRecord(
        record_id=rec.record_id,
        components=[normalize_component(c, method=method) for c in rec.components],
        metadata=rec.metadata,
        lineage=rec.lineage + ["normalized"],
        representation=rec.representation,
        node_tags=rec.node_tags,
        electro_metadata=rec.electro_metadata,
    )
