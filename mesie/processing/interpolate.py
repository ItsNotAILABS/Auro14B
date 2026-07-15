"""Spectral interpolation operations."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.io.loaders import RecordInput, load_record

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_float_array(values: ArrayLike, field_name: str) -> np.ndarray:
    """Convert to 1D float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional; got shape {arr.shape}.")
    return arr


def interpolate_component(
    component: SpectralComponent,
    target_frequency: ArrayLike,
    log_frequency: bool = False,
) -> SpectralComponent:
    """Interpolate a component onto a target frequency grid.

    Args:
        component: Input spectral component.
        target_frequency: Target frequency values.
        log_frequency: If True, interpolate in log-frequency space.

    Returns:
        New SpectralComponent interpolated to the target grid.

    Raises:
        ValueError: If log_frequency is True and frequencies are non-positive.
    """
    target = _as_float_array(target_frequency, "target_frequency")
    src_f = component.frequency
    src_a = component.amplitude

    if log_frequency:
        if np.any(src_f <= 0) or np.any(target <= 0):
            raise ValueError("Log-frequency interpolation requires strictly positive frequencies.")
        x_src = np.log10(src_f)
        x_tgt = np.log10(target)
    else:
        x_src = src_f
        x_tgt = target

    amp = np.interp(x_tgt, x_src, src_a, left=src_a[0], right=src_a[-1])
    phase = None
    if component.phase is not None:
        phase = np.interp(x_tgt, x_src, component.phase, left=component.phase[0], right=component.phase[-1])

    return SpectralComponent(**{**component.__dict__, "frequency": target, "amplitude": amp, "phase": phase})


def interpolate_record(
    record: RecordInput,
    target_frequency: ArrayLike,
    log_frequency: bool = False,
) -> MultiElementRecord:
    """Interpolate all components in a record to a common frequency grid.

    Args:
        record: Input record.
        target_frequency: Target frequency grid.
        log_frequency: If True, interpolate in log-frequency space.

    Returns:
        New MultiElementRecord with all components on the target grid.
    """
    rec = load_record(record)
    return MultiElementRecord(
        record_id=rec.record_id,
        components=[interpolate_component(c, target_frequency, log_frequency) for c in rec.components],
        metadata=rec.metadata,
        lineage=rec.lineage + ["interpolated"],
        representation=rec.representation,
        node_tags=rec.node_tags,
        electro_metadata=rec.electro_metadata,
    )
