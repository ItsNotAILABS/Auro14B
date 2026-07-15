"""Spectral processing operations."""

from mesie.processing.normalize import normalize_record, normalize_component
from mesie.processing.interpolate import interpolate_component, interpolate_record
from mesie.processing.smooth import smooth_component
from mesie.processing.transforms import log_transform, amplitude_scale

__all__ = [
    "amplitude_scale",
    "interpolate_component",
    "interpolate_record",
    "log_transform",
    "normalize_component",
    "normalize_record",
    "smooth_component",
]
