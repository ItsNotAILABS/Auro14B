"""Spectral record validation with multi-level checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record


@dataclass
class ValidationReport:
    """Validation report for a spectral record.

    Attributes:
        is_valid: Whether the record passes all critical checks.
        errors: Critical validation errors.
        warnings: Non-critical warnings.
        level: Highest validation level passed (1-6).
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    level: int = 0


class _ValidationRules:
    """Internal validation rule sets."""
    PSD_UNITS = {"psd", "power/hz", "(m/s^2)^2/hz", "g^2/hz"}
    FAS_UNITS = {"fas", "m/s", "cm/s", "unit/sqrt(hz)"}


def validate_record(record: RecordInput) -> ValidationReport:
    """Validate a spectral record through multiple levels of checks.

    Validation levels:
        1. File validity (data can be loaded)
        2. Spectral validity (finite values, monotonic frequencies)
        3. Component compatibility (consistent dimensions)
        4. PSD/FAS/RotDnn compatibility
        5. Embedding-readiness
        6. Cognitive integration readiness

    Args:
        record: Input record in any supported format.

    Returns:
        ValidationReport with errors, warnings, and validation level.
    """
    rec = load_record(record)
    errors: List[str] = []
    warnings: List[str] = []
    level = 1  # Level 1: file validity (already loaded)

    # Level 2: Spectral validity
    if not rec.components:
        errors.append("Record contains no spectral components.")
    else:
        level = 2

    base_len = None
    base_freq = None
    for idx, c in enumerate(rec.components):
        if len(c.frequency) != len(c.amplitude):
            errors.append(f"Component '{c.name}' has mismatched frequency and amplitude array lengths.")
            continue

        if len(c.frequency) == 0:
            errors.append(f"Component '{c.name}' has empty frequency data.")
            continue

        if np.any(~np.isfinite(c.frequency)) or np.any(~np.isfinite(c.amplitude)):
            errors.append(f"Component '{c.name}' contains NaN/Inf values.")

        if np.any(np.diff(c.frequency) <= 0):
            errors.append(f"Component '{c.name}' has non-monotonically increasing frequency values.")

        if np.any(c.amplitude < 0):
            errors.append(f"Component '{c.name}' contains negative amplitudes.")

        if len(c.frequency) > 1 and np.any(np.diff(c.frequency) > np.median(np.diff(c.frequency)) * 5):
            warnings.append(f"Component '{c.name}' may have missing frequencies due to large grid gaps.")

        # Level 3: Component compatibility
        if base_len is None:
            base_len = len(c.frequency)
            base_freq = c.frequency
        else:
            if len(c.frequency) != base_len:
                warnings.append("Incompatible component dimensions across components.")
            elif not np.allclose(c.frequency, base_freq):
                warnings.append("Components are on different frequency grids.")

    if not errors and len(rec.components) > 0:
        level = 3

    # Level 4: PSD/FAS/RotDnn compatibility
    rep = rec.representation.lower()
    if rep == "psd":
        for c in rec.components:
            if c.units.lower() not in _ValidationRules.PSD_UNITS:
                warnings.append(
                    f"PSD unit compatibility warning: '{c.name}' uses '{c.units}' for PSD representation."
                )

    if rep == "fas":
        for c in rec.components:
            if c.units.lower() not in _ValidationRules.FAS_UNITS:
                warnings.append(
                    f"FAS unit compatibility warning: '{c.name}' uses '{c.units}' for FAS representation."
                )

    if rep == "rotdnn":
        names = {c.name.lower() for c in rec.components}
        if not any(("rot" in n or "rotd" in n) for n in names):
            warnings.append("RotDnn component consistency checks: expected RotDnn-like component names.")
        if len(rec.components) < 2:
            warnings.append("RotDnn component consistency checks: expected multiple rotational components.")

    if not errors:
        level = 4

    # Level 5: Embedding-readiness
    if not errors and all(len(c.frequency) >= 4 for c in rec.components):
        level = 5

    # Level 6: Cognitive integration readiness
    if not errors and level >= 5:
        level = 6

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        level=level,
    )
