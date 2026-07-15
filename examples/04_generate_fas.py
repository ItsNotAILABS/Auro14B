"""Example 04: Generate an FAS-compatible spectral record."""

import numpy as np
from mesie import generate_fas, validate_record
from mesie.core.config import GenerationConfig

# Generate FAS with gaussian shape
config = GenerationConfig(seed=7, amplitude_shape="gaussian")
fas_record = generate_fas(config)

print(f"Record ID: {fas_record.record_id}")
print(f"Representation: {fas_record.representation}")
print(f"Units: {fas_record.components[0].units}")
print(f"Frequency points: {len(fas_record.components[0].frequency)}")
print(f"Peak amplitude: {fas_record.components[0].amplitude.max():.6f}")

# Validate
report = validate_record(fas_record)
print(f"\nValidation: {'PASSED' if report.is_valid else 'FAILED'}")
