"""Example 03: Generate a PSD-compatible spectral record."""

import numpy as np
from mesie import generate_psd, validate_record
from mesie.core.config import GenerationConfig

# Generate a PSD with default settings
config = GenerationConfig(seed=42, amplitude_shape="power_law")
psd_record = generate_psd(config)

print(f"Record ID: {psd_record.record_id}")
print(f"Representation: {psd_record.representation}")
print(f"Units: {psd_record.components[0].units}")
print(f"Frequency points: {len(psd_record.components[0].frequency)}")
print(f"Min amplitude: {psd_record.components[0].amplitude.min():.6e}")
print(f"Max amplitude: {psd_record.components[0].amplitude.max():.6e}")

# Validate the generated record
report = validate_record(psd_record)
print(f"\nValidation: {'PASSED' if report.is_valid else 'FAILED'}")
