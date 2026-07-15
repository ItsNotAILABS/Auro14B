"""Example 05: Generate and validate RotDnn-compatible spectra."""

from mesie import validate_record
from mesie.core.config import GenerationConfig
from mesie.generation.rotdnn import generate_rotdnn

# Generate RotDnn record with 3 components
config = GenerationConfig(
    seed=15,
    amplitude_shape="power_law",
    multi_element_blending={"RotD0": 0.25, "RotD50": 0.5, "RotD100": 0.25},
)
rotdnn_record = generate_rotdnn(config)

print(f"Record ID: {rotdnn_record.record_id}")
print(f"Representation: {rotdnn_record.representation}")
print(f"Components: {len(rotdnn_record.components)}")
for comp in rotdnn_record.components:
    print(f"  - {comp.name}: {len(comp.frequency)} points, max={comp.amplitude.max():.4f}")

# Validate
report = validate_record(rotdnn_record)
print(f"\nValidation: {'PASSED' if report.is_valid else 'FAILED'}")
if report.warnings:
    for w in report.warnings:
        print(f"  Warning: {w}")
