"""Example 01: Load and validate a spectral record."""

from mesie import load_record, validate_record

# Create a sample record from a dictionary
payload = {
    "record_id": "example_signal",
    "components": [
        {
            "name": "horizontal",
            "frequency": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "amplitude": [0.01, 0.05, 0.12, 0.25, 0.18, 0.08, 0.03, 0.01],
        }
    ],
}

# Load the record
record = load_record(payload)
print(f"Record ID: {record.record_id}")
print(f"Components: {len(record.components)}")
print(f"Representation: {record.representation}")

# Validate the record
report = validate_record(record)
print(f"\nValidation: {'PASSED' if report.is_valid else 'FAILED'}")
print(f"Level: {report.level}/6")
if report.warnings:
    print(f"Warnings: {report.warnings}")
