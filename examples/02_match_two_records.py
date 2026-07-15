"""Example 02: Match two spectral records and examine scores."""

from mesie import load_record, match_records

# Reference signal
reference = load_record({
    "record_id": "reference",
    "components": [
        {
            "name": "channel_1",
            "frequency": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "amplitude": [0.01, 0.05, 0.12, 0.25, 0.18, 0.08, 0.03, 0.01],
        }
    ],
})

# Candidate signal (slightly different)
candidate = load_record({
    "record_id": "candidate",
    "components": [
        {
            "name": "channel_1",
            "frequency": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            "amplitude": [0.012, 0.048, 0.115, 0.24, 0.175, 0.082, 0.031, 0.011],
        }
    ],
})

# Match the records
result = match_records(reference, candidate)

print(f"Composite Score: {result.composite_score:.4f}")
print(f"\nMetric Breakdown:")
for metric, value in result.metric_breakdown.items():
    print(f"  {metric}: {value:.4f}")
