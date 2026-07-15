"""Example 09: Load bundled reference and benchmark datasets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import list_benchmarks, list_references, load_benchmark, load_reference_record
from mesie import validate_record


def main() -> None:
    print("References:", list_references())
    print("Benchmarks:", list_benchmarks())
    meta = load_benchmark("embedding_training_data")
    print(f"Embedding corpus: {meta['n_samples']} samples, dim={meta['feature_dim']}")
    record = load_reference_record("rotdnn_reference")
    report = validate_record(record)
    print(f"{record.record_id}: valid={report.is_valid}, level={report.level}")


if __name__ == "__main__":
    main()