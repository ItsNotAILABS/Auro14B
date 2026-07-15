"""Tests for bundled data/ reference and benchmark loaders."""

import pytest

from data import (
    list_benchmarks,
    list_references,
    load_benchmark,
    load_reference_record,
)
from mesie import validate_record


@pytest.mark.parametrize("name", list_references())
def test_all_references_validate_level_6(name: str):
    record = load_reference_record(name)
    report = validate_record(record)
    assert report.is_valid, f"{name}: {report.errors}"
    assert report.level >= 6


def test_embedding_benchmark_shape():
    data = load_benchmark("embedding_training_data")
    assert data["n_samples"] == 200
    assert len(data["samples"]) == 200
    assert data["feature_dim"] == 128


def test_classification_benchmark_classes():
    data = load_benchmark("spectral_classification_benchmark")
    assert data["n_classes"] == 5
    assert len(data["samples"]) == 250
    assert data["n_samples_per_class"] == 50


def test_reference_amplitudes_non_negative():
    for name in list_references():
        record = load_reference_record(name)
        for comp in record.components:
            assert (comp.amplitude >= 0).all(), f"{name}/{comp.name} has negative amplitude"