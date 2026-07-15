"""Tests for spectral validation."""

import numpy as np
import pytest

from mesie.validation.validators import validate_record, ValidationReport
from mesie.validation.exceptions import MESIEValidationError


class TestValidation:
    def test_valid_record(self):
        payload = {
            "record_id": "valid",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.2, 0.3, 0.4]}
            ],
        }
        report = validate_record(payload)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_non_monotonic_frequency(self):
        payload = {
            "record_id": "bad",
            "components": [
                {"name": "a", "frequency": [1.0, 3.0, 2.0], "amplitude": [0.2, 0.3, 0.4]}
            ],
        }
        report = validate_record(payload)
        assert not report.is_valid
        assert any("non-monotonically" in e for e in report.errors)

    def test_nan_values(self):
        payload = {
            "record_id": "nan",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.2, float("nan"), 0.4]}
            ],
        }
        report = validate_record(payload)
        assert not report.is_valid
        assert any("NaN/Inf" in e for e in report.errors)

    def test_negative_amplitudes(self):
        payload = {
            "record_id": "neg",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.2, -0.1, 0.4]}
            ],
        }
        report = validate_record(payload)
        assert not report.is_valid
        assert any("negative" in e for e in report.errors)

    def test_empty_components(self):
        payload = {"record_id": "empty", "components": []}
        report = validate_record(payload)
        assert not report.is_valid

    def test_mismatched_lengths(self):
        payload = {
            "record_id": "mismatch",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0], "amplitude": [0.2, 0.3, 0.4]}
            ],
        }
        report = validate_record(payload)
        assert not report.is_valid

    def test_psd_unit_warning(self):
        payload = {
            "record_id": "psd_warn",
            "representation": "psd",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0], "amplitude": [0.2, 0.3, 0.4], "units": "linear"}
            ],
        }
        report = validate_record(payload)
        assert report.is_valid
        assert any("PSD unit" in w for w in report.warnings)

    def test_validation_level(self):
        payload = {
            "record_id": "leveled",
            "components": [
                {"name": "a", "frequency": [1.0, 2.0, 3.0, 4.0], "amplitude": [0.2, 0.3, 0.4, 0.5]}
            ],
        }
        report = validate_record(payload)
        assert report.is_valid
        assert report.level >= 4
