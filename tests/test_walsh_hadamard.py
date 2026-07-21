from __future__ import annotations

import numpy as np
import pytest

from auro_native_llm.model.walsh_hadamard import (
    diagnose,
    fwht,
    hadamard_matrix,
    next_power_of_two,
    walsh_tensor,
)


def test_fwht_round_trip_and_energy():
    value = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    transformed = fwht(value)
    np.testing.assert_allclose(transformed, [np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(fwht(transformed), value, atol=1e-12)
    assert np.linalg.norm(transformed) == pytest.approx(np.linalg.norm(value))


def test_hadamard_is_orthonormal_in_both_orderings():
    for ordering in ("natural", "sequency"):
        matrix = hadamard_matrix(16, ordering=ordering)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(16), atol=1e-12)


def test_sequency_order_is_monotonic_by_sign_changes():
    matrix = hadamard_matrix(16, ordering="sequency")
    changes = np.count_nonzero(matrix[:, 1:] != matrix[:, :-1], axis=1)
    assert np.all(changes[:-1] <= changes[1:])


def test_walsh_tensor_is_deterministic_and_supports_non_power_width():
    left = walsh_tensor(19, 13, seed=873539)
    right = walsh_tensor(19, 13, seed=873539)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(np.linalg.norm(left, axis=1), np.ones(19), atol=1e-12)
    assert next_power_of_two(13) == 16


def test_invalid_transform_length_is_rejected():
    with pytest.raises(ValueError):
        fwht(np.ones(6))


def test_diagnostics_enforce_transform_invariants():
    receipt = diagnose(32)
    assert receipt.orthogonality_max_error < 1e-12
    assert receipt.involution_max_error < 1e-12
    assert receipt.energy_error < 1e-12
    assert len(receipt.basis_sha256) == 64
