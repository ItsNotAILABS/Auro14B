import numpy as np
import pytest

from auro_native_llm.model.sparse_projections import (
    AchlioptasProjector,
    SRHTProjector,
    projection_diagnostics,
)


def test_achlioptas_is_deterministic_and_sparse():
    left = AchlioptasProjector.build(100, 30, seed=7)
    right = AchlioptasProjector.build(100, 30, seed=7)
    np.testing.assert_array_equal(left.matrix, right.matrix)
    density = left.nonzero_count / left.matrix.size
    assert 0.27 < density < 0.40


def test_achlioptas_shape_and_validation():
    projector = AchlioptasProjector.build(13, 7, seed=9)
    assert projector.transform(np.ones((4, 13))).shape == (4, 7)
    with pytest.raises(ValueError):
        projector.transform(np.ones((4, 12)))


def test_srht_is_deterministic_for_non_power_of_two_input():
    left = SRHTProjector.build(13, 7, seed=11)
    right = SRHTProjector.build(13, 7, seed=11)
    x = np.random.default_rng(5).standard_normal((3, 13))
    np.testing.assert_array_equal(left.signs, right.signs)
    np.testing.assert_array_equal(left.sample_indices, right.sample_indices)
    np.testing.assert_allclose(left.transform(x), right.transform(x))
    assert left.padded_dim == 16


def test_srht_output_shape_and_finite_values():
    projector = SRHTProjector.build(64, 24, seed=17)
    result = projector.transform(np.random.default_rng(2).standard_normal((8, 64)))
    assert result.shape == (8, 24)
    assert np.isfinite(result).all()


def test_projection_receipt_is_repeatable():
    x = np.random.default_rng(4).standard_normal((30, 64))
    projector = SRHTProjector.build(64, 24, seed=9)
    y = projector.transform(x)
    left = projection_diagnostics(x, y, method="srht", state=projector.signs)
    right = projection_diagnostics(x, y, method="srht", state=projector.signs)
    assert left == right
    assert len(left.state_sha256) == 64
    assert left.mean_relative_distance_error >= 0.0


def test_invalid_projection_configurations_are_rejected():
    with pytest.raises(ValueError):
        AchlioptasProjector.build(10, 3, density=0.0)
    with pytest.raises(ValueError):
        SRHTProjector.build(10, 32)
