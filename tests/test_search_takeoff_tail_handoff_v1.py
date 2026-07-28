import numpy as np

from cli.search_takeoff_tail_handoff_v1 import residual_design, support_distance


def test_residual_design_is_reproducible_bounded_and_includes_nominal():
    first = residual_design(32, 17, .3)
    second = residual_design(32, 17, .3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], np.zeros(4))
    assert np.max(np.abs(first)) <= .3
    assert len(np.unique(first, axis=0)) == 32


def test_support_distance_zero_at_support_state():
    feature = np.arange(16, dtype=float)
    assert support_distance(feature, feature[None, :], np.ones(16)) == 0.
