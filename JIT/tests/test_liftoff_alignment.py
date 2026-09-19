import numpy as np
import pytest

from jit_dvgc.analysis.liftoff_alignment import liftoff_index, align_xz


def test_initial_clearance_and_short_bounce_are_not_liftoff():
    clearance = np.array([.031, -.001, .02, 0., .02, .03, .04])
    assert liftoff_index(clearance, clearance, threshold=.01, hold=3) == 4


def test_no_ground_or_no_sustained_airborne_returns_none():
    assert liftoff_index(np.ones(5), np.ones(5)) is None
    assert liftoff_index([0, .02, .03], [0, 0, .03]) is None


def test_alignment_preserves_height_and_relative_displacement():
    original = np.array([[2.5, .13], [2.8, .18], [3.4, .6]])
    aligned = align_xz(original, 2.8)
    np.testing.assert_allclose(aligned[:, 0], [-.3, 0., .6], atol=1e-12)
    np.testing.assert_array_equal(aligned[:, 1], original[:, 1])
    assert original[1, 0] == 2.8


def test_invalid_clearance_rejected():
    with pytest.raises(ValueError):
        liftoff_index([0, np.nan], [0, .1])
    with pytest.raises(ValueError):
        liftoff_index([0], [0], hold=0)
