import numpy as np
import pytest

from dvgc.local_entry import calibrate_local_radii
from dvgc.stage_reachability import support_distance


def test_local_radii_keep_anchor_specific_negative_outside():
    anchors = np.asarray([[0., 0.], [10., 0.]])
    rows = []
    for anchor in range(2):
        for delta in (.1, .2, .3, .4):
            rows.append({"anchor_index": anchor, "feature": anchors[anchor] + [delta, 0.], "safe": True})
        rows.append({"anchor_index": anchor, "feature": anchors[anchor] + [.6, 0.], "safe": False})
    result = calibrate_local_radii(anchors, rows, np.ones(2))
    assert result["status"] == "PASS"
    assert result["precision"] == 1.0
    assert result["radii"] == pytest.approx([.4, .4])


def test_local_radius_rejects_anchor_without_enough_safe_support():
    result = calibrate_local_radii(
        [[0., 0.]],
        [{"anchor_index": 0, "feature": [value, 0.], "safe": value < .3}
         for value in (.1, .2, .3, .4)],
        [1., 1.],
    )
    assert result["status"] == "FAIL"
    assert result["active_anchor_indices"] == []


def test_stage_support_distance_accepts_per_anchor_radii():
    metadata = {
        "support_features": [[0., 0.], [10., 0.]],
        "stage_entry_matcher": {"center": [0., 0.], "scale": [1., 1.], "radii": [.5, 2.]},
    }
    distance, matched = support_distance(np.asarray([11., 0.]), metadata)
    assert matched
    assert distance == pytest.approx(.5)
    distance, matched = support_distance(np.asarray([1., 0.]), metadata)
    assert not matched
    assert distance == pytest.approx(2.)


def test_stage_support_distance_rejects_invalid_radii():
    metadata = {
        "support_features": [[0., 0.]],
        "stage_entry_matcher": {"center": [0., 0.], "scale": [1., 1.], "radii": [0.]},
    }
    with pytest.raises(ValueError, match="radii"):
        support_distance(np.asarray([0., 0.]), metadata)
