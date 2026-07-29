import numpy as np
import pytest

from cli.train_descent_reachability_network_v3 import (
    final_branch_success,
    fit_network,
    merge_rows,
    parent_cv,
    predict_network,
    safety_class,
    select_pilot,
)


def test_safety_class_preserves_three_way_construction_labels():
    assert safety_class(0.0) == "dead"
    assert safety_class(0.5) == "boundary"
    assert safety_class(0.75) == "safe"


def test_final_label_is_independent_of_chain_but_rejects_unhealthy_outcome():
    assert final_branch_success({"final_recovery": True, "downstream_entry": False})
    assert not final_branch_success({"final_recovery": True, "timeout": True})
    assert not final_branch_success({"final_recovery": False, "downstream_entry": True})


def test_network_is_deterministic_finite_and_bounded():
    x = np.arange(96, dtype=float).reshape(6, 16) / 20.0
    y = np.asarray([0.0, 0.0, 0.25, 0.75, 1.0, 1.0])
    first = fit_network(x, y, seed=7, steps=50)
    second = fit_network(x, y, seed=7, steps=50)
    assert all(np.array_equal(first[key], second[key]) for key in first)
    prediction = predict_network(first, x)
    assert np.isfinite(prediction).all()
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_parent_cv_holds_out_complete_parent():
    x = np.arange(128, dtype=float).reshape(8, 16) / 30.0
    y = np.asarray([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 1.0])
    parents = np.asarray(["a", "a", "b", "c", "d", "e", "f", "g"])
    prediction = parent_cv(x, y, parents, seed=11)
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()


def test_merge_rejects_conflicting_state_identity():
    old = [{"key": "same", "parent": "a", "target": 0.0, "physical": np.zeros(16),
            "history": np.zeros(140), "source": "construction"}]
    new = [{"state_hash": "same", "parent": "b", "target": 1.0,
            "physical": np.zeros(16), "history": np.zeros(140), "source": "source-a"}]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        merge_rows(old, new)


def test_pilot_selection_is_parent_disjoint():
    ranked = [
        {"candidate_id": parent, "predicted_p_safe": score}
        for parent, score in [("a", .9), ("a", .8), ("b", .7), ("c", .6), ("d", .5)]
    ]
    selected = select_pilot(ranked, count=4)
    assert [row["candidate_id"] for row in selected] == ["a", "b", "c", "d"]


def test_pilot_selection_supports_remaining_two_parent_round():
    ranked = [{"candidate_id": "a"}, {"candidate_id": "b"}]
    assert len(select_pilot(ranked, count=2)) == 2
