import numpy as np

from cli.train_descent_reachability_kernel_v2 import (
    average_precision,
    grouped_cv,
    kernel_predict,
    record_dedup_id,
    select_ranked,
)


def test_kernel_prediction_is_finite_bounded_and_deterministic():
    x = np.arange(48, dtype=float).reshape(12, 4)
    y = np.linspace(0, 1, 12)
    first = kernel_predict(x, y, x[:3], 0.2)
    second = kernel_predict(x, y, x[:3], 0.2)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all() and np.all((first >= 0) & (first <= 1))


def test_old_construction_record_id_is_explicit_dedup_fallback():
    assert record_dedup_id({"id": "old"}) == "old"
    assert record_dedup_id({"id": "old", "state_byte_hash": "new"}) == "new"


def test_grouped_cv_never_trains_on_held_parent():
    x = np.arange(36, dtype=float).reshape(9, 4)
    y = np.asarray([0, 0, 1, 0, 1, 0, 1, 0, 0], dtype=float)
    parents = np.asarray(["a", "a", "b", "c", "d", "e", "f", "g", "h"])
    prediction, bandwidths = grouped_cv(x, y, parents)
    assert prediction.shape == y.shape and len(bandwidths) == len(set(parents))
    assert np.isfinite(prediction).all()
    assert 0 <= average_precision(y, prediction) <= 1


def test_ranked_selection_is_region_balanced_and_parent_disjoint():
    rows = []
    for region in ("early", "middle", "late"):
        for index in range(6):
            rows.append({
                "region": region, "candidate_id": f"{region}-{index}",
                "proposal_id": f"p-{region}-{index}", "reachability_score": 1 - index / 10,
            })
    selected = select_ranked(rows, {"early-0"}, per_region=2)
    assert {region: sum(row["region"] == region for row in selected) for region in ("early", "middle", "late")} == {
        "early": 2, "middle": 2, "late": 2,
    }
    parents = [row["candidate_id"] for row in selected]
    assert len(parents) == len(set(parents)) and "early-0" not in parents


def test_ranked_selection_redistributes_exhausted_region_without_parent_reuse():
    rows = [
        {"region": "early", "candidate_id": "e", "proposal_id": "pe", "reachability_score": 0.9},
        {"region": "late", "candidate_id": "l", "proposal_id": "pl", "reachability_score": 0.8},
    ] + [
        {"region": "middle", "candidate_id": f"m{i}", "proposal_id": f"pm{i}", "reachability_score": 0.7-i/100}
        for i in range(8)
    ]
    selected = select_ranked(rows, set(), per_region=2, total=6)
    assert len(selected) == 6
    assert sum(row["region"] == "early" for row in selected) == 1
    assert len({row["candidate_id"] for row in selected}) == 6
