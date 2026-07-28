import numpy as np

from dvgc.descent_balanced import iterative_balanced_weights, marginal_masses


def test_iterative_weights_balance_candidate_layer_and_region():
    rows = [
        {"candidate_id": "a", "layer": 1, "region": "late"},
        {"candidate_id": "a", "layer": 2, "region": "middle"},
        {"candidate_id": "b", "layer": 2, "region": "middle"},
        {"candidate_id": "b", "layer": 3, "region": "early"},
    ]
    weights = iterative_balanced_weights(rows, iterations=500)
    assert np.isclose(weights.sum(), 1)
    for field in ("candidate_id", "layer", "region"):
        mass = list(marginal_masses(rows, weights, field).values())
        assert max(mass) - min(mass) < 1e-4
