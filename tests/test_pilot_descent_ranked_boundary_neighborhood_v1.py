from cli.pilot_descent_ranked_boundary_neighborhood_v1 import local_deltas, p0_anchor_ids


def test_local_grid_is_fixed_symmetric_and_excludes_zero():
    deltas = local_deltas()
    assert deltas.shape == (24, 2)
    assert not ((deltas == 0).all(axis=1)).any()
    assert {tuple(row) for row in deltas} == {tuple(-row) for row in deltas}
    assert abs(deltas).max() == 0.01


def test_only_p0_rows_become_local_anchors():
    certification = {"rows": [
        {"node_id": "b", "P0": {"pass": True}},
        {"node_id": "a", "P0": {"pass": True}},
        {"node_id": "c", "P0": {"pass": False}},
    ]}
    assert p0_anchor_ids(certification) == ["a", "b"]
