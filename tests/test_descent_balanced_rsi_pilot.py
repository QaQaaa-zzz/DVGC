import numpy as np

from cli.run_descent_balanced_rsi_pilot import build_training_records


def test_training_mix_has_exact_group_masses(monkeypatch):
    balanced = [{"node_id": f"b{i}", "candidate_id": f"c{i%2}", "layer": i%2+1,
                 "region": ("late", "middle")[i%2], "physical_state": {}} for i in range(4)]
    frontier = [{"node_id": "f", "candidate_id": "f", "layer": 2, "region": "middle", "physical_state": {}}]
    c_l = [{"id": "l", "final": {"label": "safe"}}]
    monkeypatch.setattr("cli.run_descent_balanced_rsi_pilot._load_node_record", lambda node, _: {"id": node["node_id"]})
    rows, audit = build_training_records(balanced, frontier, c_l, [])
    assert np.isclose(sum(row["reset_weight"] for row in rows), 1.)
    assert all(np.isclose(audit["group_mass"][key], value) for key, value in {
        "balanced_P1": .6, "non_dominant_P0_frontier": .2, "downstream_C_L_retention": .2}.items())
    assert all(row["descent_layer"] in {"late", "middle"} for row in rows if row.get("backward_tube_label") != "C_L_interface")
