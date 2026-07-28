from dvgc.backward_tube import balanced_p1_launch_gate


def test_empty_launch_cannot_be_frozen():
    gate = balanced_p1_launch_gate([])
    assert gate["status"] == "FAIL"
    assert not gate["checks"]["minimum_nodes"]
