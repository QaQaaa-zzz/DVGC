import numpy as np

from cli.capture_takeoff_tail_snapshots_v1 import selected_entries, state_error


def test_selected_entries_preserve_requested_parent_order():
    rows = [{"trajectory_parent_id": "a"}, {"trajectory_parent_id": "b"}]
    assert selected_entries(rows, ["b", "a"]) == [rows[1], rows[0]]


def test_state_error_compares_both_qpos_and_qvel():
    class Data:
        qpos = np.array([1., 2.])
        qvel = np.array([3., 4.])
    class State:
        data = Data()
    assert state_error(State(), {"qpos": [1., 2.], "qvel": [3., 4.]}) == 0.
    assert state_error(State(), {"qpos": [1., 2.5], "qvel": [3., 4.]}) == .5
