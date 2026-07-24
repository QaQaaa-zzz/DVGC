import numpy as np

from cli.analyze_apex_bridge_timing import (
    _auc,
    _first_action_tick,
    _first_divergence_tick,
)
from cli.build_descent_terminal_clusters import _deterministic_kmeans, _physical_shock


def test_failure_timing_is_strictly_before_first_nonzero_action():
    trace = [
        {"tick": 1, "action": [0, 0, 0, 0], "roll": .1, "pitch": .1,
         "angular_velocity": [1, 1, 0]},
        {"tick": 2, "action": [0, 0, 0, 0], "roll": .5, "pitch": .1,
         "angular_velocity": [5, 1, 0]},
        {"tick": 3, "action": [0, 0, .2, 0], "roll": .6, "pitch": .1,
         "angular_velocity": [6, 1, 0]},
    ]
    assert _first_action_tick(trace) == 3
    assert _first_divergence_tick(
        trace, max_roll=.7, max_pitch=1.3, max_rate=4.
    ) == 2


def test_auc_handles_ties_and_ordering():
    assert _auc([0, 0, 1, 1], [.1, .2, .8, .9]) == 1.
    assert _auc([0, 1], [.5, .5]) == .5
    assert _auc([1, 1], [.1, .2]) is None


def test_terminal_clustering_is_deterministic():
    x = np.asarray([[0., 0.], [.1, 0.], [4., 4.], [4.1, 4.]])
    labels_a, centers_a = _deterministic_kmeans(x, 2)
    labels_b, centers_b = _deterministic_kmeans(x, 2)
    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_allclose(centers_a, centers_b)
    assert len(set(labels_a.tolist())) == 2


def test_terminal_builder_keeps_final_safe_separate_from_noisy_shock_rule():
    text = open("cli/build_descent_terminal_clusters.py").read()
    assert 'row["replay_label"] == "boundary_replay"' in text
    assert "five_step_reset_shock_failure" in text
    assert "must not erase a" in text


def test_success_terminal_is_not_a_reset_shock_failure():
    assert not _physical_shock({
        "five_step_reset_shock_failure": True, "physical_failure": False,
    })
    assert _physical_shock({
        "five_step_reset_shock_failure": True, "physical_failure": True,
    })
    text = open("cli/audit_descent_support_compatibility.py").read()
    assert 'if entry["valid"]:' in text
    assert '"recovery", "chain_entry", "next_stage_entry"' in text
