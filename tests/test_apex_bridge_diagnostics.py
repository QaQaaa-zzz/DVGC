import numpy as np

from cli.analyze_apex_bridge_timing import (
    _auc,
    _first_action_tick,
    _first_divergence_tick,
)
from cli.build_descent_terminal_clusters import _deterministic_kmeans, _physical_shock
from cli.audit_apex_control_authority import _offset_ticks, _pulse_action
from cli.discover_apex_feedback_bridge import (
    _actions,
    _bridge_has_no_physical_failure,
    _terminal_distance,
)


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


def test_control_authority_offsets_cover_pre_and_event():
    offsets = _offset_ticks(15)
    assert offsets == [3, 7, 11, 15]
    short = _offset_ticks(7)
    assert short[-1] == 7
    assert len(short) == 4


def test_control_authority_pulses_are_paired_and_nonsaturated():
    plus = np.asarray(_pulse_action("opposite_positive", .25))
    minus = np.asarray(_pulse_action("opposite_negative", .25))
    np.testing.assert_allclose(plus, -minus)
    assert np.max(np.abs(plus)) == .25


def test_feedback_bridge_action_set_is_bounded_and_has_neutral():
    actions = np.asarray(_actions())
    assert np.max(np.abs(actions)) < 1.
    assert np.allclose(actions[0], 0.)
    assert actions.shape[1] == 4


def test_terminal_distance_uses_declared_physical_projection():
    feature = np.zeros(16)
    center = np.zeros(11)
    scale = np.ones(11)
    target = np.zeros((1, 11))
    assert _terminal_distance(feature, target, center, scale) == 0.


def test_feedback_bridge_supports_separate_nominal_and_fresh_runs():
    text = open("cli/discover_apex_feedback_bridge.py").read()
    assert '"deterministic", "fresh", "all"' in text
    assert "--nominal-report" in text
    assert "deterministic_stable" in text


def test_gate_a_rejects_transient_descent_followed_by_roll_failure():
    assert not _bridge_has_no_physical_failure({
        "termination_reason": "roll_limit",
    })
    assert _bridge_has_no_physical_failure({
        "termination_reason": "horizon_exhaustion",
    })
    text = open("cli/stage_next_v3_controller.py").read()
    assert "feedback_bridge_gate_reclassification_v2" in text
    assert "transient four-tick negative-vz segment" in text
