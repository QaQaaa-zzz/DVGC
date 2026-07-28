from cli.discover_apex_feedback_bridge import bridge_objective


def _score(**overrides):
    values = dict(entry=False, stable=False, apex=True, done=False,
                  pose_margin=.2, target_distance=20., pose_cost=1.,
                  joint_margin=.1, action_energy=.1)
    values.update(overrides)
    return bridge_objective(**values)


def test_safe_pose_beats_transient_stable_with_failure_precursor():
    assert _score(stable=False, pose_margin=.2) > _score(stable=True, pose_margin=-.05)


def test_formal_entry_dominates_auxiliary_cost_and_done_is_worst():
    assert _score(entry=True, target_distance=100.) > _score(target_distance=0.)
    assert _score(done=True) < _score(done=False)
