import json
import numpy as np
import pytest


def test_invalid_controller_and_events_are_rejected_before_collection():
    from jit_dvgc.pulse_schedule import controller_mode, selected_event
    assert controller_mode({}) == 'learned_residual'
    assert controller_mode({'controller_mode': 'learned'}) == 'learned_residual'
    assert controller_mode({'controller_mode': 'fixed_random'}) == 'fixed_random'
    with pytest.raises(ValueError):
        controller_mode({'controller_mode': 'random_typo'})
    with pytest.raises(ValueError):
        selected_event({'pulse_event_schedule': []})
    with pytest.raises(ValueError):
        selected_event({'pulse_event_schedule': ['touchdown']})
    assert selected_event({'pulse_event_schedule': ['start', 'apex'], 'round_index': 3}) == 'apex'
    assert selected_event({'pulse_event_schedule': ['apex'], 'nominal_source_rollout': True}) is None


def test_events_use_real_airborne_motion_and_pre_action_crossing():
    from jit_dvgc.pulse_schedule import event_ready
    args = dict(tick=9, front_clearance=np.array([.06, .06, .01]),
                rear_clearance=np.array([.07, .07, .01]),
                previous_vertical_velocity=np.array([.1, -.1, .1]),
                vertical_velocity=np.array([-.02, -.2, -.02]),
                valid_contact_seen=np.zeros(3, bool), min_airborne_clearance=.05,
                min_descent_velocity=.05)
    np.testing.assert_array_equal(event_ready('apex', **args), [True, False, False])
    np.testing.assert_array_equal(event_ready('descent', **args), [False, True, False])
    np.testing.assert_array_equal(event_ready('liftoff', **args), [True, True, False])
    args['valid_contact_seen'] = np.ones(3, bool)
    assert not np.any(event_ready('descent', **args))


def test_descent_waits_until_near_contact_without_triggering_on_the_ground():
    from jit_dvgc.pulse_schedule import event_ready
    ready = event_ready('descent', tick=20,
        front_clearance=np.array([.3, .08, .02, .12]),
        rear_clearance=np.array([.3, .08, .02, .12]),
        previous_vertical_velocity=np.array([-.1, -.1, -.1, -.1]),
        vertical_velocity=np.array([-.2, -.2, -.2, -.2]),
        valid_contact_seen=np.zeros(4, bool), min_airborne_clearance=.05,
        min_descent_velocity=.05)
    np.testing.assert_array_equal(ready, [False, True, False, False])


def test_per_lane_trigger_has_exactly_three_actions_and_dead_lanes_never_trigger():
    from jit_dvgc.pulse_schedule import pulse_activity
    trigger = np.full(3, -1, np.int32)
    counts = np.zeros(3, np.int32)
    masks = []
    for tick in range(6):
        trigger, active = pulse_activity(trigger, counts, np.array([tick >= 1, tick >= 3, True]),
                                         np.array([True, True, False]), tick, 3)
        counts = counts + np.asarray(active, np.int32)
        masks.append(np.asarray(active))
    np.testing.assert_array_equal(trigger, [1, 3, -1])
    np.testing.assert_array_equal(counts, [3, 3, 0])
    np.testing.assert_array_equal(np.asarray(masks)[:, 0], [False, True, True, True, False, False])


def test_random_update_preserves_rng_optimizer_and_reports_no_learning(tmp_path):
    from jit_dvgc.pulse_exploration_runtime import update
    source = tmp_path / 'collection'
    source.mkdir()
    frozen_state = b'exact unmodified controller RNG and optimizer state'
    (source / 'update_state.msgpack').write_bytes(frozen_state)
    np.savez(source / 'prefixes.npz', mask=np.array([[True, False], [True, True]]))
    feedback = tmp_path / 'feedback.json'
    feedback.write_text(json.dumps(dict(eligible=[True, True], rewards=[1.25, -1.],
                                        component_sums={'novelty': .25, 'quality': 0.})))
    output = tmp_path / 'updated'
    update(dict(controller_mode='fixed_random', collection=str(source), feedback=str(feedback)), output)
    assert (output / 'state.msgpack').read_bytes() == frozen_state
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['optimizer_updates'] == 0
    assert metrics['effective_training_samples'] == 0
    assert metrics['reward'] == .25
    assert metrics['update_skipped'] and metrics['skip_reason'] == 'fixed_random_controller'
    assert json.loads((output / 'optimizer_updates.json').read_text()) == []


def test_geometry_trace_ignores_shared_warp_leaves_and_keeps_true_contact():
    from collections import namedtuple
    from types import SimpleNamespace
    import jax.numpy as jp
    from jit_dvgc.geometry import GEOM_SPHERE
    from jit_dvgc.pulse_exploration_runtime import physical_trace
    # The shared contact leaf deliberately has a different leading dimension
    # from the two worlds. Mapping the whole DataWarp-shaped tree is invalid.
    Data = namedtuple('Data', 'qpos qvel geom_xpos geom_xmat time shared_contact')
    qpos=jp.tile(jp.array([0.,0.,0.,1.,0.,0.,0.]),(2,1))
    data=Data(qpos,jp.zeros((2,6)),
        jp.array([[[0.,0.,.2],[1.,0.,.2],[.5,0.,.4]],
                  [[0.,0.,.3],[1.,0.,.3],[.5,0.,.4]]]),
        jp.tile(jp.eye(3).reshape(1,1,9),(2,3,1)),jp.array([.1,.2]),jp.zeros(5))
    geometry=SimpleNamespace(robot_geom_ids=jp.arange(3),
        robot_geom_types=jp.full(3,GEOM_SPHERE),robot_geom_sizes=jp.full((3,3),.1),
        wheel_geom_positions=jp.array([0,1]),body_geom_positions=jp.array([2]),
        obstacle_front_x=10.,obstacle_back_x=11.,obstacle_half_width=1.,obstacle_top_z=.2)
    down=SimpleNamespace(valid_contact_seen=jp.array([True,False]),contact_x=jp.array([1.,0.]),
                         post_contact_ticks=jp.array([3,0]))
    state=SimpleNamespace(data=data,info={'down_events':down,'success':jp.array([False,False])})
    result=physical_trace(SimpleNamespace(_geometry=geometry),state)
    np.testing.assert_allclose(result['front_wheel_clearance'],[.1,.2],atol=1e-7)
    np.testing.assert_array_equal(result['valid_contact_seen'],[True,False])
    np.testing.assert_array_equal(result['recovery_ticks'],[3,0])
