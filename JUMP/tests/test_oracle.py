"""Hand-specified physical event sequences for the complete-task oracle."""

import copy
import math

import pytest

from jump_planning.oracle import TaskMonitor


@pytest.fixture
def config():
    return {
        "scene": {"front_x": 1.0, "length": 0.2, "height": 0.1,
                  "half_width": 0.5, "landing_near": 0.05, "landing_far": 1.5},
        "timing": {"sim_dt": 0.005, "control_dt": 0.020,
                   "episode_seconds": 8.0, "approach_timeout_seconds": 2.0,
                   "settle_seconds": 0.1, "airborne_seconds": 0.01,
                   "recovery_seconds": 2.0, "hold_seconds": 0.5},
        "limits": {"nominal_speed": 1.0, "min_forward_speed": 0.5,
                   "corridor_half_width": 0.3, "max_roll": 0.7,
                   "max_pitch": 0.8, "max_yaw": 0.6,
                   "recovery_roll": 0.1, "recovery_pitch": 0.2,
                   "recovery_rate": 0.5, "recovery_yaw": 0.1,
                   "recovery_lateral": 0.15, "recovery_speed_error": 0.2,
                   "contact_force_threshold": 0.01,
                   "contact_distance_tolerance": 0.001,
                   "airborne_clearance": 0.001},
    }


def frame(t, **changes):
    row = dict(time=t, x=0.5, y=0.0, z=0.4, vx=1.0, vy=0.0, vz=0.0,
               roll=0.0, pitch=0.0, yaw=0.0, wx=0.0, wy=0.0, wz=0.0,
               front_contact=True, rear_contact=True, body_contact=False,
               obstacle_contact=False, front_clearance=0.0, rear_clearance=0.0,
               front_touch_x=0.6, rear_touch_x=0.4, robot_back_x=0.3,
               finite=True, qpos=[0.0], qvel=[0.0], action=[0.0] * 4,
               ctrl=[0.0] * 4, signal=0.0, control_step=0, substep=0,
               phase="approach")
    row.update(changes)
    return row


def air(t, **changes):
    return frame(t, **dict(front_contact=False, rear_contact=False,
                          front_clearance=0.04, rear_clearance=0.03,
                          front_touch_x=None, rear_touch_x=None, **changes))


def launch(monitor):
    monitor.update(frame(0.0), triggered=False)
    monitor.update(frame(0.005), triggered=True)
    monitor.update(air(0.010), triggered=True)
    monitor.update(air(0.015), triggered=True)
    return monitor.update(air(0.020), triggered=True)


def land_and_pass(monitor):
    launch(monitor)
    monitor.update(frame(0.025, front_touch_x=1.4, rear_touch_x=1.3,
                         robot_back_x=1.21), triggered=False)
    return 5


def advance_recovery(monitor, start_tick=6, end_tick=405, overrides=None):
    result = None
    for tick in range(start_tick, end_tick + 1):
        change = {} if overrides is None else overrides(tick)
        result = monitor.update(frame(tick * 0.005, x=2.0,
                                      front_touch_x=1.6, rear_touch_x=1.4,
                                      robot_back_x=1.3, **change), triggered=False)
    return result


def test_trigger_is_recorded_separately_from_confirmed_liftoff(config):
    monitor = TaskMonitor(config)
    result = launch(monitor)
    assert result["status"] == "ongoing"
    assert result["trigger_time"] == pytest.approx(0.005)
    assert result["liftoff_time"] == pytest.approx(0.010)
    assert result["liftoff_confirmed_time"] == pytest.approx(0.020)
    assert result["front_landing_time"] is None


def test_trigger_receipt_survives_failure_in_its_first_physics_sample(config):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    result = monitor.update(frame(0.005, obstacle_contact=True), triggered=True)
    assert result["status"] == "failure"
    assert result["trigger_time"] == pytest.approx(0.005)


def test_exact_command_timestamp_precedes_first_triggered_physics_sample(config):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    result = monitor.update(frame(0.005, trigger_command_time=0.0), triggered=True)
    assert result["trigger_time"] == 0.0


@pytest.mark.parametrize("command_time", [math.nan, 0.010, -1.0])
def test_invalid_command_time_cannot_rewrite_event_order(config, command_time):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    result = monitor.update(frame(0.005, trigger_command_time=command_time), triggered=True)
    assert result["status"] == "unknown"


def test_first_landing_record_is_not_replaced_by_later_recontact(config):
    monitor = TaskMonitor(config)
    land_and_pass(monitor)
    monitor.update(air(0.030, robot_back_x=1.3), triggered=False)
    result = monitor.update(frame(0.035, front_touch_x=3.0, rear_touch_x=2.8,
                                  robot_back_x=2.7), triggered=False)
    assert result["status"] == "ongoing"
    assert result["front_landing_time"] == pytest.approx(0.025)
    assert result["front_landing_x"] == pytest.approx(1.4)
    assert result["rear_landing_time"] == pytest.approx(0.025)
    assert result["rear_landing_x"] == pytest.approx(1.3)


def test_initial_air_clearance_cannot_qualify_as_a_jump(config):
    monitor = TaskMonitor(config)
    for tick in range(8):
        result = monitor.update(air(tick * 0.005), triggered=False, qualified=False)
    assert result["status"] == "ongoing"
    assert result["liftoff_time"] is None
    assert monitor.finish()["reason"] == "approach_not_qualified"


def test_first_frame_airborne_does_not_fabricate_a_prior_grounded_state(config):
    monitor = TaskMonitor(config)
    for tick in range(8):
        result = monitor.update(air(tick * 0.005), triggered=True)
    assert result["liftoff_time"] is None
    assert result["status"] == "ongoing"


def test_spontaneous_jump_after_grounded_qualification_fails(config):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    for tick in range(1, 4):
        result = monitor.update(air(tick * 0.005), triggered=False)
    assert result["reason"] == "premature_liftoff"
    assert result["liftoff_time"] == pytest.approx(0.005)


def test_trigger_during_existing_airborne_run_does_not_rewrite_liftoff(config):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    monitor.update(air(0.005), triggered=False)
    monitor.update(air(0.010), triggered=True)
    result = monitor.update(air(0.015), triggered=True)
    assert result["reason"] == "premature_liftoff"


def test_short_contact_loss_and_insufficient_clearance_do_not_count(config):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=True)
    monitor.update(air(0.005), triggered=True)
    monitor.update(frame(0.010), triggered=True)
    for tick in range(3, 9):
        row = air(tick * 0.005)
        row["rear_clearance"] = 0.0005
        result = monitor.update(row, triggered=True)
    assert result["liftoff_time"] is None


@pytest.mark.parametrize("changes,reason", [
    ({"body_contact": True}, "body_contact"),
    ({"obstacle_contact": True}, "obstacle_contact"),
    ({"vx": 0.49}, "forward_speed_violation"),
    ({"vx": -0.1}, "forward_speed_violation"),
    ({"y": 0.301}, "corridor_violation"),
    ({"roll": -0.701}, "roll_limit"),
    ({"pitch": 0.801}, "pitch_limit"),
    ({"yaw": -0.601}, "yaw_limit"),
])
def test_physical_failures_count_before_approach_qualification(config, changes, reason):
    monitor = TaskMonitor(config)
    result = monitor.update(frame(0.0, **changes), triggered=False, qualified=False)
    assert result["status"] == "failure"
    assert result["reason"] == reason


def test_front_wheel_can_land_before_back_edge_has_passed(config):
    monitor = TaskMonitor(config)
    launch(monitor)
    row = air(0.025)
    row.update(front_contact=True, front_touch_x=1.3, front_clearance=0.0,
               robot_back_x=1.1)
    result = monitor.update(row, triggered=False)
    assert result["status"] == "ongoing"
    assert result["front_landing_time"] == pytest.approx(0.025)
    assert result["recovery_start_time"] is None
    result = monitor.update(frame(0.030, front_touch_x=1.4, rear_touch_x=1.3,
                                  robot_back_x=1.15), triggered=False)
    assert result["recovery_start_time"] is None
    result = monitor.update(frame(0.035, front_touch_x=1.4, rear_touch_x=1.3,
                                  robot_back_x=1.201), triggered=False)
    assert result["recovery_start_time"] == pytest.approx(0.035)


@pytest.mark.parametrize("wheel,location", [("front", 1.249), ("rear", 2.701)])
def test_each_first_wheel_landing_must_be_inside_declared_region(config, wheel, location):
    monitor = TaskMonitor(config)
    launch(monitor)
    row = frame(0.025, front_touch_x=1.4, rear_touch_x=1.3)
    row[f"{wheel}_touch_x"] = location
    result = monitor.update(row, triggered=False)
    assert result["reason"] == f"{wheel}_landing_outside_region"


def test_success_requires_full_two_second_observation(config):
    monitor = TaskMonitor(config)
    land_and_pass(monitor)
    result = advance_recovery(monitor, end_tick=404)
    assert result["status"] == "ongoing"
    result = advance_recovery(monitor, start_tick=405, end_tick=405)
    assert result["status"] == "success"
    assert result["terminal_time"] == pytest.approx(2.025)


def test_landing_does_not_hide_a_later_collision(config):
    monitor = TaskMonitor(config)
    land_and_pass(monitor)
    result = monitor.update(frame(0.030, body_contact=True), triggered=False)
    assert result["reason"] == "body_contact"
    assert result["status"] == "failure"


def test_recovery_hold_resets_on_tracking_violation(config):
    monitor = TaskMonitor(config)
    land_and_pass(monitor)
    result = advance_recovery(monitor, overrides=lambda tick: {"roll": 0.11} if tick == 360 else {})
    assert result["status"] == "failure"
    assert result["reason"] == "recovery_not_recovered"
    assert result["hold_start_time"] == pytest.approx(1.805)


def test_exact_final_half_second_hold_is_sufficient(config):
    monitor = TaskMonitor(config)
    land_and_pass(monitor)
    result = advance_recovery(monitor, overrides=lambda tick: {"roll": 0.11} if tick < 305 else {})
    assert result["status"] == "success"
    assert result["hold_start_time"] == pytest.approx(1.525)


@pytest.mark.parametrize("change", [
    {"pitch": 0.201}, {"yaw": -0.101}, {"wx": 0.501},
    {"wy": -0.501}, {"wz": 0.501}, {"y": -0.151}, {"vx": 0.799},
])
def test_recovery_requires_every_declared_tracking_quantity(config, change):
    monitor = TaskMonitor(config)
    land_and_pass(monitor)
    result = advance_recovery(monitor, overrides=lambda tick: change)
    assert result["reason"] == "recovery_not_recovered"


def test_no_physics_evidence_cannot_become_a_negative_task_label(config):
    result = TaskMonitor(config).finish()
    assert result["status"] == "unknown"
    assert result["reason"] == "empty_episode"


@pytest.mark.parametrize("changes", [
    {"vx": math.nan}, {"finite": False}, {"qvel": [0.0, math.inf]},
    {"front_touch_x": None}, {"time": math.inf},
])
def test_nonfinite_or_missing_physical_evidence_is_unknown(config, changes):
    monitor = TaskMonitor(config)
    result = monitor.update(frame(0.0, **changes), triggered=True)
    assert result["status"] == "unknown"
    assert result["done"] is True


@pytest.mark.parametrize("next_time", [0.0, -0.005, 0.020])
def test_nonmonotone_or_missing_physics_samples_cannot_prove_continuous_holds(config, next_time):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    result = monitor.update(frame(next_time), triggered=False)
    assert result["status"] == "unknown"


def test_terminal_failure_is_latched_and_finish_does_not_relabel_it(config):
    monitor = TaskMonitor(config)
    result = monitor.update(frame(0.0, obstacle_contact=True), triggered=False)
    saved = copy.deepcopy(result)
    result["reason"] = "caller_mutation"
    assert monitor.update(frame(0.005), triggered=True) == saved
    assert monitor.finish() == saved


def test_executed_episode_without_crossing_is_timeout_failure(config):
    monitor = TaskMonitor(config)
    monitor.update(frame(0.0), triggered=False)
    assert monitor.finish()["reason"] == "episode_timeout"


def test_yaw_wrap_does_not_turn_equivalent_forward_heading_into_failure(config):
    monitor = TaskMonitor(config)
    result = monitor.update(frame(0.0, yaw=2 * math.pi), triggered=False)
    assert result["status"] == "ongoing"


@pytest.mark.parametrize("field,value", [("sim_dt", 0.0), ("recovery_seconds", -1.0),
                                         ("hold_seconds", math.nan)])
def test_invalid_time_contract_is_rejected_before_evaluation(config, field, value):
    config["timing"][field] = value
    with pytest.raises(ValueError):
        TaskMonitor(config)
