from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from typing import NamedTuple
from dataclasses import replace

from dvgc.config import load_config
from dvgc.two_phase_runtime import (
    EVENT_NAMES,
    TwoPhaseThresholds,
    advance_two_phase_events,
    build_two_phase_geometry,
    collision_geom_support_bounds,
    extract_apex_band_signals,
    extract_recovery_signals,
    full_structure_metrics,
    geometry_manifest,
    validate_geometry_manifest,
    initial_two_phase_event_state,
)
from dvgc.two_phase_semantics import (
    ApexBandSignals,
    ApexBandThresholds,
    RecoverySignals,
    RecoveryThresholds,
)


XML = "assets/orange_bike_4kg_horizontal.xml"


class FakeData(NamedTuple):
    qpos: jp.ndarray
    qvel: jp.ndarray
    geom_xpos: jp.ndarray
    geom_xmat: jp.ndarray


class FakeState(NamedTuple):
    data: FakeData
    info: dict[str, jp.ndarray]
    reward: jp.ndarray


def _synthetic_geometry():
    model = mujoco.MjModel.from_xml_path(XML)
    cfg = load_config("configs/default.json")
    geometry = build_two_phase_geometry(model, cfg)
    return replace(
        geometry,
        robot_geom_ids=np.asarray([0, 1, 2], np.int32),
        robot_geom_types=jp.asarray([int(mujoco.mjtGeom.mjGEOM_BOX)] * 3, jp.int32),
        robot_geom_sizes=jp.asarray([[0.1, 0.1, 0.1]] * 3),
        robot_geom_body_ids=np.asarray([1, 2, 3], np.int32),
        wheel_mask=jp.asarray([True, True, False]),
        body_mask=jp.asarray([False, False, True]),
        root_qpos_adr=0,
        root_dof_adr=0,
        airborne_confirm_steps=3,
        support_tolerance=0.01,
        body_penetration_tolerance=0.002,
    )


def _fake_state(
    *, geom_x=None, geom_y=None, geom_z=None, x=4.5, y=0.0, z=0.5, vx=1.2, vz=0.0
):
    geom_x = [4.4, 4.6, 4.5] if geom_x is None else geom_x
    geom_y = [y, y, y] if geom_y is None else geom_y
    geom_z = [0.26, 0.26, 0.40] if geom_z is None else geom_z
    geom_xpos = jp.asarray(
        [[gx, gy, gz] for gx, gy, gz in zip(geom_x, geom_y, geom_z, strict=True)]
    )
    return FakeState(
        data=FakeData(
            qpos=jp.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0]),
            qvel=jp.asarray([vx, 0.0, vz, 0.1, -0.2, 0.2]),
            geom_xpos=geom_xpos,
            geom_xmat=jp.tile(jp.eye(3).reshape((1, 9)), (3, 1)),
        ),
        info={
            "airborne_count": jp.asarray(3, jp.int32),
            "terminated": jp.asarray(0, jp.int32),
            "truncated": jp.asarray(0, jp.int32),
            "end_code": jp.asarray(0, jp.int32),
            "jump_signal_latched": jp.asarray(True),
            "phase": jp.asarray(0, jp.int32),
            "reference_index": jp.asarray(999, jp.int32),
            "tube_distance": jp.asarray(-999.0),
        },
        reward=jp.asarray(1234.0),
    )


def test_authoritative_geometry_manifest_covers_every_collision_robot_geom():
    model = mujoco.MjModel.from_xml_path(XML)
    cfg = load_config("configs/default.json")

    geometry = build_two_phase_geometry(model, cfg)
    manifest = geometry_manifest(model, geometry)
    validation = validate_geometry_manifest(manifest)

    assert len(manifest["geoms"]) == model.ngeom
    assert validation["valid"] is True
    assert manifest["authoritative_ngeom"] == model.ngeom
    assert manifest["authoritative_geom_ids"] == list(range(model.ngeom))
    assert len(manifest["model_identity_sha256"]) == 64
    collision_robot = [
        row
        for row in manifest["geoms"]
        if row["body_ownership"] != "world" and row["collision_participation"]
    ]
    assert {row["geom_name"] for row in collision_robot} == {
        "base_collision",
        "rearwheel_collision",
        "steer_collision",
        "frontwheel_collision",
        "downarm_collision",
        "knee_motor_collision",
        "uparm_collision",
    }
    assert all(row["supported_jax_geometry_formula"] for row in collision_robot)
    assert not [
        row
        for row in collision_robot
        if row["geom_type"] == "mesh"
    ]


def test_box_cylinder_and_ellipsoid_support_bounds_are_hand_checkable():
    identity = np.eye(3, dtype=np.float32).reshape(1, 9)
    positions = jp.asarray([[1.0, 0.0, 2.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    rotations = jp.asarray(np.repeat(identity, 3, axis=0))
    types = jp.asarray(
        [
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        ],
        jp.int32,
    )
    sizes = jp.asarray([[0.5, 0.25, 1.0], [0.2, 0.5, 0.0], [0.4, 0.3, 0.2]])

    bounds = collision_geom_support_bounds(positions, rotations, types, sizes)

    np.testing.assert_allclose(bounds.min_x, [0.5, -0.2, 1.6], atol=1e-6)
    np.testing.assert_allclose(bounds.max_x, [1.5, 0.2, 2.4], atol=1e-6)
    np.testing.assert_allclose(bounds.min_y, [-0.25, -0.2, -0.3], atol=1e-6)
    np.testing.assert_allclose(bounds.max_y, [0.25, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(bounds.min_z, [1.0, 0.5, 0.8], atol=1e-6)
    np.testing.assert_allclose(bounds.max_z, [3.0, 1.5, 1.2], atol=1e-6)


def test_rotated_cylinder_uses_world_support_not_unrotated_size():
    rotate_y_90 = jp.asarray(
        [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
        jp.float32,
    ).reshape((1, 9))
    bounds = collision_geom_support_bounds(
        jp.asarray([[1.0, 0.0, 2.0]]),
        rotate_y_90,
        jp.asarray([int(mujoco.mjtGeom.mjGEOM_CYLINDER)], jp.int32),
        jp.asarray([[0.2, 0.5, 0.0]]),
    )

    assert float(bounds.min_x[0]) == pytest.approx(0.5)
    assert float(bounds.max_x[0]) == pytest.approx(1.5)
    assert float(bounds.min_z[0]) == pytest.approx(1.8)
    assert float(bounds.max_z[0]) == pytest.approx(2.2)


@pytest.mark.parametrize(
    ("frontmost_x", "expected"),
    [
        (3.4, 0.2),
        (3.6, 0.0),
        (3.8, -0.2),
    ],
)
def test_obstacle_relative_x_has_fixed_front_edge_sign(frontmost_x, expected):
    positions = jp.asarray([[frontmost_x - 0.1, 0.0, 0.5], [frontmost_x - 0.4, 0.0, 0.4]])
    rotations = jp.tile(jp.eye(3).reshape((1, 9)), (2, 1))
    types = jp.asarray([int(mujoco.mjtGeom.mjGEOM_BOX)] * 2, jp.int32)
    sizes = jp.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])

    metrics = full_structure_metrics(
        positions,
        rotations,
        types,
        sizes,
        obstacle_front_x=3.6,
        obstacle_top_z=0.16,
    )

    assert float(metrics.robot_frontmost_x) == pytest.approx(frontmost_x)
    assert float(metrics.obstacle_relative_x) == pytest.approx(expected, abs=1e-6)


def test_geometry_support_and_structure_metrics_are_jittable():
    def calculate(positions):
        metrics = full_structure_metrics(
            positions,
            jp.tile(jp.eye(3).reshape((1, 9)), (2, 1)),
            jp.asarray([int(mujoco.mjtGeom.mjGEOM_BOX)] * 2, jp.int32),
            jp.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.2]]),
            obstacle_front_x=3.6,
            obstacle_top_z=0.16,
        )
        return jp.stack(
            [
                metrics.robot_frontmost_x,
                metrics.obstacle_relative_x,
                metrics.full_structure_clearance,
            ]
        )

    fn = jax.jit(calculate)

    result = fn(jp.asarray([[3.0, 0.0, 0.5], [3.3, 0.0, 0.6]]))

    np.testing.assert_allclose(result, [3.5, 0.1, 0.24], atol=1e-6)


def test_apex_signal_extraction_uses_structure_geometry_and_not_metadata():
    geometry = _synthetic_geometry()
    airborne = _fake_state(
        geom_x=[3.2, 3.3, 3.1],
        geom_z=[0.45, 0.46, 0.50],
        x=3.0,
        vx=2.0,
        vz=0.05,
    )

    signals = extract_apex_band_signals(airborne, geometry)
    changed_metadata = airborne._replace(
        info=airborne.info
        | {
            "phase": jp.asarray(3, jp.int32),
            "reference_index": jp.asarray(-5, jp.int32),
            "tube_distance": jp.asarray(0.0),
        },
        reward=jp.asarray(-999.0),
    )
    changed = extract_apex_band_signals(changed_metadata, geometry)

    assert bool(signals.stable_airborne) is True
    assert float(signals.com_vz) == pytest.approx(0.05)
    assert float(signals.clearance) == pytest.approx(0.19)
    assert float(signals.obstacle_relative_x) == pytest.approx(0.2)
    assert float(signals.angular_speed) == pytest.approx(0.3)
    np.testing.assert_array_equal(jax.tree.leaves(signals), jax.tree.leaves(changed))


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        ({"geom_z": [0.26, 0.40, 0.40]}, "stable_wheel_support"),
        ({"geom_x": [3.7, 4.6, 4.5]}, "landing_region_valid"),
        ({"geom_z": [0.26, 0.26, 0.20]}, "no_body_contact"),
    ],
)
def test_recovery_signal_extraction_rejects_illegal_physical_support(mutation, field):
    geometry = _synthetic_geometry()
    state = _fake_state(**mutation)

    signals = extract_recovery_signals(
        state, geometry, previous_recovery_hold_count=jp.asarray(4, jp.int32)
    )

    assert bool(getattr(signals, field)) is False


def test_recovery_signals_require_current_support_and_carry_only_previous_hold():
    signals = extract_recovery_signals(
        _fake_state(),
        _synthetic_geometry(),
        previous_recovery_hold_count=jp.asarray(7, jp.int32),
    )

    assert bool(signals.stable_wheel_support) is True
    assert bool(signals.landing_region_valid) is True
    assert bool(signals.no_body_contact) is True
    assert int(signals.previous_recovery_hold_count) == 7


@pytest.mark.parametrize(
    ("axis", "wheel_index", "upper_edge"),
    [
        ("x", 0, False),
        ("x", 1, True),
        ("y", 0, True),
        ("y", 1, False),
    ],
)
def test_recovery_landing_region_contains_entire_wheel_support_volume(
    axis, wheel_index, upper_edge
):
    geometry = _synthetic_geometry()
    geom_x = [4.4, 4.6, 4.5]
    geom_y = [0.0, 0.0, 0.0]
    if axis == "x":
        geom_x[wheel_index] = (
            geometry.landing_x_max - 0.05
            if upper_edge
            else geometry.landing_x_min + 0.05
        )
    else:
        geom_y[wheel_index] = (
            geometry.landing_y_limit - 0.05
            if upper_edge
            else -geometry.landing_y_limit + 0.05
        )
    signals = extract_recovery_signals(
        _fake_state(geom_x=geom_x, geom_y=geom_y),
        geometry,
        previous_recovery_hold_count=jp.asarray(0, jp.int32),
    )

    assert bool(signals.landing_region_valid) is False


def test_signal_extractors_support_jit_and_vmap_over_batched_state():
    geometry = _synthetic_geometry()
    first = _fake_state(geom_x=[3.2, 3.3, 3.1], geom_z=[0.45, 0.46, 0.50])
    second = _fake_state(geom_x=[3.7, 3.8, 3.6], geom_z=[0.45, 0.46, 0.50])
    batched = jax.tree.map(lambda a, b: jp.stack([a, b]), first, second)
    apex_fn = jax.jit(jax.vmap(lambda state: extract_apex_band_signals(state, geometry)))
    recovery_fn = jax.jit(
        jax.vmap(
            lambda state, hold: extract_recovery_signals(
                state, geometry, previous_recovery_hold_count=hold
            )
        )
    )

    apex = apex_fn(batched)
    recovery = recovery_fn(batched, jp.asarray([2, 3], jp.int32))

    np.testing.assert_allclose(apex.obstacle_relative_x, [0.2, -0.3], atol=1e-6)
    np.testing.assert_array_equal(recovery.previous_recovery_hold_count, [2, 3])


@pytest.mark.parametrize("end_code", [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15])
def test_physical_failure_is_decoded_from_failure_end_codes(end_code):
    state = _fake_state()
    failed = state._replace(
        info=state.info
        | {
            "terminated": jp.asarray(1, jp.int32),
            "end_code": jp.asarray(end_code, jp.int32),
        }
    )

    apex = extract_apex_band_signals(failed, _synthetic_geometry())
    recovery = extract_recovery_signals(
        failed,
        _synthetic_geometry(),
        previous_recovery_hold_count=jp.asarray(0, jp.int32),
    )

    assert bool(apex.physical_failure) is True
    assert bool(recovery.physical_failure) is True


@pytest.mark.parametrize("end_code", [0, 1, 8, 14, 16])
def test_nonfailure_terminal_outcomes_are_not_physical_failures(end_code):
    state = _fake_state()
    terminal = state._replace(
        info=state.info
        | {
            "terminated": jp.asarray(end_code in (1, 14, 16), jp.int32),
            "truncated": jp.asarray(end_code == 8, jp.int32),
            "end_code": jp.asarray(end_code, jp.int32),
        }
    )

    apex = extract_apex_band_signals(terminal, _synthetic_geometry())
    recovery = extract_recovery_signals(
        terminal,
        _synthetic_geometry(),
        previous_recovery_hold_count=jp.asarray(2, jp.int32),
    )

    assert bool(apex.physical_failure) is False
    assert bool(recovery.physical_failure) is False


def test_successful_recovery_terminal_can_complete_stable_recovery():
    state = _fake_state()._replace(
        info=_fake_state().info
        | {
            "terminated": jp.asarray(1, jp.int32),
            "end_code": jp.asarray(1, jp.int32),
        }
    )
    recovery = extract_recovery_signals(
        state,
        _synthetic_geometry(),
        previous_recovery_hold_count=jp.asarray(2, jp.int32),
    )
    previous = initial_two_phase_event_state()._replace(
        jump_window_entered=jp.asarray(True),
        liftoff_seen=jp.asarray(True),
        stable_airborne=jp.asarray(True),
        ascending=jp.asarray(True),
        apex_band_entered=jp.asarray(True),
        descending=jp.asarray(True),
        pre_landing=jp.asarray(True),
        first_valid_contact=jp.asarray(True),
        impact_absorbing=jp.asarray(True),
        recovery_hold_count=jp.asarray(2, jp.int32),
    )

    result = advance_two_phase_events(
        extract_apex_band_signals(state, _synthetic_geometry()),
        recovery,
        previous,
        _event_thresholds(),
        tick=jp.asarray(9, jp.int32),
        jump_signal=jp.asarray(True),
    )

    assert bool(result.stable_recovery) is True


def test_manifest_validation_rejects_missing_and_duplicate_geom_rows():
    model = mujoco.MjModel.from_xml_path(XML)
    manifest = geometry_manifest(model, build_two_phase_geometry(model, load_config("configs/default.json")))

    missing = manifest | {"geoms": manifest["geoms"][:-1]}
    duplicate = manifest | {"geoms": manifest["geoms"][:-1] + [manifest["geoms"][0]]}

    assert "geom_identity" in validate_geometry_manifest(missing)["failed"]
    assert "geom_identity" in validate_geometry_manifest(duplicate)["failed"]


def _event_thresholds():
    return TwoPhaseThresholds(
        apex=ApexBandThresholds(
            max_abs_com_vz=0.1,
            min_clearance=0.1,
            max_abs_roll=0.2,
            max_abs_pitch=0.2,
            max_angular_speed=0.5,
            min_forward_velocity=0.5,
            relative_x_min=-0.2,
            relative_x_max=0.3,
        ),
        recovery=RecoveryThresholds(
            max_abs_roll=0.2,
            max_abs_pitch=0.2,
            max_angular_speed=0.5,
            min_forward_velocity=0.5,
            required_hold_ticks=3,
        ),
    )


def _event_apex(*, stable=False, vz=0.5, member=False):
    return ApexBandSignals(
        stable_airborne=stable,
        com_vz=vz,
        clearance=0.2 if member else 0.0,
        roll=0.0,
        pitch=0.0,
        angular_speed=0.1,
        forward_velocity=1.0,
        obstacle_relative_x=0.0,
        illegal_contact=False,
        physical_failure=False,
    )


def _event_recovery(*, support=False, region=False, previous_hold=0):
    return RecoverySignals(
        stable_wheel_support=support,
        landing_region_valid=region,
        no_body_contact=True,
        roll=0.0,
        pitch=0.0,
        angular_speed=0.1,
        forward_velocity=1.0,
        previous_recovery_hold_count=previous_hold,
        physical_failure=False,
    )


def test_event_state_enforces_declared_first_occurrence_order():
    thresholds = _event_thresholds()
    state = initial_two_phase_event_state()
    sequence = [
        (_event_apex(), _event_recovery(), True),
        (_event_apex(), _event_recovery(), True),
        (_event_apex(stable=True), _event_recovery(), True),
        (_event_apex(stable=True, vz=0.5), _event_recovery(), True),
        (_event_apex(stable=True, vz=0.0, member=True), _event_recovery(), True),
        (_event_apex(stable=True, vz=-0.2), _event_recovery(), True),
        (_event_apex(stable=True, vz=-0.2), _event_recovery(region=True), True),
        (
            _event_apex(stable=True, vz=-0.2),
            _event_recovery(support=True, region=True),
            True,
        ),
        (
            _event_apex(stable=True, vz=-0.2),
            _event_recovery(support=True, region=True),
            True,
        ),
        (
            _event_apex(stable=True, vz=-0.2),
            _event_recovery(support=True, region=True),
            True,
        ),
    ]
    for tick, (apex, recovery, jump_signal) in enumerate(sequence):
        recovery = replace(
            recovery,
            previous_recovery_hold_count=state.recovery_hold_count
        )
        state = advance_two_phase_events(
            apex,
            recovery,
            state,
            thresholds,
            tick=jp.asarray(tick, jp.int32),
            jump_signal=jp.asarray(jump_signal),
        )

    assert tuple(np.asarray(state.first_event_ticks).tolist()) == tuple(range(10))
    assert EVENT_NAMES == (
        "jump_window_entered",
        "liftoff_seen",
        "stable_airborne",
        "ascending",
        "apex_band_entered",
        "descending",
        "pre_landing",
        "first_valid_contact",
        "impact_absorbing",
        "stable_recovery",
    )
    assert int(state.recovery_hold_count) == 3


def test_event_recovery_hold_resets_when_current_support_is_lost():
    state = initial_two_phase_event_state()._replace(recovery_hold_count=jp.asarray(2))
    recovery = _event_recovery(support=False, region=True, previous_hold=2)

    next_state = advance_two_phase_events(
        _event_apex(stable=True, vz=-0.2),
        recovery,
        state,
        _event_thresholds(),
        tick=jp.asarray(9, jp.int32),
        jump_signal=jp.asarray(True),
    )

    assert int(next_state.recovery_hold_count) == 0
    assert bool(next_state.stable_recovery) is False


def test_event_transition_is_jittable_and_does_not_mutate_input():
    original = initial_two_phase_event_state()
    fn = jax.jit(
        lambda previous, tick: advance_two_phase_events(
            _event_apex(),
            _event_recovery(previous_hold=previous.recovery_hold_count),
            previous,
            _event_thresholds(),
            tick=tick,
            jump_signal=jp.asarray(True),
        )
    )

    result = fn(original, jp.asarray(0, jp.int32))

    assert bool(result.jump_window_entered) is True
    assert bool(original.jump_window_entered) is False
    assert tuple(np.asarray(original.first_event_ticks).tolist()) == (-1,) * 10
