from __future__ import annotations

from dataclasses import fields, replace

import jax
from jax import numpy as jp
import pytest

from dvgc import two_phase_semantics as semantics


def apex_thresholds():
    return semantics.ApexBandThresholds(
        max_abs_com_vz=0.20,
        min_clearance=0.20,
        max_abs_roll=0.30,
        max_abs_pitch=0.30,
        max_angular_speed=0.50,
        min_forward_velocity=0.50,
        relative_x_min=-0.40,
        relative_x_max=0.40,
    )


def valid_apex_signals():
    return semantics.ApexBandSignals(
        stable_airborne=True,
        com_vz=0.05,
        clearance=0.30,
        roll=0.10,
        pitch=-0.08,
        angular_speed=0.20,
        forward_velocity=1.20,
        obstacle_relative_x=0.10,
        illegal_contact=False,
        physical_failure=False,
    )


def test_apex_membership_accepts_a_valid_obstacle_relative_state():
    assert bool(semantics.apex_band_membership(valid_apex_signals(), apex_thresholds())) is True


def test_phase_vocabulary_has_two_experts_and_no_apex_phase():
    assert semantics.PHASES == ("propulsion_ascent", "descent_recovery")
    assert "apex" not in semantics.PHASES
    assert set(semantics.INTERNAL_EVENTS) == {
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
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stable_airborne", False),
        ("com_vz", 0.21),
        ("clearance", 0.19),
        ("roll", 0.31),
        ("pitch", -0.31),
        ("angular_speed", 0.51),
        ("forward_velocity", 0.49),
        ("obstacle_relative_x", -0.41),
        ("obstacle_relative_x", 0.41),
        ("illegal_contact", True),
        ("physical_failure", True),
    ],
)
def test_each_apex_gate_can_reject_membership(field, value):
    signals = replace(valid_apex_signals(), **{field: value})
    assert bool(semantics.apex_band_membership(signals, apex_thresholds())) is False


def test_apex_horizontal_window_is_inclusive_at_both_edges():
    for relative_x in (-0.40, 0.40):
        signals = replace(valid_apex_signals(), obstacle_relative_x=relative_x)
        assert bool(semantics.apex_band_membership(signals, apex_thresholds())) is True


def test_apex_thresholds_reject_inverted_or_nonpositive_bounds():
    with pytest.raises(ValueError, match="relative_x_min"):
        replace(apex_thresholds(), relative_x_min=0.5, relative_x_max=-0.5)
    with pytest.raises(ValueError, match="max_abs_com_vz"):
        replace(apex_thresholds(), max_abs_com_vz=0.0)
    for value in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_abs_roll"):
            replace(apex_thresholds(), max_abs_roll=value)


def recovery_thresholds():
    return semantics.RecoveryThresholds(
        max_abs_roll=0.25,
        max_abs_pitch=0.30,
        max_angular_speed=0.60,
        min_forward_velocity=0.40,
        required_hold_ticks=5,
    )


def valid_recovery_signals():
    return semantics.RecoverySignals(
        stable_wheel_support=True,
        landing_region_valid=True,
        no_body_contact=True,
        roll=0.10,
        pitch=-0.10,
        angular_speed=0.20,
        forward_velocity=0.80,
        previous_recovery_hold_count=4,
        physical_failure=False,
    )


def test_recovery_thresholds_reject_nonpositive_limits_and_hold_duration():
    with pytest.raises(ValueError, match="max_abs_roll"):
        replace(recovery_thresholds(), max_abs_roll=0.0)
    with pytest.raises(ValueError, match="required_hold_ticks"):
        replace(recovery_thresholds(), required_hold_ticks=0)
    with pytest.raises(ValueError, match="required_hold_ticks"):
        replace(recovery_thresholds(), required_hold_ticks=5.0)


def test_recovery_contract_has_no_ambiguous_valid_contact_latch():
    assert "valid_contact" not in {field.name for field in fields(semantics.RecoverySignals)}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stable_wheel_support", False),
        ("landing_region_valid", False),
        ("no_body_contact", False),
        ("roll", 0.26),
        ("pitch", -0.31),
        ("angular_speed", 0.61),
        ("forward_velocity", 0.39),
        ("previous_recovery_hold_count", 3),
        ("physical_failure", True),
    ],
)
def test_each_recovery_gate_can_reject_stable_recovery(field, value):
    signals = replace(valid_recovery_signals(), **{field: value})
    assert bool(semantics.descent_recovery_success(signals, recovery_thresholds())) is False


def test_one_instantaneous_contact_cannot_satisfy_recovery_hold():
    signals = replace(valid_recovery_signals(), previous_recovery_hold_count=0)
    assert bool(semantics.descent_recovery_success(signals, recovery_thresholds())) is False


def test_recovery_hold_transition_resets_every_gate_and_rebuilds_consecutively():
    thresholds = recovery_thresholds()
    almost_complete = valid_recovery_signals()
    assert int(semantics.advance_recovery_hold_count(almost_complete, thresholds)) == 5

    lost_support = replace(almost_complete, stable_wheel_support=False)
    assert int(semantics.advance_recovery_hold_count(lost_support, thresholds)) == 0

    first_valid_tick_after_reset = replace(
        almost_complete, previous_recovery_hold_count=0
    )
    assert int(
        semantics.advance_recovery_hold_count(first_valid_tick_after_reset, thresholds)
    ) == 1
    assert bool(
        semantics.descent_recovery_success(first_valid_tick_after_reset, thresholds)
    ) is False


def test_phase_success_dispatches_only_the_two_formal_phases():
    assert bool(
        semantics.phase_success(semantics.PHASE_UP, valid_apex_signals(), apex_thresholds())
    ) is True
    assert bool(
        semantics.phase_success(
            semantics.PHASE_DOWN, valid_recovery_signals(), recovery_thresholds()
        )
    ) is True
    with pytest.raises(ValueError, match="Unknown two-phase phase"):
        semantics.phase_success("apex", valid_apex_signals(), apex_thresholds())


def test_membership_masks_are_jittable_without_host_boolean_conversion():
    apex_fn = jax.jit(
        lambda vz, relative_x: semantics.apex_band_membership(
            replace(valid_apex_signals(), com_vz=vz, obstacle_relative_x=relative_x),
            apex_thresholds(),
        )
    )
    recovery_fn = jax.jit(
        lambda hold: semantics.descent_recovery_success(
            replace(valid_recovery_signals(), previous_recovery_hold_count=hold),
            recovery_thresholds(),
        )
    )

    assert bool(apex_fn(jp.asarray(0.0), jp.asarray(0.0))) is True
    assert bool(apex_fn(jp.asarray(0.0), jp.asarray(0.8))) is False
    assert bool(recovery_fn(jp.asarray(5))) is True


def test_negative_angular_speed_norm_is_rejected_in_both_phases():
    apex = replace(valid_apex_signals(), angular_speed=-0.1)
    recovery = replace(valid_recovery_signals(), angular_speed=-0.1)
    assert bool(semantics.apex_band_membership(apex, apex_thresholds())) is False
    assert bool(semantics.descent_recovery_success(recovery, recovery_thresholds())) is False
