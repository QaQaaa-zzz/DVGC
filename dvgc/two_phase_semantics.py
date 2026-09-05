"""Pure two-phase event and success semantics for the approved DVGC method."""
from __future__ import annotations

import math
from numbers import Integral, Real
from dataclasses import dataclass
from typing import Any, Mapping

from jax import numpy as jp


PHASE_UP = "propulsion_ascent"
PHASE_DOWN = "descent_recovery"
PHASES = (PHASE_UP, PHASE_DOWN)

INTERNAL_EVENTS = (
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


def _require_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class ApexBandSignals:
    stable_airborne: Any
    com_vz: Any
    clearance: Any
    minimum_wheel_terrain_clearance: Any
    roll: Any
    pitch: Any
    angular_speed: Any
    forward_velocity: Any
    obstacle_relative_x: Any
    illegal_contact: Any
    physical_failure: Any


@dataclass(frozen=True)
class ApexBandThresholds:
    max_abs_com_vz: float
    min_clearance: float
    max_abs_roll: float
    max_abs_pitch: float
    max_angular_speed: float
    min_forward_velocity: float
    relative_x_min: float
    relative_x_max: float

    def __post_init__(self) -> None:
        for name in (
            "max_abs_com_vz",
            "min_clearance",
            "max_abs_roll",
            "max_abs_pitch",
            "max_angular_speed",
            "min_forward_velocity",
        ):
            _require_positive(name, getattr(self, name))
        _require_finite("relative_x_min", self.relative_x_min)
        _require_finite("relative_x_max", self.relative_x_max)
        if self.relative_x_min >= self.relative_x_max:
            raise ValueError("relative_x_min must be less than relative_x_max")


@dataclass(frozen=True)
class RecoverySignals:
    stable_wheel_support: Any
    landing_region_valid: Any
    no_body_contact: Any
    roll: Any
    pitch: Any
    angular_speed: Any
    forward_velocity: Any
    previous_recovery_hold_count: Any
    physical_failure: Any


@dataclass(frozen=True)
class RecoveryThresholds:
    max_abs_roll: float
    max_abs_pitch: float
    max_angular_speed: float
    min_forward_velocity: float
    required_hold_ticks: int

    def __post_init__(self) -> None:
        for name in (
            "max_abs_roll",
            "max_abs_pitch",
            "max_angular_speed",
            "min_forward_velocity",
        ):
            _require_positive(name, getattr(self, name))
        if not isinstance(self.required_hold_ticks, Integral) or isinstance(
            self.required_hold_ticks, bool
        ) or self.required_hold_ticks <= 0:
            raise ValueError("required_hold_ticks must be a positive integer")


def apex_band_components(
    signals: ApexBandSignals, thresholds: ApexBandThresholds
) -> dict[str, Any]:
    """Return each observable Apex transition-band membership gate."""
    return {
        "stable_airborne": jp.asarray(signals.stable_airborne, dtype=bool),
        "vertical_speed": jp.abs(signals.com_vz) <= thresholds.max_abs_com_vz,
        "clearance": signals.clearance >= thresholds.min_clearance,
        "roll": jp.abs(signals.roll) <= thresholds.max_abs_roll,
        "pitch": jp.abs(signals.pitch) <= thresholds.max_abs_pitch,
        "angular_speed": jp.logical_and(
            signals.angular_speed >= 0.0,
            signals.angular_speed <= thresholds.max_angular_speed,
        ),
        "forward_velocity": signals.forward_velocity >= thresholds.min_forward_velocity,
        "relative_x_min": signals.obstacle_relative_x >= thresholds.relative_x_min,
        "relative_x_max": signals.obstacle_relative_x <= thresholds.relative_x_max,
        "no_illegal_contact": ~jp.asarray(signals.illegal_contact, dtype=bool),
        "no_physical_failure": ~jp.asarray(signals.physical_failure, dtype=bool),
    }


def _all_components(components: Mapping[str, Any]) -> Any:
    result = jp.asarray(True)
    for value in components.values():
        result = jp.logical_and(result, value)
    return result


def apex_band_membership(signals: ApexBandSignals, thresholds: ApexBandThresholds) -> Any:
    """Return whether every observable Apex transition-band gate holds."""
    return _all_components(apex_band_components(signals, thresholds))


def propulsion_ascent_success(
    signals: ApexBandSignals, thresholds: ApexBandThresholds
) -> Any:
    """Phase U succeeds only by entering the full Apex transition band."""
    return apex_band_membership(signals, thresholds)


def _descent_recovery_current_components(
    signals: RecoverySignals, thresholds: RecoveryThresholds
) -> dict[str, Any]:
    """Return gates that must hold on the current recovery tick."""
    return {
        "stable_wheel_support": jp.asarray(signals.stable_wheel_support, dtype=bool),
        "landing_region_valid": jp.asarray(signals.landing_region_valid, dtype=bool),
        "no_body_contact": jp.asarray(signals.no_body_contact, dtype=bool),
        "roll": jp.abs(signals.roll) <= thresholds.max_abs_roll,
        "pitch": jp.abs(signals.pitch) <= thresholds.max_abs_pitch,
        "angular_speed": jp.logical_and(
            signals.angular_speed >= 0.0,
            signals.angular_speed <= thresholds.max_angular_speed,
        ),
        "forward_velocity": signals.forward_velocity >= thresholds.min_forward_velocity,
        "no_physical_failure": ~jp.asarray(signals.physical_failure, dtype=bool),
    }


def advance_recovery_hold_count(
    signals: RecoverySignals, thresholds: RecoveryThresholds
) -> Any:
    """Advance a consecutive legal-support count, resetting on any failed gate."""
    current_tick_valid = _all_components(
        _descent_recovery_current_components(signals, thresholds)
    )
    previous_count = jp.asarray(signals.previous_recovery_hold_count)
    return jp.where(current_tick_valid, previous_count + 1, jp.zeros_like(previous_count))


def descent_recovery_components(
    signals: RecoverySignals, thresholds: RecoveryThresholds
) -> dict[str, Any]:
    """Return every current and sustained stable-recovery gate."""
    components = _descent_recovery_current_components(signals, thresholds)
    components["recovery_hold"] = (
        advance_recovery_hold_count(signals, thresholds)
        >= thresholds.required_hold_ticks
    )
    return components


def descent_recovery_success(
    signals: RecoverySignals, thresholds: RecoveryThresholds
) -> Any:
    """Phase D succeeds only after sustained legal recovery support."""
    return _all_components(descent_recovery_components(signals, thresholds))


def phase_success(phase: str, signals: Any, thresholds: Any) -> Any:
    """Dispatch a success predicate for one of the two formal phases."""
    if phase == PHASE_UP:
        return propulsion_ascent_success(signals, thresholds)
    if phase == PHASE_DOWN:
        return descent_recovery_success(signals, thresholds)
    raise ValueError(f"Unknown two-phase phase: {phase}")
