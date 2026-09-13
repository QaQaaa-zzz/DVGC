"""Authoritative timing-explicit two-phase snapshot round-trip validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jp
import numpy as np

from .bank import SnapshotBank
from .rollout import restore_snapshot_mode
from .two_phase_runtime import (
    TwoPhaseEventState,
    TwoPhaseGeometry,
    TwoPhaseThresholds,
    advance_two_phase_events,
    extract_apex_band_signals,
    extract_recovery_signals,
    initial_two_phase_event_state,
)


RESTORE_MODE = "timing_explicit_independent_reconstruction"


@dataclass(frozen=True)
class RoundtripTolerances:
    qpos: float = 5e-5
    qvel: float = 2e-3
    ctrl: float = 1e-5
    last_action: float = 1e-6
    actor_observation: float = 2e-2
    privileged_observation: float = 1e-3
    history: float = 2e-2
    two_phase_signals: float = 2e-3
    estimator_continuous: float = 2e-3

    def for_field(self, field: str) -> float:
        if field in {
            "obs_history",
            "actor_obs_history_pre",
            "actor_current_frame",
            "actor_obs_history_post",
            "actor_packet_fifo",
        }:
            return self.history
        if field == "two_phase_event_continuous":
            return self.two_phase_signals
        return float(getattr(self, field))


def compare_two_phase_payloads(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    tolerances: RoundtripTolerances,
) -> dict[str, Any]:
    """Compare named continuous fields and exact discrete fields."""
    continuous_names = sorted(set(left["continuous"]) | set(right["continuous"]))
    discrete_names = sorted(set(left["discrete"]) | set(right["discrete"]))
    max_abs_difference: dict[str, float] = {}
    exact_discrete: dict[str, bool] = {}
    failed: list[str] = []
    for name in continuous_names:
        if name not in left["continuous"] or name not in right["continuous"]:
            max_abs_difference[name] = float("inf")
            failed.append(name)
            continue
        lhs = np.asarray(left["continuous"][name], dtype=np.float64)
        rhs = np.asarray(right["continuous"][name], dtype=np.float64)
        if lhs.shape != rhs.shape:
            difference = float("inf")
        else:
            difference = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0
        max_abs_difference[name] = difference
        if not np.isfinite(difference) or difference > tolerances.for_field(name):
            failed.append(name)
    for name in discrete_names:
        equal = (
            name in left["discrete"]
            and name in right["discrete"]
            and np.array_equal(
                np.asarray(left["discrete"][name]),
                np.asarray(right["discrete"][name]),
            )
        )
        exact_discrete[name] = bool(equal)
        if not equal:
            failed.append(name)
    failed = sorted(set(failed))
    return {
        "valid": not failed,
        "failed_fields": failed,
        "max_abs_difference": max_abs_difference,
        "exact_discrete_state": exact_discrete,
    }


def _host(value: Any) -> np.ndarray:
    return np.array(jax.device_get(value), copy=True)


def _signal_vector(apex: Any, recovery: Any) -> np.ndarray:
    return np.asarray(
        [
            apex.com_vz,
            apex.clearance,
            apex.roll,
            apex.pitch,
            apex.angular_speed,
            apex.forward_velocity,
            apex.obstacle_relative_x,
            recovery.roll,
            recovery.pitch,
            recovery.angular_speed,
            recovery.forward_velocity,
        ],
        dtype=np.float64,
    )


def _event_latches(event: TwoPhaseEventState) -> np.ndarray:
    return np.asarray(
        [
            event.jump_window_entered,
            event.liftoff_seen,
            event.stable_airborne,
            event.ascending,
            event.apex_band_entered,
            event.descending,
            event.pre_landing,
            event.first_valid_contact,
            event.impact_absorbing,
            event.stable_recovery,
        ],
        dtype=bool,
    )


def _capture_payload(
    state: Any,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    previous_event: TwoPhaseEventState,
    *,
    tick: int,
) -> tuple[dict[str, dict[str, Any]], TwoPhaseEventState]:
    apex = extract_apex_band_signals(state, geometry)
    recovery = extract_recovery_signals(
        state,
        geometry,
        previous_recovery_hold_count=previous_event.recovery_hold_count,
    )
    event = advance_two_phase_events(
        apex,
        recovery,
        previous_event,
        thresholds,
        tick=jp.asarray(tick, jp.int32),
        jump_signal=jp.asarray(state.info["jump_signal_latched"]),
    )
    continuous = {
        "qpos": _host(state.data.qpos),
        "qvel": _host(state.data.qvel),
        "ctrl": _host(state.data.ctrl),
        "last_action": _host(state.info["last_action"]),
        "actor_observation": _host(state.obs["state"]),
        "privileged_observation": _host(state.obs["privileged_state"]),
        "obs_history": _host(state.info["obs_history"]),
        "actor_obs_history_pre": _host(state.info["actor_obs_history_pre"]),
        "actor_current_frame": _host(state.info["actor_current_frame"]),
        "actor_obs_history_post": _host(state.info["actor_obs_history_post"]),
        "actor_packet_fifo": _host(state.info["actor_packet_fifo"]),
        "two_phase_signals": _signal_vector(apex, recovery),
        "two_phase_event_continuous": np.asarray(
            [event.previous_com_vz], dtype=np.float64
        ),
        "estimator_continuous": np.concatenate(
            [
                _host(state.info["phase_probs"]).reshape(-1),
                np.asarray(
                    [
                        _host(state.info[name]).reshape(())
                        for name in (
                            "prev_acc_z",
                            "prev_vz",
                            "prev_front_tire_bottom_z",
                            "prev_rear_tire_bottom_z",
                            "jump_window_start_x",
                            "jump_window_end_x",
                        )
                    ],
                    dtype=np.float64,
                ),
            ]
        ),
    }
    info_names = (
        "phase",
        "estimated_phase",
        "end_code",
        "terminated",
        "truncated",
        "had_airborne",
        "had_valid_landing",
        "contact_age",
        "landing_entry_age",
        "landing_phase_step",
        "airborne_count",
        "prelaunch_airborne_count",
        "landing_bounce_count",
        "invalid_wheel_count",
        "recovery_count",
        "positive_pitch_count",
        "wheelie_count",
        "dual_wheel_liftoff_seen",
        "stage_entry_ever",
        "apex_seen",
        "apex_descent_stable_count",
        "jump_signal_latched",
        "chain_ever",
        "recovery_success",
        "episode_step",
        "actor_packet_fifo_valid",
    )
    discrete = {name: _host(state.info[name]) for name in info_names}
    discrete.update(
        {
            "done": _host(state.done),
            "rng_state": _host(state.info["rng"]),
            "observation_rng": _host(state.info["actor_observation_rng"]),
            "apex_signal_flags": np.asarray(
                [
                    apex.stable_airborne,
                    apex.illegal_contact,
                    apex.physical_failure,
                ],
                dtype=bool,
            ),
            "recovery_signal_flags": np.asarray(
                [
                    recovery.stable_wheel_support,
                    recovery.landing_region_valid,
                    recovery.no_body_contact,
                    recovery.physical_failure,
                ],
                dtype=bool,
            ),
            "two_phase_event_latches": _event_latches(event),
            "two_phase_first_event_ticks": _host(event.first_event_ticks),
            "two_phase_event_counters": np.asarray(
                [
                    event.recovery_hold_count,
                    event.apex_band_width,
                    event.max_apex_band_width,
                ],
                dtype=np.int32,
            ),
        }
    )
    return {"continuous": continuous, "discrete": discrete}, event


def compare_two_phase_roundtrip(
    env: Any,
    record: Mapping[str, Any],
    original_state: Any,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    *,
    seed: int,
    actions: Sequence[Any],
    tolerances: RoundtripTolerances = RoundtripTolerances(),
) -> dict[str, Any]:
    """Replay two explicit branches sequentially and compare complete state."""
    if not 1 <= len(actions) <= 3:
        raise ValueError("Round-trip requires one to three control ticks")
    restore = jax.jit(
        lambda key: restore_snapshot_mode(
            env, record, key, observation_mode=RESTORE_MODE
        )
    )
    key = jax.random.PRNGKey(int(seed))
    left_state = original_state
    jax.block_until_ready(left_state)
    right_state = restore(key)
    jax.block_until_ready(right_state)
    step = jax.jit(env.step)
    left_event = initial_two_phase_event_state()
    right_event = initial_two_phase_event_state()
    rows = []
    failed_fields: set[str] = set()

    def compare_tick(
        tick: int,
        left_payload: Mapping[str, Mapping[str, Any]],
        right_payload: Mapping[str, Mapping[str, Any]],
    ) -> None:
        comparison = compare_two_phase_payloads(
            left_payload, right_payload, tolerances
        )
        comparison["tick"] = tick
        rows.append(comparison)
        failed_fields.update(comparison["failed_fields"])

    left_payload, left_event = _capture_payload(
        left_state, geometry, thresholds, left_event, tick=0
    )
    right_payload, right_event = _capture_payload(
        right_state, geometry, thresholds, right_event, tick=0
    )
    compare_tick(0, left_payload, right_payload)
    for tick, action_value in enumerate(actions, start=1):
        action = jp.asarray(action_value, jp.float32)
        left_state = step(left_state, action)
        jax.block_until_ready(left_state)
        left_payload, left_event = _capture_payload(
            left_state, geometry, thresholds, left_event, tick=tick
        )
        # Copy every left-branch field to host before dispatching the second
        # Warp contact step; this is the same sequential comparison discipline
        # as the runtime gate and avoids shared scratch-storage races.
        right_state = step(right_state, action)
        jax.block_until_ready(right_state)
        right_payload, right_event = _capture_payload(
            right_state, geometry, thresholds, right_event, tick=tick
        )
        compare_tick(tick, left_payload, right_payload)
    failed = sorted(failed_fields)
    return {
        "status": "pass" if not failed else "gate_pause",
        "restore_mode": RESTORE_MODE,
        "comparison": "capture_time_original_vs_independent_restore",
        "snapshot_id": record.get("id"),
        "parent_trajectory_id": record.get("two_phase_context", {}).get(
            "parent_trajectory_id"
        ),
        "seed": int(seed),
        "control_ticks": len(actions),
        "failed_fields": failed,
        "ticks": rows,
    }


def select_roundtrip_representatives(
    up: SnapshotBank, down: SnapshotBank
) -> dict[str, Mapping[str, Any]]:
    """Select explicit applicable U boundaries and D transition thickness."""
    up_by_name = {row.get("guideline_selection"): row for row in up.records}
    down_by_name = {row.get("guideline_selection"): row for row in down.records}
    required_up = {"front", "back"}
    required_down = {"pre", "nearest", "post", "early_descent"}
    if not required_up.issubset(up_by_name) or not required_down.issubset(
        down_by_name
    ):
        raise ValueError("Banks lack required round-trip representatives")
    return {
        "up_boundary_front": up_by_name["front"],
        "up_boundary_back": up_by_name["back"],
        "down_pre": down_by_name["pre"],
        "down_nearest": down_by_name["nearest"],
        "down_post": down_by_name["post"],
        "down_boundary": down_by_name["early_descent"],
    }
