"""Exact capture and host-only rendering inputs for dynamic failure audits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jp
import numpy as np

from .config import STAGE_ID
from .env import END_REASON
from .reference import ReferenceTrajectory
from .reset_geometry import GroundSupportSolver
from .two_phase_guideline import _reference_action, reconstruct_guideline_state
from .two_phase_runtime import (
    EVENT_NAMES,
    TwoPhaseGeometry,
    TwoPhaseThresholds,
    extract_recovery_signals,
    extract_two_phase_events,
    initial_two_phase_event_state,
)


@dataclass(frozen=True)
class FailureScenario:
    name: str
    start_reference_index: int
    maximum_control_ticks: int


@dataclass(frozen=True)
class FailureTrace:
    scenario: FailureScenario
    frames: tuple[dict[str, np.ndarray], ...]
    telemetry: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


FAILURE_SCENARIOS = {
    "full_guideline_prelaunch_airborne": FailureScenario(
        name="full_guideline_prelaunch_airborne",
        start_reference_index=0,
        maximum_control_ticks=100,
    ),
    "launch_history_airborne_before_window": FailureScenario(
        name="launch_history_airborne_before_window",
        start_reference_index=83,
        maximum_control_ticks=8,
    ),
}


def _host_scalar(value: Any, cast):
    return cast(np.asarray(jax.device_get(value)))


def _frame(state: Any) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(jax.device_get(value)).copy()
        for name, value in (
            ("qpos", state.data.qpos),
            ("qvel", state.data.qvel),
            ("ctrl", state.data.ctrl),
        )
    }


def capture_failure_scenario(
    env: Any,
    reference: ReferenceTrajectory,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    scenario: str,
    seed: int,
) -> FailureTrace:
    """Capture one named Gate B failure without changing environment semantics."""
    if scenario not in FAILURE_SCENARIOS:
        raise ValueError(f"Unknown failure scenario: {scenario}")
    contract = FAILURE_SCENARIOS[scenario]
    stride = int(round(float(env.dt) / float(reference.dt_median)))
    if stride <= 0 or not np.isclose(
        stride * reference.dt_median, float(env.dt), atol=1e-9, rtol=0.0
    ):
        raise ValueError("Reference timing does not divide the environment control tick")
    start = contract.start_reference_index
    proposal = reconstruct_guideline_state(
        env.mj_model,
        reference,
        start,
        wheel_roll_radius=float(env._config.wheel_roll_radius),
        nominal_base_z_ground=float(env._config.nominal_base_z_ground),
    )
    initial_action = jp.asarray(_reference_action(reference, start))
    ctrl = env._action_to_ctrl(
        initial_action,
        jp.asarray(proposal.qpos)[env._joint_qpos["knee_joint"]],
    )
    support_solver = GroundSupportSolver(env._config.xml_path)
    placement = support_solver.solve(
        proposal.qpos,
        proposal.qvel,
        np.asarray(jax.device_get(ctrl)),
    )
    if not placement.accepted:
        raise ValueError(f"Failure-video ground placement rejected: {placement.reason}")
    ctrl = env._action_to_ctrl(
        initial_action,
        jp.asarray(placement.qpos)[env._joint_qpos["knee_joint"]],
    )
    state = env.reset_from_snapshot(
        jp.asarray(placement.qpos, jp.float32),
        jp.asarray(proposal.qvel, jp.float32),
        ctrl,
        jax.random.PRNGKey(int(seed)),
        jp.asarray(STAGE_ID["approach"], jp.int32),
        jp.asarray(0, jp.int32),
        jp.asarray(0, jp.int32),
        jp.asarray(0, jp.int32),
        last_action=initial_action,
        estimated_phase=jp.asarray(STAGE_ID["approach"], jp.int32),
        jump_signal_latched=jp.asarray(False),
    )
    step = jax.jit(env.step)
    event = initial_two_phase_event_state()
    frames: list[dict[str, np.ndarray]] = []
    telemetry: list[dict[str, Any]] = []
    front = float(env._config.step_front_x)
    window_min = front - float(env._config.takeoff_window_far)
    window_max = front - float(env._config.takeoff_window_near)

    def append(tick: int, action_index: int) -> None:
        frame = _frame(state)
        qpos, qvel, ctrl_values = frame["qpos"], frame["qvel"], frame["ctrl"]
        support = support_solver.measure(qpos, qvel, ctrl_values)
        recovery = extract_recovery_signals(
            state,
            geometry,
            previous_recovery_hold_count=0,
        )
        event_values = {
            name: _host_scalar(getattr(event, name), bool) for name in EVENT_NAMES
        }
        x = float(qpos[geometry.root_qpos_adr])
        end_code = _host_scalar(state.info["end_code"], int)
        telemetry.append(
            {
                "tick": int(tick),
                "action_reference_index": int(action_index),
                "x": x,
                "z": float(qpos[geometry.root_qpos_adr + 2]),
                "vx": float(qvel[geometry.root_dof_adr]),
                "vz": float(qvel[geometry.root_dof_adr + 2]),
                "jump_window_min_x": window_min,
                "jump_window_max_x": window_max,
                "inside_jump_window": bool(window_min <= x <= window_max),
                "host_wheel_contacts": int(support["wheel_contacts"]),
                "host_body_contacts": int(support["body_contacts"]),
                "host_wheel_clearance_min_m": float(support["wheel_min"]),
                "deployable_wheel_support": _host_scalar(
                    recovery.stable_wheel_support, bool
                ),
                "jump_signal_latched": _host_scalar(
                    state.info["jump_signal_latched"], bool
                ),
                "phase": _host_scalar(state.info["phase"], int),
                "terminated": _host_scalar(state.info["terminated"], bool),
                "truncated": _host_scalar(state.info["truncated"], bool),
                "end_code": end_code,
                "termination_reason": END_REASON.get(end_code, f"unknown_{end_code}"),
                "events": event_values,
            }
        )
        frames.append(frame)

    append(0, start)
    executed = 0
    for tick in range(1, contract.maximum_control_ticks + 1):
        action_index = min(start + tick * stride, len(reference.df) - 1)
        state = step(state, jp.asarray(_reference_action(reference, action_index)))
        jax.block_until_ready(state)
        event = extract_two_phase_events(
            state,
            geometry,
            event,
            thresholds,
            tick=jp.asarray(tick, jp.int32),
        )
        executed = tick
        append(tick, action_index)
        if _host_scalar(state.done, bool):
            break

    terminal = bool(telemetry[-1]["terminated"] or telemetry[-1]["truncated"])
    if scenario == "full_guideline_prelaunch_airborne":
        failure_reason = telemetry[-1]["termination_reason"]
    else:
        invalid_entry = any(
            row["inside_jump_window"]
            and not row["deployable_wheel_support"]
            and not row["jump_signal_latched"]
            for row in telemetry
        )
        if not invalid_entry:
            raise ValueError("Launch-history scenario did not reproduce lost support")
        failure_reason = "airborne_before_jump_window_latch"
    summary = {
        "scenario": scenario,
        "seed": int(seed),
        "start_reference_index": start,
        "reference_rows_per_control_tick": stride,
        "environment_transitions": executed,
        "formal_training_transitions": 0,
        "terminal": terminal,
        "terminated": bool(telemetry[-1]["terminated"]),
        "truncated": bool(telemetry[-1]["truncated"]),
        "end_code": int(telemetry[-1]["end_code"]),
        "failure_reason": failure_reason,
        "initial_ground_support": placement.summary(),
    }
    return FailureTrace(
        scenario=contract,
        frames=tuple(frames),
        telemetry=tuple(telemetry),
        summary=summary,
    )
