"""Exact capture and host-only rendering inputs for dynamic failure audits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import STAGE_ID
from .env import END_PRETAKEOFF_AIRBORNE, END_REASON
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
    initial_action_offset_ticks: int
    first_step_action_offset_ticks: int


@dataclass(frozen=True)
class FailureTrace:
    scenario: FailureScenario
    frames: tuple[dict[str, np.ndarray], ...]
    telemetry: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


FAILURE_SCENARIOS = {
    "full_guideline_continuation": FailureScenario(
        name="full_guideline_continuation",
        start_reference_index=0,
        maximum_control_ticks=100,
        initial_action_offset_ticks=0,
        first_step_action_offset_ticks=1,
    ),
    "launch_history_window_latch": FailureScenario(
        name="launch_history_window_latch",
        start_reference_index=83,
        maximum_control_ticks=8,
        initial_action_offset_ticks=-1,
        first_step_action_offset_ticks=0,
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


def _state_trace_sha256(frames: tuple[dict[str, np.ndarray], ...]) -> str:
    digest = hashlib.sha256()
    for tick, frame in enumerate(frames):
        digest.update(int(tick).to_bytes(8, "little", signed=False))
        for name in ("qpos", "qvel", "ctrl"):
            values = np.ascontiguousarray(frame[name])
            digest.update(name.encode("ascii") + b"\0")
            digest.update(values.dtype.str.encode("ascii") + b"\0")
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def _load_state_trace(path: Path) -> tuple[dict[str, np.ndarray], ...]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"qpos", "qvel", "ctrl"}:
            raise ValueError("Failure-video state trace fields are incomplete")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    counts = {values.shape[0] for values in arrays.values() if values.ndim >= 1}
    if len(counts) != 1 or any(values.ndim < 2 for values in arrays.values()):
        raise ValueError("Failure-video state trace arrays are not frame-aligned")
    count = counts.pop()
    return tuple(
        {name: values[index] for name, values in arrays.items()}
        for index in range(count)
    )


def _capture_failure_scenario(
    env: Any,
    reference: ReferenceTrajectory,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    scenario: str,
    seed: int,
    *,
    step: Any,
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
    initial_action_index = start + contract.initial_action_offset_ticks * stride
    if initial_action_index < 0:
        raise ValueError("Failure scenario cannot form its initial action history")
    initial_action = jp.asarray(_reference_action(reference, initial_action_index))
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

    append(0, initial_action_index)
    executed = 0
    for tick in range(1, contract.maximum_control_ticks + 1):
        action_index = min(
            start + (contract.first_step_action_offset_ticks + tick - 1) * stride,
            len(reference.df) - 1,
        )
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
    if scenario == "full_guideline_continuation":
        audit_outcome = telemetry[-1]["termination_reason"]
    else:
        valid_continuation = any(
            row["inside_jump_window"]
            and not row["deployable_wheel_support"]
            and row["jump_signal_latched"]
            for row in telemetry
        )
        if not valid_continuation:
            raise ValueError("Launch-history scenario did not latch after lost support")
        audit_outcome = "window_latched_after_early_airborne"
    first_ticks_values = np.asarray(jax.device_get(event.first_event_ticks), int)
    first_event_ticks = {
        name: int(first_ticks_values[index]) for index, name in enumerate(EVENT_NAMES)
    }
    summary = {
        "scenario": scenario,
        "seed": int(seed),
        "start_reference_index": start,
        "initial_action_reference_index": initial_action_index,
        "reference_rows_per_control_tick": stride,
        "environment_transitions": executed,
        "formal_training_transitions": 0,
        "terminal": terminal,
        "terminated": bool(telemetry[-1]["terminated"]),
        "truncated": bool(telemetry[-1]["truncated"]),
        "end_code": int(telemetry[-1]["end_code"]),
        "terminal_reason": telemetry[-1]["termination_reason"],
        "audit_outcome": audit_outcome,
        "first_event_ticks": first_event_ticks,
        "initial_ground_support": placement.summary(),
    }
    return FailureTrace(
        scenario=contract,
        frames=tuple(frames),
        telemetry=tuple(telemetry),
        summary=summary,
    )


def capture_failure_scenario(
    env: Any,
    reference: ReferenceTrajectory,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    scenario: str,
    seed: int,
) -> FailureTrace:
    """Capture one named Gate B failure through the unchanged environment step."""
    return _capture_failure_scenario(
        env,
        reference,
        geometry,
        thresholds,
        scenario,
        seed,
        step=jax.jit(env.step),
    )


def _decorate_failure_frame(
    frame: np.ndarray, telemetry: dict[str, Any], scenario: str
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width = image.size[0]
    box_width = min(width - 16, 700)
    draw.rounded_rectangle((8, 8, box_width, 160), radius=7, fill=(0, 0, 0, 185))
    try:
        font = ImageFont.load_default(size=max(11, min(18, width // 45)))
    except TypeError:
        font = ImageFont.load_default()
    events = telemetry["events"]
    active = [name for name in EVENT_NAMES if events[name]]
    lines = [
        f"FAILURE AUDIT | {scenario}",
        (
            f"tick={telemetry['tick']:02d} ref_action={telemetry['action_reference_index']} "
            f"phase={telemetry['phase']} end={telemetry['end_code']}:{telemetry['termination_reason']}"
        ),
        (
            f"x={telemetry['x']:.3f} z={telemetry['z']:.3f} "
            f"vx={telemetry['vx']:+.3f} vz={telemetry['vz']:+.3f} m/s"
        ),
        (
            f"jump window=[{telemetry['jump_window_min_x']:.3f},"
            f" {telemetry['jump_window_max_x']:.3f}] inside={telemetry['inside_jump_window']}"
        ),
        (
            f"host wheel/body contacts={telemetry['host_wheel_contacts']}/"
            f"{telemetry['host_body_contacts']} deployable_support="
            f"{telemetry['deployable_wheel_support']} jump_latch="
            f"{telemetry['jump_signal_latched']}"
        ),
        f"two-phase events={','.join(active) if active else 'none'}",
    ]
    for index, line in enumerate(lines):
        color = (255, 190, 80, 255) if index == 0 else (255, 255, 255, 255)
        draw.text((18, 15 + 23 * index), line, font=font, fill=color)
    if telemetry["terminated"] or telemetry["truncated"]:
        draw.rounded_rectangle(
            (max(8, width - 245), 8, width - 8, 48),
            radius=7,
            fill=(170, 0, 0, 220),
        )
        draw.text(
            (max(16, width - 232), 18),
            "TERMINAL FAILURE",
            font=font,
            fill=(255, 255, 255, 255),
        )
    return np.asarray(image)


def render_failure_trace(
    env: Any,
    trace: FailureTrace,
    output_path: str | Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: int = 25,
) -> dict[str, Any]:
    """Render captured state values; this function never advances dynamics."""
    if len(trace.frames) != len(trace.telemetry) or not trace.frames:
        raise ValueError("Failure trace frames and telemetry must be nonempty and aligned")
    if min(width, height, fps) <= 0:
        raise ValueError("Video dimensions and fps must be positive")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = env.mj_model
    data = mujoco.MjData(model)
    root = int(model.jnt_qposadr[int(model.joint("floating_base_joint").id)])
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 90.0, -10.0, 2.4
    rendered: list[np.ndarray] = []
    try:
        for values, telemetry in zip(trace.frames, trace.telemetry):
            data.qpos[:] = values["qpos"]
            data.qvel[:] = values["qvel"]
            if model.nu:
                data.ctrl[:] = values["ctrl"]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [
                float(values["qpos"][root]) + 0.30,
                float(values["qpos"][root + 1]),
                0.24,
            ]
            renderer.update_scene(data, camera=camera)
            rendered.append(
                _decorate_failure_frame(
                    renderer.render(), telemetry, trace.scenario.name
                )
            )
    finally:
        renderer.close()
    playback = [rendered[0]] * 12
    for frame in rendered:
        playback.extend((frame, frame))
    playback.extend([rendered[-1]] * 25)
    media.write_video(path, playback, fps=int(fps), codec="h264", crf=18)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state_path = path.with_suffix(".states.npz")
    np.savez_compressed(
        state_path,
        **{
            name: np.stack([frame[name] for frame in trace.frames])
            for name in ("qpos", "qvel", "ctrl")
        },
    )
    return {
        "scenario": trace.scenario.name,
        "video": str(path.resolve()),
        "video_sha256": digest,
        "video_bytes": path.stat().st_size,
        "fps": int(fps),
        "playback": "0.5x control-tick playback with initial/terminal holds",
        "environment_transitions": trace.summary["environment_transitions"],
        "formal_training_transitions": 0,
        "frame_count": len(trace.frames),
        "state_trace_sha256": _state_trace_sha256(trace.frames),
        "state_trace": str(state_path.resolve()),
        "state_trace_file_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "summary": trace.summary,
        "telemetry": list(trace.telemetry),
    }


def write_failure_video_manifest(
    path: str | Path,
    videos: list[dict[str, Any]],
    *,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Validate and write the closed two-scenario failure-video manifest."""
    expected = set(FAILURE_SCENARIOS)
    actual = {str(row.get("scenario", "")) for row in videos}
    if actual != expected or len(videos) != len(expected):
        raise ValueError("Failure-video manifest requires both named scenarios")
    if set(source_hashes) != {"xml", "config", "reference"}:
        raise ValueError("Failure-video manifest source hashes are incomplete")
    required_report_fields = {
        "scenario",
        "video",
        "video_sha256",
        "formal_training_transitions",
        "environment_transitions",
        "frame_count",
        "state_trace_sha256",
        "state_trace",
        "state_trace_file_sha256",
        "summary",
        "telemetry",
    }
    required_telemetry_fields = {
        "tick",
        "action_reference_index",
        "inside_jump_window",
        "host_wheel_contacts",
        "host_body_contacts",
        "deployable_wheel_support",
        "jump_signal_latched",
        "terminated",
        "truncated",
        "end_code",
        "events",
    }
    for row in videos:
        if not required_report_fields.issubset(row):
            raise ValueError("Failure-video audit report is incomplete")
        video_path = Path(row["video"])
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise ValueError(f"Failure video is absent or empty: {video_path}")
        actual_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
        if row.get("video_sha256") != actual_hash:
            raise ValueError(f"Failure video hash mismatch: {video_path}")
        state_path = Path(row["state_trace"])
        if not state_path.is_file() or state_path.stat().st_size <= 0:
            raise ValueError(f"Failure-video state trace is absent: {state_path}")
        state_file_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
        if row.get("state_trace_file_sha256") != state_file_hash:
            raise ValueError(f"Failure-video state trace file hash mismatch: {state_path}")
        state_frames = _load_state_trace(state_path)
        if row.get("state_trace_sha256") != _state_trace_sha256(state_frames):
            raise ValueError("Failure-video state trace content hash mismatch")
        if int(row.get("formal_training_transitions", -1)) != 0:
            raise ValueError("Failure video cannot report training transitions")
        scenario = FAILURE_SCENARIOS[str(row["scenario"])]
        summary = row["summary"]
        telemetry = row["telemetry"]
        if not isinstance(summary, dict) or not isinstance(telemetry, list) or not telemetry:
            raise ValueError("Failure-video audit report is incomplete")
        if any(not required_telemetry_fields.issubset(item) for item in telemetry):
            raise ValueError("Failure-video telemetry is incomplete")
        transitions = int(summary.get("environment_transitions", -1))
        if (
            int(row["environment_transitions"]) != transitions
            or int(row["frame_count"]) != len(telemetry)
            or int(row["frame_count"]) != len(state_frames)
            or len(telemetry) != transitions + 1
            or [int(item["tick"]) for item in telemetry] != list(range(len(telemetry)))
        ):
            raise ValueError("Failure-video frame/transition accounting is inconsistent")
        if (
            summary.get("scenario") != scenario.name
            or int(summary.get("start_reference_index", -1))
            != scenario.start_reference_index
            or int(summary.get("formal_training_transitions", -1)) != 0
            or set(summary.get("first_event_ticks", {})) != set(EVENT_NAMES)
            or len(str(row["state_trace_sha256"])) != 64
        ):
            raise ValueError("Failure-video scenario contract is incomplete")
        stride = int(summary.get("reference_rows_per_control_tick", -1))
        initial_index = (
            scenario.start_reference_index
            + scenario.initial_action_offset_ticks * stride
        )
        expected_actions = [initial_index] + [
            scenario.start_reference_index
            + (scenario.first_step_action_offset_ticks + tick - 1) * stride
            for tick in range(1, len(telemetry))
        ]
        actual_actions = [int(item["action_reference_index"]) for item in telemetry]
        if (
            stride <= 0
            or int(summary.get("initial_action_reference_index", -1)) != initial_index
            or actual_actions != expected_actions
        ):
            raise ValueError("Failure-video action schedule does not match the scenario")
        if any(set(item["events"]) != set(EVENT_NAMES) for item in telemetry):
            raise ValueError("Failure-video event telemetry is incomplete")
        derived_first_ticks = {
            name: next(
                (
                    int(item["tick"])
                    for item in telemetry
                    if bool(item["events"][name])
                ),
                -1,
            )
            for name in EVENT_NAMES
        }
        if summary["first_event_ticks"] != derived_first_ticks:
            raise ValueError("Failure-video first event ticks do not match telemetry")
        if int(summary.get("end_code", -1)) == END_PRETAKEOFF_AIRBORNE:
            raise ValueError("Retired prelaunch-airborne terminal was emitted")
        if scenario.name == "full_guideline_continuation":
            if not (
                int(summary.get("end_code", -1)) == int(telemetry[-1]["end_code"])
                and summary.get("terminal_reason")
                == telemetry[-1]["termination_reason"]
                and summary.get("audit_outcome") == summary.get("terminal_reason")
                and bool(summary.get("terminal"))
                == bool(telemetry[-1]["terminated"] or telemetry[-1]["truncated"])
            ):
                raise ValueError("Full-guideline observed outcome does not match")
        elif not (
            summary.get("audit_outcome") == "window_latched_after_early_airborne"
            and any(
                bool(item["inside_jump_window"])
                and not bool(item["deployable_wheel_support"])
                and bool(item["jump_signal_latched"])
                for item in telemetry
            )
        ):
            raise ValueError("Launch-history continuation state does not match")
    manifest = {
        "contract_version": 2,
        "status": "pass",
        "artifact_role": "dynamic_gate_b_outcome_audit_only",
        "source_hashes": dict(source_hashes),
        "formal_training_transitions": 0,
        "environment_transitions": sum(
            int(row["environment_transitions"]) for row in videos
        ),
        "videos": videos,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def render_failure_archive(
    env: Any,
    reference: ReferenceTrajectory,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    *,
    output_dir: str | Path,
    seed: int,
    source_hashes: dict[str, str],
    width: int = 960,
    height: int = 540,
    fps: int = 25,
) -> dict[str, Any]:
    """Capture and render the closed pair of authoritative Gate B outcomes."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    videos = []
    step = jax.jit(env.step)
    for offset, scenario in enumerate(FAILURE_SCENARIOS):
        trace = _capture_failure_scenario(
            env,
            reference,
            geometry,
            thresholds,
            scenario=scenario,
            seed=int(seed) + offset,
            step=step,
        )
        videos.append(
            render_failure_trace(
                env,
                trace,
                output / f"{scenario}.mp4",
                width=width,
                height=height,
                fps=fps,
            )
        )
    return write_failure_video_manifest(
        output / "failure_video_manifest.json",
        videos,
        source_hashes=source_hashes,
    )
