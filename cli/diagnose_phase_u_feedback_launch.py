#!/usr/bin/env python3
"""Run one frozen-budget Phase U feedback-braking physical diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import jax
import mujoco
import numpy as np

from dvgc.config import AUTHORITATIVE_XML_SHA256, file_sha256
from dvgc.env import END_REASON
from dvgc.phase_expert_training import (
    PHASE_PROPULSION_ASCENT,
    PhaseExpertRunSpec,
    _event_from_info,
    _phase_expert_frame,
    build_phase_expert_environment,
    phase_expert_source_tree_sha256,
    phase_u_reward_contract_hash,
    validate_phase_expert_run_spec,
)
from dvgc.phase_u_launch_diagnostic import (
    FeedbackLaunchSpec,
    close_diagnostic_outcomes,
    feedback_launch_action,
    feedback_launch_specs,
    rank_diagnostic_row,
    select_representative_rows,
)
from dvgc.two_phase_runtime import apex_band_membership


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIAGNOSTIC_SOURCE_PATHS = (
    "cli/diagnose_phase_u_feedback_launch.py",
    "dvgc/phase_u_launch_diagnostic.py",
    "dvgc/phase_expert_training.py",
    "dvgc/two_phase_runtime.py",
    "dvgc/two_phase_semantics.py",
    "dvgc/env.py",
    "dvgc/signals.py",
    "dvgc/rewards.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _diagnostic_source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for relative in _DIAGNOSTIC_SOURCE_PATHS:
        path = _REPOSITORY_ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def frozen_manifest_payload(
    *,
    run_id: str,
    seed: int,
    horizon: int,
    source_head: str,
    source_tree_sha256: str,
    xml_sha256: str,
    config_sha256: str,
    training_config_sha256: str,
    threshold_manifest_canonical_hash: str,
    reward_contract_hash: str,
) -> dict[str, Any]:
    specs = feedback_launch_specs()
    return {
        "schema": "dvgc.phase_u_feedback_braking_diagnostic.v1",
        "status": "frozen_before_outcomes",
        "run_id": run_id,
        "seed": seed,
        "horizon": horizon,
        "branch_count": len(specs),
        "maximum_diagnostic_environment_transitions": len(specs) * horizon,
        "ppo_training_transitions": 0,
        "controller_provenance": "bounded_physical_feedback_diagnostic",
        "guideline_action_replay": False,
        "parameter_adaptation_after_outcomes": False,
        "controller_grid": [asdict(spec) for spec in specs],
        "source_head": source_head,
        "phase_expert_source_tree_sha256": phase_expert_source_tree_sha256(),
        "source_hashes": {
            "source_tree": source_tree_sha256,
            "xml": xml_sha256,
            "config": config_sha256,
            "training_config": training_config_sha256,
            "threshold_manifest": threshold_manifest_canonical_hash,
            "reward_contract": reward_contract_hash,
        },
        "claims": {
            "expert": False,
            "reachable": False,
            "safe": False,
            "tube": False,
            "training_reset": False,
        },
    }


def write_diagnostic_run(
    run: str | Path,
    *,
    manifest: Mapping[str, Any],
    specs: Sequence[FeedbackLaunchSpec],
    branch_runner: Callable[[int, FeedbackLaunchSpec], Mapping[str, Any]],
    media_renderer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze inputs, execute every branch once, close outcomes, then render."""
    root = Path(run)
    if root.exists():
        raise ValueError(f"refusing overwrite {root}")
    root.mkdir(parents=True)
    expected_grid = [_jsonable(asdict(spec)) for spec in specs]
    if list(manifest.get("controller_grid", ())) != expected_grid:
        raise ValueError("manifest controller grid does not match execution grid")
    if int(manifest.get("branch_count", -1)) != len(specs):
        raise ValueError("manifest branch count does not match execution grid")
    _write_json_atomic(root / "frozen_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    transitions = 0
    outcomes_path = root / "outcomes.jsonl"
    with outcomes_path.open("a", encoding="utf-8") as handle:
        for index, spec in enumerate(specs):
            row = dict(branch_runner(index, spec))
            row.setdefault("branch_id", f"branch_{index:03d}")
            row.setdefault("spec_index", index)
            row.setdefault("spec", asdict(spec))
            if row["spec_index"] != index or row["spec"] != asdict(spec):
                raise ValueError("branch result identity does not match frozen spec")
            used = int(row.get("diagnostic_environment_transitions", -1))
            if used < 0:
                raise ValueError("branch transition accounting is missing")
            transitions += used
            rows.append(row)
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    ceiling = int(manifest["maximum_diagnostic_environment_transitions"])
    if transitions > ceiling:
        raise ValueError("diagnostic transition ceiling exceeded")
    closed = close_diagnostic_outcomes(rows)
    reasons = Counter(str(row.get("terminal_reason", "unknown")) for row in rows)
    ranked = sorted(rows, key=rank_diagnostic_row)
    report = {
        "status": "completed",
        **closed,
        "termination_reason_counts": dict(sorted(reasons.items())),
        "diagnostic_environment_transitions": transitions,
        "maximum_diagnostic_environment_transitions": ceiling,
        "ppo_training_transitions": 0,
        "best_branch": ranked[0] if ranked else None,
        "apex_success_branch_ids": [
            row["branch_id"] for row in ranked if bool(row.get("success"))
        ],
        "stable_ascending_nonphysical_branch_ids": [
            row["branch_id"]
            for row in ranked
            if bool(row.get("stable_airborne_reached"))
            and bool(row.get("ascending_reached"))
            and not bool(row.get("physical_failure"))
        ],
        "claims": dict(manifest["claims"]),
    }
    _write_json_atomic(root / "diagnostic_report.json", report)

    media: list[Mapping[str, Any]] = []
    if media_renderer is not None:
        for row in select_representative_rows(rows, maximum=8):
            media.append(dict(media_renderer(row)))
    _write_json_atomic(root / "representative_media.json", media)
    return report


def _apex_contract_residual(signals: Any, thresholds: Any) -> float:
    """Normalized zero-at-membership residual for diagnostic ranking only."""
    relative_width = max(thresholds.relative_x_max - thresholds.relative_x_min, 1e-6)
    values = (
        max(0.0, abs(float(signals.com_vz)) / thresholds.max_abs_com_vz - 1.0),
        max(0.0, (thresholds.min_clearance - float(signals.clearance)) /
            max(abs(thresholds.min_clearance), 1e-6)),
        max(0.0, abs(float(signals.roll)) / thresholds.max_abs_roll - 1.0),
        max(0.0, abs(float(signals.pitch)) / thresholds.max_abs_pitch - 1.0),
        max(0.0, float(signals.angular_speed) / thresholds.max_angular_speed - 1.0),
        max(0.0, (thresholds.min_forward_velocity - float(signals.forward_velocity)) /
            max(thresholds.min_forward_velocity, 1e-6)),
        max(0.0, (thresholds.relative_x_min - float(signals.obstacle_relative_x)) /
            relative_width),
        max(0.0, (float(signals.obstacle_relative_x) - thresholds.relative_x_max) /
            relative_width),
        1.0 if bool(signals.illegal_contact) or bool(signals.physical_failure) else 0.0,
    )
    return float(math.sqrt(sum(value * value for value in values)))


class _PhysicalBranchRunner:
    def __init__(self, environment: Any, *, seed: int, horizon: int) -> None:
        self.environment = environment
        self.seed = seed
        self.horizon = horizon
        self.step = jax.jit(environment.step)
        self.frames: dict[str, list[dict[str, np.ndarray]]] = {}
        self.actions: dict[str, list[np.ndarray]] = {}

    def __call__(self, index: int, spec: FeedbackLaunchSpec) -> dict[str, Any]:
        branch_id = f"branch_{index:03d}"
        state = self.environment.reset(jax.random.PRNGKey(self.seed))
        frames = [_phase_expert_frame(state)]
        actions = [np.zeros((4,), np.float32)]
        active_age = 0
        minimum_residual = math.inf
        maximum_root_z = -math.inf
        maximum_vz = -math.inf
        maximum_clearance = -math.inf
        maximum_abs_roll = 0.0
        maximum_abs_pitch = 0.0
        maximum_angular_speed = 0.0
        minimum_forward_velocity = math.inf
        action_energy = 0.0
        apex_member_seen = False
        for tick in range(self.horizon):
            event_before = _event_from_info(state.info)
            _, _, _, pitch, _ = self.environment._base_env._root_state(state.data)
            gyro = self.environment._base_env._sensor_vec(state.data, "gyro_local", 3)
            action = feedback_launch_action(
                spec,
                pitch=pitch,
                pitch_rate=gyro[1],
                window_latched=event_before.jump_window_entered,
                active_age=active_age,
            )
            action_host = np.asarray(jax.device_get(action), np.float32)
            state = self.step(state, action)
            jax.block_until_ready(state)
            frames.append(_phase_expert_frame(state))
            actions.append(action_host.copy())
            action_energy += float(np.sum(np.square(action_host)))
            event_after = _event_from_info(state.info)
            if bool(event_after.jump_window_entered):
                active_age = active_age + 1 if bool(event_before.jump_window_entered) else 0
            apex, _ = self.environment._extract_signals(
                state, self.environment._geometry, event_after.recovery_hold_count
            )
            apex = jax.device_get(apex)
            root_qpos, _, _, _, _ = self.environment._base_env._root_state(state.data)
            residual = _apex_contract_residual(
                apex, self.environment._thresholds.apex
            )
            minimum_residual = min(minimum_residual, residual)
            maximum_root_z = max(maximum_root_z, float(root_qpos[2]))
            maximum_vz = max(maximum_vz, float(apex.com_vz))
            maximum_clearance = max(maximum_clearance, float(apex.clearance))
            maximum_abs_roll = max(maximum_abs_roll, abs(float(apex.roll)))
            maximum_abs_pitch = max(maximum_abs_pitch, abs(float(apex.pitch)))
            maximum_angular_speed = max(
                maximum_angular_speed, float(apex.angular_speed)
            )
            minimum_forward_velocity = min(
                minimum_forward_velocity, float(apex.forward_velocity)
            )
            apex_member_seen = apex_member_seen or bool(
                apex_band_membership(apex, self.environment._thresholds.apex)
            )
            if bool(state.done):
                break
        info = jax.device_get(state.info)
        event = _event_from_info(info)
        success = bool(info["phase_expert/success"])
        physical = bool(info["phase_expert/physical_failure"])
        environment_timeout = bool(info["phase_expert/timeout"])
        exhausted = not bool(state.done) and tick + 1 >= self.horizon
        timeout = environment_timeout or exhausted
        end_code = int(info["end_code"])
        terminal_reason = (
            "apex_transition_band"
            if success
            else "diagnostic_horizon"
            if exhausted
            else END_REASON.get(end_code, f"unknown_{end_code}")
        )
        self.frames[branch_id] = frames
        self.actions[branch_id] = actions
        return {
            "branch_id": branch_id,
            "spec_index": index,
            "spec": asdict(spec),
            "seed": self.seed,
            "diagnostic_environment_transitions": tick + 1,
            "success": success,
            "physical_failure": physical,
            "timeout": timeout,
            "task_failure": bool(info["phase_expert/task_failure"]),
            "end_code": end_code,
            "terminal_reason": terminal_reason,
            "first_event_ticks": np.asarray(event.first_event_ticks).tolist(),
            "jump_window_reached": bool(event.jump_window_entered),
            "liftoff_reached": bool(event.liftoff_seen),
            "stable_airborne_reached": bool(event.stable_airborne),
            "ascending_reached": bool(event.ascending),
            "apex_member_seen": apex_member_seen,
            "minimum_apex_contract_residual": minimum_residual,
            "maximum_root_z": maximum_root_z,
            "maximum_vertical_velocity": maximum_vz,
            "maximum_clearance": maximum_clearance,
            "maximum_abs_roll": maximum_abs_roll,
            "maximum_abs_pitch": maximum_abs_pitch,
            "maximum_angular_speed": maximum_angular_speed,
            "minimum_forward_velocity": minimum_forward_velocity,
            "action_energy": action_energy,
            "action_saturation_fraction": float(
                np.mean(np.abs(np.stack(actions[1:])) >= 0.999)
            ),
        }


def _render_branch(
    base_environment: Any,
    row: Mapping[str, Any],
    frames: Sequence[Mapping[str, np.ndarray]],
    actions: Sequence[np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    import mediapy as media

    branch_id = str(row["branch_id"])
    video_path = output_dir / f"{branch_id}_{row['terminal_reason']}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    model = base_environment.mj_model
    data = mujoco.MjData(model)
    root = int(model.jnt_qposadr[int(model.joint("floating_base_joint").id)])
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 90.0, -10.0, 2.4
    images = []
    try:
        for frame in frames:
            data.qpos[:] = frame["qpos"]
            data.qvel[:] = frame["qvel"]
            data.ctrl[:] = frame["ctrl"]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [
                float(frame["qpos"][root]) + 0.3,
                float(frame["qpos"][root + 1]),
                0.24,
            ]
            renderer.update_scene(data, camera=camera)
            images.append(renderer.render().copy())
    finally:
        renderer.close()
    playback = [images[0]] * 10 + [image for image in images for _ in range(2)]
    playback += [images[-1]] * 20
    media.write_video(video_path, playback, fps=25, codec="h264", crf=18)
    trace_path = video_path.with_suffix(".states.npz")
    np.savez_compressed(
        trace_path,
        **{
            name: np.stack([frame[name] for frame in frames])
            for name in ("qpos", "qvel", "ctrl")
        },
        action=np.stack(actions),
    )
    return {
        "branch_id": branch_id,
        "terminal_reason": row["terminal_reason"],
        "end_code": row["end_code"],
        "status": "rendered",
        "video": str(video_path.resolve()),
        "video_sha256": file_sha256(video_path),
        "state_trace": str(trace_path.resolve()),
        "state_trace_sha256": file_sha256(trace_path),
        "captured_control_ticks": len(frames) - 1,
        "rendering_environment_transitions": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--threshold-manifest", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--horizon", default=80, type=int)
    parser.add_argument("--run", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.horizon != 80:
        raise ValueError("feedback diagnostic horizon is frozen at 80")
    run = (_REPOSITORY_ROOT / args.run).resolve()
    required_parent = (
        _REPOSITORY_ROOT / "runs" / "two_phase" / "diagnostics"
    ).resolve()
    if run.parent != required_parent:
        raise ValueError("run must be runs/two_phase/diagnostics/<run_id>")
    config_path = (_REPOSITORY_ROOT / args.config).resolve()
    training_path = (_REPOSITORY_ROOT / args.training_config).resolve()
    threshold_path = (_REPOSITORY_ROOT / args.threshold_manifest).resolve()
    training_config = json.loads(training_path.read_text(encoding="utf-8"))
    preflight_output = (
        _REPOSITORY_ROOT
        / "runs"
        / "two_phase"
        / "phase_experts"
        / f"diagnostic_preflight_{run.name}"
    )
    spec = PhaseExpertRunSpec(
        phase=PHASE_PROPULSION_ASCENT,
        experiment_level="formal_expert",
        requested_total_transitions=6400,
        seed=args.seed,
        config_path=str(config_path),
        training_config_path=str(training_path),
        threshold_manifest_path=str(threshold_path),
        authorization_manifest_path=None,
        output_dir=str(preflight_output),
        descent_seed_bank=None,
        descent_seed_manifest=None,
        resume_run=None,
        restore_checkpoint=None,
    )
    validated = validate_phase_expert_run_spec(spec, preflight_only=True)
    environment = build_phase_expert_environment(validated)
    manifest = frozen_manifest_payload(
        run_id=run.name,
        seed=args.seed,
        horizon=args.horizon,
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_tree_sha256=_diagnostic_source_tree_sha256(),
        xml_sha256=AUTHORITATIVE_XML_SHA256,
        config_sha256=file_sha256(config_path),
        training_config_sha256=file_sha256(training_path),
        threshold_manifest_canonical_hash=validated.thresholds.canonical_manifest_hash,
        reward_contract_hash=phase_u_reward_contract_hash(training_config),
    )
    runner = _PhysicalBranchRunner(
        environment, seed=args.seed, horizon=args.horizon
    )

    def renderer(row: Mapping[str, Any]) -> Mapping[str, Any]:
        branch_id = str(row["branch_id"])
        return _render_branch(
            environment._base_env,
            row,
            runner.frames[branch_id],
            runner.actions[branch_id],
            run / "representative_media",
        )

    report = write_diagnostic_run(
        run,
        manifest=manifest,
        specs=feedback_launch_specs(),
        branch_runner=runner,
        media_renderer=renderer,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
