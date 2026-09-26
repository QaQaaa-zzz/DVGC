"""Deterministic guideline envelopes and host-only geometry audit contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from .bank import SnapshotBank
from .config import ACTION_MAPPING_VERSION, STAGE_ID
from .feasibility import validate_phase_snapshot
from .reference import ReferenceAnchors, ReferenceTrajectory
from .reset_geometry import GroundSupportSolver
from .two_phase_runtime import (
    EVENT_NAMES,
    TwoPhaseThresholds,
    TwoPhaseGeometry,
    collision_geom_support_bounds,
    extract_apex_band_signals,
    extract_recovery_signals,
    extract_two_phase_events,
    full_structure_metrics,
    initial_two_phase_event_state,
)
from .two_phase_semantics import ApexBandThresholds, RecoveryThresholds


_APEX_FIELDS = frozenset(
    {
        "com_vz",
        "clearance",
        "roll",
        "pitch",
        "angular_speed",
        "forward_velocity",
        "obstacle_relative_x",
    }
)
_RECOVERY_FIELDS = frozenset(
    {"roll", "pitch", "angular_speed", "forward_velocity"}
)
_SOURCE_HASHES = frozenset(
    {"xml", "reference", "config", "code", "geometry_manifest"}
)
_SOURCE_PATHS = frozenset({"xml", "reference", "config", "code"})
_FORBIDDEN_CONTROLLER_TERMS = ("expert", "pi up", "pi down", "trained policy")


@dataclass(frozen=True)
class GuidelineMargins:
    apex_abs_vz: float
    apex_clearance: float
    apex_abs_roll: float
    apex_abs_pitch: float
    apex_angular_speed: float
    apex_forward_velocity: float
    apex_relative_x: float
    recovery_abs_roll: float
    recovery_abs_pitch: float
    recovery_angular_speed: float
    recovery_forward_velocity: float

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Guideline margins must be finite and nonnegative")


@dataclass(frozen=True)
class GuidelineSelection:
    launch: dict[str, int]
    apex: dict[str, int]
    early_descent: int
    recovery: tuple[int, int, int]


@dataclass(frozen=True)
class ReconstructedGuidelineState:
    qpos: np.ndarray
    qvel: np.ndarray
    normalized_action: np.ndarray
    reference_index: int
    time: float


@dataclass(frozen=True)
class GuidelinePerturbation:
    name: str
    root_x_offset_m: float
    root_z_offset_m: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Guideline perturbation name must be nonempty")
        values = np.asarray(
            [self.root_x_offset_m, self.root_z_offset_m], dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 0.01:
            raise ValueError("Guideline perturbations must be finite and at most 1 cm")


DEFAULT_GUIDELINE_PERTURBATIONS = (
    GuidelinePerturbation("nominal", 0.0, 0.0),
    GuidelinePerturbation("root_x_minus_5mm", -0.005, 0.0),
    GuidelinePerturbation("root_x_plus_5mm", 0.005, 0.0),
)


@dataclass(frozen=True)
class CapturedGuidelineSnapshot:
    record: dict[str, Any]
    cost: dict[str, int]
    original_state: Any


@dataclass(frozen=True)
class GuidelineBankBuild:
    phase_up: SnapshotBank
    phase_down: SnapshotBank
    report: dict[str, Any]
    original_states: dict[str, Any]


def _three_indices(indices: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in indices)
    if not values:
        raise ValueError("Guideline slice is empty")
    return values[0], values[(len(values) - 1) // 2], values[-1]


def select_guideline_indices(
    reference: ReferenceTrajectory, anchors: ReferenceAnchors
) -> GuidelineSelection:
    """Select fixed launch, Apex, early-descent, and recovery reference rows."""
    count = len(reference.df)
    values = tuple(anchors.as_dict().values())
    if min(values) < 0 or max(values) >= count:
        raise ValueError("Reference anchors are outside the trajectory")
    if tuple(sorted(values)) != values:
        raise ValueError("Reference anchors are not ordered")

    launch_indices = range(anchors.approach_end, anchors.takeoff_end + 1)
    launch_front, launch_middle, launch_back = _three_indices(launch_indices)
    flight_start = anchors.takeoff_end
    flight_stop = min(count, anchors.apex + 2)
    velocity = reference.df["vel_z"].to_numpy(float)
    transitions = [
        index
        for index in range(flight_start, flight_stop - 1)
        if velocity[index] > 0.0 and velocity[index + 1] < 0.0
    ]
    if not transitions:
        raise ValueError("Apex slice lacks a positive-to-negative sign transition")
    transition = min(transitions, key=lambda index: abs(index - anchors.apex))
    if abs(velocity[transition]) <= abs(velocity[transition + 1]):
        nearest = transition
        pre = nearest - 1
        post = nearest + 1
    else:
        nearest = transition + 1
        pre = nearest - 1
        post = nearest + 1
    if pre < flight_start or post >= flight_stop:
        raise ValueError("Apex nearest row lacks fixed pre/post neighbors")
    if not (velocity[pre] > 0.0 and velocity[post] < 0.0):
        raise ValueError("Apex pre/post rows do not straddle the velocity sign transition")
    early_descent = post + 1
    if early_descent >= anchors.landing_start or velocity[early_descent] >= 0.0:
        raise ValueError("Early-descent slice is unavailable")
    recovery = _three_indices(range(anchors.recovery_start, anchors.recovery_end + 1))
    return GuidelineSelection(
        launch={"front": launch_front, "middle": launch_middle, "back": launch_back},
        apex={"pre": pre, "nearest": nearest, "post": post},
        early_descent=early_descent,
        recovery=recovery,
    )


def _validated_samples(
    samples: Sequence[Mapping[str, Any]], allowed: frozenset[str], category: str
) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError(f"{category} samples are empty")
    for sample in samples:
        extras = set(sample) - allowed
        missing = allowed - set(sample)
        if extras or missing:
            raise ValueError(
                f"{category} unregistered fields: extras={sorted(extras)}, missing={sorted(missing)}"
            )
    arrays = {
        field: np.asarray([sample[field] for sample in samples], dtype=np.float64)
        for field in sorted(allowed)
    }
    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError(f"{category} samples must be finite")
    return arrays


def _extrema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        field: {"min": float(values.min()), "max": float(values.max())}
        for field, values in sorted(arrays.items())
    }


_FEATURE_DEFINITIONS = {
    "com_vz": {"definition": "root or CoM vertical velocity", "unit": "m/s"},
    "clearance": {"definition": "full collision structure clearance above obstacle top", "unit": "m"},
    "roll": {"definition": "root roll", "unit": "rad"},
    "pitch": {"definition": "root pitch", "unit": "rad"},
    "angular_speed": {"definition": "root angular-speed norm", "unit": "rad/s"},
    "forward_velocity": {"definition": "root or CoM forward velocity", "unit": "m/s"},
    "obstacle_relative_x": {
        "definition": "obstacle_front_x minus robot_frontmost_x; positive before edge",
        "unit": "m",
    },
}


def canonical_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash canonical JSON while excluding the hash field itself."""
    payload = {key: value for key, value in manifest.items() if key != "canonical_manifest_hash"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_threshold_manifest(
    *,
    selection: GuidelineSelection,
    apex_samples: Sequence[Mapping[str, Any]],
    recovery_samples: Sequence[Mapping[str, Any]],
    margins: GuidelineMargins,
    required_recovery_hold_ticks: int,
    source_hashes: Mapping[str, str],
    source_paths: Mapping[str, str],
    geometry_manifest: Mapping[str, Any],
    reference_anchors: ReferenceAnchors,
    extraction_code_version: str,
    controller_provenance: str,
    creation_seed: int,
) -> dict[str, Any]:
    """Build an auditable threshold contract from physical guideline extrema."""
    normalized_controller = " ".join(
        controller_provenance.casefold().replace("_", " ").replace("-", " ").split()
    )
    if normalized_controller != "kinematic guideline envelope":
        raise ValueError(
            "controller provenance must be exactly 'kinematic guideline envelope'"
        )
    if any(term in normalized_controller for term in _FORBIDDEN_CONTROLLER_TERMS):
        raise ValueError("Invalid controller provenance claim")
    if set(source_hashes) != _SOURCE_HASHES or not all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
        for value in source_hashes.values()
    ):
        raise ValueError("Source hashes must bind the complete authoritative input set")
    if set(source_paths) != _SOURCE_PATHS or not all(
        isinstance(value, str) and value for value in source_paths.values()
    ):
        raise ValueError("Source paths must bind XML, reference, config, and code")
    actual_hashes = {
        name: hashlib.sha256(Path(source_paths[name]).read_bytes()).hexdigest()
        for name in sorted(_SOURCE_PATHS)
    }
    actual_hashes["geometry_manifest"] = _canonical_payload_hash(geometry_manifest)
    mismatched = sorted(
        name for name in _SOURCE_HASHES if source_hashes[name] != actual_hashes[name]
    )
    if mismatched:
        raise ValueError(f"Authoritative source hash mismatch: {mismatched}")
    if not isinstance(extraction_code_version, str) or not extraction_code_version:
        raise ValueError("Extraction code version must be nonempty")
    if required_recovery_hold_ticks <= 0:
        raise ValueError("Recovery hold must be positive")
    apex = _validated_samples(apex_samples, _APEX_FIELDS, "apex")
    recovery = _validated_samples(recovery_samples, _RECOVERY_FIELDS, "recovery")
    margin = asdict(margins)
    apex_thresholds = ApexBandThresholds(
        max_abs_com_vz=float(np.max(np.abs(apex["com_vz"])) + margin["apex_abs_vz"]),
        min_clearance=float(np.min(apex["clearance"]) - margin["apex_clearance"]),
        max_abs_roll=float(np.max(np.abs(apex["roll"])) + margin["apex_abs_roll"]),
        max_abs_pitch=float(np.max(np.abs(apex["pitch"])) + margin["apex_abs_pitch"]),
        max_angular_speed=float(np.max(apex["angular_speed"]) + margin["apex_angular_speed"]),
        min_forward_velocity=float(np.min(apex["forward_velocity"]) - margin["apex_forward_velocity"]),
        relative_x_min=float(np.min(apex["obstacle_relative_x"]) - margin["apex_relative_x"]),
        relative_x_max=float(np.max(apex["obstacle_relative_x"]) + margin["apex_relative_x"]),
    )
    recovery_thresholds = RecoveryThresholds(
        max_abs_roll=float(np.max(np.abs(recovery["roll"])) + margin["recovery_abs_roll"]),
        max_abs_pitch=float(np.max(np.abs(recovery["pitch"])) + margin["recovery_abs_pitch"]),
        max_angular_speed=float(np.max(recovery["angular_speed"]) + margin["recovery_angular_speed"]),
        min_forward_velocity=float(np.min(recovery["forward_velocity"]) - margin["recovery_forward_velocity"]),
        required_hold_ticks=int(required_recovery_hold_ticks),
    )
    manifest: dict[str, Any] = {
        "contract_version": 1,
        "source_category": "guideline_physical_envelope",
        "source_hashes": dict(sorted(source_hashes.items())),
        "source_paths": dict(sorted(source_paths.items())),
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "extraction_code_version": extraction_code_version,
        "feature_definitions": _FEATURE_DEFINITIONS,
        "fixed_selection": asdict(selection),
        "reference_anchors": reference_anchors.as_dict(),
        "raw_physical_extrema": {
            "apex": _extrema(apex),
            "recovery": _extrema(recovery),
        },
        "engineering_margins": margin,
        "selected_thresholds": {
            "apex": asdict(apex_thresholds),
            "recovery": asdict(recovery_thresholds),
        },
        "controller_provenance": controller_provenance,
        "reference_rollout_source": "kinematic_guideline_envelope",
        "creation_seed": int(creation_seed),
    }
    manifest["canonical_manifest_hash"] = canonical_manifest_hash(manifest)
    return manifest


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def reconstruct_guideline_state(
    model: mujoco.MjModel,
    reference: ReferenceTrajectory,
    reference_index: int,
    *,
    wheel_roll_radius: float,
    nominal_base_z_ground: float,
) -> ReconstructedGuidelineState:
    """Reconstruct one deterministic proposal with an explicit vertical frame."""
    index = int(reference_index)
    if index < 0 or index >= len(reference.df):
        raise ValueError("Reference index is out of range")
    if not np.isfinite(wheel_roll_radius) or wheel_roll_radius <= 0.0:
        raise ValueError("Wheel roll radius must be finite and positive")
    if not np.isfinite(nominal_base_z_ground):
        raise ValueError("Nominal grounded base z must be finite")
    row = reference.df.iloc[index]
    qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    qvel = np.zeros(model.nv, dtype=np.float64)
    root = int(model.joint("floating_base_joint").id)
    root_qpos = int(model.jnt_qposadr[root])
    root_qvel = int(model.jnt_dofadr[root])
    reference_z_delta = float(row["pos_z"] - reference.df.iloc[0]["pos_z"])
    qpos[root_qpos : root_qpos + 3] = np.asarray(
        [
            row["pos_x"],
            row.get("pos_y", 0.0),
            float(nominal_base_z_ground) + reference_z_delta,
        ],
        dtype=np.float64,
    )
    angles = row[["roll_angle", "pitch_angle", "yaw_angle"]].to_numpy(float)
    if reference.angle_unit == "degree":
        angles = np.deg2rad(angles)
    qpos[root_qpos + 3 : root_qpos + 7] = _quaternion_from_euler(*angles)
    qvel[root_qvel : root_qvel + 3] = np.asarray(
        [row["vel_x"], row.get("vel_y", 0.0), row["vel_z"]], dtype=np.float64
    )
    angle_frame = reference.df[["roll_angle", "pitch_angle", "yaw_angle"]].to_numpy(float)
    if reference.angle_unit == "degree":
        angle_frame = np.deg2rad(angle_frame)
    angular_velocity = np.gradient(
        np.unwrap(angle_frame, axis=0), reference.df["time"].to_numpy(float), axis=0
    )[index]
    qvel[root_qvel + 3 : root_qvel + 6] = angular_velocity
    for joint_name, position_field, velocity_field in (
        ("hip_joint", "hip_position", "hip_velocity"),
        ("knee_joint", "knee_position", "knee_velocity"),
    ):
        joint = int(model.joint(joint_name).id)
        qpos[int(model.jnt_qposadr[joint])] = float(row[position_field])
        qvel[int(model.jnt_dofadr[joint])] = float(row[velocity_field])
    wheel_speed = float(row["vel_x"]) / float(wheel_roll_radius)
    for joint_name in ("frontwheel_joint", "rearwheel_joint"):
        joint = int(model.joint(joint_name).id)
        qvel[int(model.jnt_dofadr[joint])] = wheel_speed
    action = row[
        ["action_steering", "action_rearwheel", "action_hip", "action_knee"]
    ].to_numpy(float)
    return ReconstructedGuidelineState(
        qpos=qpos,
        qvel=qvel,
        normalized_action=np.clip(action, -1.0, 1.0).astype(np.float32),
        reference_index=index,
        time=float(row["time"]),
    )


def _reference_action(reference: ReferenceTrajectory, index: int) -> np.ndarray:
    row = reference.df.iloc[int(index)]
    return np.clip(
        row[
            ["action_steering", "action_rearwheel", "action_hip", "action_knee"]
        ].to_numpy(float),
        -1.0,
        1.0,
    ).astype(np.float32)


def validate_guideline_snapshot(
    record: Mapping[str, Any], *, expected_source_phase: str
) -> dict[str, Any]:
    """Require an explicit formal phase; never infer it from legacy fields."""
    context = record.get("two_phase_context")
    explicit = isinstance(context, Mapping)
    formal_match = explicit and context.get("source_phase") == expected_source_phase
    phase_result = validate_phase_snapshot(record) if explicit else {"valid": False}
    checks = {
        "explicit_two_phase_context": explicit,
        "formal_source_phase": formal_match,
        "phase_snapshot_contract": bool(phase_result.get("valid", False)),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "valid": not failed,
        "checks": checks,
        "failed": failed,
        "phase_snapshot": phase_result,
    }


def capture_guideline_snapshot(
    env: Any,
    reference: ReferenceTrajectory,
    *,
    target_index: int,
    source_phase: str,
    event_names: Sequence[str],
    event_position: str,
    parent_trajectory_id: str,
    trajectory_id: str,
    provenance: Mapping[str, Any],
    rng: Any,
    step_fn: Any | None = None,
    perturbation: GuidelinePerturbation = DEFAULT_GUIDELINE_PERTURBATIONS[0],
) -> CapturedGuidelineSnapshot:
    """Capture v4 state t after real consecutive packets t-2, t-1, t."""
    legacy_by_formal = {
        "propulsion_ascent": "takeoff",
        "descent_recovery": "flight",
    }
    if source_phase not in legacy_by_formal:
        raise ValueError(f"Unknown formal two-phase source: {source_phase}")
    target = int(target_index)
    stride = int(round(float(env.dt) / float(reference.dt_median)))
    if stride <= 0 or not np.isclose(
        stride * reference.dt_median, float(env.dt), atol=1e-9, rtol=0.0
    ):
        raise ValueError("Reference timing does not divide the environment control tick")
    start = target - 3 * stride
    if start < stride or target >= len(reference.df):
        raise ValueError("Target index cannot form a real three-tick history")
    proposal = reconstruct_guideline_state(
        env.mj_model,
        reference,
        start,
        wheel_roll_radius=float(env._config.wheel_roll_radius),
        nominal_base_z_ground=float(env._config.nominal_base_z_ground),
    )
    proposal_qpos = proposal.qpos.copy()
    root_joint = int(env.mj_model.joint("floating_base_joint").id)
    root_qpos = int(env.mj_model.jnt_qposadr[root_joint])
    proposal_qpos[root_qpos] += perturbation.root_x_offset_m
    proposal_qpos[root_qpos + 2] += perturbation.root_z_offset_m
    previous_action = jp.asarray(_reference_action(reference, start - stride))
    ctrl = env._action_to_ctrl(
        previous_action,
        jp.asarray(proposal_qpos)[env._joint_qpos["knee_joint"]],
    )
    legacy_phase = legacy_by_formal[source_phase]
    ground_support = None
    if source_phase == "propulsion_ascent":
        ground_support = GroundSupportSolver(env._config.xml_path).solve(
            proposal_qpos,
            proposal.qvel,
            np.asarray(jax.device_get(ctrl)),
        )
        if not ground_support.accepted:
            raise ValueError(
                f"Guideline Phase U ground placement rejected: {ground_support.reason}"
            )
        proposal_qpos = ground_support.qpos
        ctrl = env._action_to_ctrl(
            previous_action,
            jp.asarray(proposal_qpos)[env._joint_qpos["knee_joint"]],
        )
    airborne = int(source_phase == "descent_recovery")
    state = env.reset_from_snapshot(
        jp.asarray(proposal_qpos, jp.float32),
        jp.asarray(proposal.qvel, jp.float32),
        ctrl,
        rng,
        jp.asarray(STAGE_ID[legacy_phase], jp.int32),
        jp.asarray(airborne, jp.int32),
        jp.asarray(0, jp.int32),
        jp.asarray(0, jp.int32),
        last_action=previous_action,
        estimated_phase=jp.asarray(STAGE_ID[legacy_phase], jp.int32),
        airborne_count=jp.asarray(
            env._config.airborne_confirm_steps if airborne else 0, jp.int32
        ),
        jump_signal_latched=jp.asarray(bool(airborne)),
    )
    advance = jax.jit(env.step) if step_fn is None else step_fn
    for action_index in (start, start + stride, start + 2 * stride):
        state = advance(state, jp.asarray(_reference_action(reference, action_index)))
        jax.block_until_ready(state)
    policy_action = jp.asarray(_reference_action(reference, target))
    record = env.snapshot_record_v4(
        state, legacy_phase, policy_action, dict(provenance)
    )
    terminated = bool(np.asarray(jax.device_get(state.info["terminated"])))
    truncated = bool(np.asarray(jax.device_get(state.info["truncated"])))
    end_code = int(np.asarray(jax.device_get(state.info["end_code"])))
    if terminated or truncated:
        outcome = "physical_failure" if terminated else "timeout"
        raise ValueError(
            f"Terminal guideline snapshot rejected: {outcome}, end_code={end_code}"
        )
    record["two_phase_context"] = {
        "contract_version": 1,
        "source_phase": source_phase,
        "parent_trajectory_id": str(parent_trajectory_id),
        "trajectory_id": str(trajectory_id),
        "time_index": target,
        "event_names": list(event_names),
        "event_position": str(event_position),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": "none" if not (terminated or truncated) else f"end_code_{end_code}",
        "source_policy_hash": provenance["policy_params_sha256"],
        "source_xml_hash": provenance["xml_sha256"],
        "source_config_hash": provenance["config_sha256"],
    }
    record["guideline_controller_provenance"] = {
        "controller": "guideline_open_loop_action_sequence",
        "reference_rollout_source": "data/reference_jump.csv",
        "target_reference_index": target,
        "history_reference_indices": [
            target - 2 * stride,
            target - stride,
            target,
        ],
        "reference_rows_per_control_tick": stride,
        "perturbation": asdict(perturbation),
        "vertical_frame": {
            "reference_origin_z_m": float(reference.df.iloc[0]["pos_z"]),
            "nominal_base_z_ground_m": float(env._config.nominal_base_z_ground),
            "mapping": "nominal_base_z_ground + (reference_z - reference_initial_z)",
        },
        "ground_support": None if ground_support is None else ground_support.summary(),
    }
    validation = validate_guideline_snapshot(
        record, expected_source_phase=source_phase
    )
    if not validation["valid"]:
        raise ValueError(f"Guideline snapshot contract failed: {validation['failed']}")
    return CapturedGuidelineSnapshot(
        record=record,
        cost={"environment_transitions": 3, "training_transitions": 0},
        original_state=state,
    )


def _stable_snapshot_id(record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
        digest.update(np.ascontiguousarray(np.asarray(record[name], np.float32)).tobytes())
    digest.update(
        json.dumps(
            record["two_phase_context"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    digest.update(
        json.dumps(
            record["guideline_controller_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()[:32]


def build_guideline_banks(
    env: Any,
    reference: ReferenceTrajectory,
    selection: GuidelineSelection,
    *,
    provenance: Mapping[str, Any],
    seed: int,
    perturbations: Sequence[GuidelinePerturbation] = DEFAULT_GUIDELINE_PERTURBATIONS,
) -> GuidelineBankBuild:
    """Build small deterministic U/D banks from fixed guideline selections."""
    if not perturbations:
        raise ValueError("At least one declared guideline perturbation is required")
    step_fn = jax.jit(env.step)
    up_records: list[dict[str, Any]] = []
    down_records: list[dict[str, Any]] = []
    original_states: dict[str, Any] = {}
    transitions = 0

    specifications: list[tuple[str, str, int, list[str], str]] = []
    for name, index in selection.launch.items():
        specifications.append(
            ("propulsion_ascent", name, index, ["jump_window_entered"], "event")
        )
    for name, index in selection.apex.items():
        specifications.append(
            ("descent_recovery", name, index, ["apex_band_entered"], name)
        )
    specifications.append(
        (
            "descent_recovery",
            "early_descent",
            selection.early_descent,
            ["descending"],
            "event",
        )
    )
    for spec_index, (phase, selection_name, index, events, position) in enumerate(
        specifications
    ):
        for perturbation_index, perturbation in enumerate(perturbations):
            trajectory_id = (
                f"guideline:{phase}:{selection_name}:{index}:{perturbation.name}"
            )
            captured = capture_guideline_snapshot(
                env,
                reference,
                target_index=index,
                source_phase=phase,
                event_names=events,
                event_position=position,
                parent_trajectory_id="guideline:reference_jump",
                trajectory_id=trajectory_id,
                provenance=provenance,
                rng=jax.random.PRNGKey(
                    int(seed) + spec_index * 100 + perturbation_index
                ),
                step_fn=step_fn,
                perturbation=perturbation,
            )
            record = captured.record
            record["guideline_selection"] = selection_name
            record["guideline_perturbation"] = perturbation.name
            record["id"] = _stable_snapshot_id(record)
            original_states[record["id"]] = captured.original_state
            transitions += captured.cost["environment_transitions"]
            (up_records if phase == "propulsion_ascent" else down_records).append(
                record
            )
    metadata = {
        "method": "two_phase_guideline",
        "controller_provenance": "guideline_open_loop_action_sequence",
        "formal_training_transitions": 0,
        "construction_seed": int(seed),
        "perturbations": [asdict(item) for item in perturbations],
        "created_at": 0.0,
        "reproducible_build": True,
    }
    up = SnapshotBank(up_records, metadata | {"formal_source_phase": "propulsion_ascent"})
    down = SnapshotBank(
        down_records, metadata | {"formal_source_phase": "descent_recovery"}
    )
    report = {
        "phase_up_records": len(up.records),
        "phase_down_records": len(down.records),
        "construction_environment_transitions": transitions,
        "formal_training_transitions": 0,
        "perturbations": [asdict(item) for item in perturbations],
    }
    return GuidelineBankBuild(
        phase_up=up,
        phase_down=down,
        report=report,
        original_states=original_states,
    )


def extract_guideline_threshold_samples(
    model: mujoco.MjModel,
    reference: ReferenceTrajectory,
    selection: GuidelineSelection,
    geometry: TwoPhaseGeometry,
    *,
    wheel_roll_radius: float,
    nominal_base_z_ground: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Extract threshold-only physical values from reconstructed host states."""
    data = mujoco.MjData(model)

    def state_at(index: int) -> Any:
        proposal = reconstruct_guideline_state(
            model,
            reference,
            index,
            wheel_roll_radius=wheel_roll_radius,
            nominal_base_z_ground=nominal_base_z_ground,
        )
        data.qpos[:] = proposal.qpos
        data.qvel[:] = proposal.qvel
        mujoco.mj_forward(model, data)
        runtime_data = SimpleNamespace(
            qpos=jp.asarray(data.qpos),
            qvel=jp.asarray(data.qvel),
            geom_xpos=jp.asarray(data.geom_xpos),
            geom_xmat=jp.asarray(data.geom_xmat),
        )
        return SimpleNamespace(
            data=runtime_data,
            info={
                "airborne_count": jp.asarray(geometry.airborne_confirm_steps),
                "terminated": jp.asarray(0),
                "truncated": jp.asarray(0),
                "end_code": jp.asarray(0),
            },
        )

    apex_samples = []
    for index in selection.apex.values():
        signals = extract_apex_band_signals(state_at(index), geometry)
        apex_samples.append(
            {
                name: float(np.asarray(getattr(signals, name)))
                for name in sorted(_APEX_FIELDS)
            }
        )
    recovery_samples = []
    for index in selection.recovery:
        signals = extract_recovery_signals(
            state_at(index), geometry, previous_recovery_hold_count=0
        )
        recovery_samples.append(
            {
                name: float(np.asarray(getattr(signals, name)))
                for name in sorted(_RECOVERY_FIELDS)
            }
        )
    return apex_samples, recovery_samples


def run_guideline_event_trace(
    env: Any,
    reference: ReferenceTrajectory,
    geometry: TwoPhaseGeometry,
    thresholds: TwoPhaseThresholds,
    *,
    seed: int,
    maximum_control_ticks: int,
) -> dict[str, Any]:
    """Run the fixed open-loop guideline once and close physical event order."""
    stride = int(round(float(env.dt) / float(reference.dt_median)))
    if stride <= 0 or not np.isclose(
        stride * reference.dt_median, float(env.dt), atol=1e-9, rtol=0.0
    ):
        raise ValueError("Reference timing does not divide the environment control tick")
    proposal = reconstruct_guideline_state(
        env.mj_model,
        reference,
        0,
        wheel_roll_radius=float(env._config.wheel_roll_radius),
        nominal_base_z_ground=float(env._config.nominal_base_z_ground),
    )
    initial_action = jp.asarray(_reference_action(reference, 0))
    ctrl = env._action_to_ctrl(
        initial_action,
        jp.asarray(proposal.qpos)[env._joint_qpos["knee_joint"]],
    )
    placement = GroundSupportSolver(env._config.xml_path).solve(
        proposal.qpos,
        proposal.qvel,
        np.asarray(jax.device_get(ctrl)),
    )
    if not placement.accepted:
        raise ValueError(
            f"Guideline event initial ground placement rejected: {placement.reason}"
        )
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
    executed = 0
    terminal = False
    for tick in range(1, int(maximum_control_ticks) + 1):
        reference_index = min(tick * stride, len(reference.df) - 1)
        state = step(state, jp.asarray(_reference_action(reference, reference_index)))
        jax.block_until_ready(state)
        event = extract_two_phase_events(
            state,
            geometry,
            event,
            thresholds,
            tick=jp.asarray(tick, jp.int32),
        )
        executed = tick
        terminal = bool(np.asarray(jax.device_get(state.done)))
        if terminal or bool(np.asarray(jax.device_get(event.stable_recovery))):
            break
    first_ticks_values = np.asarray(jax.device_get(event.first_event_ticks), int)
    first_ticks = {
        name: int(first_ticks_values[index]) for index, name in enumerate(EVENT_NAMES)
    }
    validation = validate_guideline_event_order(
        first_ticks,
        apex_band_width=int(np.asarray(jax.device_get(event.max_apex_band_width))),
        recovery_hold_ticks=int(np.asarray(jax.device_get(event.recovery_hold_count))),
        required_apex_width=1,
        required_recovery_hold=int(thresholds.recovery.required_hold_ticks),
    )
    end_code = int(np.asarray(jax.device_get(state.info["end_code"])))
    return {
        **validation,
        "controller_provenance": "guideline_open_loop_action_sequence",
        "reference_rows_per_control_tick": stride,
        "environment_transitions": executed,
        "formal_training_transitions": 0,
        "maximum_control_ticks": int(maximum_control_ticks),
        "terminal": terminal,
        "end_code": end_code,
        "pre_nearest_post_counts": {"pre": 1, "nearest": 1, "post": 1},
        "initial_ground_support": placement.summary(),
        "vertical_frame": {
            "reference_origin_z_m": float(reference.df.iloc[0]["pos_z"]),
            "nominal_base_z_ground_m": float(env._config.nominal_base_z_ground),
            "mapping": "nominal_base_z_ground + (reference_z - reference_initial_z)",
        },
    }


def validate_guideline_event_order(
    first_event_ticks: Mapping[str, int],
    *,
    apex_band_width: int,
    recovery_hold_ticks: int,
    required_apex_width: int,
    required_recovery_hold: int,
) -> dict[str, Any]:
    """Close event presence, order, Apex width, and recovery-hold accounting."""
    missing = [name for name in EVENT_NAMES if name not in first_event_ticks or int(first_event_ticks[name]) < 0]
    ticks = [int(first_event_ticks[name]) for name in EVENT_NAMES if name not in missing]
    checks = {
        "event_presence": not missing,
        "event_order": not missing
        and all(left < right for left, right in zip(ticks, ticks[1:])),
        "apex_band_width": int(apex_band_width) >= int(required_apex_width),
        "recovery_hold": int(recovery_hold_ticks) >= int(required_recovery_hold),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed else "gate_pause",
        "checks": checks,
        "failed": failed,
        "missing_events": missing,
        "first_event_ticks": {name: int(first_event_ticks.get(name, -1)) for name in EVENT_NAMES},
        "apex_band_width": int(apex_band_width),
        "recovery_hold_ticks": int(recovery_hold_ticks),
    }


def audit_geometry_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: TwoPhaseGeometry,
    *,
    representative: str,
    tolerance: float,
) -> dict[str, Any]:
    """Cross-audit one comparable above-obstacle host state and close a gate."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Geometry audit tolerance must be finite and positive")
    ids = np.asarray(geometry.robot_geom_ids, dtype=np.int32)
    positions = np.asarray(data.geom_xpos)[ids]
    rotations = np.asarray(data.geom_xmat)[ids]
    bounds = collision_geom_support_bounds(
        positions,
        rotations,
        geometry.robot_geom_types,
        geometry.robot_geom_sizes,
    )
    metrics = full_structure_metrics(
        positions,
        rotations,
        geometry.robot_geom_types,
        geometry.robot_geom_sizes,
        obstacle_front_x=geometry.obstacle_front_x,
        obstacle_top_z=geometry.obstacle_top_z,
    )
    nearest_distance = np.inf
    nearest_robot = -1
    for robot in ids:
        distance = float(
            mujoco.mj_geomDistance(
                model, data, int(robot), int(geometry.obstacle_geom_id), 100.0, None
            )
        )
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_robot = int(robot)
    jax_clearance = float(metrics.full_structure_clearance)
    comparable_projection = bool(
        np.all(
            (np.asarray(bounds.max_x) >= geometry.obstacle_front_x)
            & (np.asarray(bounds.min_x) <= geometry.obstacle_back_x)
            & (np.asarray(bounds.max_y) >= -geometry.obstacle_half_width)
            & (np.asarray(bounds.min_y) <= geometry.obstacle_half_width)
        )
    )
    absolute_difference = abs(jax_clearance - float(nearest_distance))
    sign_agreement = bool(
        np.sign(jax_clearance) == np.sign(float(nearest_distance))
    )
    checks = {
        "comparable_projection": comparable_projection,
        "finite": bool(np.isfinite(jax_clearance) and np.isfinite(nearest_distance)),
        "sign_agreement": sign_agreement,
        "absolute_difference": absolute_difference <= float(tolerance),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "representative": representative,
        "status": "pass" if not failed else "gate_pause",
        "checks": checks,
        "failed": failed,
        "comparable_projection": comparable_projection,
        "jax_clearance": jax_clearance,
        "mujoco_reference_distance": float(nearest_distance),
        "absolute_difference": absolute_difference,
        "tolerance": float(tolerance),
        "sign_agreement": sign_agreement,
        "nearest_geom_pair": {
            "robot": model.geom(nearest_robot).name,
            "obstacle": model.geom(int(geometry.obstacle_geom_id)).name,
        },
    }
