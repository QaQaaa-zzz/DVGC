"""Deterministic guideline envelopes and host-only geometry audit contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from .reference import ReferenceAnchors, ReferenceTrajectory
from .two_phase_runtime import (
    EVENT_NAMES,
    TwoPhaseGeometry,
    full_structure_metrics,
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
_FORBIDDEN_CONTROLLER_TERMS = ("expert", "pi_up", "pi_down", "trained policy")


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
    flight_indices = np.arange(flight_start, flight_stop, dtype=np.int32)
    flight_vz = np.abs(reference.df.iloc[flight_indices]["vel_z"].to_numpy(float))
    nearest = int(flight_indices[int(np.argmin(flight_vz))])
    if nearest <= flight_start or nearest >= flight_stop - 1:
        raise ValueError("Apex nearest row lacks fixed pre/post neighbors")
    post = nearest + 1
    early_descent = post + 1
    if early_descent > anchors.landing_start:
        raise ValueError("Early-descent slice is unavailable")
    recovery = _three_indices(range(anchors.recovery_start, anchors.recovery_end + 1))
    return GuidelineSelection(
        launch={"front": launch_front, "middle": launch_middle, "back": launch_back},
        apex={"pre": nearest - 1, "nearest": nearest, "post": post},
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


def build_threshold_manifest(
    *,
    selection: GuidelineSelection,
    apex_samples: Sequence[Mapping[str, Any]],
    recovery_samples: Sequence[Mapping[str, Any]],
    margins: GuidelineMargins,
    required_recovery_hold_ticks: int,
    source_hashes: Mapping[str, str],
    source_paths: Mapping[str, str],
    reference_anchors: ReferenceAnchors,
    extraction_code_version: str,
    controller_provenance: str,
    creation_seed: int,
) -> dict[str, Any]:
    """Build an auditable threshold contract from physical guideline extrema."""
    lower_controller = controller_provenance.casefold()
    if any(term in lower_controller for term in _FORBIDDEN_CONTROLLER_TERMS):
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
        "action_mapping_version": "steer_rearwheel_hip_knee_v1",
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
        "reference_rollout_source": "guideline_open_loop_actions",
        "creation_seed": int(creation_seed),
    }
    manifest["canonical_manifest_hash"] = canonical_manifest_hash(manifest)
    return manifest


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
) -> dict[str, Any]:
    """Cross-audit one host state; never use this function in online JAX paths."""
    ids = np.asarray(geometry.robot_geom_ids, dtype=np.int32)
    metrics = full_structure_metrics(
        np.asarray(data.geom_xpos)[ids],
        np.asarray(data.geom_xmat)[ids],
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
    return {
        "representative": representative,
        "jax_clearance": jax_clearance,
        "mujoco_reference_distance": float(nearest_distance),
        "absolute_difference": abs(jax_clearance - float(nearest_distance)),
        "sign_agreement": bool(np.signbit(jax_clearance) == np.signbit(nearest_distance)),
        "nearest_geom_pair": {
            "robot": model.geom(nearest_robot).name,
            "obstacle": model.geom(int(geometry.obstacle_geom_id)).name,
        },
    }
