"""Build a real-rollout nominal jump centerline on a 0.1 m x grid.

The centerline is a geometric scaffold for Jump-Tube identification, not a new
controller target and not a reward modification.  It is extracted only from a
completed successful canonical natural rollout.  No qpos/qvel interpolation is
allowed: every centerline point is one real captured simulator frame.

The longitudinal corridor is nominally [2.5 m, 4.2 m].  The usable end is the
first valid landing/contact if it occurs earlier.  Upstream points are selected
before first Apex.  Downstream points are selected after Apex only when root
vertical velocity is negative.  Post-landing recovery never enters the nominal
Jump-Tube centerline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "jit_nominal_jump_centerline_v1"
X_MIN_M = 2.5
X_MAX_M = 4.2
DX_M = 0.1
MAX_X_MATCH_ERROR_M = 0.05 + 1.0e-9


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_true(values: np.ndarray) -> int:
    indices = np.flatnonzero(np.asarray(values) > 0.5)
    return int(indices[0]) if indices.size else -1


def _metric_array(data: Mapping[str, np.ndarray], metadata: Mapping[str, Any], name: str) -> np.ndarray:
    key = metadata.get("metric_keys", {}).get(name)
    if not key or key not in data:
        raise ValueError(f"trace is missing required metric: {name}")
    return np.asarray(data[key], dtype=np.float64)


def _centers(last_x: float) -> list[float]:
    last = min(float(last_x), X_MAX_M)
    values = []
    index = 0
    while True:
        value = X_MIN_M + index * DX_M
        if value > last + DX_M * 0.5 + 1.0e-12:
            break
        values.append(round(value, 10))
        index += 1
    return values


def build_nominal_jump_centerline(
    *,
    canonical_evaluation_report: Path,
    output_dir: Path,
) -> dict[str, Any]:
    report_path = Path(canonical_evaluation_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "jit_pi_unified_canonical_natural_eval_v1":
        raise ValueError("nominal centerline requires canonical unified natural evaluation")
    if report.get("status") != "completed":
        raise ValueError("canonical evaluation is not completed")
    rollout = report.get("canonical_rollout")
    if not isinstance(rollout, dict) or rollout.get("full_recovery_success") is not True:
        raise ValueError("nominal centerline requires one successful full-recovery rollout")

    npz_path = Path(str(report["canonical_trace_npz"]))
    metadata_path = Path(str(report["canonical_trace_metadata"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _file_sha256(npz_path) != str(metadata["npz_sha256"]):
        raise ValueError("canonical trace NPZ hash drift")

    loaded = np.load(npz_path, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    if qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) != len(qvel):
        raise ValueError("canonical trace physical arrays have invalid shape")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("canonical trace contains nonfinite physical state")

    apex_seen = _metric_array(data, metadata, "event/apex_seen")
    contact_seen = _metric_array(data, metadata, "event/descent_valid_contact_seen")
    apex_index = _first_true(apex_seen)
    contact_index = _first_true(contact_seen)
    if apex_index < 0 or contact_index < 0 or apex_index >= contact_index:
        raise ValueError("successful centerline trace has invalid Apex/contact ordering")

    x = qpos[:, 0]
    z = qpos[:, 2]
    vx = qvel[:, 0]
    vz = qvel[:, 2]
    apex_x = float(x[apex_index])
    contact_x = float(x[contact_index])
    if contact_x < X_MIN_M:
        raise ValueError("landing occurs before the declared jump corridor")

    targets = _centers(contact_x)
    if not targets:
        raise ValueError("no nominal x slices lie before first valid landing")
    points: list[dict[str, Any]] = []
    previous_index = -1
    for target in targets:
        if target < apex_x - DX_M * 0.5:
            branch = "upstream"
            eligible = np.arange(0, apex_index + 1)
        elif target > apex_x + DX_M * 0.5:
            branch = "downstream"
            eligible = np.arange(apex_index, contact_index + 1)
            eligible = eligible[vz[eligible] < 0.0]
        else:
            branch = "apex"
            eligible = np.asarray([apex_index], dtype=np.int64)
        eligible = eligible[eligible > previous_index]
        if eligible.size == 0:
            raise ValueError(f"no real trace frame available for nominal slice x={target:.1f}")
        errors = np.abs(x[eligible] - target)
        best = int(eligible[int(np.argmin(errors))])
        error = abs(float(x[best]) - target)
        if error > MAX_X_MATCH_ERROR_M:
            raise ValueError(
                f"successful rollout does not resolve x={target:.1f} within 0.05 m; error={error:.6f}"
            )
        if branch == "downstream" and not float(vz[best]) < 0.0:
            raise ValueError("nominal downstream centerline selected a non-descending frame")
        previous_index = best
        points.append(
            {
                "x_target_m": float(target),
                "frame_index": int(best),
                "phase_semantics": branch,
                "root_x_m": float(x[best]),
                "root_z_m": float(z[best]),
                "root_vx_mps": float(vx[best]),
                "root_vz_mps": float(vz[best]),
                "x_match_error_m": float(error),
            }
        )

    payload = {
        "schema": SCHEMA,
        "status": "completed",
        "purpose": "real-rollout geometric centerline for trajectory-centered Jump-Tube widening",
        "canonical_evaluation_report": str(report_path),
        "canonical_evaluation_report_file_sha256": _file_sha256(report_path),
        "canonical_trace_npz": str(npz_path),
        "canonical_trace_npz_sha256": str(metadata["npz_sha256"]),
        "x_min_m": X_MIN_M,
        "x_hard_max_m": X_MAX_M,
        "x_step_m": DX_M,
        "first_apex_frame_index": int(apex_index),
        "first_apex_x_m": apex_x,
        "first_valid_landing_frame_index": int(contact_index),
        "first_valid_landing_x_m": contact_x,
        "effective_centerline_max_x_m": float(points[-1]["x_target_m"]),
        "point_count": len(points),
        "points": points,
        "real_frames_only": True,
        "qpos_qvel_interpolation_used": False,
        "post_landing_recovery_included": False,
        "reward_changed": False,
        "actor_observation_changed": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    payload["centerline_sha256"] = _canonical_sha256(payload)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"centerline output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "centerline.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_nominal_jump_centerline(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "completed":
        raise ValueError("invalid nominal jump centerline")
    declared = str(payload.get("centerline_sha256", ""))
    base = {key: value for key, value in payload.items() if key != "centerline_sha256"}
    if len(declared) != 64 or _canonical_sha256(base) != declared:
        raise ValueError("nominal jump centerline self-hash drift")
    if payload.get("real_frames_only") is not True or payload.get("qpos_qvel_interpolation_used") is not False:
        raise ValueError("nominal centerline provenance drift")
    if payload.get("post_landing_recovery_included") is not False:
        raise ValueError("nominal centerline contains post-landing recovery")
    return payload
