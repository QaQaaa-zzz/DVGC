"""Build and lock one fixed-jump-start nominal jump centerline.

The centerline is a method scaffold for Jump-Tube identification, not a policy
intent, tracking target, or reward change. It is extracted from one completed
successful canonical fixed-jump-start rollout and then reused unchanged by later
prospective JIT iterations unless a new method version is explicitly declared.

Every centerline point is a real simulator frame. No qpos/qvel interpolation is
allowed. The longitudinal corridor is nominally [2.5 m, 4.2 m] and ends at the
first valid landing if landing occurs earlier. Downstream centerline points must
be post-Apex and descending. Post-landing recovery is excluded.

Conditional provenance is explicit: each selected frame records a physical-state
SHA-256 over the raw saved qpos/qvel bytes and the number of real env.step
transitions from the fixed ground jump start. It does not claim reachability from
the earlier natural reset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "jit_nominal_jump_centerline_v3"
LEGACY_SCHEMAS = {
    "jit_nominal_jump_centerline_v1",
    "jit_nominal_jump_centerline_v2",
}
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


def physical_state_sha256_from_arrays(qpos: np.ndarray, qvel: np.ndarray) -> str:
    """Hash raw physical arrays using the same qpos/qvel byte contract as snapshots."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(qpos)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(qvel)).tobytes())
    return digest.hexdigest()


def _first_true(values: np.ndarray) -> int:
    indices = np.flatnonzero(np.asarray(values) > 0.5)
    return int(indices[0]) if indices.size else -1


def _metric_array(
    data: Mapping[str, np.ndarray], metadata: Mapping[str, Any], name: str
) -> np.ndarray:
    key = metadata.get("metric_keys", {}).get(name)
    if not key or key not in data:
        raise ValueError(f"trace is missing required metric: {name}")
    return np.asarray(data[key], dtype=np.float64)


def _centers(last_x: float) -> list[float]:
    last = min(float(last_x), X_MAX_M)
    values: list[float] = []
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
    if report.get("schema") != "jit_pi_unified_canonical_jump_start_eval_v1":
        raise ValueError("nominal centerline requires canonical jump-start evaluation")
    if report.get("status") != "completed":
        raise ValueError("canonical evaluation is not completed")
    start_contract = report.get("start_contract")
    if not isinstance(start_contract, dict):
        raise ValueError("canonical evaluation lacks jump-start contract")
    if start_contract.get("start_kind") != "fixed_ground_jump_start":
        raise ValueError("canonical evaluation did not use the fixed ground jump start")
    if float(start_contract.get("jump_start_x_m", float("nan"))) != X_MIN_M:
        raise ValueError("canonical evaluation jump-start x drift")
    if start_contract.get("natural_start_connected") is not False:
        raise ValueError("jump-start centerline must not claim natural-start connection")
    if start_contract.get("tube_or_rsi_reset_used") is not False:
        raise ValueError("jump-start centerline cannot use a Tube or RSI reset")
    rollout = report.get("canonical_rollout")
    if not isinstance(rollout, dict) or rollout.get("jump_trajectory_success") is not True:
        raise ValueError("nominal centerline requires one successful jump-to-landing trajectory")

    npz_path = Path(str(report["canonical_trace_npz"]))
    metadata_path = Path(str(report["canonical_trace_metadata"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _file_sha256(npz_path) != str(metadata["npz_sha256"]):
        raise ValueError("canonical trace NPZ hash drift")

    loaded = np.load(npz_path, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}
    qpos_raw = np.asarray(data["qpos"])
    qvel_raw = np.asarray(data["qvel"])
    if qpos_raw.ndim != 2 or qvel_raw.ndim != 2 or len(qpos_raw) != len(qvel_raw):
        raise ValueError("canonical trace physical arrays have invalid shape")
    if not np.isfinite(qpos_raw).all() or not np.isfinite(qvel_raw).all():
        raise ValueError("canonical trace contains nonfinite physical state")
    qpos = np.asarray(qpos_raw, dtype=np.float64)
    qvel = np.asarray(qvel_raw, dtype=np.float64)
    if abs(float(qpos[0, 0]) - X_MIN_M) > 1.0e-6:
        raise ValueError("canonical trace does not begin at the fixed jump start")

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
                f"successful rollout does not resolve x={target:.1f} within 0.05 m; "
                f"error={error:.6f}"
            )
        if branch == "downstream" and not float(vz[best]) < 0.0:
            raise ValueError("nominal downstream centerline selected a non-descending frame")

        transitions_from_previous = (
            int(best) if previous_index < 0 else int(best - previous_index)
        )
        point = {
            "x_target_m": float(target),
            "frame_index": int(best),
            "phase_semantics": branch,
            "physical_state_sha256": physical_state_sha256_from_arrays(
                qpos_raw[best], qvel_raw[best]
            ),
            "root_x_m": float(x[best]),
            "root_z_m": float(z[best]),
            "root_vx_mps": float(vx[best]),
            "root_vz_mps": float(vz[best]),
            "x_match_error_m": float(error),
            "environment_transitions_from_jump_start": int(best),
            "environment_transitions_from_previous_centerline_point": transitions_from_previous,
            "jump_start_reachable_by_saved_forward_rollout": True,
        }
        points.append(point)
        previous_index = best

    jump_start_sha = physical_state_sha256_from_arrays(qpos_raw[0], qvel_raw[0])
    reference_payload_sha = str(report.get("checkpoint_payload_sha256", ""))
    if reference_payload_sha and len(reference_payload_sha) != 64:
        raise ValueError("canonical evaluation checkpoint payload SHA drift")

    payload = {
        "schema": SCHEMA,
        "status": "completed_locked_method_reference",
        "purpose": (
            "fixed-jump-start real-rollout centerline for conditional trajectory-centered "
            "Jump-Tube widening"
        ),
        "canonical_evaluation_report": str(report_path),
        "canonical_evaluation_report_file_sha256": _file_sha256(report_path),
        "canonical_trace_npz": str(npz_path),
        "canonical_trace_npz_sha256": str(metadata["npz_sha256"]),
        "reference_policy_checkpoint_payload_sha256": reference_payload_sha or None,
        "jump_start_state_sha256": jump_start_sha,
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
        "jump_start_connected": True,
        "natural_start_connected": False,
        "rsi_used_to_establish_centerline_reachability": False,
        "post_landing_recovery_included": False,
        "centerline_recomputed_each_iteration": False,
        "centerline_is_policy_intent": False,
        "centerline_is_reward_target": False,
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
    schema = payload.get("schema")
    if schema in LEGACY_SCHEMAS:
        raise ValueError(
            "legacy nominal centerline uses the superseded start contract; "
            "rebuild it under jit_nominal_jump_centerline_v3"
        )
    if schema != SCHEMA or payload.get("status") != "completed_locked_method_reference":
        raise ValueError("invalid causal nominal jump centerline")
    declared = str(payload.get("centerline_sha256", ""))
    base = {key: value for key, value in payload.items() if key != "centerline_sha256"}
    if len(declared) != 64 or _canonical_sha256(base) != declared:
        raise ValueError("nominal jump centerline self-hash drift")
    if payload.get("real_frames_only") is not True:
        raise ValueError("nominal centerline is not real-frame-only")
    if payload.get("qpos_qvel_interpolation_used") is not False:
        raise ValueError("nominal centerline used qpos/qvel interpolation")
    if payload.get("jump_start_connected") is not True:
        raise ValueError("nominal centerline lacks fixed jump-start reachability")
    if payload.get("natural_start_connected") is not False:
        raise ValueError("jump-start centerline improperly claims natural-start connection")
    if payload.get("rsi_used_to_establish_centerline_reachability") is not False:
        raise ValueError("nominal centerline reachability used RSI")
    if payload.get("post_landing_recovery_included") is not False:
        raise ValueError("nominal centerline contains post-landing recovery")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("nominal centerline has no points")
    if any(
        len(str(point.get("physical_state_sha256", ""))) != 64
        or point.get("jump_start_reachable_by_saved_forward_rollout") is not True
        for point in points
    ):
        raise ValueError("nominal centerline point reachability provenance drift")
    return payload
