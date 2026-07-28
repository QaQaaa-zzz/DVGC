"""Timing-explicit snapshot schema and causal packet validation."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


SNAPSHOT_SCHEMA_NAME = "dvgc_physical_policy_state_v4_timing_explicit"
SNAPSHOT_SCHEMA_VERSION = 4
LEGACY_SNAPSHOT_SCHEMA = "dvgc_physical_policy_state_v3_warmstart"
ACTION_ORDER = ("steer", "drive", "hip", "knee")
ACTOR_FRAME_DIMENSION_NAMES = (
    "steering_joint_pos", "hip_joint_pos", "knee_joint_pos",
    "rearwheel_joint_velocity", "steering_joint_velocity",
    "hip_joint_velocity", "knee_joint_velocity", "frontwheel_joint_velocity",
    "gyro_x", "gyro_y", "gyro_z", "gravity_body_x", "gravity_body_y",
    "gravity_body_z", "last_action_steer", "last_action_drive",
    "last_action_hip", "last_action_knee", "task_distance_to_front",
    "task_distance_to_back", "task_step_height", "task_target_speed",
    "task_terrain_relative_height", "accel_x", "accel_y", "accel_z",
    "accel_z_delta", "wheel_velocity_estimate", "support_estimate",
    "phase_approach_probability", "phase_takeoff_probability",
    "phase_flight_probability", "phase_landing_probability",
    "contact_age_normalized", "recovery_count_normalized",
)
J12_DELAY_SEQUENCE = (1, 2, 1, 1, 2, 1, 2, 1) * 3
REPLAY_MODES = (
    "legacy_logged_replay",
    "timing_explicit_logged_replay",
    "timing_explicit_independent_reconstruction",
)


def authority_replay_mode(record: Mapping[str, Any]) -> str:
    """Return the declared authority mode; never silently reconstruct legacy rows."""
    if record.get("schema_name") == SNAPSHOT_SCHEMA_NAME:
        return "timing_explicit_independent_reconstruction"
    return "legacy_logged_replay"
ESTIMATOR_FIELDS = (
    "phase", "estimated_phase", "phase_probs", "had_airborne",
    "had_valid_landing", "airborne_count", "prelaunch_airborne_count",
    "landing_bounce_count", "invalid_wheel_count", "recovery_count",
    "contact_age", "landing_entry_age", "landing_phase_step",
    "prev_acc_z", "prev_vz", "prev_front_tire_bottom_z",
    "prev_rear_tire_bottom_z", "positive_pitch_count", "wheelie_count",
    "dual_wheel_liftoff_seen", "stage_entry_ever", "apex_seen",
    "jump_signal_latched", "jump_window_start_x", "jump_window_end_x",
    "chain_ever", "recovery_success", "episode_step", "end_code",
)


def causal_prior_packet(actor_observation: Any, delay_ticks: int, *, frame_dim: int = 35) -> np.ndarray:
    """Legacy diagnostic only: synthesize an unavailable prior packet.

    V4 authority and delay audits must use ``actor_packet_fifo_t`` and never
    call this helper.  It remains solely to reproduce the declared legacy
    padding diagnostic.
    """
    packet = np.asarray(actor_observation, np.float32)
    frames = packet.reshape((-1, int(frame_dim)))
    delay = int(delay_ticks)
    if delay < 0 or delay >= len(frames):
        raise ValueError("delay_ticks must be within the packet history")
    if delay == 0:
        return packet.copy()
    return np.concatenate((np.repeat(frames[:1], delay, axis=0), frames[:-delay]), axis=0).reshape(packet.shape)


def select_delayed_packet(packet_queue: Any, delay_ticks: int) -> np.ndarray:
    """Select one complete observation packet; never shifts single signals."""
    queue = np.asarray(packet_queue)
    delay = int(delay_ticks)
    if queue.ndim < 2 or queue.shape[0] != 3 or delay not in (0, 1, 2):
        raise ValueError("v4 packet FIFO must contain exactly t-2,t-1,t")
    return np.asarray(queue[2 - delay]).copy()


def snapshot_v4_contract() -> dict[str, Any]:
    return {
        "schema_name": SNAPSHOT_SCHEMA_NAME,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "boundary": "policy evaluation at physical_state_t before policy_action_t is applied",
        "replay_modes": list(REPLAY_MODES),
        "physical_state_fields": ["qpos", "qvel", "act", "ctrl_previous", "qacc_warmstart", "sensordata", "time"],
        "timing_fields": ["obs_history_pre_t", "current_frame_t", "actor_observation_t", "obs_history_post_t", "actor_packet_fifo_t", "policy_action_t", "ctrl_applied_t", "rng_state_t", "field_ticks", "simulation_timestamps"],
        "estimator_fields": list(ESTIMATOR_FIELDS),
        "required_hashes": ["xml_sha256", "config_sha256", "action_mapping_version", "policy_params_sha256", "policy_config_sha256", "policy_manifest_sha256", "normalizer_sha256", "source_fingerprint"],
        "forbidden": ["silent replay-mode fallback", "actor sidecar override in independent reconstruction", "synthetic v4 packet padding", "post-history used as pre-history"],
    }


# Backward-compatible public name used by the v2 report generator.
snapshot_v2_contract = snapshot_v4_contract


def _finite(value: Any) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.dtype.kind not in "fc" or bool(np.isfinite(array).all())


def _eq(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _ordered_float32_bits(value: Any) -> int:
    raw = int(np.asarray(value, np.float32).reshape(()).view(np.uint32))
    return 0xFFFFFFFF - raw if raw & 0x80000000 else raw + 0x80000000


def fieldwise_float32_comparison(left: Any, right: Any) -> dict[str, Any]:
    """Return exact float32 field evidence, including first mismatch and ULPs."""
    lhs = np.asarray(left, np.float32).reshape((-1,))
    rhs = np.asarray(right, np.float32).reshape((-1,))
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: {lhs.shape} != {rhs.shape}")
    if lhs.size != len(ACTOR_FRAME_DIMENSION_NAMES):
        raise ValueError(f"actor frame must have {len(ACTOR_FRAME_DIMENSION_NAMES)} fields")
    rows = []
    for index, (a, b) in enumerate(zip(lhs, rhs, strict=True)):
        a_bits = int(np.asarray(a).view(np.uint32))
        b_bits = int(np.asarray(b).view(np.uint32))
        rows.append({
            "index": index,
            "semantic_name": ACTOR_FRAME_DIMENSION_NAMES[index],
            "left_value": float(a), "right_value": float(b),
            "left_uint32": a_bits, "right_uint32": b_bits,
            "left_hex": f"0x{a_bits:08x}", "right_hex": f"0x{b_bits:08x}",
            "bit_exact": a_bits == b_bits,
            "absolute_error": float(abs(np.float64(a) - np.float64(b))),
            "ulp_distance": abs(_ordered_float32_bits(a) - _ordered_float32_bits(b)),
        })
    mismatch = [row for row in rows if not row["bit_exact"]]
    return {
        "bit_exact": not mismatch,
        "mismatch_dimension_count": len(mismatch),
        "first_mismatching_dimension": None if not mismatch else mismatch[0]["index"],
        "first_mismatching_semantic": None if not mismatch else mismatch[0]["semantic_name"],
        "rows": rows,
    }


def validate_snapshot_v4(
    record: Mapping[str, Any], *, frame_dim: int = 35,
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
    expected_hashes: Mapping[str, Any] | None = None,
    actor_action_fn: Callable[[np.ndarray], Any] | None = None,
    ctrl_from_action_fn: Callable[[np.ndarray], Any] | None = None,
    current_frame_fn: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Validate every timing, action, provenance and self-containment identity."""
    contract = snapshot_v4_contract()
    required = {
        "schema_name", "schema_version", "physical_state_t", "obs_history_pre_t",
        "current_frame_t", "actor_observation_t", "obs_history_post_t",
        "actor_packet_fifo_t", "estimator_state_pre_t", "estimator_state_post_t",
        "last_normalized_command_t", "policy_action_t", "ctrl_applied_t",
        "rng_state_t", "field_ticks", "simulation_timestamps", "provenance",
    }
    missing = sorted(required - set(record))
    checks: dict[str, bool] = {
        "required_fields": not missing,
        "schema_name": record.get("schema_name") == SNAPSHOT_SCHEMA_NAME,
        "schema_version": record.get("schema_version") == SNAPSHOT_SCHEMA_VERSION,
    }
    if missing:
        return {"valid": False, "missing": missing, "checks": checks, "failed": sorted(key for key, value in checks.items() if not value)}
    physical = record["physical_state_t"]
    checks["physical_fields"] = all(name in physical for name in contract["physical_state_fields"])
    checks["finite"] = all(_finite(value) for group in (physical, record["estimator_state_pre_t"], record["estimator_state_post_t"]) for value in group.values()) and all(_finite(record[name]) for name in ("obs_history_pre_t", "current_frame_t", "actor_observation_t", "obs_history_post_t", "actor_packet_fifo_t", "last_normalized_command_t", "policy_action_t", "ctrl_applied_t", "rng_state_t"))
    if expected_shapes:
        checks["physical_shapes"] = all(tuple(np.asarray(physical[name]).shape) == tuple(shape) for name, shape in expected_shapes.items())
    else:
        checks["physical_shapes"] = np.asarray(physical.get("qpos", [])).ndim == 1 and np.asarray(physical.get("qvel", [])).ndim == 1
    pre = np.asarray(record["obs_history_pre_t"], np.float32)
    frame = np.asarray(record["current_frame_t"], np.float32).reshape((-1,))
    actor = np.asarray(record["actor_observation_t"], np.float32).reshape((-1,))
    post = np.asarray(record["obs_history_post_t"], np.float32)
    fifo = np.asarray(record["actor_packet_fifo_t"], np.float32)
    checks["frame_semantics"] = frame_dim == len(ACTOR_FRAME_DIMENSION_NAMES) or frame_dim != 35
    checks["history_shapes"] = pre.ndim == 2 and pre.shape[1:] == (frame_dim,) and post.shape == pre.shape and actor.shape == ((len(pre) + 1) * frame_dim,) and fifo.shape == (3, actor.size)
    checks["actor_identity"] = checks["history_shapes"] and _eq(actor, np.concatenate((pre.reshape(-1), frame)))
    expected_post = np.concatenate((pre[1:], frame[None]), axis=0) if len(pre) else pre
    checks["post_history_identity"] = checks["history_shapes"] and _eq(post, expected_post)
    checks["fifo_current_identity"] = checks["history_shapes"] and _eq(fifo[2], actor)
    if checks["history_shapes"]:
        frames = fifo.reshape((3, len(pre) + 1, frame_dim))
        checks["fifo_overlap"] = _eq(frames[0, 1:], frames[1, :-1]) and _eq(frames[1, 1:], frames[2, :-1])
    else:
        checks["fifo_overlap"] = False
    estimator_pre, estimator_post = record["estimator_state_pre_t"], record["estimator_state_post_t"]
    checks["estimator_fields"] = all(name in estimator_pre and name in estimator_post for name in ESTIMATOR_FIELDS)
    ticks = record["field_ticks"]
    tick = ticks.get("physical_state_t") if isinstance(ticks, Mapping) else None
    checks["field_ticks"] = isinstance(tick, int) and all(ticks.get(name) == tick for name in ("actor_observation_t", "current_frame_t", "policy_action_t", "ctrl_applied_t")) and ticks.get("ctrl_previous") == tick - 1 and ticks.get("actor_packet_fifo_t") == [tick - 2, tick - 1, tick]
    checks["estimator_ticks"] = (
        isinstance(tick, int)
        and int(estimator_pre.get("episode_step", -1)) == tick
        and int(estimator_post.get("episode_step", -1)) == tick
    )
    timestamps = record["simulation_timestamps"]
    checks["timestamps"] = isinstance(timestamps, Mapping) and _finite(timestamps.get("physical_state_t", np.nan)) and _finite(timestamps.get("ctrl_applied_t", np.nan))
    provenance = record["provenance"]
    checks["provenance_fields"] = all(name in provenance for name in contract["required_hashes"])
    checks["provenance_match"] = expected_hashes is None or all(provenance.get(name) == value for name, value in expected_hashes.items())
    checks["policy_action"] = actor_action_fn is None or _eq(actor_action_fn(actor), record["policy_action_t"])
    checks["ctrl_mapping"] = ctrl_from_action_fn is None or _eq(ctrl_from_action_fn(np.asarray(record["policy_action_t"], np.float32)), record["ctrl_applied_t"])
    checks["current_frame_reconstruction"] = current_frame_fn is None or _eq(current_frame_fn(record), frame)
    failed = sorted(name for name, value in checks.items() if not value)
    return {"valid": not failed, "missing": missing, "checks": checks, "failed": failed}


def validate_snapshot_v2(record: Mapping[str, Any], *, frame_dim: int = 35) -> dict[str, Any]:
    """Deprecated compatibility validator for old unit consumers."""
    return validate_snapshot_v4(record, frame_dim=frame_dim)


def validate_transfer_eligibility(
    expected_rows: list[Mapping[str, Any]], actual_rows: list[Mapping[str, Any]],
    *, expected_artifact_sha256: str, actual_artifact_sha256: str,
) -> dict[str, Any]:
    """Require exact immutable identities and semantics before pair replay."""
    identity_fields = (
        "source_snapshot_index", "target_snapshot_index",
        "source_snapshot_hash", "target_snapshot_hash", "correction_sha256",
        "phase", "contact_mode", "failure_precursor", "delay_semantics",
    )
    def key(row):
        return (int(row["source_snapshot_index"]), int(row["target_snapshot_index"]))
    expected = {key(row): row for row in expected_rows}
    actual = {key(row): row for row in actual_rows}
    unique = len(expected) == len(expected_rows) and len(actual) == len(actual_rows)
    keys_equal = set(expected) == set(actual)
    row_identity = keys_equal and all(
        all(expected[pair].get(name) == actual[pair].get(name) for name in identity_fields)
        for pair in expected
    )
    checks = {
        "artifact_hash": expected_artifact_sha256 == actual_artifact_sha256,
        "pair_count": len(expected_rows) == len(actual_rows),
        "pair_key_unique": unique,
        "pair_keys": keys_equal,
        "identity_and_semantics": row_identity,
    }
    return {"valid": all(checks.values()), "checks": checks, "failed": sorted(name for name, value in checks.items() if not value)}
