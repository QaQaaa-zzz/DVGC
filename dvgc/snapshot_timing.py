"""Causal observation-packet helpers and the policy snapshot v2 contract."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


SNAPSHOT_SCHEMA_NAME = "dvgc_policy_snapshot"
SNAPSHOT_SCHEMA_VERSION = 2
ACTION_ORDER = ("steer", "drive", "hip", "knee")
J12_DELAY_SEQUENCE = (1, 2, 1, 1, 2, 1, 2, 1) * 3


def causal_prior_packet(actor_observation: Any, delay_ticks: int, *, frame_dim: int = 35) -> np.ndarray:
    """Construct a causal packet before a logged packet using oldest-frame hold.

    A legacy four-frame packet contains no frame older than its first frame.
    The missing warm-start prefix is therefore filled by holding that oldest
    measured frame, never by borrowing a future frame or shifting one signal
    independently from the rest of the packet.
    """
    packet = np.asarray(actor_observation, np.float32)
    frames = packet.reshape((-1, int(frame_dim)))
    delay = int(delay_ticks)
    if delay < 0 or delay >= len(frames):
        raise ValueError("delay_ticks must be within the packet history")
    if delay == 0:
        return packet.copy()
    prefix = np.repeat(frames[:1], delay, axis=0)
    return np.concatenate((prefix, frames[:-delay]), axis=0).reshape(packet.shape)


def select_delayed_packet(packet_queue: Any, delay_ticks: int) -> np.ndarray:
    """Select one complete observation packet from an oldest-to-current FIFO."""
    queue = np.asarray(packet_queue)
    delay = int(delay_ticks)
    if queue.ndim < 2 or delay < 0 or delay >= queue.shape[0]:
        raise ValueError("invalid complete-packet delay")
    return np.asarray(queue[-1 - delay]).copy()


def snapshot_v2_contract() -> dict[str, Any]:
    """Return the mandatory fields and exact control-boundary semantics."""
    return {
        "schema_name": SNAPSHOT_SCHEMA_NAME,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "time_boundary": "policy evaluation at physical_state_t before command_action_t is applied",
        "required_fields": {
            "physical_state_t": "qpos/qvel/act/ctrl_previous/qacc_warmstart at tick t",
            "obs_history_pre_t": "history before current_frame_t is appended",
            "current_frame_t": "complete sensor/estimator frame from physical_state_t",
            "actor_observation_t": "exact online actor input; validation and logged replay authority",
            "obs_history_post_t": "advance(obs_history_pre_t,current_frame_t)",
            "policy_action_t": "deterministic normalized command generated from actor_observation_t",
            "ctrl_applied_t": "actuator control produced from policy_action_t and applied on t->t+1",
            "estimator_state_pre_t": "phase/contact/filter state used to build current_frame_t",
            "estimator_state_post_t": "estimator state after the t->t+1 transition",
            "field_ticks": "integer control tick and simulation timestamp for every field group",
        },
        "replay_modes": {
            "logged_observation_replay": "uses actor_observation_t and declares that dependency",
            "independent_reconstruction": "uses only physical_state_t, estimator_state_pre_t, obs_history_pre_t and current_frame_t",
        },
        "forbidden": ["silent sidecar fallback", "mixing obs_history_post_t with current_frame_t", "per-signal delay shifts"],
    }


def validate_snapshot_v2(record: Mapping[str, Any], *, frame_dim: int = 35) -> dict[str, Any]:
    """Validate self-contained history and logged-input identities without fallback."""
    contract = snapshot_v2_contract()
    missing = sorted(set(contract["required_fields"]) - set(record))
    if missing:
        return {"valid": False, "missing": missing}
    pre = np.asarray(record["obs_history_pre_t"], np.float32)
    frame = np.asarray(record["current_frame_t"], np.float32).reshape((int(frame_dim),))
    logged = np.asarray(record["actor_observation_t"], np.float32)
    post = np.asarray(record["obs_history_post_t"], np.float32)
    reconstructed = np.concatenate((pre.reshape(-1), frame))
    expected_post = np.concatenate((pre[1:], frame[None]), axis=0) if len(pre) else pre
    checks = {
        "logged_equals_independent_reconstruction": bool(np.array_equal(logged, reconstructed)),
        "post_history_equals_causal_advance": bool(np.array_equal(post, expected_post)),
        "ticks_declared": bool(record.get("field_ticks")),
    }
    return {"valid": all(checks.values()), "missing": [], "checks": checks}
