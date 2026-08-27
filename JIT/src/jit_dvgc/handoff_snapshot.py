"""Auditable, next-step-ready handoff snapshots for phase boundaries.

Snapshots intentionally contain the real actor FIFO rather than reconstructing
history from a single observation.  They are data artifacts, not checkpoints
for training parameters and do not alter physics or reward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping

import numpy as np
import jax

from .constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS


def compatibility_identity(env: Any) -> dict[str, Any]:
    """Return the stable contract required to execute a snapshot safely."""
    mapping = env._bundle.action_mapping
    mapping_payload = {
        "ctrl_min": np.asarray(mapping.ctrl_min).tolist(),
        "ctrl_max": np.asarray(mapping.ctrl_max).tolist(),
        "hip_initial": float(mapping.hip_initial), "knee_initial": float(mapping.knee_initial),
        "base_rear_speed": float(mapping.base_rear_speed), "rear_speed_delta": float(mapping.rear_speed_delta),
        "joint_target_semantics": mapping.joint_target_semantics, "knee_target_delta": float(mapping.knee_target_delta),
    }
    return {"xml_sha256": env._bundle.xml_sha256, "actor_frame_fields": list(ACTOR_FRAME_FIELDS),
            "actor_task_fields": list(ACTOR_TASK_FIELDS), "history_length": 3,
            "action_order": list(ACTION_ORDER), "action_mapping": mapping_payload,
            "sim_dt": float(env.sim_dt), "control_dt": float(env.dt),
            "event_schema": ["jump_signal", "jump_zone_seen", "jump_zone_consumed", "ascending_seen", "height_seen", "apex_seen", "stuck_anchor_x", "stuck_ticks", "stuck", "episode_step"]}


def _array(value: Any) -> np.ndarray:
    if getattr(value, "dtype", None) is not None and str(value.dtype).startswith("key<"):
        value = jax.random.key_data(value)
    return np.asarray(value).copy()


def _event_dict(events: Any) -> dict[str, Any]:
    names = (
        "jump_signal", "jump_zone_seen", "jump_zone_consumed", "ascending_seen",
        "height_seen", "apex_seen", "stuck_anchor_x", "stuck_ticks", "stuck",
        "episode_step",
    )
    return {name: np.asarray(getattr(events, name)).copy() for name in names}


@dataclass(eq=False)
class HandoffSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    observation_fifo: np.ndarray
    history_valid_count: int
    observation: np.ndarray
    last_action: np.ndarray
    ctrl: np.ndarray
    rng: np.ndarray
    events: dict[str, Any]
    tick: int
    parent_trajectory: str
    parent_tick: int | None
    config_sha256: str
    xml_sha256: str
    policy_sha256: str
    policy_identity: str | None = None
    compatibility_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("qpos", "qvel", "observation_fifo", "observation", "last_action", "ctrl", "rng"):
            setattr(self, name, _array(getattr(self, name)))
        self.tick = int(self.tick)
        self.history_valid_count = int(self.history_valid_count)
        if self.history_valid_count < 0 or self.history_valid_count > self.observation_fifo.shape[0]:
            raise ValueError("snapshot history_valid_count is out of range")
        if self.tick < 0:
            raise ValueError("snapshot tick must be nonnegative")
        if not self.parent_trajectory:
            raise ValueError("snapshot parent_trajectory is required")
        for name in ("config_sha256", "xml_sha256", "policy_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"snapshot {name} must be a SHA-256 hex digest")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandoffSnapshot):
            return NotImplemented
        scalar = ("tick", "history_valid_count", "parent_trajectory", "parent_tick", "config_sha256", "xml_sha256", "policy_sha256", "policy_identity", "compatibility_identity")
        return all(getattr(self, n) == getattr(other, n) for n in scalar) and all(
            np.array_equal(getattr(self, n), getattr(other, n))
            for n in ("qpos", "qvel", "observation_fifo", "observation", "last_action", "ctrl", "rng")
        ) and self.events.keys() == other.events.keys() and all(
            np.array_equal(self.events[k], other.events[k]) for k in self.events
        )


def capture_snapshot(
    state: Any,
    *,
    config_sha256: str,
    xml_sha256: str,
    policy_sha256: str,
    parent_trajectory: str,
    parent_tick: int | None = None,
    policy_identity: str | None = None,
    compatibility: Mapping[str, Any] | None = None,
) -> HandoffSnapshot:
    """Capture a live environment state without losing FIFO/event context."""
    info = state.info
    history = info["history"]
    tick = int(np.asarray(info.get("episode_step", history.valid_count)))
    return HandoffSnapshot(
        qpos=_array(state.data.qpos), qvel=_array(state.data.qvel),
        observation_fifo=_array(history.frames),
        history_valid_count=int(np.asarray(history.valid_count)),
        observation=_array(state.obs["state"]),
        last_action=_array(info["last_action"]), ctrl=_array(state.data.ctrl),
        rng=_array(info["rng"]),
        events=_event_dict(info["events"]), tick=tick,
        parent_trajectory=parent_trajectory, parent_tick=parent_tick,
        config_sha256=config_sha256, xml_sha256=xml_sha256,
        policy_sha256=policy_sha256, policy_identity=policy_identity,
        compatibility_identity=compatibility,
    )


def save_snapshot(path: Path, snapshot: HandoffSnapshot) -> None:
    """Atomically save payload plus a hash-bearing JSON provenance sidecar."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(stream.name)
    payload = directory / "snapshot.pkl"
    os.replace(temporary, payload)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    identity = {
        "schema": "handoff_snapshot_v1", "payload_sha256": digest,
        "config_sha256": snapshot.config_sha256, "xml_sha256": snapshot.xml_sha256,
        "policy_sha256": snapshot.policy_sha256, "policy_identity": snapshot.policy_identity,
        "parent_trajectory": snapshot.parent_trajectory, "parent_tick": snapshot.parent_tick,
        "tick": snapshot.tick, "history_valid_count": snapshot.history_valid_count,
        "compatibility_identity": snapshot.compatibility_identity,
    }
    (directory / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_snapshot(path: Path) -> HandoffSnapshot:
    directory = Path(path)
    sidecar = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
    payload = directory / "snapshot.pkl"
    if hashlib.sha256(payload.read_bytes()).hexdigest() != sidecar.get("payload_sha256"):
        raise ValueError("snapshot payload_sha256 mismatch")
    with payload.open("rb") as stream:
        snapshot = pickle.load(stream)
    if not isinstance(snapshot, HandoffSnapshot):
        raise ValueError("snapshot payload has the wrong type")
    for field in ("config_sha256", "xml_sha256", "policy_sha256", "policy_identity", "parent_trajectory", "parent_tick", "tick", "history_valid_count", "compatibility_identity"):
        if getattr(snapshot, field) != sidecar.get(field):
            raise ValueError(f"snapshot {field} provenance mismatch")
    return snapshot


def restore_snapshot(snapshot: HandoffSnapshot, env: Any | None = None) -> Any:
    """Return restore data, or a fully reconstructed MJX State when ``env`` is given."""
    values = {"qpos": snapshot.qpos.copy(), "qvel": snapshot.qvel.copy(), "ctrl": snapshot.ctrl.copy(),
              "observation_fifo": snapshot.observation_fifo.copy(), "history_valid_count": snapshot.history_valid_count,
              "observation": snapshot.observation.copy(), "last_action": snapshot.last_action.copy(),
              "ctrl": snapshot.ctrl.copy(), "rng": snapshot.rng.copy(), "events": snapshot.events.copy(), "tick": snapshot.tick}
    if env is None:
        return values
    from jax import numpy as jp
    from mujoco_playground._src import mjx_env
    data = mjx_env.make_data(env.mj_model, qpos=jp.asarray(snapshot.qpos), qvel=jp.asarray(snapshot.qvel), ctrl=jp.asarray(snapshot.ctrl), impl=env.mjx_model.impl.value, naconmax=int(env.resolved_config.model["naconmax"]), naccdmax=(int(env.resolved_config.model["naccdmax"]) if env.resolved_config.schema.endswith("_v4") else None), njmax=int(env.resolved_config.model["njmax"]))
    data = __import__("mujoco").mjx.forward(env.mjx_model, data)
    from .observation import HistoryState, ObservableGeometry, actor_observation, privileged_observation
    from .semantics import EventState
    from .geometry import extract_geometry
    geometry = extract_geometry(data, env._geometry)
    observable_geometry = env._observable_geometry(data, geometry)
    history = HistoryState(jp.asarray(snapshot.observation_fifo), jp.asarray(snapshot.history_valid_count, dtype=jp.int32))
    events = EventState(**{k: jp.asarray(v) for k, v in snapshot.events.items()})
    actor_obs = actor_observation(history, events.jump_signal)
    critic_obs = privileged_observation(data, actor_obs, observable_geometry)
    reward_state = env._reward_state(data, geometry)
    false = jp.asarray(False)
    info = {"history": history, "events": events, "last_action": jp.asarray(snapshot.last_action), "reward_state": reward_state,
            "reset_source_airborne_rsi": false, "terminated": false, "truncated": false, "time_out": jp.asarray(0.0, jp.float32),
            "end_code": jp.asarray(0, jp.int32), "success": false, "physical_failure": false, "roll_limit": false,
            "pitch_limit": false, "jump_zone_missed": false, "stuck": false, "yaw_limit": false, "timeout": false,
            "episode_step": jp.asarray(snapshot.tick, jp.int32), "episode_return": jp.asarray(0.0, jp.float32), "rng": jax.random.wrap_key_data(jp.asarray(snapshot.rng, dtype=jp.uint32))}
    metrics = env._zero_metrics()
    metrics.update(env._state_metrics(reward_state, geometry, events, false))
    return mjx_env.State(data=data, obs={"state": actor_obs, "privileged_state": critic_obs}, reward=jp.asarray(0.0, jp.float32), done=jp.asarray(0.0, jp.float32), metrics=metrics, info=info)
