"""Next-step-ready snapshots for policy-conditioned envelope iteration.

Unlike the bootstrap ``HandoffSnapshot``, this schema preserves both unified
phase event states, active-phase metadata, the real actor FIFO, and episode
return.  It is therefore safe to restore a boundary candidate and continue the
same frozen unified policy without reconstructing history or phase context.
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

import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

from .descent_semantics import DescentEventState
from .geometry import extract_geometry
from .handoff_snapshot import compatibility_identity
from .observation import (
    HistoryState,
    actor_observation,
    privileged_observation,
)
from .semantics import EventState
from .tube_rsi import PHASE_UPSTREAM


UNIFIED_ENVELOPE_SNAPSHOT_SCHEMA = "jit_unified_envelope_snapshot_v1"
UP_EVENT_FIELDS = (
    "jump_signal",
    "jump_zone_seen",
    "jump_zone_consumed",
    "ascending_seen",
    "height_seen",
    "apex_seen",
    "stuck_anchor_x",
    "stuck_ticks",
    "stuck",
    "episode_step",
)
DOWN_EVENT_FIELDS = (
    "airborne_seen",
    "valid_contact_seen",
    "contact_x",
    "post_contact_ticks",
    "recovery_success",
)


def _array(value: Any) -> np.ndarray:
    if getattr(value, "dtype", None) is not None and str(value.dtype).startswith("key<"):
        value = jax.random.key_data(value)
    return np.asarray(jax.device_get(value)).copy()


def _event_dict(events: Any, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: _array(getattr(events, name)) for name in fields}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(eq=False)
class UnifiedEnvelopeSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    observation_fifo: np.ndarray
    history_valid_count: int
    observation: np.ndarray
    last_action: np.ndarray
    ctrl: np.ndarray
    rng: np.ndarray
    up_events: Mapping[str, Any]
    down_events: Mapping[str, Any]
    active_phase: int
    start_phase: int
    phase_transitioned: bool
    episode_step: int
    phase_episode_step: int
    episode_return: float
    reset_from_soft_tube: bool
    source_tick: int
    parent_group_index: int
    tube_entry_index: int
    tube_global_index: int
    parent_trajectory: str
    parent_state_sha256: str
    config_sha256: str
    xml_sha256: str
    policy_actor_sha256: str
    policy_payload_sha256: str
    policy_iteration: int
    compatibility_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "qpos",
            "qvel",
            "observation_fifo",
            "observation",
            "last_action",
            "ctrl",
            "rng",
        ):
            setattr(self, name, _array(getattr(self, name)))
        self.up_events = {key: _array(value) for key, value in self.up_events.items()}
        self.down_events = {key: _array(value) for key, value in self.down_events.items()}
        if tuple(self.up_events) != UP_EVENT_FIELDS:
            raise ValueError("unified envelope snapshot upstream event schema drift")
        if tuple(self.down_events) != DOWN_EVENT_FIELDS:
            raise ValueError("unified envelope snapshot downstream event schema drift")
        self.history_valid_count = int(self.history_valid_count)
        if not 0 <= self.history_valid_count <= int(self.observation_fifo.shape[0]):
            raise ValueError("unified envelope history_valid_count is out of range")
        for name in (
            "active_phase",
            "start_phase",
            "episode_step",
            "phase_episode_step",
            "source_tick",
            "parent_group_index",
            "tube_entry_index",
            "tube_global_index",
            "policy_iteration",
        ):
            setattr(self, name, int(getattr(self, name)))
        if self.active_phase not in (0, 1) or self.start_phase not in (0, 1):
            raise ValueError("unified envelope snapshot phase must be upstream or downstream")
        if self.episode_step < 0 or self.phase_episode_step < 0 or self.policy_iteration < 0:
            raise ValueError("unified envelope snapshot counters must be nonnegative")
        self.phase_transitioned = bool(self.phase_transitioned)
        self.reset_from_soft_tube = bool(self.reset_from_soft_tube)
        self.episode_return = float(self.episode_return)
        if not np.isfinite(self.episode_return):
            raise ValueError("unified envelope snapshot episode_return must be finite")
        if not self.parent_trajectory:
            raise ValueError("unified envelope snapshot parent trajectory is required")
        for name in (
            "parent_state_sha256",
            "config_sha256",
            "xml_sha256",
            "policy_actor_sha256",
            "policy_payload_sha256",
        ):
            value = str(getattr(self, name)).lower()
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"unified envelope snapshot {name} must be SHA-256")
            setattr(self, name, value)


def physical_state_sha256(snapshot: UnifiedEnvelopeSnapshot) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(snapshot.qpos).tobytes())
    digest.update(np.ascontiguousarray(snapshot.qvel).tobytes())
    return digest.hexdigest()


def capture_unified_envelope_snapshot(
    state: Any,
    *,
    env: Any,
    parent_trajectory: str,
    parent_state_sha256: str,
    config_sha256: str,
    policy_actor_sha256: str,
    policy_payload_sha256: str,
    policy_iteration: int,
) -> UnifiedEnvelopeSnapshot:
    """Capture a nonterminal unified state without dropping continuation context."""
    if bool(np.asarray(jax.device_get(state.done))):
        raise ValueError("cannot capture a terminal unified envelope state")
    info = state.info
    if bool(np.asarray(jax.device_get(info["expert_switching_used"]))):
        raise ValueError("unified envelope snapshot cannot originate from expert switching")
    history = info["history"]
    return UnifiedEnvelopeSnapshot(
        qpos=_array(state.data.qpos),
        qvel=_array(state.data.qvel),
        observation_fifo=_array(history.frames),
        history_valid_count=int(np.asarray(jax.device_get(history.valid_count))),
        observation=_array(state.obs["state"]),
        last_action=_array(info["last_action"]),
        ctrl=_array(state.data.ctrl),
        rng=_array(info["rng"]),
        up_events=_event_dict(info["up_events"], UP_EVENT_FIELDS),
        down_events=_event_dict(info["down_events"], DOWN_EVENT_FIELDS),
        active_phase=int(np.asarray(jax.device_get(info["active_phase"]))),
        start_phase=int(np.asarray(jax.device_get(info["start_phase"]))),
        phase_transitioned=bool(np.asarray(jax.device_get(info["phase_transitioned"]))),
        episode_step=int(np.asarray(jax.device_get(info["episode_step"]))),
        phase_episode_step=int(np.asarray(jax.device_get(info["phase_episode_step"]))),
        episode_return=float(np.asarray(jax.device_get(info["episode_return"]))),
        reset_from_soft_tube=bool(
            np.asarray(jax.device_get(info["reset_from_soft_tube"]))
        ),
        source_tick=int(np.asarray(jax.device_get(info["source_tick"]))),
        parent_group_index=int(np.asarray(jax.device_get(info["parent_group_index"]))),
        tube_entry_index=int(np.asarray(jax.device_get(info["tube_entry_index"]))),
        tube_global_index=int(np.asarray(jax.device_get(info["tube_global_index"]))),
        parent_trajectory=str(parent_trajectory),
        parent_state_sha256=str(parent_state_sha256),
        config_sha256=str(config_sha256),
        xml_sha256=str(env._bundle.xml_sha256),
        policy_actor_sha256=str(policy_actor_sha256),
        policy_payload_sha256=str(policy_payload_sha256),
        policy_iteration=int(policy_iteration),
        compatibility_identity=compatibility_identity(env),
    )


def save_unified_envelope_snapshot(path: Path, snapshot: UnifiedEnvelopeSnapshot) -> None:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(stream.name)
    payload = directory / "snapshot.pkl"
    os.replace(temporary, payload)
    identity = {
        "schema": UNIFIED_ENVELOPE_SNAPSHOT_SCHEMA,
        "payload_sha256": _sha256(payload),
        "physical_state_sha256": physical_state_sha256(snapshot),
        "parent_trajectory": snapshot.parent_trajectory,
        "parent_state_sha256": snapshot.parent_state_sha256,
        "config_sha256": snapshot.config_sha256,
        "xml_sha256": snapshot.xml_sha256,
        "policy_actor_sha256": snapshot.policy_actor_sha256,
        "policy_payload_sha256": snapshot.policy_payload_sha256,
        "policy_iteration": snapshot.policy_iteration,
        "active_phase": snapshot.active_phase,
        "episode_step": snapshot.episode_step,
        "phase_episode_step": snapshot.phase_episode_step,
        "compatibility_identity": snapshot.compatibility_identity,
    }
    (directory / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_unified_envelope_snapshot(path: Path) -> UnifiedEnvelopeSnapshot:
    directory = Path(path)
    sidecar = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
    if sidecar.get("schema") != UNIFIED_ENVELOPE_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported unified envelope snapshot schema")
    payload_path = directory / "snapshot.pkl"
    if _sha256(payload_path) != sidecar.get("payload_sha256"):
        raise ValueError("unified envelope snapshot payload SHA-256 mismatch")
    with payload_path.open("rb") as stream:
        snapshot = pickle.load(stream)
    if not isinstance(snapshot, UnifiedEnvelopeSnapshot):
        raise ValueError("unified envelope snapshot payload has the wrong type")
    expected = {
        "physical_state_sha256": physical_state_sha256(snapshot),
        "parent_trajectory": snapshot.parent_trajectory,
        "parent_state_sha256": snapshot.parent_state_sha256,
        "config_sha256": snapshot.config_sha256,
        "xml_sha256": snapshot.xml_sha256,
        "policy_actor_sha256": snapshot.policy_actor_sha256,
        "policy_payload_sha256": snapshot.policy_payload_sha256,
        "policy_iteration": snapshot.policy_iteration,
        "active_phase": snapshot.active_phase,
        "episode_step": snapshot.episode_step,
        "phase_episode_step": snapshot.phase_episode_step,
        "compatibility_identity": snapshot.compatibility_identity,
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError(f"unified envelope snapshot {key} provenance mismatch")
    return snapshot


def restore_unified_envelope_snapshot(snapshot: UnifiedEnvelopeSnapshot, env: Any) -> mjx_env.State:
    """Restore one candidate with full unified metadata for the very next step."""
    if snapshot.compatibility_identity != compatibility_identity(env):
        raise ValueError("unified envelope snapshot runtime compatibility mismatch")
    if snapshot.xml_sha256 != env._bundle.xml_sha256:
        raise ValueError("unified envelope snapshot XML mismatch")

    model = env._require_runtime_model()
    data = mjx_env.make_data(
        env.mj_model,
        qpos=jp.asarray(snapshot.qpos),
        qvel=jp.asarray(snapshot.qvel),
        ctrl=jp.asarray(snapshot.ctrl),
        impl=model.impl.value,
        naconmax=int(env.resolved_config.model["naconmax"]),
        naccdmax=env._reset_data_naccdmax(),
        njmax=int(env.resolved_config.model["njmax"]),
    )
    data = mjx.forward(model, data)
    geometry = extract_geometry(data, env._geometry)
    observable_geometry = env._observable_geometry(data, geometry)
    history = HistoryState(
        frames=jp.asarray(snapshot.observation_fifo),
        valid_count=jp.asarray(snapshot.history_valid_count, jp.int32),
    )
    up_events = EventState(
        **{name: jp.asarray(snapshot.up_events[name]) for name in UP_EVENT_FIELDS}
    )
    down_events = DescentEventState(
        **{name: jp.asarray(snapshot.down_events[name]) for name in DOWN_EVENT_FIELDS}
    )
    active_phase = jp.asarray(snapshot.active_phase, jp.int32)
    start_phase = jp.asarray(snapshot.start_phase, jp.int32)
    task_signal = jp.where(
        active_phase == PHASE_UPSTREAM,
        up_events.jump_signal,
        jp.asarray(False),
    )
    actor_obs = actor_observation(history, task_signal)
    critic_obs = privileged_observation(data, actor_obs, observable_geometry)
    reward_state = env._reward_state(data, geometry)
    false = jp.asarray(False)
    soft = jp.asarray(snapshot.reset_from_soft_tube, dtype=bool)
    info = {
        "rng": jax.random.wrap_key_data(jp.asarray(snapshot.rng, dtype=jp.uint32)),
        "history": history,
        "up_events": up_events,
        "down_events": down_events,
        "active_phase": active_phase,
        "start_phase": start_phase,
        "phase_transitioned": jp.asarray(snapshot.phase_transitioned),
        "expert_switching_used": false,
        "last_action": jp.asarray(snapshot.last_action),
        "reward_state": reward_state,
        "episode_step": jp.asarray(snapshot.episode_step, jp.int32),
        "phase_episode_step": jp.asarray(snapshot.phase_episode_step, jp.int32),
        "source_tick": jp.asarray(snapshot.source_tick, jp.int32),
        "parent_group_index": jp.asarray(snapshot.parent_group_index, jp.int32),
        "tube_entry_index": jp.asarray(snapshot.tube_entry_index, jp.int32),
        "tube_global_index": jp.asarray(snapshot.tube_global_index, jp.int32),
        "reset_from_soft_tube": soft,
        "reset_from_jump_start": false,
        "reset_from_continuation": jp.asarray(True),
        "terminated": false,
        "truncated": false,
        "time_out": jp.asarray(0.0, jp.float32),
        "end_code": jp.asarray(0, jp.int32),
        "success": false,
        "physical_failure": false,
        "roll_limit": false,
        "pitch_limit": false,
        "jump_zone_missed": false,
        "stuck": false,
        "yaw_limit": false,
        "timeout": false,
        "episode_return": jp.asarray(snapshot.episode_return, jp.float32),
    }
    metrics = env._unified_zero_metrics()
    metrics.update(env._state_metrics(reward_state, geometry, up_events, jp.asarray(False)))
    metrics.update(
        {
            "event/descent_airborne_seen": down_events.airborne_seen.astype(jp.float32),
            "event/descent_valid_contact_seen": down_events.valid_contact_seen.astype(jp.float32),
            "event/descent_post_contact_ticks": down_events.post_contact_ticks.astype(jp.float32),
            "reset/source_soft_tube": soft.astype(jp.float32),
            "reset/source_natural": jp.asarray(0.0, jp.float32),
            "reset/source_jump_start": jp.asarray(0.0, jp.float32),
            "reset/source_continuation": jp.asarray(1.0, jp.float32),
            "reset/tube_phase_upstream": (
                soft & (start_phase == jp.asarray(0, jp.int32))
            ).astype(jp.float32),
            "reset/tube_phase_downstream": (
                soft & (start_phase == jp.asarray(1, jp.int32))
            ).astype(jp.float32),
            "event/tube_phase_transition": jp.asarray(0.0, jp.float32),
            "state/active_phase": active_phase.astype(jp.float32),
        }
    )
    return mjx_env.State(
        data=data,
        obs={"state": actor_obs, "privileged_state": critic_obs},
        reward=jp.asarray(0.0, jp.float32),
        done=jp.asarray(0.0, jp.float32),
        metrics=metrics,
        info=info,
    )
