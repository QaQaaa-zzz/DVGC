"""Independent MJX-Warp Propulsion-Ascent environment."""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .action_mapping import map_action
from .config import ResolvedConfig
from .constants import (
    ACTOR_OBSERVATION_SIZE,
    CTRL_DT,
    END_ONGOING,
    PRIVILEGED_OBSERVATION_SIZE,
    REWARD_COMPONENT_KEYS,
    SIM_DT,
)
from .geometry import GeometrySignals, build_geometry_contract, extract_geometry
from .model import ModelBundle, load_host_model, put_warp_model
from .observation import (
    HistoryState,
    ObservableGeometry,
    actor_observation,
    advance_history,
    initial_history,
    observable_frame,
    privileged_observation,
)
from .rewards import RewardInputs, RewardResult, RewardState, phase_u_reward
from .semantics import (
    PhaseUSignals,
    TerminalInputs,
    TerminalState,
    advance_events,
    classify_terminal,
    initial_event_state,
)
from .handoff_snapshot import HandoffSnapshot, capture_snapshot, compatibility_identity, restore_snapshot
from .snapshot_pool import SnapshotPool
from .descent_semantics import DescentSignals, initial_descent_events, advance_descent_events, classify_descent_terminal
from .descent_rewards import DescentRewardInputs, descent_recovery_reward


def _finite_stuck_window_progress(
    current_x: jax.Array, anchor_x: jax.Array
) -> jax.Array:
    progress = jp.asarray(current_x) - jp.asarray(anchor_x)
    return jp.where(
        jp.isfinite(progress),
        progress,
        jp.asarray(0.0, dtype=progress.dtype),
    )


class TwoPhaseBikeEnv(mjx_env.MjxEnv):
    """The JIT Propulsion-Ascent training environment."""

    def __init__(self, config: ResolvedConfig, *, convert_model: bool = True, snapshot_pool: SnapshotPool | None = None):
        runtime = config_dict.create(
            ctrl_dt=CTRL_DT,
            sim_dt=SIM_DT,
            episode_length=config.ppo.episode_horizon,
            action_repeat=1,
            vision=False,
            impl=str(config.model["mjx_impl"]),
            naconmax=int(config.model["naconmax"]),
            njmax=int(config.model["njmax"]),
        )
        super().__init__(runtime)
        self._resolved_config = config
        self._snapshot_pool = snapshot_pool
        if config.phase == "descent_recovery" and snapshot_pool is None:
            raise ValueError("descent_recovery requires a snapshot pool")
        bundle = load_host_model(config)
        self._bundle: ModelBundle = put_warp_model(bundle) if convert_model else bundle
        self._geometry = build_geometry_contract(bundle.mj_model)
        self._initial_x = float(
            bundle.mj_model.key_qpos[
                bundle.model_index.keyframe_id,
                bundle.model_index.root_qpos_address,
            ]
        )

    @property
    def xml_path(self) -> str:
        return str(self._bundle.xml_path)

    @property
    def action_size(self) -> int:
        return 4

    @property
    def actor_observation_size(self) -> int:
        return ACTOR_OBSERVATION_SIZE

    @property
    def privileged_observation_size(self) -> int:
        return PRIVILEGED_OBSERVATION_SIZE

    @property
    def mj_model(self):
        return self._bundle.mj_model

    @property
    def mjx_model(self):
        return self._bundle.mjx_model

    @property
    def resolved_config(self) -> ResolvedConfig:
        return self._resolved_config

    def capture_handoff_snapshot(
        self,
        state: mjx_env.State,
        *,
        policy_sha256: str,
        parent_trajectory: str,
        parent_tick: int | None = None,
        policy_identity: str | None = None,
    ) -> HandoffSnapshot:
        """Capture an online state with all context needed for continuation."""
        return capture_snapshot(
            state,
            config_sha256=self._resolved_config.config_sha256,
            xml_sha256=self._bundle.xml_sha256,
            policy_sha256=policy_sha256,
            parent_trajectory=parent_trajectory,
            parent_tick=parent_tick,
            policy_identity=policy_identity,
            compatibility=compatibility_identity(self),
        )

    def restore_handoff_snapshot(self, snapshot: HandoffSnapshot) -> mjx_env.State:
        """Restore a snapshot and rebuild derived observations for the next step."""
        if snapshot.compatibility_identity != compatibility_identity(self):
            raise ValueError("handoff snapshot runtime compatibility mismatch")
        return restore_snapshot(snapshot, self)

    def _require_runtime_model(self):
        if self.mjx_model is None:
            raise RuntimeError("MJX model conversion was disabled for this Host-only environment")
        return self.mjx_model

    @staticmethod
    def _sample_range(key: jax.Array, lower: float, upper: float) -> jax.Array:
        return jax.random.uniform(key, (), minval=lower, maxval=upper)

    @staticmethod
    def _observable_geometry(data, signals: GeometrySignals) -> ObservableGeometry:
        return ObservableGeometry(
            obstacle_relative_x=signals.obstacle_relative_x,
            root_height=data.qpos[2],
            roll=signals.roll,
            pitch=signals.pitch,
            yaw=signals.yaw,
            illegal_contact=signals.prohibited_contact,
        )

    def _reward_state(self, data, signals: GeometrySignals) -> RewardState:
        index = self._bundle.model_index
        angular_address = index.sensor_addresses[9]
        angular_velocity = data.sensordata[angular_address : angular_address + 3]
        return RewardState(
            x=data.qpos[index.root_qpos_address],
            y=data.qpos[index.root_qpos_address + 1],
            z=data.qpos[index.root_qpos_address + 2],
            roll=signals.roll,
            pitch=signals.pitch,
            yaw=signals.yaw,
            forward_velocity=data.qvel[index.root_dof_address],
            lateral_velocity=data.qvel[index.root_dof_address + 1],
            vertical_velocity=data.qvel[index.root_dof_address + 2],
            roll_rate=angular_velocity[0],
            pitch_rate=angular_velocity[1],
            yaw_rate=angular_velocity[2],
            hip_velocity=data.qvel[index.hip_dof_address],
            knee_velocity=data.qvel[index.knee_dof_address],
            hip_force=data.actuator_force[2],
            knee_force=data.actuator_force[3],
        )

    @staticmethod
    def _zero_metrics() -> dict[str, jax.Array]:
        zero = jp.asarray(0.0, jp.float32)
        metrics = {
            "reward": zero,
            "reward/unclipped": zero,
            "reward/pre_episode_return_override": zero,
            "reward/unclipped_pre_episode_return_override": zero,
            "reward/episode_return_override": zero,
        }
        metrics.update({f"reward/{key}": zero for key in REWARD_COMPONENT_KEYS})
        for key in (
            "event/jump_signal",
            "event/jump_zone_seen",
            "event/jump_zone_consumed",
            "event/ascending_seen",
            "event/height_seen",
            "event/apex_seen",
            "event/stuck",
            "event/stuck_ticks",
            "event/stuck_window_progress",
            "reset/source_airborne_rsi",
            "signal/root_x",
            "signal/root_y",
            "signal/root_z",
            "signal/forward_velocity",
            "signal/lateral_velocity",
            "signal/vertical_velocity",
            "signal/roll",
            "signal/pitch",
            "signal/yaw",
            "signal/roll_rate",
            "signal/pitch_rate",
            "signal/yaw_rate",
            "signal/angular_speed",
            "signal/obstacle_relative_x",
            "signal/front_wheel_terrain_clearance",
            "signal/rear_wheel_terrain_clearance",
            "signal/maximum_wheel_penetration",
            "signal/hip_velocity",
            "signal/knee_velocity",
            "signal/hip_force",
            "signal/knee_force",
            "signal/joint_power",
            "terminal/physical_failure",
            "terminal/roll_limit",
            "terminal/pitch_limit",
            "terminal/jump_zone_missed",
            "terminal/stuck",
            "terminal/yaw_limit",
            "terminal/timeout",
            "terminal/success",
        ):
            metrics[key] = zero
        return metrics

    @staticmethod
    def _state_metrics(
        reward_state: RewardState,
        geometry: GeometrySignals,
        events,
        reset_source: jax.Array,
    ) -> dict[str, jax.Array]:
        return {
            "event/jump_signal": events.jump_signal.astype(jp.float32),
            "event/jump_zone_seen": events.jump_zone_seen.astype(jp.float32),
            "event/jump_zone_consumed": events.jump_zone_consumed.astype(jp.float32),
            "event/ascending_seen": events.ascending_seen.astype(jp.float32),
            "event/height_seen": events.height_seen.astype(jp.float32),
            "event/apex_seen": events.apex_seen.astype(jp.float32),
            "event/stuck": events.stuck.astype(jp.float32),
            "event/stuck_ticks": events.stuck_ticks.astype(jp.float32),
            "event/stuck_window_progress": _finite_stuck_window_progress(
                reward_state.x, events.stuck_anchor_x
            ),
            "reset/source_airborne_rsi": jp.asarray(reset_source, jp.float32),
            "signal/root_x": reward_state.x,
            "signal/root_y": reward_state.y,
            "signal/root_z": reward_state.z,
            "signal/forward_velocity": reward_state.forward_velocity,
            "signal/lateral_velocity": reward_state.lateral_velocity,
            "signal/vertical_velocity": reward_state.vertical_velocity,
            "signal/roll": reward_state.roll,
            "signal/pitch": reward_state.pitch,
            "signal/yaw": reward_state.yaw,
            "signal/roll_rate": reward_state.roll_rate,
            "signal/pitch_rate": reward_state.pitch_rate,
            "signal/yaw_rate": reward_state.yaw_rate,
            "signal/angular_speed": jp.sqrt(
                jp.square(reward_state.roll_rate)
                + jp.square(reward_state.pitch_rate)
                + jp.square(reward_state.yaw_rate)
            ),
            "signal/obstacle_relative_x": geometry.obstacle_relative_x,
            "signal/front_wheel_terrain_clearance": (
                geometry.front_wheel_terrain_clearance
            ),
            "signal/rear_wheel_terrain_clearance": (
                geometry.rear_wheel_terrain_clearance
            ),
            "signal/maximum_wheel_penetration": geometry.maximum_wheel_penetration,
            "signal/hip_velocity": reward_state.hip_velocity,
            "signal/knee_velocity": reward_state.knee_velocity,
            "signal/hip_force": reward_state.hip_force,
            "signal/knee_force": reward_state.knee_force,
            "signal/joint_power": jp.abs(
                reward_state.hip_force * reward_state.hip_velocity
            )
            + jp.abs(reward_state.knee_force * reward_state.knee_velocity),
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self._resolved_config.phase == "descent_recovery":
            return self._reset_descent(rng)
        decision_key, state_key = jax.random.split(rng)
        use_airborne_rsi = jax.random.bernoulli(
            decision_key, self._resolved_config.reset.airborne_rsi_probability
        )
        return self._reset(state_key, use_airborne_rsi)

    def _reset_descent(self, rng):
        sample = self._snapshot_pool.sample(rng)
        model = self._require_runtime_model(); data = mjx_env.make_data(self.mj_model, qpos=sample["qpos"], qvel=sample["qvel"], ctrl=sample["ctrl"], impl=model.impl.value, naconmax=int(self._resolved_config.model["naconmax"]), naccdmax=(int(self._resolved_config.model["naccdmax"]) if "naccdmax" in self._resolved_config.model else None), njmax=int(self._resolved_config.model["njmax"]))
        data = mjx.forward(model, data); geometry = extract_geometry(data, self._geometry); og = self._observable_geometry(data, geometry)
        history = HistoryState(frames=sample["observation_fifo"], valid_count=sample["history_valid_count"])
        events = initial_descent_events(data.qpos[self._bundle.model_index.root_qpos_address])
        actor_obs = actor_observation(history, sample["observation"][ -1])
        critic_obs = privileged_observation(data, actor_obs, og); rs = self._reward_state(data, geometry); false=jp.asarray(False)
        info={"rng":sample["rng"],"history":history,"events":events,"last_action":sample["last_action"],"reward_state":rs,"episode_step":jp.asarray(0,jp.int32),"source_tick":sample["tick"],"parent_group_index":sample["parent_group_index"],"reset_source_airborne_rsi":false,"terminated":false,"truncated":false,"time_out":jp.asarray(0.,jp.float32),"end_code":jp.asarray(0,jp.int32),"success":false,"physical_failure":false,"timeout":false,"episode_return":jp.asarray(0.,jp.float32)}
        metrics=self._zero_metrics()
        # Keep reset and step metric pytrees identical for Brax lax.scan/vmap.
        metrics.update({
            "reward": jp.asarray(0., jp.float32),
            "reward/unclipped": jp.asarray(0., jp.float32),
            "reward/pre_episode_return_override": jp.asarray(0., jp.float32),
            "reward/unclipped_pre_episode_return_override": jp.asarray(0., jp.float32),
            "reward/episode_return_override": jp.asarray(0., jp.float32),
            "reward/descent_forward_progress": jp.asarray(0., jp.float32),
            "reward/descent_contact": jp.asarray(0., jp.float32),
            "reward/descent_recovery_tick": jp.asarray(0., jp.float32),
            "reward/descent_success": jp.asarray(0., jp.float32),
            "reward/descent_bad_contact": jp.asarray(0., jp.float32),
            "reward/descent_failure": jp.asarray(0., jp.float32),
            "reward/descent_timeout": jp.asarray(0., jp.float32),
            "event/descent_airborne_seen": events.airborne_seen.astype(jp.float32),
            "event/descent_valid_contact_seen": events.valid_contact_seen.astype(jp.float32),
            "event/descent_post_contact_ticks": jp.asarray(0., jp.float32),
            "terminal/descent_success": jp.asarray(0., jp.float32),
            "terminal/descent_physical_failure": jp.asarray(0., jp.float32),
            "terminal/descent_timeout": jp.asarray(0., jp.float32),
            "terminal/physical_failure": jp.asarray(0., jp.float32),
            "terminal/roll_limit": jp.asarray(0., jp.float32),
            "terminal/pitch_limit": jp.asarray(0., jp.float32),
            "terminal/jump_zone_missed": jp.asarray(0., jp.float32),
            "terminal/stuck": jp.asarray(0., jp.float32),
            "terminal/yaw_limit": jp.asarray(0., jp.float32),
            "terminal/timeout": jp.asarray(0., jp.float32),
            "terminal/success": jp.asarray(0., jp.float32),
        })
        return mjx_env.State(data=data,obs={"state":actor_obs,"privileged_state":critic_obs},reward=jp.asarray(0.,jp.float32),done=jp.asarray(0.,jp.float32),metrics=metrics,info=info)

    def reset_natural(self, rng: jax.Array) -> mjx_env.State:
        return self._reset(rng, jp.asarray(False))

    def reset_airborne_rsi(self, rng: jax.Array) -> mjx_env.State:
        """Resets from the approved airborne distribution for diagnostics."""
        return self._reset(rng, jp.asarray(True))

    def _reset(self, rng: jax.Array, use_airborne_rsi: jax.Array) -> mjx_env.State:
        model = self._require_runtime_model()
        index = self._bundle.model_index
        reset_config = self._resolved_config.reset
        qpos = jp.asarray(
            self.mj_model.key_qpos[index.keyframe_id], dtype=jp.float32
        )
        qvel = jp.asarray(
            self.mj_model.key_qvel[index.keyframe_id], dtype=jp.float32
        )
        x_key, z_key, vx_key, vz_key = jax.random.split(rng, 4)
        rsi_x = self._sample_range(
            x_key, reset_config.airborne_rsi_x_min, reset_config.airborne_rsi_x_max
        )
        rsi_z = self._sample_range(
            z_key, reset_config.airborne_rsi_z_min, reset_config.airborne_rsi_z_max
        )
        rsi_vx = self._sample_range(
            vx_key, reset_config.airborne_rsi_vx_min, reset_config.airborne_rsi_vx_max
        )
        rsi_vz = self._sample_range(
            vz_key, reset_config.airborne_rsi_vz_min, reset_config.airborne_rsi_vz_max
        )
        qpos = qpos.at[index.root_qpos_address].set(
            jp.where(use_airborne_rsi, rsi_x, qpos[index.root_qpos_address])
        )
        qpos = qpos.at[index.root_qpos_address + 2].set(
            jp.where(use_airborne_rsi, rsi_z, qpos[index.root_qpos_address + 2])
        )
        qvel = qvel.at[index.root_dof_address].set(
            jp.where(
                use_airborne_rsi,
                rsi_vx,
                jp.asarray(reset_config.initial_forward_velocity, jp.float32),
            )
        )
        qvel = qvel.at[index.root_dof_address + 2].set(
            jp.where(use_airborne_rsi, rsi_vz, jp.asarray(0.0, jp.float32))
        )
        last_action = jp.zeros((4,), dtype=jp.float32)
        ctrl = map_action(
            last_action,
            qpos[index.knee_qpos_address],
            self._bundle.action_mapping,
        )
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=model.impl.value,
            naconmax=int(self._resolved_config.model["naconmax"]),
            naccdmax=(
                int(self._resolved_config.model["naccdmax"])
                if self._resolved_config.schema.endswith("_v4")
                else None
            ),
            njmax=int(self._resolved_config.model["njmax"]),
        )
        data = mjx.forward(model, data)
        geometry = extract_geometry(data, self._geometry)
        observable_geometry = self._observable_geometry(data, geometry)
        history = initial_history()
        events = initial_event_state(qpos[index.root_qpos_address], self._resolved_config)
        actor_obs = actor_observation(history, events.jump_signal)
        critic_obs = privileged_observation(data, actor_obs, observable_geometry)
        reward_state = self._reward_state(data, geometry)
        false = jp.asarray(False)
        info: dict[str, Any] = {
            "rng": rng,
            "history": history,
            "events": events,
            "last_action": last_action,
            "reward_state": reward_state,
            "reset_source_airborne_rsi": jp.asarray(use_airborne_rsi, dtype=bool),
            "terminated": false,
            "truncated": false,
            "time_out": jp.asarray(0.0, jp.float32),
            "end_code": jp.asarray(END_ONGOING, jp.int32),
            "success": false,
            "physical_failure": false,
            "roll_limit": false,
            "pitch_limit": false,
            "jump_zone_missed": false,
            "stuck": false,
            "yaw_limit": false,
            "timeout": false,
            "episode_step": jp.asarray(0, jp.int32),
            "episode_return": jp.asarray(0.0, jp.float32),
        }
        metrics = self._zero_metrics()
        metrics.update(
            self._state_metrics(reward_state, geometry, events, use_airborne_rsi)
        )
        return mjx_env.State(
            data=data,
            obs={"state": actor_obs, "privileged_state": critic_obs},
            reward=jp.asarray(0.0, jp.float32),
            done=jp.asarray(0.0, jp.float32),
            metrics=metrics,
            info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._resolved_config.phase == "descent_recovery":
            return self._step_descent(state, action)
        model = self._require_runtime_model()
        index = self._bundle.model_index
        normalized_action = jp.clip(jp.asarray(action, jp.float32), -1.0, 1.0)
        ctrl = map_action(
            normalized_action,
            state.data.qpos[index.knee_qpos_address],
            self._bundle.action_mapping,
        )
        data = mjx_env.step(model, state.data, ctrl, self.n_substeps)
        geometry = extract_geometry(data, self._geometry)
        nonfinite = (
            ~jp.isfinite(data.qpos).all()
            | ~jp.isfinite(data.qvel).all()
            | ~jp.isfinite(normalized_action).all()
        )
        backward_exit = (
            data.qpos[index.root_qpos_address]
            < self._initial_x
            - self._resolved_config.physical_limits.max_backward_distance
        )
        previous_events = state.info["events"]
        preliminary_inputs = TerminalInputs(
            episode_step=state.info["events"].episode_step,
            nonfinite=nonfinite,
            roll=geometry.roll,
            pitch=geometry.pitch,
            illegal_contact=geometry.prohibited_contact,
            backward_exit=backward_exit,
            stuck=previous_events.stuck,
            yaw=geometry.yaw,
            jump_zone_seen=previous_events.jump_zone_seen,
        )
        preliminary = classify_terminal(preliminary_inputs, self._resolved_config)
        events = advance_events(
            previous_events,
            PhaseUSignals(
                x=data.qpos[index.root_qpos_address],
                z=data.qpos[index.root_qpos_address + 2],
                forward_velocity=data.qvel[index.root_dof_address],
                vertical_velocity=data.qvel[index.root_dof_address + 2],
                physical_failure=preliminary.terminated,
            ),
            self._resolved_config,
        )
        first_apex = events.apex_seen & ~previous_events.apex_seen
        terminal: TerminalState = classify_terminal(
            preliminary_inputs.replace(
                stuck=events.stuck, jump_zone_seen=events.jump_zone_seen
            ),
            self._resolved_config,
        )
        reward_state = self._reward_state(data, geometry)
        reward_result: RewardResult = phase_u_reward(
            RewardInputs(
                current=reward_state,
                action=normalized_action,
                last_action=state.info["last_action"],
                jump_signal=events.jump_signal,
                first_apex_success=first_apex,
                illegal_contact=geometry.prohibited_contact,
                physical_failure_transition=terminal.physical_failure
                & ~state.info["physical_failure"],
                roll_pitch_failure_transition=(
                    terminal.roll_limit | terminal.pitch_limit
                )
                & ~(state.info["roll_limit"] | state.info["pitch_limit"]),
                jump_zone_missed_transition=terminal.jump_zone_missed
                & ~state.info["jump_zone_missed"],
                stuck_transition=terminal.stuck & ~state.info["stuck"],
                yaw_limit_transition=terminal.yaw_limit & ~state.info["yaw_limit"],
                timeout_transition=terminal.timeout & ~state.info["timeout"],
            ),
            self._resolved_config.reward,
            self._resolved_config.physical_limits,
        )
        task_failure = terminal.stuck | terminal.yaw_limit
        failed_episode_return = self._resolved_config.reward.failed_episode_return
        if failed_episode_return is None:
            reward = reward_result.total
        else:
            reward = jp.where(
                task_failure,
                jp.asarray(failed_episode_return, jp.float32)
                - state.info["episode_return"],
                reward_result.total,
            )
        episode_return = state.info["episode_return"] + reward
        episode_return_override = reward - reward_result.total
        observable_geometry = self._observable_geometry(data, geometry)
        frame = observable_frame(
            data,
            index,
            observable_geometry,
            normalized_action,
            jp.asarray(True),
        )
        history = advance_history(state.info["history"], frame)
        actor_obs = actor_observation(history, events.jump_signal)
        critic_obs = privileged_observation(data, actor_obs, observable_geometry)
        reset_source = state.info["reset_source_airborne_rsi"]
        metrics = {
            "reward": reward,
            "reward/unclipped": jp.where(
                task_failure, reward, reward_result.unclipped_total
            ),
            "reward/pre_episode_return_override": reward_result.total,
            "reward/unclipped_pre_episode_return_override": reward_result.unclipped_total,
            "reward/episode_return_override": episode_return_override,
            **{
                f"reward/{key}": value
                for key, value in reward_result.components.as_dict().items()
            },
            **self._state_metrics(reward_state, geometry, events, reset_source),
            "terminal/physical_failure": terminal.physical_failure.astype(jp.float32),
            "terminal/roll_limit": terminal.roll_limit.astype(jp.float32),
            "terminal/pitch_limit": terminal.pitch_limit.astype(jp.float32),
            "terminal/jump_zone_missed": terminal.jump_zone_missed.astype(jp.float32),
            "terminal/stuck": terminal.stuck.astype(jp.float32),
            "terminal/yaw_limit": terminal.yaw_limit.astype(jp.float32),
            "terminal/timeout": terminal.timeout.astype(jp.float32),
            "terminal/success": terminal.success.astype(jp.float32),
        }
        info = {
            **state.info,
            "rng": state.info["rng"],
            "history": history,
            "events": events,
            "last_action": normalized_action,
            "reward_state": reward_state,
            "reset_source_airborne_rsi": reset_source,
            "terminated": terminal.terminated,
            "truncated": terminal.truncated,
            "time_out": terminal.truncated.astype(jp.float32),
            "end_code": terminal.end_code,
            "success": terminal.success,
            "physical_failure": terminal.physical_failure,
            "roll_limit": terminal.roll_limit,
            "pitch_limit": terminal.pitch_limit,
            "jump_zone_missed": terminal.jump_zone_missed,
            "stuck": terminal.stuck,
            "yaw_limit": terminal.yaw_limit,
            "timeout": terminal.timeout,
            "episode_step": events.episode_step,
            "episode_return": episode_return,
        }
        return mjx_env.State(
            data=data,
            obs={"state": actor_obs, "privileged_state": critic_obs},
            reward=reward,
            done=(terminal.terminated | terminal.truncated).astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def _step_descent(self, state, action):
        model=self._require_runtime_model(); index=self._bundle.model_index; a=jp.clip(jp.asarray(action,jp.float32),-1.,1.); ctrl=map_action(a,state.data.qpos[index.knee_qpos_address],self._bundle.action_mapping); data=mjx_env.step(model,state.data,ctrl,self.n_substeps); geometry=extract_geometry(data,self._geometry); rs=self._reward_state(data,geometry); prev=state.info["events"]
        signals=DescentSignals(data.qpos[index.root_qpos_address],geometry.front_wheel_terrain_clearance,geometry.rear_wheel_terrain_clearance,geometry.maximum_wheel_penetration,geometry.prohibited_contact,jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all() & jp.isfinite(a).all(),geometry.roll,geometry.pitch,data.qpos[index.root_qpos_address] < state.info["reward_state"].x-self._resolved_config.physical_limits.max_backward_distance)
        events=advance_descent_events(prev,signals,self._resolved_config.descent); terminal=classify_descent_terminal(signals,events,self._resolved_config.descent,self._resolved_config.physical_limits, state.info["episode_step"]+1,self._resolved_config.ppo.episode_horizon); rr=descent_recovery_reward(DescentRewardInputs(rs.x-state.info["reward_state"].x,events.valid_contact_seen,prev.valid_contact_seen,events.valid_contact_seen,events.recovery_success,prev.recovery_success,signals.body_contact,terminal.physical_failure,terminal.truncated),self._resolved_config.descent); og=self._observable_geometry(data,geometry); frame=observable_frame(data,index,og,a,jp.asarray(True)); history=advance_history(state.info["history"],frame); actor=actor_observation(history,jp.asarray(False)); critic=privileged_observation(data,actor,og); info={**state.info,"history":history,"events":events,"last_action":a,"reward_state":rs,"terminated":terminal.terminated,"truncated":terminal.truncated,"time_out":terminal.truncated.astype(jp.float32),"end_code":terminal.end_code,"success":terminal.success,"physical_failure":terminal.physical_failure,"timeout":terminal.timeout,"episode_step":state.info["episode_step"]+1,"episode_return":state.info["episode_return"]+rr.total}; metrics={**self._zero_metrics(),"reward":rr.total,"reward/descent_forward_progress":rr.components.forward_progress,"reward/descent_contact":rr.components.contact,"reward/descent_recovery_tick":rr.components.recovery_tick,"reward/descent_success":rr.components.success,"reward/descent_bad_contact":rr.components.bad_contact,"reward/descent_failure":rr.components.failure,"reward/descent_timeout":rr.components.timeout,"event/descent_airborne_seen":events.airborne_seen.astype(jp.float32),"event/descent_valid_contact_seen":events.valid_contact_seen.astype(jp.float32),"event/descent_post_contact_ticks":events.post_contact_ticks.astype(jp.float32),"terminal/descent_success":terminal.success.astype(jp.float32),"terminal/descent_physical_failure":terminal.physical_failure.astype(jp.float32),"terminal/descent_timeout":terminal.truncated.astype(jp.float32)}
        return mjx_env.State(data=data,obs={"state":actor,"privileged_state":critic},reward=rr.total,done=(terminal.terminated|terminal.truncated).astype(jp.float32),metrics=metrics,info=info)
