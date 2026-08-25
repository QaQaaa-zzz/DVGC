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


class TwoPhaseBikeEnv(mjx_env.MjxEnv):
    """The JIT Propulsion-Ascent training environment."""

    def __init__(self, config: ResolvedConfig, *, convert_model: bool = True):
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
            illegal_contact=signals.prohibited_contact | signals.illegal_wheel_contact,
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
        metrics = {"reward": zero, "reward/unclipped": zero}
        metrics.update({f"reward/{key}": zero for key in REWARD_COMPONENT_KEYS})
        for key in (
            "event/jump_signal",
            "event/jump_zone_seen",
            "event/jump_zone_consumed",
            "event/ascending_seen",
            "event/height_seen",
            "event/apex_seen",
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
            "signal/hip_velocity",
            "signal/knee_velocity",
            "signal/hip_force",
            "signal/knee_force",
            "signal/joint_power",
            "terminal/physical_failure",
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
        decision_key, state_key = jax.random.split(rng)
        use_airborne_rsi = jax.random.bernoulli(
            decision_key, self._resolved_config.reset.airborne_rsi_probability
        )
        return self._reset(state_key, use_airborne_rsi)

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
            "timeout": false,
            "episode_step": jp.asarray(0, jp.int32),
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
        preliminary_inputs = TerminalInputs(
            episode_step=state.info["events"].episode_step,
            nonfinite=nonfinite,
            roll=geometry.roll,
            pitch=geometry.pitch,
            illegal_contact=geometry.prohibited_contact,
            illegal_wheel_contact=geometry.illegal_wheel_contact,
            backward_exit=backward_exit,
            apex_success=jp.asarray(False),
        )
        preliminary = classify_terminal(preliminary_inputs, self._resolved_config)
        previous_events = state.info["events"]
        events = advance_events(
            previous_events,
            PhaseUSignals(
                x=data.qpos[index.root_qpos_address],
                z=data.qpos[index.root_qpos_address + 2],
                vertical_velocity=data.qvel[index.root_dof_address + 2],
                physical_failure=preliminary.physical_failure,
            ),
            self._resolved_config,
        )
        first_apex = events.apex_seen & ~previous_events.apex_seen
        terminal: TerminalState = classify_terminal(
            preliminary_inputs.replace(apex_success=first_apex),
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
                illegal_contact=geometry.prohibited_contact
                | geometry.illegal_wheel_contact,
                physical_failure_transition=terminal.physical_failure
                & ~state.info["physical_failure"],
                timeout_transition=terminal.timeout & ~state.info["timeout"],
            ),
            self._resolved_config.reward,
            self._resolved_config.physical_limits,
        )
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
            "reward": reward_result.total,
            "reward/unclipped": reward_result.unclipped_total,
            **{
                f"reward/{key}": value
                for key, value in reward_result.components.as_dict().items()
            },
            **self._state_metrics(reward_state, geometry, events, reset_source),
            "terminal/physical_failure": terminal.physical_failure.astype(jp.float32),
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
            "timeout": terminal.timeout,
            "episode_step": events.episode_step,
        }
        return mjx_env.State(
            data=data,
            obs={"state": actor_obs, "privileged_state": critic_obs},
            reward=reward_result.total,
            done=(terminal.terminated | terminal.truncated).astype(jp.float32),
            metrics=metrics,
            info=info,
        )
