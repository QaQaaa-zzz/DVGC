"""Minimal independent MJX-Warp environment for Propulsion-Ascent."""

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
from .geometry import (
    GeometrySignals,
    build_geometry_contract,
    extract_geometry,
)
from .model import ModelBundle, load_host_model, put_warp_model
from .observation import (
    ObservableGeometry,
    actor_observation,
    advance_history,
    initial_history,
    observable_frame,
    privileged_observation,
)
from .rewards import RewardInputs, RewardState, phase_u_reward
from .semantics import (
    ApexSignals,
    TerminalInputs,
    advance_events,
    classify_terminal,
    initial_event_state,
)


class TwoPhaseBikeEnv(mjx_env.MjxEnv):
    """The first-delivery Propulsion-Ascent training environment."""

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

    def _observable_geometry(self, signals: GeometrySignals) -> ObservableGeometry:
        return ObservableGeometry(
            obstacle_relative_x=signals.obstacle_relative_x,
            structure_clearance=signals.structure_clearance,
            front_wheel_support=signals.front_wheel_support,
            rear_wheel_support=signals.rear_wheel_support,
            roll=signals.roll,
            pitch=signals.pitch,
            yaw=signals.yaw,
            illegal_contact=signals.prohibited_contact | signals.illegal_wheel_contact,
        )

    @staticmethod
    def _reward_state(data, signals: GeometrySignals) -> RewardState:
        return RewardState(
            x=data.qpos[0],
            z=data.qpos[2],
            clearance=signals.structure_clearance,
            roll=signals.roll,
            pitch=signals.pitch,
            angular_speed=signals.angular_speed,
            vertical_velocity=signals.vertical_velocity,
            forward_velocity=signals.forward_velocity,
            obstacle_relative_x=signals.obstacle_relative_x,
        )

    @staticmethod
    def _zero_metrics() -> dict[str, jax.Array]:
        zero = jp.asarray(0.0, jp.float32)
        metrics = {"reward": zero}
        metrics.update({f"reward/{key}": zero for key in REWARD_COMPONENT_KEYS})
        metrics.update(
            {
                "event/window_latched": zero,
                "event/liftoff_seen": zero,
                "event/stable_airborne_seen": zero,
                "event/ascending_seen": zero,
                "event/apex_seen": zero,
                "signal/obstacle_relative_x": zero,
                "signal/structure_clearance": zero,
                "signal/roll": zero,
                "signal/pitch": zero,
                "signal/angular_speed": zero,
                "terminal/physical_failure": zero,
                "terminal/timeout": zero,
                "terminal/success": zero,
            }
        )
        return metrics

    def reset(self, rng: jax.Array) -> mjx_env.State:
        model = self._require_runtime_model()
        index = self._bundle.model_index
        qpos = jp.asarray(
            self.mj_model.key_qpos[index.keyframe_id], dtype=jp.float32
        )
        qvel = jp.asarray(
            self.mj_model.key_qvel[index.keyframe_id], dtype=jp.float32
        ).at[index.root_dof_address].set(
            float(self._resolved_config.reset["initial_forward_velocity"])
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
        observable_geometry = self._observable_geometry(geometry)
        history = initial_history()
        actor_obs = actor_observation(history)
        critic_obs = privileged_observation(data, actor_obs, observable_geometry)
        events = initial_event_state()
        reward_state = self._reward_state(data, geometry)
        false = jp.asarray(False)
        info: dict[str, Any] = {
            "rng": rng,
            "history": history,
            "events": events,
            "last_action": last_action,
            "reward_state": reward_state,
            "terminated": false,
            "truncated": false,
            "time_out": jp.asarray(0.0, jp.float32),
            "end_code": jp.asarray(END_ONGOING, jp.int32),
            "success": false,
            "physical_failure": false,
            "timeout": false,
            "episode_step": jp.asarray(0, jp.int32),
        }
        return mjx_env.State(
            data=data,
            obs={"state": actor_obs, "privileged_state": critic_obs},
            reward=jp.asarray(0.0, jp.float32),
            done=jp.asarray(0.0, jp.float32),
            metrics=self._zero_metrics(),
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
            < self._initial_x - self._resolved_config.physical_limits.max_backward_distance
        )
        platform_overrun = (
            data.qpos[index.root_qpos_address]
            > self._geometry.obstacle_back_x
            + self._resolved_config.physical_limits.platform_back_margin
        )
        preliminary_inputs = TerminalInputs(
            episode_step=state.info["events"].episode_step,
            nonfinite=nonfinite,
            roll=geometry.roll,
            pitch=geometry.pitch,
            illegal_contact=geometry.prohibited_contact,
            illegal_wheel_contact=geometry.illegal_wheel_contact,
            backward_exit=backward_exit,
            platform_overrun=platform_overrun,
            apex_success=jp.asarray(False),
        )
        preliminary = classify_terminal(preliminary_inputs, self._resolved_config)
        apex_signals = ApexSignals(
            stable_airborne=jp.asarray(False),
            vertical_velocity=geometry.vertical_velocity,
            clearance=geometry.structure_clearance,
            roll=geometry.roll,
            pitch=geometry.pitch,
            angular_speed=geometry.angular_speed,
            forward_velocity=geometry.forward_velocity,
            obstacle_relative_x=geometry.obstacle_relative_x,
            illegal_contact=geometry.prohibited_contact | geometry.illegal_wheel_contact,
            physical_failure=preliminary.physical_failure,
        )
        supported = geometry.front_wheel_support | geometry.rear_wheel_support
        previous_events = state.info["events"]
        events = advance_events(
            previous_events,
            apex_signals,
            self._resolved_config,
            supported=supported,
        )
        first_apex = events.apex_seen & ~previous_events.apex_seen
        terminal = classify_terminal(
            preliminary_inputs.replace(apex_success=first_apex),
            self._resolved_config,
        )
        current_reward_state = self._reward_state(data, geometry)
        reward_result = phase_u_reward(
            RewardInputs(
                previous=state.info["reward_state"],
                current=current_reward_state,
                action=normalized_action,
                last_action=state.info["last_action"],
                window_latched=events.window_latched,
                first_window_entry=events.window_latched
                & ~previous_events.window_latched,
                first_liftoff=events.liftoff_seen & ~previous_events.liftoff_seen,
                first_stable_airborne=events.stable_airborne_seen
                & ~previous_events.stable_airborne_seen,
                stable_airborne=events.stable_airborne_current,
                ascending_seen=events.ascending_seen,
                first_apex_success=first_apex,
                illegal_contact=geometry.prohibited_contact
                | geometry.illegal_wheel_contact,
                physical_failure_transition=terminal.physical_failure
                & ~state.info["physical_failure"],
                timeout_transition=terminal.timeout & ~state.info["timeout"],
            ),
            self._resolved_config.reward,
            self._resolved_config.apex,
            self._resolved_config.physical_limits,
        )
        observable_geometry = self._observable_geometry(geometry)
        frame = observable_frame(
            data,
            index,
            observable_geometry,
            normalized_action,
            jp.asarray(True),
        )
        history = advance_history(state.info["history"], frame)
        actor_obs = actor_observation(history)
        critic_obs = privileged_observation(data, actor_obs, observable_geometry)
        metrics = {
            "reward": reward_result.total,
            **{
                f"reward/{key}": value
                for key, value in reward_result.components.as_dict().items()
            },
            "event/window_latched": events.window_latched.astype(jp.float32),
            "event/liftoff_seen": events.liftoff_seen.astype(jp.float32),
            "event/stable_airborne_seen": events.stable_airborne_seen.astype(jp.float32),
            "event/ascending_seen": events.ascending_seen.astype(jp.float32),
            "event/apex_seen": events.apex_seen.astype(jp.float32),
            "signal/obstacle_relative_x": geometry.obstacle_relative_x,
            "signal/structure_clearance": geometry.structure_clearance,
            "signal/roll": geometry.roll,
            "signal/pitch": geometry.pitch,
            "signal/angular_speed": geometry.angular_speed,
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
            "reward_state": current_reward_state,
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
