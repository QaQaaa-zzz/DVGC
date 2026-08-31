"""Single-policy two-phase environment driven by learned Soft-Tube resets."""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .action_mapping import map_action
from .config import ResolvedConfig
from .descent_rewards import DescentRewardInputs, descent_recovery_reward
from .descent_semantics import (
    DescentSignals,
    advance_descent_events,
    classify_descent_terminal,
    initial_descent_events,
)
from .env import TwoPhaseBikeEnv
from .geometry import extract_geometry
from .handoff_snapshot import compatibility_identity
from .observation import (
    HistoryState,
    actor_observation,
    advance_history,
    observable_frame,
    privileged_observation,
)
from .rewards import RewardInputs, phase_u_reward
from .semantics import (
    EventState,
    PhaseUSignals,
    TerminalInputs,
    advance_events,
    classify_terminal,
)
from .soft_tube import SoftTubeArtifact
from .tube_rsi import PHASE_DOWNSTREAM, PHASE_UPSTREAM, TubeRSIPool


DESCENT_REWARD_KEYS = (
    "roll_posture",
    "pitch_posture",
    "roll_rate",
    "pitch_rate",
    "action_smoothness",
    "forward_progress",
    "contact",
    "recovery_tick",
    "success",
    "bad_contact",
    "failure",
    "timeout",
)


def next_training_phase(
    active_phase: jax.Array | int,
    previous_apex_seen: jax.Array | bool,
    current_apex_seen: jax.Array | bool,
    physical_failure: jax.Array | bool,
) -> jax.Array:
    """Transition training semantics at first valid Apex; never select a policy."""
    active = jp.asarray(active_phase, dtype=jp.int32)
    first_apex = ~jp.asarray(previous_apex_seen, dtype=bool) & jp.asarray(
        current_apex_seen, dtype=bool
    )
    transition = (
        (active == PHASE_UPSTREAM)
        & first_apex
        & ~jp.asarray(physical_failure, dtype=bool)
    )
    return jp.where(
        transition, jp.asarray(PHASE_DOWNSTREAM, jp.int32), active
    )


class UnifiedTubeRSIEnv(TwoPhaseBikeEnv):
    """One Actor runtime with phase semantics and configurable reset coverage."""

    def __init__(
        self,
        up_config: ResolvedConfig,
        down_config: ResolvedConfig,
        artifact: SoftTubeArtifact,
        *,
        runtime_naccdmax: int | None = None,
        natural_reset_probability: float = 0.0,
    ):
        if up_config.phase != "propulsion_ascent":
            raise ValueError("unified environment requires a propulsion_ascent config")
        if down_config.phase != "descent_recovery" or down_config.descent is None:
            raise ValueError("unified environment requires a descent_recovery config")
        if up_config.model["xml_sha256"] != down_config.model["xml_sha256"]:
            raise ValueError("unified environment XML identity mismatch")
        if up_config.action != down_config.action:
            raise ValueError("unified environment action mapping mismatch")
        for field in ("max_abs_roll", "max_abs_pitch", "max_backward_distance"):
            if getattr(up_config.physical_limits, field) != getattr(
                down_config.physical_limits, field
            ):
                raise ValueError(f"unified environment shared {field} mismatch")
        configured_naccdmax = int(up_config.model["naccdmax"])
        self._runtime_naccdmax = (
            configured_naccdmax
            if runtime_naccdmax is None
            else int(runtime_naccdmax)
        )
        if self._runtime_naccdmax < configured_naccdmax:
            raise ValueError("unified runtime naccdmax cannot reduce source capacity")
        if self._runtime_naccdmax > int(up_config.model["naconmax"]):
            raise ValueError("unified runtime naccdmax must not exceed naconmax")
        probability = float(natural_reset_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("unified natural reset probability must be in [0, 1]")
        self._natural_reset_probability = probability
        super().__init__(up_config)
        if self._bundle.xml_sha256 != down_config.model["xml_sha256"]:
            raise ValueError("unified environment runtime XML identity mismatch")
        self._down_config = down_config
        self._tube_pool = TubeRSIPool.from_artifact(
            artifact, compatibility=compatibility_identity(self)
        )

    @property
    def tube_pool(self) -> TubeRSIPool:
        return self._tube_pool

    @property
    def natural_reset_probability(self) -> float:
        return self._natural_reset_probability

    def _reset_data_naccdmax(self) -> int | None:
        """Use one runtime CCD capacity for every unified reset source."""
        return self._runtime_naccdmax

    def _unified_zero_metrics(self) -> dict[str, jax.Array]:
        metrics = self._zero_metrics()
        zero = jp.asarray(0.0, jp.float32)
        for key in DESCENT_REWARD_KEYS:
            metrics[f"reward/descent_{key}"] = zero
        for key in (
            "event/descent_airborne_seen",
            "event/descent_valid_contact_seen",
            "event/descent_post_contact_ticks",
            "terminal/descent_success",
            "terminal/descent_physical_failure",
            "terminal/descent_timeout",
            "reset/source_soft_tube",
            "reset/source_natural",
            "reset/tube_phase_upstream",
            "reset/tube_phase_downstream",
            "event/tube_phase_transition",
            "state/active_phase",
        ):
            metrics[key] = zero
        return metrics

    @staticmethod
    def _with_reset_source(state: mjx_env.State, *, soft_tube) -> mjx_env.State:
        soft = jp.asarray(soft_tube, dtype=bool)
        natural = ~soft
        start_phase = jp.asarray(state.info["start_phase"], dtype=jp.int32)
        info = {**state.info, "reset_from_soft_tube": soft}
        metrics = {
            **state.metrics,
            "reset/source_soft_tube": soft.astype(jp.float32),
            "reset/source_natural": natural.astype(jp.float32),
            "reset/tube_phase_upstream": (
                soft & (start_phase == PHASE_UPSTREAM)
            ).astype(jp.float32),
            "reset/tube_phase_downstream": (
                soft & (start_phase == PHASE_DOWNSTREAM)
            ).astype(jp.float32),
        }
        return state.replace(info=info, metrics=metrics)

    def _reset_tube(self, rng: jax.Array) -> mjx_env.State:
        state = self._reset_from_tube_sample(self._tube_pool.sample(rng))
        return self._with_reset_source(state, soft_tube=True)

    def _reset_natural_unified(self, rng: jax.Array) -> mjx_env.State:
        """Reuse the Phase-U natural reset, adapting only unified metadata."""
        phase_state = TwoPhaseBikeEnv.reset_natural(self, rng)
        up_events = phase_state.info["events"]
        active_phase = jp.asarray(PHASE_UPSTREAM, jp.int32)
        false = jp.asarray(False)
        root_x = phase_state.data.qpos[self._bundle.model_index.root_qpos_address]
        info = {
            "rng": phase_state.info["rng"],
            "history": phase_state.info["history"],
            "up_events": up_events,
            "down_events": initial_descent_events(root_x),
            "active_phase": active_phase,
            "start_phase": active_phase,
            "phase_transitioned": false,
            "expert_switching_used": false,
            "last_action": phase_state.info["last_action"],
            "reward_state": phase_state.info["reward_state"],
            "episode_step": phase_state.info["episode_step"],
            "phase_episode_step": jp.asarray(0, jp.int32),
            "source_tick": jp.asarray(-1, jp.int32),
            "parent_group_index": jp.asarray(-1, jp.int32),
            "tube_entry_index": jp.asarray(-1, jp.int32),
            "tube_global_index": jp.asarray(-1, jp.int32),
            "terminated": phase_state.info["terminated"],
            "truncated": phase_state.info["truncated"],
            "time_out": phase_state.info["time_out"],
            "end_code": phase_state.info["end_code"],
            "success": phase_state.info["success"],
            "physical_failure": phase_state.info["physical_failure"],
            "roll_limit": phase_state.info["roll_limit"],
            "pitch_limit": phase_state.info["pitch_limit"],
            "jump_zone_missed": phase_state.info["jump_zone_missed"],
            "stuck": phase_state.info["stuck"],
            "yaw_limit": phase_state.info["yaw_limit"],
            "timeout": phase_state.info["timeout"],
            "episode_return": phase_state.info["episode_return"],
        }
        metrics = self._unified_zero_metrics()
        for key, value in phase_state.metrics.items():
            if key in metrics:
                metrics[key] = value
        metrics.update(
            {
                "reset/source_soft_tube": jp.asarray(0.0, jp.float32),
                "reset/source_natural": jp.asarray(1.0, jp.float32),
                "reset/tube_phase_upstream": jp.asarray(0.0, jp.float32),
                "reset/tube_phase_downstream": jp.asarray(0.0, jp.float32),
                "event/tube_phase_transition": jp.asarray(0.0, jp.float32),
                "state/active_phase": active_phase.astype(jp.float32),
            }
        )
        return self._with_reset_source(
            phase_state.replace(info=info, metrics=metrics), soft_tube=False
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self._natural_reset_probability == 0.0:
            return self._reset_tube(rng)
        decision_key, reset_key = jax.random.split(rng)
        use_natural = jax.random.bernoulli(
            decision_key, self._natural_reset_probability
        )
        return jax.lax.cond(
            use_natural,
            self._reset_natural_unified,
            self._reset_tube,
            reset_key,
        )

    def reset_tube_index(
        self, phase_index: jax.Array | int, entry_index: jax.Array | int
    ) -> mjx_env.State:
        state = self._reset_from_tube_sample(
            self._tube_pool.sample_at(phase_index, entry_index)
        )
        return self._with_reset_source(state, soft_tube=True)

    def _reset_from_tube_sample(self, sample) -> mjx_env.State:
        model = self._require_runtime_model()
        data = mjx_env.make_data(
            self.mj_model,
            qpos=sample["qpos"],
            qvel=sample["qvel"],
            ctrl=sample["ctrl"],
            impl=model.impl.value,
            naconmax=int(self._resolved_config.model["naconmax"]),
            naccdmax=self._reset_data_naccdmax(),
            njmax=int(self._resolved_config.model["njmax"]),
        )
        data = mjx.forward(model, data)
        geometry = extract_geometry(data, self._geometry)
        observable_geometry = self._observable_geometry(data, geometry)
        history = HistoryState(
            frames=sample["observation_fifo"],
            valid_count=sample["history_valid_count"],
        )
        up_events = EventState(
            **{name: jp.asarray(value) for name, value in sample["events"].items()}
        ).replace(episode_step=jp.asarray(0, jp.int32))
        down_events = initial_descent_events(
            data.qpos[self._bundle.model_index.root_qpos_address]
        )
        active_phase = jp.asarray(sample["tube_phase"], dtype=jp.int32)
        task_signal = jp.where(
            active_phase == PHASE_UPSTREAM,
            up_events.jump_signal,
            jp.asarray(False),
        )
        actor_obs = actor_observation(history, task_signal)
        critic_obs = privileged_observation(
            data, actor_obs, observable_geometry
        )
        reward_state = self._reward_state(data, geometry)
        false = jp.asarray(False)
        info: dict[str, Any] = {
            "rng": sample["rng"],
            "history": history,
            "up_events": up_events,
            "down_events": down_events,
            "active_phase": active_phase,
            "start_phase": active_phase,
            "phase_transitioned": false,
            "expert_switching_used": false,
            "last_action": sample["last_action"],
            "reward_state": reward_state,
            "episode_step": jp.asarray(0, jp.int32),
            "phase_episode_step": jp.asarray(0, jp.int32),
            "source_tick": sample["tick"],
            "parent_group_index": sample["parent_group_index"],
            "tube_entry_index": sample["tube_entry_index"],
            "tube_global_index": sample["tube_global_index"],
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
            "episode_return": jp.asarray(0.0, jp.float32),
        }
        metrics = self._unified_zero_metrics()
        metrics.update(
            self._state_metrics(reward_state, geometry, up_events, false)
        )
        metrics.update(
            {
                "reset/source_soft_tube": jp.asarray(1.0, jp.float32),
                "reset/source_natural": jp.asarray(0.0, jp.float32),
                "reset/tube_phase_upstream": (
                    active_phase == PHASE_UPSTREAM
                ).astype(jp.float32),
                "reset/tube_phase_downstream": (
                    active_phase == PHASE_DOWNSTREAM
                ).astype(jp.float32),
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
        reward_state = self._reward_state(data, geometry)
        finite = (
            jp.isfinite(data.qpos).all()
            & jp.isfinite(data.qvel).all()
            & jp.isfinite(normalized_action).all()
        )
        active_up = state.info["active_phase"] == PHASE_UPSTREAM

        previous_up = state.info["up_events"]
        up_backward = (
            data.qpos[index.root_qpos_address]
            < self._initial_x
            - self._resolved_config.physical_limits.max_backward_distance
        )
        preliminary_inputs = TerminalInputs(
            episode_step=state.info["phase_episode_step"],
            nonfinite=~finite,
            roll=geometry.roll,
            pitch=geometry.pitch,
            illegal_contact=geometry.prohibited_contact,
            backward_exit=up_backward,
            stuck=previous_up.stuck,
            yaw=geometry.yaw,
            jump_zone_seen=previous_up.jump_zone_seen,
        )
        preliminary = classify_terminal(preliminary_inputs, self._resolved_config)
        up_events = advance_events(
            previous_up,
            PhaseUSignals(
                x=data.qpos[index.root_qpos_address],
                z=data.qpos[index.root_qpos_address + 2],
                forward_velocity=data.qvel[index.root_dof_address],
                vertical_velocity=data.qvel[index.root_dof_address + 2],
                physical_failure=preliminary.terminated,
            ),
            self._resolved_config,
        )
        up_terminal = classify_terminal(
            preliminary_inputs.replace(
                stuck=up_events.stuck,
                jump_zone_seen=up_events.jump_zone_seen,
            ),
            self._resolved_config,
        )
        first_apex = up_events.apex_seen & ~previous_up.apex_seen
        up_reward_result = phase_u_reward(
            RewardInputs(
                current=reward_state,
                action=normalized_action,
                last_action=state.info["last_action"],
                jump_signal=up_events.jump_signal,
                first_apex_success=first_apex,
                illegal_contact=geometry.prohibited_contact,
                physical_failure_transition=up_terminal.physical_failure
                & ~state.info["physical_failure"],
                roll_pitch_failure_transition=(
                    up_terminal.roll_limit | up_terminal.pitch_limit
                )
                & ~(state.info["roll_limit"] | state.info["pitch_limit"]),
                jump_zone_missed_transition=up_terminal.jump_zone_missed
                & ~state.info["jump_zone_missed"],
                stuck_transition=up_terminal.stuck & ~state.info["stuck"],
                yaw_limit_transition=up_terminal.yaw_limit
                & ~state.info["yaw_limit"],
                timeout_transition=up_terminal.timeout & ~state.info["timeout"],
            ),
            self._resolved_config.reward,
            self._resolved_config.physical_limits,
        )
        up_task_failure = up_terminal.stuck | up_terminal.yaw_limit
        failed_return = self._resolved_config.reward.failed_episode_return
        up_reward = up_reward_result.total
        if failed_return is not None:
            up_reward = jp.where(
                up_task_failure,
                jp.asarray(failed_return, jp.float32)
                - state.info["episode_return"],
                up_reward,
            )

        previous_down = state.info["down_events"]
        down_signals = DescentSignals(
            x=data.qpos[index.root_qpos_address],
            front_clearance=geometry.front_wheel_terrain_clearance,
            rear_clearance=geometry.rear_wheel_terrain_clearance,
            maximum_wheel_penetration=geometry.maximum_wheel_penetration,
            body_contact=geometry.prohibited_contact,
            finite=finite,
            roll=geometry.roll,
            pitch=geometry.pitch,
            backward_exit=(
                data.qpos[index.root_qpos_address]
                < state.info["reward_state"].x
                - self._down_config.physical_limits.max_backward_distance
            ),
        )
        advanced_down = advance_descent_events(
            previous_down, down_signals, self._down_config.descent
        )
        down_terminal = classify_descent_terminal(
            down_signals,
            advanced_down,
            self._down_config.descent,
            self._down_config.physical_limits,
            state.info["phase_episode_step"] + 1,
            self._down_config.ppo.episode_horizon,
        )
        down_reward_result = descent_recovery_reward(
            DescentRewardInputs(
                x_delta=reward_state.x - state.info["reward_state"].x,
                valid_contact=advanced_down.valid_contact_seen,
                previous_valid_contact=previous_down.valid_contact_seen,
                post_contact=advanced_down.valid_contact_seen,
                recovery_success=advanced_down.recovery_success,
                previous_recovery_success=previous_down.recovery_success,
                bad_contact=down_signals.body_contact,
                physical_failure=down_terminal.physical_failure,
                timeout=down_terminal.truncated,
                roll=geometry.roll,
                pitch=geometry.pitch,
                roll_rate=reward_state.roll_rate,
                pitch_rate=reward_state.pitch_rate,
                action=normalized_action,
                last_action=state.info["last_action"],
            ),
            self._down_config.descent,
        )

        next_phase = next_training_phase(
            state.info["active_phase"],
            previous_up.apex_seen,
            up_events.apex_seen,
            up_terminal.physical_failure,
        )
        transitioned = active_up & (next_phase == PHASE_DOWNSTREAM)
        phase_episode_step = jp.where(
            transitioned,
            jp.asarray(0, jp.int32),
            state.info["phase_episode_step"] + 1,
        )
        down_events = jax.tree.map(
            lambda retained, advanced, initial: jp.where(
                active_up,
                jp.where(transitioned, initial, retained),
                advanced,
            ),
            previous_down,
            advanced_down,
            initial_descent_events(data.qpos[index.root_qpos_address]),
        )
        reward = jp.where(active_up, up_reward, down_reward_result.total)
        terminated = jp.where(
            active_up, up_terminal.terminated, down_terminal.terminated
        )
        truncated = jp.where(
            active_up, up_terminal.truncated, down_terminal.truncated
        )
        success = jp.where(active_up, up_terminal.success, down_terminal.success)
        physical_failure = jp.where(
            active_up, up_terminal.physical_failure, down_terminal.physical_failure
        )
        timeout = jp.where(active_up, up_terminal.timeout, down_terminal.timeout)
        end_code = jp.where(active_up, up_terminal.end_code, down_terminal.end_code)
        roll_limit = jp.abs(geometry.roll) > self._resolved_config.physical_limits.max_abs_roll
        pitch_limit = jp.abs(geometry.pitch) > self._resolved_config.physical_limits.max_abs_pitch

        observable_geometry = self._observable_geometry(data, geometry)
        frame = observable_frame(
            data,
            index,
            observable_geometry,
            normalized_action,
            jp.asarray(True),
        )
        history = advance_history(state.info["history"], frame)
        task_signal = jp.where(
            next_phase == PHASE_UPSTREAM,
            up_events.jump_signal,
            jp.asarray(False),
        )
        actor_obs = actor_observation(history, task_signal)
        critic_obs = privileged_observation(
            data, actor_obs, observable_geometry
        )
        episode_return = state.info["episode_return"] + reward
        info = {
            **state.info,
            "history": history,
            "up_events": up_events,
            "down_events": down_events,
            "active_phase": next_phase,
            "phase_transitioned": state.info["phase_transitioned"] | transitioned,
            "expert_switching_used": jp.asarray(False),
            "last_action": normalized_action,
            "reward_state": reward_state,
            "terminated": terminated,
            "truncated": truncated,
            "time_out": truncated.astype(jp.float32),
            "end_code": end_code,
            "success": success,
            "physical_failure": physical_failure,
            "roll_limit": roll_limit,
            "pitch_limit": pitch_limit,
            "jump_zone_missed": jp.where(
                active_up, up_terminal.jump_zone_missed, jp.asarray(False)
            ),
            "stuck": jp.where(active_up, up_terminal.stuck, jp.asarray(False)),
            "yaw_limit": jp.where(
                active_up, up_terminal.yaw_limit, jp.asarray(False)
            ),
            "timeout": timeout,
            "episode_step": state.info["episode_step"] + 1,
            "phase_episode_step": phase_episode_step,
            "episode_return": episode_return,
        }
        metrics = self._unified_zero_metrics()
        metrics.update(
            self._state_metrics(
                reward_state, geometry, up_events, jp.asarray(False)
            )
        )
        soft_reset = jp.asarray(state.info["reset_from_soft_tube"], dtype=bool)
        natural_reset = ~soft_reset
        start_phase = jp.asarray(state.info["start_phase"], dtype=jp.int32)
        metrics.update(
            {
                "reward": reward,
                "reward/unclipped": jp.where(
                    active_up,
                    up_reward_result.unclipped_total,
                    down_reward_result.total,
                ),
                "reward/pre_episode_return_override": jp.where(
                    active_up, up_reward_result.total, down_reward_result.total
                ),
                "reward/unclipped_pre_episode_return_override": jp.where(
                    active_up,
                    up_reward_result.unclipped_total,
                    down_reward_result.total,
                ),
                "reward/episode_return_override": jp.where(
                    active_up, up_reward - up_reward_result.total, jp.asarray(0.0)
                ),
                **{
                    f"reward/{key}": jp.where(
                        active_up,
                        value,
                        jp.asarray(0.0, jp.float32),
                    )
                    for key, value in up_reward_result.components.as_dict().items()
                },
                **{
                    f"reward/descent_{key}": jp.where(
                        active_up,
                        jp.asarray(0.0, jp.float32),
                        value,
                    )
                    for key, value in down_reward_result.components.__dict__.items()
                },
                "event/descent_airborne_seen": advanced_down.airborne_seen.astype(
                    jp.float32
                ),
                "event/descent_valid_contact_seen": advanced_down.valid_contact_seen.astype(
                    jp.float32
                ),
                "event/descent_post_contact_ticks": advanced_down.post_contact_ticks.astype(
                    jp.float32
                ),
                "terminal/physical_failure": physical_failure.astype(jp.float32),
                "terminal/roll_limit": roll_limit.astype(jp.float32),
                "terminal/pitch_limit": pitch_limit.astype(jp.float32),
                "terminal/jump_zone_missed": info["jump_zone_missed"].astype(
                    jp.float32
                ),
                "terminal/stuck": info["stuck"].astype(jp.float32),
                "terminal/yaw_limit": info["yaw_limit"].astype(jp.float32),
                "terminal/timeout": timeout.astype(jp.float32),
                "terminal/success": success.astype(jp.float32),
                "terminal/descent_success": jp.where(
                    active_up, jp.asarray(False), down_terminal.success
                ).astype(jp.float32),
                "terminal/descent_physical_failure": jp.where(
                    active_up, jp.asarray(False), down_terminal.physical_failure
                ).astype(jp.float32),
                "terminal/descent_timeout": jp.where(
                    active_up, jp.asarray(False), down_terminal.timeout
                ).astype(jp.float32),
                "reset/source_soft_tube": soft_reset.astype(jp.float32),
                "reset/source_natural": natural_reset.astype(jp.float32),
                "reset/tube_phase_upstream": (
                    soft_reset & (start_phase == PHASE_UPSTREAM)
                ).astype(jp.float32),
                "reset/tube_phase_downstream": (
                    soft_reset & (start_phase == PHASE_DOWNSTREAM)
                ).astype(jp.float32),
                "event/tube_phase_transition": transitioned.astype(jp.float32),
                "state/active_phase": next_phase.astype(jp.float32),
            }
        )
        return mjx_env.State(
            data=data,
            obs={"state": actor_obs, "privileged_state": critic_obs},
            reward=reward,
            done=(terminated | truncated).astype(jp.float32),
            metrics=metrics,
            info=info,
        )
