"""Auditable Gate C1 contracts and runtime for phase-expert smoke runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import jax
from jax import numpy as jp
import numpy as np

from .config import (
    ACTION_MAPPING_VERSION,
    AUTHORITATIVE_XML_PATH,
    AUTHORITATIVE_XML_SHA256,
)
from .training_budget import PPOBudgetReport, build_ppo_budget_report
from .two_phase_guideline import canonical_manifest_hash
from .two_phase_runtime import (
    TwoPhaseEventState,
    TwoPhaseThresholds,
    advance_two_phase_events,
    extract_apex_band_signals,
    extract_recovery_signals,
    initial_two_phase_event_state,
)
from .two_phase_semantics import (
    ApexBandSignals,
    ApexBandThresholds,
    RecoveryThresholds,
)


PHASE_PROPULSION_ASCENT = "propulsion_ascent"
PHASE_DESCENT_RECOVERY = "descent_recovery"
PHASE_EXPERT_PHASES = (PHASE_PROPULSION_ASCENT, PHASE_DESCENT_RECOVERY)
_PHASE_U_PHYSICAL_FAILURE_END_CODES = (2, 3, 4, 5, 6, 7, 15)
_PHASE_U_POST_LATCH_TASK_FAILURE_END_CODES = (10, 11, 12, 13)
_THRESHOLD_SOURCE_HASHES = frozenset(
    {"xml", "reference", "config", "code", "geometry_manifest"}
)
_THRESHOLD_SOURCE_PATHS = frozenset({"xml", "reference", "config", "code"})
_DESCENT_SEED_TIERS = frozenset(
    {"physically_validated_descent_seed", "pi_up_online_apex_snapshot"}
)
_BASE_MODE = MappingProxyType(
    {
        "training_stage": "full",
        "use_bank_resets": False,
        "expert_chain_termination": False,
        "stage_reachability_objective": "",
        "domain_randomization": False,
    }
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PHASE_EXPERT_SOURCE_PATHS = (
    "configs/default.json",
    "configs/phase_expert_phase_u.json",
    "configs/phase_expert_smoke.json",
    "dvgc/config.py",
    "dvgc/bank.py",
    "dvgc/env.py",
    "dvgc/feasibility.py",
    "dvgc/observation_audit.py",
    "dvgc/phase_expert_training.py",
    "dvgc/phase_candidate_acquisition.py",
    "dvgc/rewards.py",
    "dvgc/rollout.py",
    "dvgc/runtime.py",
    "dvgc/snapshot_timing.py",
    "dvgc/training_budget.py",
    "dvgc/two_phase_guideline.py",
    "dvgc/two_phase_runtime.py",
    "dvgc/two_phase_semantics.py",
    "dvgc/wrappers.py",
    "cli/train_phase_expert.py",
)


@dataclass(frozen=True)
class PhaseExpertRunSpec:
    phase: str
    experiment_level: str
    requested_total_transitions: int
    seed: int
    config_path: str
    training_config_path: str
    threshold_manifest_path: str
    authorization_manifest_path: str | None
    output_dir: str
    descent_seed_bank: str | None
    descent_seed_manifest: str | None
    resume_run: str | None
    restore_checkpoint: str | None


@dataclass(frozen=True)
class PhaseExpertResetProtocol:
    phase: str
    mode: str
    seed_tier: str | None
    source_hash: str | None


@dataclass(frozen=True)
class ResolvedThresholdManifest:
    manifest: Mapping[str, Any]
    canonical_manifest_hash: str
    action_mapping_version: str
    reference_rollout_source: str
    apex_thresholds: ApexBandThresholds
    recovery_thresholds: RecoveryThresholds


@dataclass(frozen=True)
class PhaseExpertSeedNamespaces:
    training_namespace: str
    training_seeds: tuple[int, ...]
    evaluation_namespace: str
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class PhaseExpertInteractionBudget:
    training: PPOBudgetReport
    brax_evaluation_transition_ceiling: int
    fixed_evaluation_transition_ceiling: int
    combined_transition_ceiling: int
    candidate_acquisition_transition_ceiling: int
    continuation_labeling_transition_ceiling: int
    total_environment_transition_ceiling: int


@dataclass(frozen=True)
class PhaseCheckpointMilestone:
    requested: int
    effective: int


class PhaseCheckpointTracker:
    """Claim configured host-side checkpoint callbacks exactly once."""

    def __init__(self, milestones: tuple[PhaseCheckpointMilestone, ...]) -> None:
        self._by_effective = {item.effective: item for item in milestones}
        if len(self._by_effective) != len(milestones):
            raise ValueError("checkpoint milestones must have unique effective steps")
        self._claimed: set[int] = set()

    def claim(self, effective_step: int) -> PhaseCheckpointMilestone | None:
        milestone = self._by_effective.get(int(effective_step))
        if milestone is None or milestone.effective in self._claimed:
            return None
        self._claimed.add(milestone.effective)
        return milestone

    @property
    def claimed_effective(self) -> tuple[int, ...]:
        return tuple(sorted(self._claimed))


@dataclass(frozen=True)
class ValidatedPhaseExpertRunSpec:
    spec: PhaseExpertRunSpec
    thresholds: ResolvedThresholdManifest
    seeds: PhaseExpertSeedNamespaces
    interaction_budget: PhaseExpertInteractionBudget
    authorization: Mapping[str, Any] | None
    cumulative_training_start: int


@dataclass(frozen=True)
class PhaseURewardConfig:
    """Bounded Phase U reward terms with task progress gated by the window."""

    forward_propulsion_weight: float = 0.5
    jump_window_progress_weight: float = 2.0
    liftoff_bonus_weight: float = 0.0
    stable_airborne_bonus_weight: float = 0.0
    ascent_progress_weight: float = 4.0
    dual_wheel_lift_progress_weight: float = 0.0
    dual_wheel_lift_progress_target: float = 0.015
    clearance_progress_weight: float = 2.0
    apex_approach_weight: float = 2.0
    attitude_penalty_weight: float = 0.5
    angular_rate_penalty_weight: float = 0.25
    angular_rate_penalty_cap_ratio: float = 1.0
    illegal_contact_penalty_weight: float = 20.0
    action_smoothness_weight: float = 0.02
    action_magnitude_weight: float = 0.005
    success_bonus: float = 30.0
    physical_failure_penalty: float = 20.0
    task_failure_penalty: float = 20.0
    target_forward_velocity: float = 3.75
    target_vertical_velocity: float = 1.0
    clearance_floor: float = -0.30
    total_min: float = -50.0
    total_max: float = 50.0

    def __post_init__(self) -> None:
        for name in (
            "forward_propulsion_weight",
            "jump_window_progress_weight",
            "liftoff_bonus_weight",
            "stable_airborne_bonus_weight",
            "ascent_progress_weight",
            "dual_wheel_lift_progress_weight",
            "clearance_progress_weight",
            "apex_approach_weight",
            "attitude_penalty_weight",
            "angular_rate_penalty_weight",
            "illegal_contact_penalty_weight",
            "action_smoothness_weight",
            "action_magnitude_weight",
            "success_bonus",
            "physical_failure_penalty",
            "task_failure_penalty",
            "target_forward_velocity",
            "target_vertical_velocity",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.total_min) or not math.isfinite(self.total_max):
            raise ValueError("reward bounds must be finite")
        if not math.isfinite(self.clearance_floor):
            raise ValueError("clearance_floor must be finite")
        if (
            not math.isfinite(self.angular_rate_penalty_cap_ratio)
            or self.angular_rate_penalty_cap_ratio <= 0.0
        ):
            raise ValueError(
                "angular_rate_penalty_cap_ratio must be finite and positive"
            )
        if (
            not math.isfinite(self.dual_wheel_lift_progress_target)
            or self.dual_wheel_lift_progress_target <= 0.0
        ):
            raise ValueError(
                "dual_wheel_lift_progress_target must be finite and positive"
            )
        if self.total_min >= self.total_max:
            raise ValueError("total_min must be less than total_max")


for _jax_dataclass in (ApexBandThresholds, PhaseURewardConfig):
    try:
        jax.tree_util.register_dataclass(
            _jax_dataclass,
            data_fields=[],
            meta_fields=[field.name for field in dataclass_fields(_jax_dataclass)],
        )
    except ValueError:
        pass


def resolve_phase_u_reward_config(
    training_config: Mapping[str, Any],
) -> PhaseURewardConfig:
    payload = training_config.get("phase_u_reward")
    bounds = training_config.get("reward_bounds")
    if not isinstance(payload, Mapping) or not isinstance(bounds, Mapping):
        raise ValueError("phase_u_reward and reward_bounds mappings are required")
    expected = {
        field.name
        for field in dataclass_fields(PhaseURewardConfig)
        if field.name not in {"total_min", "total_max"}
    }
    if set(payload) != expected:
        raise ValueError(
            "phase_u_reward must define exactly the approved reward fields"
        )
    if set(bounds) != {"total_min", "total_max"}:
        raise ValueError("reward_bounds must define total_min and total_max")
    return PhaseURewardConfig(**dict(payload), **dict(bounds))


def phase_u_reward_contract_hash(training_config: Mapping[str, Any]) -> str:
    return _canonical_payload_hash(
        {
            "semantics": PHASE_U_REWARD_SEMANTICS,
            "config": asdict(resolve_phase_u_reward_config(training_config)),
        }
    )


def resolve_policy_initial_action_std(
    training_config: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    """Resolve the explicit steer/drive/hip/knee exploration prior."""
    value = training_config.get("policy_initial_action_std")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(
            "policy_initial_action_std must be a four-value action-order list"
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                "policy_initial_action_std must contain finite numeric values"
            )
        result.append(float(item))
    if any(
        not math.isfinite(item) or item <= 0.001 or item >= 1.0
        for item in result
    ):
        raise ValueError(
            "policy_initial_action_std values must be strictly between 0.001 and 1.0"
        )
    return (result[0], result[1], result[2], result[3])


_PHASE_U_REWARD_COMPONENTS = (
    "forward_propulsion",
    "jump_window_progress",
    "legal_liftoff_bonus",
    "stable_airborne_bonus",
    "ascent_progress",
    "dual_wheel_lift_progress",
    "clearance_progress",
    "apex_approach",
    "apex_success_bonus",
    "attitude_penalty",
    "angular_rate_penalty",
    "illegal_contact_penalty",
    "action_smoothness_penalty",
    "action_magnitude_penalty",
    "physical_failure_penalty",
    "task_failure_penalty",
)

PHASE_U_REWARD_SEMANTICS = "phase_u.rate_qualified_dual_wheel_lift_credit.v6"


def _interval_proximity(value: Any, lower: float, upper: float) -> Any:
    width = jp.maximum(jp.asarray(upper - lower), 1.0e-6)
    distance = jp.maximum(lower - value, jp.maximum(value - upper, 0.0))
    return jp.clip(1.0 - distance / width, 0.0, 1.0)


def phase_u_reward_components(
    signals: ApexBandSignals,
    thresholds: ApexBandThresholds,
    legal_window_entered: Any,
    airborne_progress_enabled: Any,
    window_entry_transition: Any,
    legal_liftoff_transition: Any,
    stable_airborne_transition: Any,
    apex_eligible: Any,
    success_transition: Any,
    physical_failure: Any,
    task_failure: Any,
    action: Any,
    last_action: Any,
    config: PhaseURewardConfig,
) -> dict[str, Any]:
    """Return bounded observable Phase U terms with monotonic legal-window gating."""
    window = jp.asarray(legal_window_entered, dtype=bool)
    airborne_progress = jp.asarray(airborne_progress_enabled, dtype=bool)
    forward_scale = jp.maximum(config.target_forward_velocity, 1.0e-6)
    vertical_scale = jp.maximum(config.target_vertical_velocity, 1.0e-6)
    forward = config.forward_propulsion_weight * jp.clip(
        jp.asarray(signals.forward_velocity) / forward_scale, 0.0, 1.0
    )
    ascent_rate_limit = jp.maximum(
        thresholds.max_angular_speed * config.angular_rate_penalty_cap_ratio,
        1.0e-6,
    )
    ascent_rate_quality = jp.clip(
        1.0 - jp.asarray(signals.angular_speed) / ascent_rate_limit,
        0.0,
        1.0,
    )
    ascent = config.ascent_progress_weight * jp.where(
        window,
        jp.clip(jp.asarray(signals.com_vz) / vertical_scale, 0.0, 1.0)
        * ascent_rate_quality,
        0.0,
    )
    dual_wheel_lift = config.dual_wheel_lift_progress_weight * jp.where(
        window,
        jp.clip(
            jp.asarray(signals.minimum_wheel_terrain_clearance)
            / config.dual_wheel_lift_progress_target,
            0.0,
            1.0,
        )
        * ascent_rate_quality,
        0.0,
    )
    clearance_denominator = jp.maximum(
        thresholds.min_clearance - config.clearance_floor, 1.0e-6
    )
    clearance = config.clearance_progress_weight * jp.where(
        airborne_progress,
        jp.clip(
            (jp.asarray(signals.clearance) - config.clearance_floor)
            / clearance_denominator,
            0.0,
            1.0,
        ),
        0.0,
    )
    apex_scores = jp.stack(
        [
            jp.clip(
                1.0
                - jp.abs(jp.asarray(signals.com_vz))
                / thresholds.max_abs_com_vz,
                0.0,
                1.0,
            ),
            jp.clip(
                jp.asarray(signals.clearance) / thresholds.min_clearance,
                0.0,
                1.0,
            ),
            jp.clip(1.0 - jp.abs(signals.roll) / thresholds.max_abs_roll, 0.0, 1.0),
            jp.clip(1.0 - jp.abs(signals.pitch) / thresholds.max_abs_pitch, 0.0, 1.0),
            jp.clip(
                1.0 - signals.angular_speed / thresholds.max_angular_speed,
                0.0,
                1.0,
            ),
            jp.clip(
                signals.forward_velocity / thresholds.min_forward_velocity,
                0.0,
                1.0,
            ),
            _interval_proximity(
                signals.obstacle_relative_x,
                thresholds.relative_x_min,
                thresholds.relative_x_max,
            ),
        ]
    )
    apex_approach = config.apex_approach_weight * jp.where(
        airborne_progress & jp.asarray(apex_eligible, dtype=bool),
        jp.mean(apex_scores),
        0.0,
    )
    attitude = -config.attitude_penalty_weight * jp.mean(
        jp.stack(
            [
                jp.clip(jp.abs(signals.roll) / thresholds.max_abs_roll, 0.0, 1.0),
                jp.clip(jp.abs(signals.pitch) / thresholds.max_abs_pitch, 0.0, 1.0),
            ]
        )
    )
    angular_rate = -config.angular_rate_penalty_weight * jp.clip(
        signals.angular_speed / thresholds.max_angular_speed,
        0.0,
        config.angular_rate_penalty_cap_ratio,
    )
    smoothness = -config.action_smoothness_weight * jp.mean(
        jp.square(jp.asarray(action) - jp.asarray(last_action))
    )
    magnitude = -config.action_magnitude_weight * jp.mean(
        jp.square(jp.clip(jp.asarray(action), -1.0, 1.0))
    )
    return {
        "forward_propulsion": forward,
        "jump_window_progress": config.jump_window_progress_weight
        * jp.asarray(window_entry_transition, jp.float32),
        "legal_liftoff_bonus": config.liftoff_bonus_weight
        * jp.asarray(legal_liftoff_transition, jp.float32),
        "stable_airborne_bonus": config.stable_airborne_bonus_weight
        * jp.asarray(stable_airborne_transition, jp.float32),
        "ascent_progress": ascent,
        "dual_wheel_lift_progress": dual_wheel_lift,
        "clearance_progress": clearance,
        "apex_approach": apex_approach,
        "apex_success_bonus": config.success_bonus
        * jp.asarray(success_transition, jp.float32),
        "attitude_penalty": attitude,
        "angular_rate_penalty": angular_rate,
        "illegal_contact_penalty": -config.illegal_contact_penalty_weight
        * jp.asarray(signals.illegal_contact, jp.float32),
        "action_smoothness_penalty": smoothness,
        "action_magnitude_penalty": magnitude,
        "physical_failure_penalty": -config.physical_failure_penalty
        * jp.asarray(physical_failure, jp.float32),
        "task_failure_penalty": -config.task_failure_penalty
        * jp.asarray(task_failure, jp.float32),
    }


def _contains_end_code(end_code: Any, values: tuple[int, ...]) -> Any:
    code = jp.asarray(end_code)
    return jp.any(code[..., None] == jp.asarray(values, dtype=code.dtype), axis=-1)


def _event_info(event: TwoPhaseEventState) -> dict[str, Any]:
    return {
        f"phase_expert/event/{name}": value
        for name, value in zip(TwoPhaseEventState._fields, event)
    }


def _event_from_info(info: Mapping[str, Any]) -> TwoPhaseEventState:
    return TwoPhaseEventState(
        *(info[f"phase_expert/event/{name}"] for name in TwoPhaseEventState._fields)
    )


def _finite_tree(value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(value)
    valid = jp.asarray(True)
    for leaf in leaves:
        valid = valid & jp.all(jp.isfinite(jp.asarray(leaf)))
    return valid


class PhaseExpertEnvAdapter:
    """Pure-JAX Phase U ownership layer around the unchanged DVGC environment."""

    phase = PHASE_PROPULSION_ASCENT

    def __init__(
        self,
        base_env: Any,
        *,
        geometry: Any,
        thresholds: TwoPhaseThresholds,
        reward_config: PhaseURewardConfig,
        episode_horizon: int,
        signal_extractor: Any | None = None,
    ) -> None:
        if episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive")
        self._base_env = base_env
        self._geometry = geometry
        self._thresholds = thresholds
        self._reward_config = reward_config
        self._episode_horizon = int(episode_horizon)
        self._signal_extractor = signal_extractor or self._extract_signals
        self.action_size = base_env.action_size

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_env, name)

    def _extract_signals(
        self, state: Any, geometry: Any, previous_hold_count: Any
    ) -> tuple[Any, Any]:
        return (
            extract_apex_band_signals(state, geometry),
            extract_recovery_signals(
                state,
                geometry,
                previous_recovery_hold_count=previous_hold_count,
            ),
        )

    def _window_active(self, state: Any) -> Any:
        """Reconstruct the existing deployable latch/end-x window without env edits."""
        if "jump_window_active" in state.info:  # test adapters may expose it directly
            return jp.asarray(state.info["jump_window_active"], dtype=bool)
        root_x = state.data.qpos[..., self._geometry.root_qpos_adr]
        return jp.asarray(state.info["jump_signal_latched"], dtype=bool) & (
            root_x <= jp.asarray(state.info["jump_window_end_x"])
        )

    @staticmethod
    def _metrics(
        *,
        reward: Any,
        success: Any,
        physical: Any,
        task: Any,
        timeout: Any,
        reward_components: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        components = reward_components or {
            name: jp.asarray(0.0, jp.float32) for name in _PHASE_U_REWARD_COMPONENTS
        }
        return {
            "reward": jp.asarray(reward),
            "phase_expert/reward": jp.asarray(reward),
            "phase_expert/success": jp.asarray(success, jp.float32),
            "phase_expert/physical_failure": jp.asarray(physical, jp.float32),
            "phase_expert/task_failure": jp.asarray(task, jp.float32),
            "phase_expert/timeout": jp.asarray(timeout, jp.float32),
            **{
                f"phase_expert/reward_component/{name}": jp.asarray(value)
                for name, value in components.items()
            },
        }

    def reset(self, rng: Any) -> Any:
        state = self._base_env.reset(rng)
        event = initial_two_phase_event_state()
        apex, recovery = self._signal_extractor(
            state, self._geometry, event.recovery_hold_count
        )
        neutral = jp.asarray(self._base_env._neutral_action)
        reset_valid = (
            _finite_tree((state.data.qpos, state.data.qvel, state.data.ctrl, state.obs))
            & _finite_tree((state.info["last_action"], state.info["obs_history"]))
            & jp.all(jp.asarray(state.info["last_action"]) == neutral)
            & (jp.asarray(state.info["actor_packet_fifo_valid"]) > 0)
            & jp.asarray(recovery.stable_wheel_support, bool)
            & jp.asarray(recovery.no_body_contact, bool)
            & ~jp.asarray(apex.illegal_contact, bool)
            & ~jp.asarray(apex.physical_failure, bool)
        )
        info = state.info | _event_info(event) | {
            "phase_expert/source_phase_id": jp.asarray(0, jp.int32),
            "phase_expert/reset_valid": reset_valid,
            "phase_expert/episode_step": jp.asarray(0, jp.int32),
            "phase_expert/success": jp.asarray(False),
            "phase_expert/physical_failure": jp.asarray(False),
            "phase_expert/task_failure": jp.asarray(False),
            "phase_expert/timeout": jp.asarray(False),
        }
        zero = jp.asarray(0.0, jp.float32)
        return state.replace(
            reward=zero,
            done=(~reset_valid).astype(state.done.dtype),
            metrics=self._metrics(
                reward=zero,
                success=False,
                physical=False,
                task=False,
                timeout=False,
            ),
            info=info,
        )

    def step(self, state: Any, action: Any) -> Any:
        previous = _event_from_info(state.info)
        previous_action = state.info["last_action"]
        raw = self._base_env.step(
            state.replace(reward=jp.zeros_like(state.reward), done=jp.zeros_like(state.done)),
            action,
        )
        tick = jp.asarray(state.info["phase_expert/episode_step"], jp.int32) + 1
        apex, recovery = self._signal_extractor(
            raw, self._geometry, previous.recovery_hold_count
        )
        event = advance_two_phase_events(
            apex,
            recovery,
            previous,
            self._thresholds,
            tick=tick,
            jump_signal=raw.info["jump_signal_latched"],
        )
        success_transition = ~jp.asarray(previous.apex_band_entered) & jp.asarray(
            event.apex_band_entered
        )
        success = jp.asarray(event.apex_band_entered)
        window_entry_transition = ~jp.asarray(previous.jump_window_entered) & jp.asarray(
            event.jump_window_entered
        )
        physical = _contains_end_code(
            raw.info["end_code"], _PHASE_U_PHYSICAL_FAILURE_END_CODES
        )
        legal_liftoff_transition = (
            ~jp.asarray(previous.liftoff_seen)
            & jp.asarray(event.liftoff_seen)
            & jp.asarray(event.jump_window_entered)
            & ~physical
        )
        stable_airborne_transition = (
            ~jp.asarray(previous.stable_airborne)
            & jp.asarray(event.stable_airborne)
            & jp.asarray(event.jump_window_entered)
            & ~physical
        )
        task = jp.asarray(event.jump_window_entered) & _contains_end_code(
            raw.info["end_code"], _PHASE_U_POST_LATCH_TASK_FAILURE_END_CODES
        )
        terminated = success | physical | task
        timeout = (tick >= self._episode_horizon) & ~terminated
        done = terminated | timeout
        reward_terms = phase_u_reward_components(
            apex,
            self._thresholds.apex,
            event.jump_window_entered,
            jp.asarray(event.jump_window_entered) & jp.asarray(event.liftoff_seen),
            window_entry_transition,
            legal_liftoff_transition,
            stable_airborne_transition,
            jp.asarray(event.stable_airborne) & jp.asarray(event.ascending),
            success_transition,
            physical,
            task,
            action,
            previous_action,
            self._reward_config,
        )
        reward = sum(reward_terms.values())
        reward = jp.clip(
            reward, self._reward_config.total_min, self._reward_config.total_max
        )
        info = raw.info | _event_info(event) | {
            "phase_expert/source_phase_id": jp.asarray(0, jp.int32),
            "phase_expert/reset_valid": state.info["phase_expert/reset_valid"],
            "phase_expert/episode_step": tick,
            "phase_expert/success": success,
            "phase_expert/physical_failure": physical,
            "phase_expert/task_failure": task,
            "phase_expert/timeout": timeout,
        }
        return raw.replace(
            reward=reward,
            done=done.astype(raw.done.dtype),
            metrics=self._metrics(
                reward=reward,
                success=success,
                physical=physical,
                task=task,
                timeout=timeout,
                reward_components=reward_terms,
            ),
            info=info,
        )


_END_REASON = {
    0: "none",
    1: "recovery",
    2: "prohibited_contact",
    3: "invalid_wheel_step_contact",
    4: "roll_limit",
    5: "pitch_limit",
    6: "backward",
    7: "platform_back_edge_exit",
    8: "stage_timeout",
    9: "prelaunch_airborne",
    10: "takeoff_positive_pitch_failure",
    11: "takeoff_wheelie_failure",
    12: "takeoff_missed_liftoff_deadline",
    13: "takeoff_missed_wheel_clearance_deadline",
    14: "chain_entry",
    15: "nonfinite",
    16: "next_stage_entry",
}


def summarize_phase_expert_evaluation(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close fixed-evaluation outcome accounting without promotion semantics."""
    counts = {
        "success": 0,
        "physical_failure": 0,
        "timeout": 0,
        "other_failure": 0,
    }
    reasons: dict[str, int] = {}
    for row in rows:
        flags = [
            bool(row.get("success")),
            bool(row.get("physical_failure")),
            bool(row.get("timeout")),
        ]
        if sum(flags) > 1:
            raise ValueError("evaluation outcomes must be mutually exclusive")
        outcome = (
            "success"
            if flags[0]
            else "physical_failure"
            if flags[1]
            else "timeout"
            if flags[2]
            else "other_failure"
        )
        counts[outcome] += 1
        reason = _END_REASON.get(int(row.get("end_code", 0)), "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(rows)
    return {
        "num_rollouts": total,
        "outcome_counts": counts,
        "termination_reason_counts": dict(sorted(reasons.items())),
        "empirical_success_rate": counts["success"] / total if total else 0.0,
        "physical_failure_rate": counts["physical_failure"] / total if total else 0.0,
        "timeout_rate": counts["timeout"] / total if total else 0.0,
        "evidence_level": "smoke_diagnostic_only",
        "promotion_authorized": False,
    }


def align_phase_u_checkpoints(
    requested_checkpoints: tuple[int, ...],
    *,
    rollout_block_size: int,
    transition_ceiling: int,
) -> tuple[PhaseCheckpointMilestone, ...]:
    """Align requested reporting milestones upward without crossing the ceiling."""
    if (
        isinstance(rollout_block_size, bool)
        or not isinstance(rollout_block_size, int)
        or rollout_block_size <= 0
    ):
        raise ValueError("rollout_block_size must be a positive integer")
    if (
        isinstance(transition_ceiling, bool)
        or not isinstance(transition_ceiling, int)
        or transition_ceiling <= 0
    ):
        raise ValueError("transition ceiling must be a positive integer")
    if (
        not requested_checkpoints
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in requested_checkpoints
        )
        or any(
            right <= left
            for left, right in zip(requested_checkpoints, requested_checkpoints[1:])
        )
    ):
        raise ValueError("requested checkpoints must be strictly increasing integers")
    milestones = tuple(
        PhaseCheckpointMilestone(
            requested=value,
            effective=(
                0
                if value == 0
                else int(math.ceil(value / rollout_block_size) * rollout_block_size)
            ),
        )
        for value in requested_checkpoints
    )
    if any(item.effective > transition_ceiling for item in milestones):
        raise ValueError("aligned checkpoint exceeds transition ceiling")
    effective = [item.effective for item in milestones]
    if len(set(effective)) != len(effective):
        raise ValueError("aligned checkpoints must remain strictly increasing")
    return milestones


def phase_u_invocation_milestones(
    requested_checkpoints: tuple[int, ...],
    *,
    rollout_block_size: int,
    cumulative_start: int,
    invocation_transitions: int,
) -> tuple[PhaseCheckpointMilestone, ...]:
    """Select global milestones written by one initial or warm-start invocation."""
    if (
        isinstance(rollout_block_size, bool)
        or not isinstance(rollout_block_size, int)
        or rollout_block_size <= 0
    ):
        raise ValueError("rollout_block_size must be a positive integer")
    for name, value in (
        ("cumulative_start", cumulative_start),
        ("invocation_transitions", invocation_transitions),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if invocation_transitions <= 0:
        raise ValueError("invocation_transitions must be positive")
    if cumulative_start % rollout_block_size or invocation_transitions % rollout_block_size:
        raise ValueError("warm-start transition accounting must be rollout-block aligned")
    cumulative_end = cumulative_start + invocation_transitions
    if cumulative_end > 1_000_000:
        raise ValueError("cumulative Phase U training exceeds 1,000,000 transitions")
    aligned = align_phase_u_checkpoints(
        requested_checkpoints,
        rollout_block_size=rollout_block_size,
        transition_ceiling=1_000_000,
    )
    return tuple(
        milestone
        for milestone in aligned
        if (
            milestone.effective == 0 and cumulative_start == 0
        ) or cumulative_start < milestone.effective <= cumulative_end
    )


def summarize_phase_u_physical_evaluation(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate held-out physical progress separately from scalar return."""
    total = len(rows)

    def rate(field: str) -> float:
        return sum(bool(row.get(field)) for row in rows) / total if total else 0.0

    reward_keys = sorted(
        {key for row in rows for key in row.get("reward_component_sums", {})}
    )
    reward_means = {
        key: (
            sum(float(row.get("reward_component_sums", {}).get(key, 0.0)) for row in rows)
            / total
            if total
            else 0.0
        )
        for key in reward_keys
    }
    saturation = [float(row["action_saturation_fraction"]) for row in rows]
    episode_returns = [float(row["episode_return"]) for row in rows]
    return {
        "num_rollouts": total,
        "jump_window_reach_rate": rate("jump_window_reached"),
        "liftoff_rate": rate("liftoff_reached"),
        "stable_airborne_rate": rate("stable_airborne_reached"),
        "ascending_rate": rate("ascending_reached"),
        "clearance_success_rate": rate("clearance_success"),
        "apex_band_success_rate": rate("success"),
        "physical_failure_rate": rate("physical_failure"),
        "roll_violation_rate": rate("roll_violation"),
        "pitch_violation_rate": rate("pitch_violation"),
        "illegal_contact_rate": rate("illegal_contact"),
        "clearance_margin_distribution": sorted(
            float(row["clearance_margin"]) for row in rows
        ),
        "minimum_post_window_forward_velocity_distribution": sorted(
            float(row["minimum_post_window_forward_velocity"]) for row in rows
        ),
        "forward_velocity_retention_rate": rate("forward_velocity_retained"),
        "mean_action_saturation_fraction": (
            sum(saturation) / total if total else 0.0
        ),
        "mean_episode_return": (
            sum(episode_returns) / total if total else 0.0
        ),
        "reward_component_mean_sums": reward_means,
    }


def evaluate_phase_u_checkpoint_gate(
    checkpoint_reports: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply conservative automatic pauses to fixed held-out checkpoint evidence."""
    reasons: list[str] = []
    if not checkpoint_reports:
        return {"pause": False, "reasons": reasons}

    metric_names = (
        "apex_band_success_rate",
        "jump_window_reach_rate",
        "liftoff_rate",
        "clearance_success_rate",
        "forward_velocity_retention_rate",
    )

    def physical(report: Mapping[str, Any]) -> Mapping[str, Any]:
        evaluation = report.get("fixed_evaluation", {})
        return evaluation.get("physical_metrics", {}) if isinstance(
            evaluation, Mapping
        ) else {}

    def value(metrics: Mapping[str, Any], name: str) -> float:
        try:
            result = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            return math.nan
        return result

    current = physical(checkpoint_reports[-1])
    current_values = [
        value(current, name)
        for name in metric_names
        + (
            "physical_failure_rate",
            "mean_action_saturation_fraction",
            "mean_episode_return",
        )
    ]
    if not all(math.isfinite(item) for item in current_values):
        reasons.append("nonfinite_checkpoint_evaluation")
    elif value(current, "mean_action_saturation_fraction") >= 0.98:
        reasons.append("severe_action_saturation")

    if len(checkpoint_reports) >= 3:
        recent = [physical(report) for report in checkpoint_reports[-3:]]
        scores = [
            sum(value(metrics, name) for name in metric_names) / len(metric_names)
            for metrics in recent
        ]
        failures = [value(metrics, "physical_failure_rate") for metrics in recent]
        returns = [value(metrics, "mean_episode_return") for metrics in recent]
        finite = all(
            math.isfinite(item) for item in scores + failures + returns
        )
        clearly_plateaued = finite and max(scores) - min(scores) <= 0.01
        if clearly_plateaued:
            reasons.append("held_out_physical_performance_plateau")
        clearly_degrading = finite and all(
            left - right >= 0.05 for left, right in zip(scores, scores[1:])
        ) and all(
            right >= left for left, right in zip(failures, failures[1:])
        )
        if clearly_degrading:
            reasons.append("held_out_physical_performance_degradation")
            if all(
                right > left for left, right in zip(returns, returns[1:])
            ):
                reasons.append("reward_hacking_return_up_physics_down")
    pause_reasons = [
        reason
        for reason in reasons
        if reason != "held_out_physical_performance_plateau"
    ]
    return {"pause": bool(pause_reasons), "reasons": reasons}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_phase_expert_artifacts(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    initial_status: Mapping[str, Any],
) -> None:
    """Create a collision-safe run root and its immutable initial records."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(root / "run_manifest.json", manifest)
    _write_json_atomic(root / "status.json", initial_status)


def update_phase_expert_status(
    output_dir: str | Path, status: Mapping[str, Any]
) -> None:
    if status.get("status") not in {
        "initialized",
        "running",
        "completed",
        "failed",
        "gate_pause",
    }:
        raise ValueError("invalid phase expert status")
    _write_json_atomic(Path(output_dir) / "status.json", status)


def append_phase_expert_metrics(
    output_dir: str | Path, metrics: Mapping[str, Any]
) -> None:
    try:
        line = json.dumps(metrics, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise ValueError("metrics must contain only finite JSON values") from exc
    with (Path(output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def recursive_path_sha256(path: str | Path) -> str:
    """Hash a checkpoint tree by relative path and bytes in stable order."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError("checkpoint path must be a directory")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    for candidate in files:
        digest.update(candidate.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_CHECKPOINT_CONTRACT_FIELDS = frozenset(
    {
        "phase",
        "cumulative_training_transitions",
        "checkpoint_payload",
        "optimizer_state_included",
        "environment_step_state_included",
        "resume_semantics",
        "prng_lineage",
        "reset_contract_hash",
        "reward_contract_hash",
        "evaluation_contract_hash",
        "xml_sha256",
        "action_schema_hash",
        "observation_schema_hash",
        "history_schema_hash",
        "parent_checkpoint",
    }
)


def _checkpoint_sidecar_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path.with_name(path.name + ".phase_expert.json")


def write_phase_expert_checkpoint_sidecar(
    checkpoint: str | Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if set(contract) != _CHECKPOINT_CONTRACT_FIELDS:
        raise ValueError("checkpoint contract fields are incomplete")
    if (
        contract.get("checkpoint_payload") != "normalizer_policy_value"
        or contract.get("optimizer_state_included") is not False
        or contract.get("environment_step_state_included") is not False
        or contract.get("resume_semantics")
        != "policy_normalizer_value_warm_start"
    ):
        raise ValueError(
            "checkpoint must declare a policy/normalizer/value warm start, not full training state"
        )
    payload = dict(contract)
    payload["recursive_checkpoint_sha256"] = recursive_path_sha256(checkpoint)
    _write_json_atomic(_checkpoint_sidecar_path(checkpoint), payload)
    return payload


def validate_phase_expert_checkpoint_sidecar(
    checkpoint: str | Path, expected_contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    payload = _read_json(_checkpoint_sidecar_path(checkpoint), "checkpoint sidecar")
    if (
        payload.get("checkpoint_payload") != "normalizer_policy_value"
        or payload.get("optimizer_state_included") is not False
        or payload.get("environment_step_state_included") is not False
        or payload.get("resume_semantics")
        != "policy_normalizer_value_warm_start"
    ):
        raise ValueError("checkpoint sidecar does not describe a truthful warm start")
    cumulative = payload.get("cumulative_training_transitions")
    if (
        isinstance(cumulative, bool)
        or not isinstance(cumulative, int)
        or not 0 <= cumulative <= 1_000_000
        or cumulative % 1_600
    ):
        raise ValueError("checkpoint cumulative training transitions are invalid")
    for field in _CHECKPOINT_CONTRACT_FIELDS:
        if payload.get(field) != expected_contract.get(field):
            raise ValueError(f"checkpoint contract drift: {field}")
    if payload.get("recursive_checkpoint_sha256") != recursive_path_sha256(checkpoint):
        raise ValueError("checkpoint recursive identity mismatch")
    return _freeze(payload)


def _load_and_validate_checkpoint_sidecar(checkpoint: str | Path) -> Mapping[str, Any]:
    payload = _read_json(_checkpoint_sidecar_path(checkpoint), "checkpoint sidecar")
    contract = {field: payload.get(field) for field in _CHECKPOINT_CONTRACT_FIELDS}
    return validate_phase_expert_checkpoint_sidecar(checkpoint, contract)


def _parent_resume_progress(parent: Path) -> tuple[int, Path]:
    """Return latest valid checkpoint progress and reject hidden later consumption."""
    checkpoints: list[tuple[int, Path]] = []
    for sidecar_path in sorted(parent.rglob("*.phase_expert.json")):
        checkpoint = sidecar_path.with_name(
            sidecar_path.name.removesuffix(".phase_expert.json")
        )
        if not checkpoint.is_dir():
            raise ValueError("parent checkpoint sidecar has no checkpoint directory")
        sidecar = _load_and_validate_checkpoint_sidecar(checkpoint)
        checkpoints.append(
            (int(sidecar["cumulative_training_transitions"]), checkpoint.resolve())
        )
    if not checkpoints:
        raise ValueError("resume run has no valid phase expert checkpoint")
    latest_transition, latest_checkpoint = max(checkpoints, key=lambda item: item[0])

    observed = [latest_transition]
    status_path = parent / "status.json"
    if status_path.is_file():
        status = _read_json(status_path, "parent run status")
        for field in (
            "observed_training_progress",
            "cumulative_training_transitions",
            "training_transitions",
        ):
            value = status.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                observed.append(value)
    metrics_path = parent / "metrics.jsonl"
    if metrics_path.is_file():
        for line_number, line in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"parent metrics line {line_number} is invalid JSON"
                ) from exc
            for field in ("cumulative_training_step", "training_step"):
                value = row.get(field) if isinstance(row, Mapping) else None
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    observed.append(value)
    if max(observed) > latest_transition:
        raise ValueError(
            "parent consumed training beyond its latest checkpoint; warm start would repeat transitions"
        )
    return latest_transition, latest_checkpoint


def _load_parent_checkpoint_evaluations(parent: Path) -> list[dict[str, Any]]:
    """Carry held-out degradation evidence across warm-start run boundaries."""
    aggregate = parent / "checkpoint_evaluations.json"
    if aggregate.is_file():
        payload = _read_json(aggregate, "parent checkpoint evaluations")
        reports = payload.get("checkpoints")
        if not isinstance(reports, list) or not all(
            isinstance(report, Mapping) for report in reports
        ):
            raise ValueError("parent checkpoint evaluation history is invalid")
        return [dict(report) for report in reports]
    reports = []
    for path in sorted((parent / "evaluations").glob("*/fixed_evaluation.json")):
        report = _read_json(path, "parent checkpoint evaluation")
        if not isinstance(report.get("fixed_evaluation"), Mapping):
            raise ValueError("parent checkpoint evaluation is incomplete")
        reports.append(dict(report))
    return sorted(
        reports,
        key=lambda report: int(report.get("effective_training_transitions", -1)),
    )


_DESCENT_EVIDENCE_FIELDS = (
    "mujoco_forward_valid",
    "finite_state_valid",
    "no_penetration",
    "legal_geometry",
    "short_horizon_dynamic_valid",
    "real_three_frame_fifo_valid",
    "timing_explicit_snapshot_valid",
)
_FORBIDDEN_DESCENT_CLAIMS = ("reachable", "expert_snapshot", "Tube", "safe")


def validate_descent_seed_manifest(
    bank_path: str | Path, manifest_path: str | Path
) -> Mapping[str, Any]:
    bank = Path(bank_path)
    if not bank.is_file():
        raise ValueError("descent seed bank does not exist")
    manifest = _read_json(manifest_path, "descent seed manifest")
    if manifest.get("reset_mode") == "natural_start":
        raise ValueError("Phase D natural reset fallback is forbidden")
    if manifest.get("reset_mode") != "bank":
        raise ValueError("Phase D descent seed manifest must use bank reset")
    tier = manifest.get("seed_tier")
    if tier not in _DESCENT_SEED_TIERS:
        raise ValueError("Phase D descent seed manifest has an invalid seed tier")
    if manifest.get("source_hash") != _sha256_file(bank):
        raise ValueError("Phase D descent seed source hash mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("descent seed manifest requires records")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("descent seed record must be an object")
        if any(name in record for name in _FORBIDDEN_DESCENT_CLAIMS):
            raise ValueError("descent seed record contains a forbidden claim")
        if not all(record.get(name) is True for name in _DESCENT_EVIDENCE_FIELDS):
            raise ValueError("descent seed record lacks physical validation evidence")
        if tier == "physically_validated_descent_seed" and record.get("seed_role") != tier:
            raise ValueError("preliminary descent seed role is invalid")
    if tier == "pi_up_online_apex_snapshot":
        online = [row for row in records if row.get("seed_role") == tier]
        mass = sum(float(row.get("sampling_mass", 0.0)) for row in online)
        if len(online) * 2 <= len(records) or mass <= 0.5:
            raise ValueError("formal Phase D seeds must be dominated by pi_up online snapshots")
        required = {"apex_pre", "apex_nearest", "apex_post", "early_descent"}
        if not required.issubset({row.get("event_position") for row in online}):
            raise ValueError("formal Phase D seed event coverage is incomplete")
    return _freeze(manifest)


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def phase_expert_source_tree_sha256() -> str:
    """Hash every Gate C1 managed source, including uncommitted file contents."""
    rows = []
    for relative in _PHASE_EXPERT_SOURCE_PATHS:
        path = _REPOSITORY_ROOT / relative
        digest = _sha256_file(path) if path.is_file() else "missing"
        rows.append(f"{relative}:{digest}\n")
    return hashlib.sha256("".join(rows).encode("ascii")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def load_phase_expert_threshold_manifest(path: str | Path) -> ResolvedThresholdManifest:
    """Load a current, provenance-complete threshold contract without mutation."""
    manifest_path = Path(path)
    manifest = _read_json(manifest_path, "threshold manifest")
    recorded_hash = manifest.get("canonical_manifest_hash")
    if not _is_sha256(recorded_hash) or recorded_hash != canonical_manifest_hash(manifest):
        raise ValueError("threshold manifest canonical hash is invalid")
    source_hashes = manifest.get("source_hashes")
    source_paths = manifest.get("source_paths")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != _THRESHOLD_SOURCE_HASHES
        or not all(_is_sha256(value) for value in source_hashes.values())
    ):
        raise ValueError("threshold manifest source hashes are incomplete")
    if (
        not isinstance(source_paths, Mapping)
        or set(source_paths) != _THRESHOLD_SOURCE_PATHS
        or not all(isinstance(value, str) and value for value in source_paths.values())
    ):
        raise ValueError("threshold manifest source paths are incomplete")
    if _resolve_repository_path(source_paths["xml"]) != _resolve_repository_path(
        AUTHORITATIVE_XML_PATH
    ):
        raise ValueError("threshold manifest must use the authoritative XML path")
    for source in sorted(_THRESHOLD_SOURCE_PATHS):
        source_path = _resolve_repository_path(source_paths[source])
        if not source_path.is_file() or _sha256_file(source_path) != source_hashes[source]:
            raise ValueError(f"threshold manifest source hash mismatch: {source}")
    if source_hashes["xml"] != AUTHORITATIVE_XML_SHA256:
        raise ValueError("threshold manifest XML is not the authoritative model")
    geometry_path = manifest_path.with_name("geometry_manifest.json")
    if not geometry_path.is_file():
        raise ValueError("threshold manifest geometry identity is unavailable")
    geometry = _read_json(geometry_path, "geometry manifest")
    if _canonical_payload_hash(geometry) != source_hashes["geometry_manifest"]:
        raise ValueError("threshold manifest geometry identity mismatch")
    if manifest.get("action_mapping_version") != ACTION_MAPPING_VERSION:
        raise ValueError("threshold manifest action mapping does not match current configuration")
    if manifest.get("reference_rollout_source") != "kinematic_guideline_envelope":
        raise ValueError("threshold manifest reference rollout provenance is invalid")
    if manifest.get("controller_provenance") != "kinematic guideline envelope":
        raise ValueError("threshold manifest controller provenance is invalid")
    if manifest.get("source_category") != "guideline_physical_envelope":
        raise ValueError("threshold manifest source category is invalid")
    selected = manifest.get("selected_thresholds")
    if not isinstance(selected, Mapping):
        raise ValueError("threshold manifest selected thresholds are missing")
    try:
        apex = ApexBandThresholds(**dict(selected["apex"]))
        recovery = RecoveryThresholds(**dict(selected["recovery"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("threshold manifest selected thresholds are invalid") from exc
    return ResolvedThresholdManifest(
        manifest=_freeze(manifest),
        canonical_manifest_hash=recorded_hash,
        action_mapping_version=ACTION_MAPPING_VERSION,
        reference_rollout_source="kinematic_guideline_envelope",
        apex_thresholds=apex,
        recovery_thresholds=recovery,
    )


def resolve_gate_c1_base_mode(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable physical base-mode overrides for the Gate C1 adapter."""
    if not isinstance(config, Mapping) or dict(config.get("base_mode", {})) != dict(
        _BASE_MODE
    ):
        raise ValueError("Gate C1 base mode does not match the frozen contract")
    if config.get("adapter_ownership") != {
        "reward": True,
        "done": True,
        "timeout": True,
    }:
        raise ValueError("Gate C1 adapter ownership contract is invalid")
    return dict(_BASE_MODE)


def _layout_value(layout: Mapping[str, Any], name: str) -> Any:
    if name not in layout:
        raise ValueError(f"PPO layout is missing {name}")
    return layout[name]


def build_phase_expert_budget(
    spec: PhaseExpertRunSpec, layout: Mapping[str, Any]
) -> PPOBudgetReport:
    """Build an aligned Phase U budget under its experiment-level ceiling."""
    if not isinstance(layout, Mapping):
        raise ValueError("PPO layout must be a mapping")
    if (
        spec.experiment_level == "formal_expert"
        and spec.requested_total_transitions > 1_000_000
    ):
        raise ValueError("formal Phase U budget exceeds the 1,000,000 transition ceiling")
    unroll_length = _layout_value(layout, "unroll_length")
    batch_size = _layout_value(layout, "batch_size")
    num_minibatches = _layout_value(layout, "num_minibatches")
    if spec.experiment_level == "formal_expert":
        rollout_block_size = (
            int(unroll_length) * int(batch_size) * int(num_minibatches)
        )
        if (
            rollout_block_size <= 0
            or spec.requested_total_transitions % rollout_block_size
        ):
            raise ValueError(
                "requested_total_transitions must be aligned to a PPO rollout block"
            )
        num_evals = spec.requested_total_transitions // rollout_block_size + 1
    else:
        num_evals = _layout_value(layout, "num_evals")
    report = build_ppo_budget_report(
        requested_total_transitions=spec.requested_total_transitions,
        num_parallel_envs=_layout_value(layout, "num_parallel_envs"),
        episode_horizon=_layout_value(layout, "episode_horizon"),
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=_layout_value(layout, "num_updates_per_batch"),
        num_evals=num_evals,
        experiment_level=spec.experiment_level,
    )
    assert report.requested_timesteps == report.requested_total_transitions
    assert report.effective_timesteps == report.effective_total_transitions
    if report.alignment_overhead != 0:
        raise ValueError("requested_total_transitions must be aligned to a PPO rollout block")
    if spec.experiment_level == "smoke":
        if not 1 <= report.ppo_rollout_blocks <= 4:
            raise ValueError("smoke budget must use one through four PPO rollout blocks")
    elif spec.experiment_level == "formal_expert":
        if report.effective_total_transitions > 1_000_000:
            raise ValueError("formal Phase U budget exceeds the 1,000,000 transition ceiling")
    else:
        raise ValueError("only smoke or formal_expert Phase U execution is authorized")
    return report


def _derive_training_seeds(root_seed: int, namespace: str, count: int) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{root_seed}:{namespace}:{index}".encode()).digest()[:8],
            "big",
        )
        for index in range(count)
    )


def validate_phase_expert_seed_namespaces(
    spec: PhaseExpertRunSpec, training_config: Mapping[str, Any]
) -> PhaseExpertSeedNamespaces:
    """Derive reproducible training seeds and reject fixed-evaluation overlap."""
    namespace = training_config.get("training_seed_namespace")
    train_count = training_config.get("training_seed_count")
    evaluation = training_config.get("evaluation")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("training seed namespace is required")
    train_count = _positive_int("training_seed_count", train_count)
    if not isinstance(evaluation, Mapping):
        raise ValueError("fixed evaluation configuration is required")
    evaluation_namespace = evaluation.get("seed_namespace")
    evaluation_seeds = evaluation.get("seeds")
    if not isinstance(evaluation_namespace, str) or not evaluation_namespace:
        raise ValueError("evaluation seed namespace is required")
    if evaluation_namespace == namespace:
        raise ValueError("training and evaluation seed namespaces must be disjoint")
    if (
        not isinstance(evaluation_seeds, list)
        or not evaluation_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in evaluation_seeds)
        or len(set(evaluation_seeds)) != len(evaluation_seeds)
    ):
        raise ValueError("evaluation seeds must be unique integers")
    training_seeds = _derive_training_seeds(spec.seed, namespace, train_count)
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("training and evaluation seeds must be disjoint")
    return PhaseExpertSeedNamespaces(
        training_namespace=namespace,
        training_seeds=training_seeds,
        evaluation_namespace=evaluation_namespace,
        evaluation_seeds=tuple(evaluation_seeds),
    )


def build_phase_expert_interaction_budget(
    spec: PhaseExpertRunSpec, training_config: Mapping[str, Any]
) -> PhaseExpertInteractionBudget:
    """Bind declared training and fixed-evaluation ceilings into one total cost."""
    layout = training_config.get("ppo_layout")
    maximum = training_config.get("maximum_interaction_cost")
    evaluation = training_config.get("evaluation")
    if (
        not isinstance(layout, Mapping)
        or not isinstance(maximum, Mapping)
        or not isinstance(evaluation, Mapping)
    ):
        raise ValueError(
            "smoke config must declare PPO layout, fixed evaluation, and interaction ceilings"
        )
    report = build_phase_expert_budget(spec, layout)
    training_ceiling = _positive_int(
        "training transition ceiling", maximum.get("training_transitions")
    )
    evaluation_ceiling = _positive_int(
        "fixed evaluation transition ceiling", maximum.get("fixed_evaluation_transitions")
    )
    brax_evaluation_ceiling = _nonnegative_int(
        "Brax evaluation transition ceiling",
        maximum.get("brax_evaluation_transitions"),
    )
    maximum_combined_ceiling = _positive_int(
        "combined interaction transition ceiling", maximum.get("combined_transitions")
    )
    brax_evaluation_environments = _positive_int(
        "Brax evaluation environment count", layout.get("num_eval_envs")
    )
    run_brax_evaluation = training_config.get("run_brax_evaluation", True)
    if not isinstance(run_brax_evaluation, bool):
        raise ValueError("run_brax_evaluation must be boolean")
    declared_brax_evaluation_cost = (
        brax_evaluation_environments * report.episode_horizon * report.num_evals
        if run_brax_evaluation
        else 0
    )
    if brax_evaluation_ceiling != declared_brax_evaluation_cost:
        raise ValueError(
            "Brax evaluation transition ceiling must equal evaluation "
            "environments times horizon times evaluations"
        )
    evaluation_environments = _positive_int(
        "fixed evaluation environment count", evaluation.get("environment_count")
    )
    evaluation_horizon = _positive_int(
        "fixed evaluation episode horizon", evaluation.get("episode_horizon")
    )
    evaluation_episodes = _positive_int(
        "fixed evaluation episode count", evaluation.get("episodes")
    )
    checkpoint_evaluations = _positive_int(
        "fixed checkpoint evaluation count",
        evaluation.get("checkpoint_evaluations", 1),
    )
    declared_evaluation_cost = (
        evaluation_environments
        * evaluation_horizon
        * evaluation_episodes
        * checkpoint_evaluations
    )
    candidate_config = training_config.get("candidate_acquisition", {})
    continuation_config = training_config.get("continuation_diagnostic", {})
    if not isinstance(candidate_config, Mapping) or not isinstance(
        continuation_config, Mapping
    ):
        raise ValueError("candidate and continuation configurations must be mappings")
    candidate_per_checkpoint = _nonnegative_int(
        "candidate acquisition transition ceiling per checkpoint",
        candidate_config.get("transition_ceiling_per_checkpoint", 0),
    )
    continuation_per_checkpoint = _nonnegative_int(
        "continuation labeling transition ceiling per checkpoint",
        continuation_config.get("transition_ceiling_per_checkpoint", 0),
    )
    candidate_ceiling = candidate_per_checkpoint * checkpoint_evaluations
    continuation_ceiling = continuation_per_checkpoint * checkpoint_evaluations
    if evaluation_ceiling != declared_evaluation_cost:
        raise ValueError(
            "fixed evaluation transition ceiling must equal environments times horizon times episodes"
        )
    if training_ceiling % report.ppo_rollout_block_size != 0:
        raise ValueError("training transition ceiling must be PPO-rollout-block aligned")
    if spec.experiment_level == "smoke" and not 1 <= (
        training_ceiling // report.ppo_rollout_block_size
    ) <= 4:
        raise ValueError(
            "training transition ceiling must be aligned to one through four PPO rollout blocks"
        )
    if spec.experiment_level == "formal_expert" and training_ceiling > 1_000_000:
        raise ValueError("formal training ceiling exceeds 1,000,000 transitions")
    if maximum_combined_ceiling != (
        training_ceiling + brax_evaluation_ceiling + evaluation_ceiling
    ):
        raise ValueError(
            "combined interaction ceiling must equal training plus Brax and fixed "
            "evaluation ceilings"
        )
    if report.effective_total_transitions > training_ceiling:
        raise ValueError("training budget exceeds its interaction ceiling")
    return PhaseExpertInteractionBudget(
        training=report,
        brax_evaluation_transition_ceiling=brax_evaluation_ceiling,
        fixed_evaluation_transition_ceiling=evaluation_ceiling,
        combined_transition_ceiling=(
            report.effective_total_transitions
            + brax_evaluation_ceiling
            + evaluation_ceiling
        ),
        candidate_acquisition_transition_ceiling=candidate_ceiling,
        continuation_labeling_transition_ceiling=continuation_ceiling,
        total_environment_transition_ceiling=(
            report.effective_total_transitions
            + brax_evaluation_ceiling
            + evaluation_ceiling
            + candidate_ceiling
            + continuation_ceiling
        ),
    )


def completed_phase_expert_interaction_accounting(
    interaction_budget: PhaseExpertInteractionBudget,
    *,
    fixed_evaluation_transitions: int,
    candidate_acquisition_transitions: int = 0,
    continuation_labeling_transitions: int = 0,
) -> dict[str, int]:
    """Close actual interaction totals for a successfully completed smoke."""
    if (
        isinstance(fixed_evaluation_transitions, bool)
        or not isinstance(fixed_evaluation_transitions, int)
        or not 0
        <= fixed_evaluation_transitions
        <= interaction_budget.fixed_evaluation_transition_ceiling
    ):
        raise ValueError("fixed evaluation transitions exceed the authorized ceiling")
    for name, value, ceiling in (
        (
            "candidate acquisition",
            candidate_acquisition_transitions,
            interaction_budget.candidate_acquisition_transition_ceiling,
        ),
        (
            "continuation labeling",
            continuation_labeling_transitions,
            interaction_budget.continuation_labeling_transition_ceiling,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= ceiling
        ):
            raise ValueError(f"{name} transitions exceed the authorized ceiling")
    training = interaction_budget.training.effective_total_transitions
    brax_evaluation = interaction_budget.brax_evaluation_transition_ceiling
    combined = training + brax_evaluation + fixed_evaluation_transitions
    if combined > interaction_budget.combined_transition_ceiling:
        raise ValueError("combined actual transitions exceed the authorized ceiling")
    total = (
        combined
        + candidate_acquisition_transitions
        + continuation_labeling_transitions
    )
    if total > interaction_budget.total_environment_transition_ceiling:
        raise ValueError("total actual environment transitions exceed the authorized ceiling")
    return {
        "training_transitions": training,
        "brax_evaluation_transitions": brax_evaluation,
        "fixed_evaluation_transitions": fixed_evaluation_transitions,
        "candidate_acquisition_transitions": candidate_acquisition_transitions,
        "continuation_labeling_transitions": continuation_labeling_transitions,
        "combined_environment_transitions": combined,
        "total_environment_transitions": total,
    }


def partial_phase_expert_interaction_accounting(
    *,
    cumulative_training_start: int,
    observed_training_progress: int,
    fixed_evaluation_transitions: int,
    candidate_acquisition_transitions: int,
    continuation_labeling_transitions: int,
) -> dict[str, int]:
    """Report a pause-path lower bound without erasing callback-first work."""
    values = (
        cumulative_training_start,
        observed_training_progress,
        fixed_evaluation_transitions,
        candidate_acquisition_transitions,
        continuation_labeling_transitions,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("partial interaction counters must be non-negative integers")
    if observed_training_progress < cumulative_training_start:
        raise ValueError("observed training progress precedes the invocation start")
    training = observed_training_progress - cumulative_training_start
    known_total = (
        training
        + fixed_evaluation_transitions
        + candidate_acquisition_transitions
        + continuation_labeling_transitions
    )
    return {
        "training_transitions_consumed": training,
        "fixed_evaluation_transitions_consumed": fixed_evaluation_transitions,
        "candidate_acquisition_transitions_consumed": candidate_acquisition_transitions,
        "continuation_labeling_transitions_consumed": continuation_labeling_transitions,
        "known_environment_transitions_consumed_lower_bound": known_total,
    }


def _validate_descent_seed_inputs(spec: PhaseExpertRunSpec) -> Mapping[str, Any] | None:
    if spec.phase != PHASE_DESCENT_RECOVERY:
        return None
    if not spec.descent_seed_bank or not spec.descent_seed_manifest:
        raise ValueError("descent seed bank and manifest are required")
    return validate_descent_seed_manifest(
        spec.descent_seed_bank, spec.descent_seed_manifest
    )


def _current_source_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _validate_authorization(
    spec: PhaseExpertRunSpec,
    thresholds: ResolvedThresholdManifest,
    interaction: PhaseExpertInteractionBudget,
    *,
    cumulative_training_start: int,
) -> Mapping[str, Any]:
    if not spec.authorization_manifest_path:
        raise ValueError("normal execution requires an authorization manifest")
    authorization = _read_json(spec.authorization_manifest_path, "authorization manifest")
    expected = {
        "decision": "authorize",
        "run_id": Path(spec.output_dir).name,
        "phase": spec.phase,
        "experiment_level": spec.experiment_level,
        "source_head": _current_source_head(),
        "source_tree_sha256": phase_expert_source_tree_sha256(),
        "seed": spec.seed,
        "output_directory": str(Path(spec.output_dir).resolve()),
        "xml_sha256": AUTHORITATIVE_XML_SHA256,
        "threshold_manifest_canonical_hash": thresholds.canonical_manifest_hash,
        "training_config_sha256": _sha256_file(spec.training_config_path),
        "requested_training_transition_ceiling": interaction.training.requested_total_transitions,
        "effective_training_transition_ceiling": interaction.training.effective_total_transitions,
        "cumulative_training_start": cumulative_training_start,
        "cumulative_training_end": (
            cumulative_training_start
            + interaction.training.effective_total_transitions
        ),
        "brax_evaluation_transition_ceiling": interaction.brax_evaluation_transition_ceiling,
        "fixed_evaluation_transition_ceiling": interaction.fixed_evaluation_transition_ceiling,
        "combined_interaction_transition_ceiling": interaction.combined_transition_ceiling,
        "candidate_acquisition_transition_ceiling": interaction.candidate_acquisition_transition_ceiling,
        "continuation_labeling_transition_ceiling": interaction.continuation_labeling_transition_ceiling,
        "total_environment_transition_ceiling": interaction.total_environment_transition_ceiling,
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            label = "run id" if field == "run_id" else field.replace("_", " ")
            raise ValueError(f"authorization manifest {label} does not match this run")
    if not isinstance(authorization.get("issuer"), str) or not authorization["issuer"]:
        raise ValueError("authorization manifest issuer is required")
    if not isinstance(authorization.get("issued_at"), str) or not authorization["issued_at"]:
        raise ValueError("authorization manifest issue time is required")
    return _freeze(authorization)


def validate_phase_expert_run_spec(
    spec: PhaseExpertRunSpec, *, preflight_only: bool
) -> ValidatedPhaseExpertRunSpec:
    """Validate a no-overwrite Phase U run contract before environment work."""
    if not isinstance(spec, PhaseExpertRunSpec):
        raise TypeError("spec must be a PhaseExpertRunSpec")
    if spec.phase not in PHASE_EXPERT_PHASES:
        raise ValueError(f"phase must be one of {PHASE_EXPERT_PHASES}")
    if spec.experiment_level not in {"smoke", "formal_expert"}:
        raise ValueError("only smoke or formal_expert Phase U execution is authorized")
    _positive_int("requested_total_transitions", spec.requested_total_transitions)
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
        raise ValueError("seed must be an integer")
    output_path = _resolve_repository_path(spec.output_dir)
    required_output_parent = (
        _REPOSITORY_ROOT / "runs" / "two_phase" / "phase_experts"
    ).resolve()
    if output_path.parent != required_output_parent or not output_path.name:
        raise ValueError(
            "output directory must be runs/two_phase/phase_experts/<run_id>"
        )
    if output_path.exists():
        raise ValueError("output directory must not already exist")
    if bool(spec.resume_run) != bool(spec.restore_checkpoint):
        raise ValueError("resume run and restore checkpoint must be paired")
    thresholds = load_phase_expert_threshold_manifest(spec.threshold_manifest_path)
    project_config_path = _resolve_repository_path(spec.config_path)
    threshold_config_path = _resolve_repository_path(
        thresholds.manifest["source_paths"]["config"]
    )
    if not project_config_path.is_file() or project_config_path != threshold_config_path:
        raise ValueError(
            "project config path must match the threshold manifest source config"
        )
    training_config = _read_json(spec.training_config_path, "phase expert training config")
    resolve_policy_initial_action_std(training_config)
    resolve_gate_c1_base_mode(training_config)
    seeds = validate_phase_expert_seed_namespaces(spec, training_config)
    interaction = build_phase_expert_interaction_budget(spec, training_config)
    _validate_descent_seed_inputs(spec)
    if spec.phase == PHASE_DESCENT_RECOVERY and not preflight_only:
        raise ValueError("Phase D is preflight-only at Gate C1")
    cumulative_training_start = 0
    if spec.resume_run and spec.restore_checkpoint:
        parent = Path(spec.resume_run).resolve()
        checkpoint = Path(spec.restore_checkpoint).resolve()
        if not parent.is_dir() or not checkpoint.is_dir():
            raise ValueError("resume run and checkpoint must exist")
        if not checkpoint.is_relative_to(parent):
            raise ValueError("restore checkpoint must belong to resume run")
        parent_manifest = _read_json(parent / "run_manifest.json", "parent run manifest")
        sidecar = _load_and_validate_checkpoint_sidecar(checkpoint)
        latest_transition, latest_checkpoint = _parent_resume_progress(parent)
        if checkpoint != latest_checkpoint:
            raise ValueError("restore checkpoint must be the latest parent checkpoint")
        if int(sidecar["cumulative_training_transitions"]) != latest_transition:
            raise ValueError("latest parent checkpoint transition identity mismatch")
        expected_resume = {
            "phase": spec.phase,
            "xml_sha256": AUTHORITATIVE_XML_SHA256,
            "reward_contract_hash": phase_u_reward_contract_hash(training_config),
            "evaluation_contract_hash": _canonical_payload_hash(
                training_config.get("evaluation", {})
            ),
            "reset_contract_hash": _canonical_payload_hash(dict(_BASE_MODE)),
            "action_schema_hash": hashlib.sha256(
                ACTION_MAPPING_VERSION.encode()
            ).hexdigest(),
            "observation_schema_hash": _canonical_payload_hash(
                {
                    "env": _sha256_file("dvgc/env.py"),
                    "audit": _sha256_file("dvgc/observation_audit.py"),
                }
            ),
            "history_schema_hash": hashlib.sha256(
                b"actor_packet_fifo.three_frame.v4.t_minus_2_to_t"
            ).hexdigest(),
        }
        for field, value in expected_resume.items():
            if sidecar.get(field) != value:
                raise ValueError(f"resume contract drift: {field}")
        if parent_manifest.get("run_id") != parent.name:
            raise ValueError("parent run manifest identity mismatch")
        cumulative_training_start = int(sidecar["cumulative_training_transitions"])
    cumulative_training_end = (
        cumulative_training_start + interaction.training.effective_total_transitions
    )
    if cumulative_training_end > 1_000_000:
        raise ValueError("cumulative Phase U training exceeds 1,000,000 transitions")
    authorization = (
        None
        if preflight_only
        else _validate_authorization(
            spec,
            thresholds,
            interaction,
            cumulative_training_start=cumulative_training_start,
        )
    )
    return ValidatedPhaseExpertRunSpec(
        spec=spec,
        thresholds=thresholds,
        seeds=seeds,
        interaction_budget=interaction,
        authorization=authorization,
        cumulative_training_start=cumulative_training_start,
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "shape"):
        array = np.asarray(jax.device_get(value))
        return float(array) if array.ndim == 0 else array.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_phase_expert_environment(
    validated: ValidatedPhaseExpertRunSpec,
) -> PhaseExpertEnvAdapter:
    """Construct the unchanged physical environment plus the external adapter."""
    if validated.spec.phase != PHASE_PROPULSION_ASCENT:
        raise ValueError("Phase D environment construction is blocked until Gate C2")
    from .bank import SnapshotBank
    from .config import load_config
    from .env import OrangeBikeDVGC
    from .two_phase_runtime import build_two_phase_geometry

    training_config = _read_json(
        validated.spec.training_config_path, "phase expert training config"
    )
    layout = training_config["ppo_layout"]
    overrides = resolve_gate_c1_base_mode(training_config) | {
        "episode_length": int(layout["episode_horizon"]),
        "obs_noise_enable": False,
    }
    config = load_config(validated.spec.config_path, overrides)
    base = OrangeBikeDVGC(config, snapshot_bank=SnapshotBank())
    geometry = build_two_phase_geometry(base.mj_model, config)
    reward_config = resolve_phase_u_reward_config(training_config)
    return PhaseExpertEnvAdapter(
        base,
        geometry=geometry,
        thresholds=TwoPhaseThresholds(
            apex=validated.thresholds.apex_thresholds,
            recovery=validated.thresholds.recovery_thresholds,
        ),
        reward_config=reward_config,
        episode_horizon=int(layout["episode_horizon"]),
    )


def _fixed_phase_u_evaluation(
    environment: PhaseExpertEnvAdapter,
    params: Any,
    seeds: tuple[int, ...],
    horizon: int,
    failure_video_dir: str | Path,
    transition_observer: Any | None = None,
) -> tuple[dict[str, Any], int]:
    from .runtime import build_inference

    inference = build_inference(environment, params, deterministic=True)
    step = jax.jit(environment.step)
    rows: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    transitions = 0
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        state = environment.reset(key)
        frames = [_phase_expert_frame(state)]
        episode_return = 0.0
        reward_component_sums = {
            name: 0.0 for name in _PHASE_U_REWARD_COMPONENTS
        }
        maximum_clearance = -math.inf
        maximum_forward_velocity = -math.inf
        maximum_abs_roll = 0.0
        maximum_abs_pitch = 0.0
        maximum_angular_speed = 0.0
        post_window_forward_velocities: list[float] = []
        clearance_success = False
        illegal_contact_seen = False
        saturated_actions = 0
        action_values = 0
        for tick in range(horizon):
            key, action_key = jax.random.split(key)
            action, _ = inference(state.obs, action_key)
            state = step(state, action)
            jax.block_until_ready(state)
            frames.append(_phase_expert_frame(state))
            transitions += 1
            if transition_observer is not None:
                transition_observer(1)
            episode_return += float(state.reward)
            action_host = np.asarray(jax.device_get(action))
            saturated_actions += int(np.count_nonzero(np.abs(action_host) >= 0.98))
            action_values += int(action_host.size)
            event_now = _event_from_info(state.info)
            apex_now, _ = environment._extract_signals(
                state, environment._geometry, event_now.recovery_hold_count
            )
            apex_host = jax.device_get(apex_now)
            clearance_value = float(apex_host.clearance)
            forward_value = float(apex_host.forward_velocity)
            maximum_clearance = max(maximum_clearance, clearance_value)
            maximum_forward_velocity = max(maximum_forward_velocity, forward_value)
            maximum_abs_roll = max(maximum_abs_roll, abs(float(apex_host.roll)))
            maximum_abs_pitch = max(maximum_abs_pitch, abs(float(apex_host.pitch)))
            maximum_angular_speed = max(
                maximum_angular_speed, float(apex_host.angular_speed)
            )
            clearance_success = clearance_success or (
                clearance_value >= environment._thresholds.apex.min_clearance
            )
            illegal_contact_seen = illegal_contact_seen or bool(
                apex_host.illegal_contact
            )
            if bool(event_now.jump_window_entered):
                post_window_forward_velocities.append(forward_value)
            for name in _PHASE_U_REWARD_COMPONENTS:
                reward_component_sums[name] += float(
                    state.metrics[f"phase_expert/reward_component/{name}"]
                )
            if bool(state.done):
                break
        info = jax.device_get(state.info)
        event = _event_from_info(info)
        end_code = int(info["end_code"])
        minimum_post_window_forward_velocity = (
            min(post_window_forward_velocities)
            if post_window_forward_velocities
            else float(maximum_forward_velocity)
        )
        row = {
                "seed": seed,
                "episode_length": tick + 1,
                "episode_return": episode_return,
                "success": bool(info["phase_expert/success"]),
                "physical_failure": bool(info["phase_expert/physical_failure"]),
                "timeout": bool(info["phase_expert/timeout"]),
                "end_code": end_code,
                "first_event_ticks": np.asarray(
                    info["phase_expert/event/first_event_ticks"]
                ).tolist(),
                "jump_window_reached": bool(event.jump_window_entered),
                "liftoff_reached": bool(event.liftoff_seen),
                "stable_airborne_reached": bool(event.stable_airborne),
                "ascending_reached": bool(event.ascending),
                "clearance_success": clearance_success,
                "roll_violation": end_code == 4,
                "pitch_violation": end_code == 5,
                "illegal_contact": illegal_contact_seen,
                "maximum_clearance": maximum_clearance,
                "clearance_margin": maximum_clearance
                - environment._thresholds.apex.min_clearance,
                "maximum_forward_velocity": maximum_forward_velocity,
                "minimum_post_window_forward_velocity": minimum_post_window_forward_velocity,
                "forward_velocity_retained": minimum_post_window_forward_velocity
                >= environment._thresholds.apex.min_forward_velocity,
                "maximum_abs_roll": maximum_abs_roll,
                "maximum_abs_pitch": maximum_abs_pitch,
                "maximum_angular_speed": maximum_angular_speed,
                "action_saturation_fraction": (
                    saturated_actions / action_values if action_values else 0.0
                ),
                "reward_component_sums": reward_component_sums,
            }
        rows.append(row)
        if not row["success"]:
            outcome = (
                "physical_failure"
                if row["physical_failure"]
                else "timeout"
                if row["timeout"]
                else "other_failure"
            )
            video_path = Path(failure_video_dir) / f"seed_{seed}_{outcome}.mp4"
            try:
                videos.append(
                    _render_phase_expert_failure_video(
                        environment._base_env,
                        frames,
                        video_path,
                        seed=seed,
                        outcome=outcome,
                        end_code=row["end_code"],
                    )
                )
            except Exception as exc:
                videos.append(
                    {
                        "seed": seed,
                        "outcome": outcome,
                        "status": "render_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    report = summarize_phase_expert_evaluation(rows)
    report["physical_metrics"] = summarize_phase_u_physical_evaluation(rows)
    report["rows"] = rows
    report["failure_videos"] = videos
    report["actual_environment_transitions"] = transitions
    return report, transitions


def _phase_expert_frame(state: Any) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(jax.device_get(value)).copy()
        for name, value in (
            ("qpos", state.data.qpos),
            ("qvel", state.data.qvel),
            ("ctrl", state.data.ctrl),
        )
    }


def _render_phase_expert_failure_video(
    base_environment: Any,
    frames: list[dict[str, np.ndarray]],
    output_path: str | Path,
    *,
    seed: int,
    outcome: str,
    end_code: int,
) -> dict[str, Any]:
    """Render captured evaluation states only; never advance environment dynamics."""
    import mediapy as media
    import mujoco

    if not frames:
        raise ValueError("failure video requires captured states")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = base_environment.mj_model
    data = mujoco.MjData(model)
    root = int(model.jnt_qposadr[int(model.joint("floating_base_joint").id)])
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 90.0, -10.0, 2.4
    images = []
    try:
        for frame in frames:
            data.qpos[:] = frame["qpos"]
            data.qvel[:] = frame["qvel"]
            if model.nu:
                data.ctrl[:] = frame["ctrl"]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [
                float(frame["qpos"][root]) + 0.3,
                float(frame["qpos"][root + 1]),
                0.24,
            ]
            renderer.update_scene(data, camera=camera)
            images.append(renderer.render().copy())
    finally:
        renderer.close()
    playback = [images[0]] * 10 + [image for image in images for _ in range(2)]
    playback += [images[-1]] * 20
    media.write_video(path, playback, fps=25, codec="h264", crf=18)
    state_path = path.with_suffix(".states.npz")
    np.savez_compressed(
        state_path,
        **{
            name: np.stack([frame[name] for frame in frames])
            for name in ("qpos", "qvel", "ctrl")
        },
    )
    return {
        "seed": seed,
        "outcome": outcome,
        "end_code": end_code,
        "status": "rendered",
        "video": str(path.resolve()),
        "video_sha256": _sha256_file(path),
        "state_trace": str(state_path.resolve()),
        "state_trace_sha256": _sha256_file(state_path),
        "captured_control_ticks": len(frames) - 1,
        "rendering_environment_transitions": 0,
    }


def _checkpoint_contract(
    validated: ValidatedPhaseExpertRunSpec,
    training_config: Mapping[str, Any],
    *,
    cumulative_transitions: int,
    parent_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "phase": validated.spec.phase,
        "cumulative_training_transitions": cumulative_transitions,
        "checkpoint_payload": "normalizer_policy_value",
        "optimizer_state_included": False,
        "environment_step_state_included": False,
        "resume_semantics": "policy_normalizer_value_warm_start",
        "prng_lineage": f"{validated.seeds.training_namespace}:{validated.spec.seed}",
        "reset_contract_hash": _canonical_payload_hash(dict(_BASE_MODE)),
        "reward_contract_hash": phase_u_reward_contract_hash(training_config),
        "evaluation_contract_hash": _canonical_payload_hash(
            training_config.get("evaluation", {})
        ),
        "xml_sha256": AUTHORITATIVE_XML_SHA256,
        "action_schema_hash": hashlib.sha256(ACTION_MAPPING_VERSION.encode()).hexdigest(),
        "observation_schema_hash": _canonical_payload_hash(
            {
                "env": _sha256_file("dvgc/env.py"),
                "audit": _sha256_file("dvgc/observation_audit.py"),
            }
        ),
        "history_schema_hash": hashlib.sha256(
            b"actor_packet_fifo.three_frame.v4.t_minus_2_to_t"
        ).hexdigest(),
        "parent_checkpoint": parent_checkpoint,
    }


def _save_phase_expert_inference_checkpoint(
    environment: PhaseExpertEnvAdapter,
    params: Any,
    checkpoint_root: str | Path,
    *,
    step: int,
    initial_action_std: tuple[float, float, float, float],
) -> Path:
    """Save Brax normalizer/policy/value params at one claimed host milestone."""
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    from .runtime import build_network_factory

    sample = environment.reset(jax.random.PRNGKey(0))
    observation_size = jax.tree_util.tree_map(lambda value: value.shape, sample.obs)
    network_config = ppo_checkpoint.network_config(
        observation_size=observation_size,
        action_size=environment.action_size,
        normalize_observations=True,
        network_factory=build_network_factory(
            initial_action_std=initial_action_std
        ),
    )
    root = Path(checkpoint_root).resolve()
    ppo_checkpoint.save(root, int(step), params, network_config)
    checkpoint = root / f"{int(step):012d}"
    if not checkpoint.is_dir():
        raise RuntimeError("Brax checkpoint save did not create the expected directory")
    return checkpoint


def run_phase_expert(validated: ValidatedPhaseExpertRunSpec) -> dict[str, Any]:
    """Execute one already-authorized Phase U run and leave promotion disabled."""
    if validated.authorization is None:
        raise ValueError("normal execution requires validated authorization")
    if validated.spec.phase != PHASE_PROPULSION_ASCENT:
        raise ValueError("Phase D execution is blocked until Gate C2")
    from .config import load_config
    from .runtime import assert_brax_metric_contract, make_ppo_train_fn

    root = Path(validated.spec.output_dir)
    training_config = _read_json(
        validated.spec.training_config_path, "phase expert training config"
    )
    layout = training_config["ppo_layout"]
    optimization = training_config["optimization"]
    reward_contract = asdict(resolve_phase_u_reward_config(training_config))
    policy_initial_action_std = resolve_policy_initial_action_std(training_config)
    resolved_project_config = load_config(
        validated.spec.config_path,
        resolve_gate_c1_base_mode(training_config)
        | {
            "episode_length": int(layout["episode_horizon"]),
            "obs_noise_enable": False,
        },
    ).to_dict()
    manifest = {
        "schema": "dvgc_phase_expert_run_v1",
        "run_id": root.name,
        "phase": validated.spec.phase,
        "experiment_level": validated.spec.experiment_level,
        "source_head": _current_source_head(),
        "source_tree_sha256": phase_expert_source_tree_sha256(),
        "xml_sha256": AUTHORITATIVE_XML_SHA256,
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "threshold_manifest_canonical_hash": validated.thresholds.canonical_manifest_hash,
        "training_config_sha256": _sha256_file(validated.spec.training_config_path),
        "authorization": _jsonable(validated.authorization),
        "interaction_budget": _jsonable(validated.interaction_budget),
        "cumulative_training_start": validated.cumulative_training_start,
        "cumulative_training_end": (
            validated.cumulative_training_start
            + validated.interaction_budget.training.effective_total_transitions
        ),
        "seed_namespaces": _jsonable(validated.seeds),
        "reward_contract": reward_contract,
        "policy_initial_action_std": policy_initial_action_std,
        "fixed_evaluation_contract": training_config["evaluation"],
        "reference_role": "kinematic_guideline_and_weak_prior_only",
        "promotion_authorized": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    initialize_phase_expert_artifacts(
        root,
        manifest,
        {
            "status": "initialized",
            "invocation_training_transitions": 0,
            "cumulative_training_transitions": validated.cumulative_training_start,
            "evaluation_transitions": 0,
        },
    )
    _write_json_atomic(root / "resolved_config.json", resolved_project_config)
    _write_json_atomic(root / "resolved_training_config.json", training_config)
    observed_training_progress = validated.cumulative_training_start
    fixed_evaluation_transitions = 0
    inherited_checkpoint_evaluations = (
        _load_parent_checkpoint_evaluations(Path(validated.spec.resume_run).resolve())
        if validated.spec.resume_run
        else []
    )
    checkpoint_evaluations: list[dict[str, Any]] = list(
        inherited_checkpoint_evaluations
    )
    candidate_acquisition_reports: list[dict[str, Any]] = []
    candidate_acquisition_transitions = 0
    continuation_labeling_transitions = 0
    last_checkpoint: str | None = validated.spec.restore_checkpoint
    try:
        environment = build_phase_expert_environment(validated)
        reset_state = jax.jit(environment.reset)(jax.random.PRNGKey(validated.spec.seed))
        jax.block_until_ready(reset_state)
        if not bool(reset_state.info["phase_expert/reset_valid"]):
            raise RuntimeError("audited natural reset is invalid")
        assert_brax_metric_contract(environment)
        update_phase_expert_status(
            root,
            {
                "status": "running",
                "invocation_training_transitions": 0,
                "cumulative_training_transitions": validated.cumulative_training_start,
                "evaluation_transitions": 0,
            },
        )

        def progress(step: int, metrics: Mapping[str, Any]) -> None:
            nonlocal observed_training_progress
            cumulative_step = validated.cumulative_training_start + int(step)
            observed_training_progress = max(
                observed_training_progress, cumulative_step
            )
            append_phase_expert_metrics(
                root,
                {
                    "invocation_training_step": int(step),
                    "cumulative_training_step": cumulative_step,
                    "metrics": _jsonable(metrics),
                },
            )

        formal = validated.spec.experiment_level == "formal_expert"
        checkpoint_callback = None
        if formal:
            requested_milestones = tuple(
                int(value) for value in training_config["checkpoint_schedule_requested"]
            )
            milestones = phase_u_invocation_milestones(
                requested_milestones,
                rollout_block_size=validated.interaction_budget.training.ppo_rollout_block_size,
                cumulative_start=validated.cumulative_training_start,
                invocation_transitions=validated.interaction_budget.training.effective_total_transitions,
            )
            tracker = PhaseCheckpointTracker(milestones)

            def checkpoint_callback(step: int, _make_policy: Any, params: Any) -> None:
                nonlocal observed_training_progress
                nonlocal fixed_evaluation_transitions, last_checkpoint
                nonlocal candidate_acquisition_transitions
                nonlocal continuation_labeling_transitions
                cumulative_step = validated.cumulative_training_start + int(step)
                observed_training_progress = max(
                    observed_training_progress, cumulative_step
                )
                if int(step) == 0 and validated.cumulative_training_start > 0:
                    return
                checkpoint = _save_phase_expert_inference_checkpoint(
                    environment,
                    params,
                    root / "orbax",
                    step=cumulative_step,
                    initial_action_std=policy_initial_action_std,
                )
                checkpoint_contract = _checkpoint_contract(
                    validated,
                    training_config,
                    cumulative_transitions=cumulative_step,
                    parent_checkpoint=last_checkpoint,
                )
                sidecar = write_phase_expert_checkpoint_sidecar(
                    checkpoint, checkpoint_contract
                )
                last_checkpoint = str(checkpoint.resolve())
                update_phase_expert_status(
                    root,
                    {
                        "status": "running",
                        "invocation_training_transitions": int(step),
                        "cumulative_training_transitions": cumulative_step,
                        "fixed_evaluation_transitions": fixed_evaluation_transitions,
                        "candidate_acquisition_transitions": candidate_acquisition_transitions,
                        "continuation_labeling_transitions": continuation_labeling_transitions,
                        "last_checkpoint": last_checkpoint,
                        "checkpoint_cadence_transitions": validated.interaction_budget.training.ppo_rollout_block_size,
                    },
                )
                milestone = tracker.claim(cumulative_step)
                if milestone is None:
                    return
                evaluation_root = (
                    root / "evaluations" / f"{milestone.effective:012d}"
                )
                fixed_before = fixed_evaluation_transitions

                def observe_fixed(count: int) -> None:
                    nonlocal fixed_evaluation_transitions
                    fixed_evaluation_transitions += int(count)

                evaluation, consumed = _fixed_phase_u_evaluation(
                    environment,
                    params,
                    validated.seeds.evaluation_seeds,
                    int(training_config["evaluation"]["episode_horizon"]),
                    evaluation_root / "failure_videos",
                    transition_observer=observe_fixed,
                )
                if fixed_evaluation_transitions - fixed_before != consumed:
                    raise RuntimeError("fixed evaluation interaction accounting drift")
                report = {
                    "requested_training_transitions": milestone.requested,
                    "effective_training_transitions": milestone.effective,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sidecar["recursive_checkpoint_sha256"],
                    "fixed_evaluation": evaluation,
                    "fixed_evaluation_transitions": consumed,
                }
                checkpoint_evaluations.append(report)
                checkpoint_gate = evaluate_phase_u_checkpoint_gate(
                    checkpoint_evaluations
                )
                report["checkpoint_gate"] = checkpoint_gate
                _write_json_atomic(evaluation_root / "fixed_evaluation.json", report)
                if checkpoint_gate["pause"]:
                    append_phase_expert_metrics(
                        root,
                        {
                            "training_step": milestone.effective,
                            "checkpoint_evaluation": report,
                        },
                    )
                    raise RuntimeError(
                        "Phase U checkpoint gate pause: "
                        + ", ".join(checkpoint_gate["reasons"])
                    )
                acquisition_config = training_config.get("candidate_acquisition", {})
                physical = evaluation.get("physical_metrics", {})
                if float(physical.get("apex_band_success_rate", 0.0)) > 0.0:
                    from .bank import SnapshotBank
                    from .phase_candidate_acquisition import (
                        acquire_phase_u_candidate_parents,
                        probe_phase_u_continuations,
                        pytree_sha256,
                        require_candidate_acquisition_integrity,
                    )

                    parent_count = int(
                        acquisition_config["maximum_parent_rollouts_per_checkpoint"]
                    )
                    acquisition_seeds = tuple(
                        int(validated.spec.seed + 1_000_000 + milestone.effective + index)
                        for index in range(parent_count)
                    )
                    provenance = {
                        "xml_sha256": AUTHORITATIVE_XML_SHA256,
                        "config_sha256": _sha256_file(validated.spec.config_path),
                        "action_mapping_version": ACTION_MAPPING_VERSION,
                        "policy_params_sha256": pytree_sha256(params),
                        "policy_config_sha256": _sha256_file(
                            validated.spec.training_config_path
                        ),
                        "policy_manifest_sha256": _sha256_file(root / "run_manifest.json"),
                        "normalizer_sha256": pytree_sha256(params[0]),
                        "source_fingerprint": phase_expert_source_tree_sha256(),
                    }
                    acquisition_before = candidate_acquisition_transitions

                    def observe_acquisition(count: int) -> None:
                        nonlocal candidate_acquisition_transitions
                        candidate_acquisition_transitions += int(count)

                    acquisition = acquire_phase_u_candidate_parents(
                        environment,
                        params,
                        fixed_evaluation=evaluation,
                        seeds=acquisition_seeds,
                        horizon=int(acquisition_config["episode_horizon"]),
                        provenance=provenance,
                        minimum_independent_successful_parents=int(
                            acquisition_config[
                                "minimum_independent_successful_parents"
                            ]
                        ),
                        transition_observer=observe_acquisition,
                    )
                    if (
                        candidate_acquisition_transitions - acquisition_before
                        != acquisition.environment_transitions
                    ):
                        raise RuntimeError("candidate acquisition interaction accounting drift")
                    require_candidate_acquisition_integrity(acquisition.gate)
                    ceiling = int(
                        acquisition_config["transition_ceiling_per_checkpoint"]
                    )
                    if acquisition.environment_transitions > ceiling:
                        raise RuntimeError(
                            "candidate acquisition exceeded its checkpoint transition ceiling"
                        )
                    acquisition_report = {
                        "requested_training_transitions": milestone.requested,
                        "effective_training_transitions": milestone.effective,
                        "gate": dict(acquisition.gate),
                        "parents": [asdict(parent) for parent in acquisition.parents],
                        "candidate_record_count": len(acquisition.records),
                        "candidate_acquisition_transitions": acquisition.environment_transitions,
                        "formal_v_up": False,
                        "formal_tube_up": False,
                    }
                    if acquisition.gate["eligible"]:
                        bank_path = evaluation_root / "phase_u_candidates.bank"
                        SnapshotBank(
                            list(acquisition.records),
                            {
                                "artifact_role": "unlabeled_phase_u_candidates",
                                "source_policy_hash": provenance[
                                    "policy_params_sha256"
                                ],
                                "training_guidance_only": False,
                                "certified_safe": False,
                            },
                        ).save(bank_path)
                        acquisition_report["candidate_bank"] = str(
                            bank_path.resolve()
                        )
                        acquisition_report["candidate_bank_sha256"] = _sha256_file(
                            bank_path
                        )
                        continuation_config = training_config[
                            "continuation_diagnostic"
                        ]
                        selected_records = tuple(acquisition.records)[
                            : int(continuation_config["maximum_states_per_checkpoint"])
                        ]
                        protocol_payload = {
                            "phase": PHASE_PROPULSION_ASCENT,
                            "branches_per_state": 1,
                            "episode_horizon": int(
                                continuation_config["episode_horizon"]
                            ),
                            "restore_mode": "timing_explicit_independent_reconstruction",
                            "checkpoint_policy_hash": provenance[
                                "policy_params_sha256"
                            ],
                            "policy_mode": "stochastic",
                            "branch_seed_namespace": "phase_u_checkpoint_continuation_v1",
                            "branch_seed_derivation": "acquisition_seed_plus_2000000",
                        }
                        protocol_hash = _canonical_payload_hash(protocol_payload)
                        continuation_seeds = tuple(
                            int(seed + 2_000_000) for seed in acquisition_seeds
                        )[: len(selected_records)]
                        continuation_before = continuation_labeling_transitions

                        def observe_continuation(count: int) -> None:
                            nonlocal continuation_labeling_transitions
                            continuation_labeling_transitions += int(count)

                        labeled, continuation_consumed = probe_phase_u_continuations(
                            environment,
                            params,
                            selected_records,
                            seeds=continuation_seeds,
                            horizon=protocol_payload["episode_horizon"],
                            source_policy_hash=provenance[
                                "policy_params_sha256"
                            ],
                            protocol_hash=protocol_hash,
                            seed_namespace=protocol_payload[
                                "branch_seed_namespace"
                            ],
                            transition_observer=observe_continuation,
                        )
                        if (
                            continuation_labeling_transitions - continuation_before
                            != continuation_consumed
                        ):
                            raise RuntimeError("continuation interaction accounting drift")
                        continuation_ceiling = int(
                            continuation_config[
                                "transition_ceiling_per_checkpoint"
                            ]
                        )
                        if continuation_consumed > continuation_ceiling:
                            raise RuntimeError(
                                "continuation diagnostic exceeded its checkpoint transition ceiling"
                            )
                        provisional_path = (
                            evaluation_root / "phase_u_provisional_continuations.bank"
                        )
                        SnapshotBank(
                            list(labeled),
                            {
                                "artifact_role": "provisional_phase_u_continuation_diagnostic",
                                "label_source_policy_hash": provenance[
                                    "policy_params_sha256"
                                ],
                                "label_protocol_hash": protocol_hash,
                                "formal_v_up": False,
                                "formal_tube_up": False,
                                "training_guidance_only": False,
                                "certified_safe": False,
                            },
                        ).save(provisional_path)
                        acquisition_report["continuation_diagnostic"] = {
                            "record_count": len(labeled),
                            "environment_transitions": continuation_consumed,
                            "protocol": protocol_payload,
                            "protocol_hash": protocol_hash,
                            "bank": str(provisional_path.resolve()),
                            "bank_sha256": _sha256_file(provisional_path),
                            "provisional": True,
                            "formal_v_up": False,
                            "formal_tube_up": False,
                        }
                    candidate_acquisition_reports.append(acquisition_report)
                    _write_json_atomic(
                        evaluation_root / "candidate_acquisition.json",
                        acquisition_report,
                    )
                append_phase_expert_metrics(
                    root,
                    {
                        "training_step": milestone.effective,
                        "checkpoint_evaluation": report,
                    },
                )
                update_phase_expert_status(
                    root,
                    {
                        "status": "running",
                        "invocation_training_transitions": (
                            milestone.effective - validated.cumulative_training_start
                        ),
                        "cumulative_training_transitions": milestone.effective,
                        "fixed_evaluation_transitions": fixed_evaluation_transitions,
                        "candidate_acquisition_transitions": candidate_acquisition_transitions,
                        "continuation_labeling_transitions": continuation_labeling_transitions,
                        "last_checkpoint": last_checkpoint,
                        "last_requested_checkpoint": milestone.requested,
                    },
                )

        train_fn = make_ppo_train_fn(
            timesteps=validated.interaction_budget.training.effective_total_transitions,
            episode_length=int(layout["episode_horizon"]),
            num_envs=int(layout["num_parallel_envs"]),
            num_eval_envs=int(layout["num_eval_envs"]),
            num_evals=validated.interaction_budget.training.num_evals,
            seed=validated.spec.seed,
            learning_rate=float(optimization["learning_rate"]),
            entropy_cost=float(optimization["entropy_cost"]),
            reward_scaling=float(optimization["reward_scaling"]),
            checkpoint_dir=(None if formal else root / "orbax"),
            unroll_length=int(layout["unroll_length"]),
            batch_size=int(layout["batch_size"]),
            num_minibatches=int(layout["num_minibatches"]),
            num_updates_per_batch=int(layout["num_updates_per_batch"]),
            discounting=float(optimization["discounting"]),
            gae_lambda=float(optimization["gae_lambda"]),
            clipping_epsilon=float(optimization["clipping_epsilon"]),
            max_grad_norm=float(optimization["max_grad_norm"]),
            initial_action_std=policy_initial_action_std,
            restore_checkpoint_path=validated.spec.restore_checkpoint,
            policy_params_fn=checkpoint_callback,
            full_reset=True,
            run_evals=bool(training_config.get("run_brax_evaluation", True)),
        )
        _, params, final_metrics = train_fn(
            environment=environment, progress_fn=progress, eval_env=environment
        )
        checkpoints = [
            path
            for path in (root / "orbax").iterdir()
            if path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))
        ] if (root / "orbax").is_dir() else []
        if not formal:
            if not checkpoints:
                raise RuntimeError("smoke training completed without a checkpoint")
            checkpoint_contract = _checkpoint_contract(
                validated,
                training_config,
                cumulative_transitions=(
                    validated.cumulative_training_start
                    + validated.interaction_budget.training.effective_total_transitions
                ),
                parent_checkpoint=validated.spec.restore_checkpoint,
            )
            for checkpoint in checkpoints:
                write_phase_expert_checkpoint_sidecar(checkpoint, checkpoint_contract)
        if formal:
            if tracker.claimed_effective != tuple(
                milestone.effective for milestone in milestones
            ):
                raise RuntimeError("formal Phase U run missed a fixed checkpoint callback")
            evaluation = checkpoint_evaluations[-1]["fixed_evaluation"]
            fixed_transitions = fixed_evaluation_transitions
            _write_json_atomic(
                root / "checkpoint_evaluations.json",
                {"checkpoints": checkpoint_evaluations},
            )
        else:
            fixed_before = fixed_evaluation_transitions

            def observe_smoke_fixed(count: int) -> None:
                nonlocal fixed_evaluation_transitions
                fixed_evaluation_transitions += int(count)

            evaluation, fixed_transitions = _fixed_phase_u_evaluation(
                environment,
                params,
                validated.seeds.evaluation_seeds,
                int(training_config["evaluation"]["episode_horizon"]),
                root / "failure_videos",
                transition_observer=observe_smoke_fixed,
            )
            if fixed_evaluation_transitions - fixed_before != fixed_transitions:
                raise RuntimeError("fixed evaluation interaction accounting drift")
            _write_json_atomic(root / "fixed_evaluation.json", evaluation)
        actual_interactions = completed_phase_expert_interaction_accounting(
            validated.interaction_budget,
            fixed_evaluation_transitions=fixed_transitions,
            candidate_acquisition_transitions=candidate_acquisition_transitions,
            continuation_labeling_transitions=continuation_labeling_transitions,
        )
        result = {
            "status": "completed",
            "evidence_level": (
                "phase_u_checkpoint_training" if formal else "engineering_smoke_only"
            ),
            "brax_evaluation_transition_ceiling": validated.interaction_budget.brax_evaluation_transition_ceiling,
            "combined_interaction_ceiling": validated.interaction_budget.combined_transition_ceiling,
            **actual_interactions,
            "final_metrics": _jsonable(final_metrics),
            "fixed_evaluation": evaluation,
            "checkpoint_evaluations": checkpoint_evaluations,
            "inherited_checkpoint_evaluation_count": len(
                inherited_checkpoint_evaluations
            ),
            "candidate_acquisition": candidate_acquisition_reports,
            "candidate_acquisition_transitions": candidate_acquisition_transitions,
            "continuation_labeling_transitions": continuation_labeling_transitions,
            "promotion_authorized": False,
            "next_gate_authorized": False,
            "cumulative_training_start": validated.cumulative_training_start,
            "cumulative_training_end": (
                validated.cumulative_training_start
                + validated.interaction_budget.training.effective_total_transitions
            ),
        }
        update_phase_expert_status(root, result)
        return result
    except Exception as exc:
        preserved_videos = (
            checkpoint_evaluations[-1]
            .get("fixed_evaluation", {})
            .get("failure_videos", [])
            if checkpoint_evaluations
            else []
        )
        update_phase_expert_status(
            root,
            {
                "status": "gate_pause",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "observed_training_progress": observed_training_progress,
                "fixed_evaluation_transitions": fixed_evaluation_transitions,
                "candidate_acquisition_transitions": candidate_acquisition_transitions,
                "continuation_labeling_transitions": continuation_labeling_transitions,
                "failure_video_status": (
                    "preserved_from_last_checkpoint_evaluation"
                    if preserved_videos
                    else "not_applicable_without_captured_dynamic_failure_frames"
                ),
                "failure_videos": preserved_videos,
                "promotion_authorized": False,
                **partial_phase_expert_interaction_accounting(
                    cumulative_training_start=validated.cumulative_training_start,
                    observed_training_progress=observed_training_progress,
                    fixed_evaluation_transitions=fixed_evaluation_transitions,
                    candidate_acquisition_transitions=candidate_acquisition_transitions,
                    continuation_labeling_transitions=continuation_labeling_transitions,
                ),
            },
        )
        raise
