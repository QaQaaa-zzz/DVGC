"""Strict configuration loading for the JIT Phase U engineering stack."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .constants import ACTION_ORDER, CTRL_DT, N_SUBSTEPS, SIM_DT


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(name: str, value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


@dataclass(frozen=True)
class PPOConfig:
    num_parallel_envs: int
    episode_horizon: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    requested_transitions: int
    num_eval_envs: int
    num_evals: int
    learning_rate: float
    entropy_cost: float
    reward_scaling: float
    discounting: float
    gae_lambda: float
    clipping_epsilon: float
    max_grad_norm: float
    seed: int
    held_out_seeds: tuple[int, ...]

    @property
    def block_transitions(self) -> int:
        return self.unroll_length * self.batch_size * self.num_minibatches


@dataclass(frozen=True)
class ApexConfig:
    max_abs_vertical_velocity: float
    min_clearance: float
    max_abs_roll: float
    max_abs_pitch: float
    max_angular_speed: float
    min_forward_velocity: float
    relative_x_min: float
    relative_x_max: float


@dataclass(frozen=True)
class ActionConfig:
    base_rear_speed: float
    rear_speed_delta: float
    knee_target_delta: float


@dataclass(frozen=True)
class EventConfig:
    window_relative_x_min: float
    window_relative_x_max: float
    airborne_confirm_ticks: int
    stable_airborne_min_clearance: float
    ascending_min_vertical_velocity: float


@dataclass(frozen=True)
class PhysicalLimits:
    max_abs_roll: float
    max_abs_pitch: float
    max_backward_distance: float
    platform_back_margin: float


@dataclass(frozen=True)
class RewardConfig:
    drive_weight: float
    window_bonus: float
    liftoff_bonus: float
    stable_airborne_bonus: float
    ascent_weight: float
    clearance_weight: float
    apex_progress_weight: float
    apex_success_bonus: float
    attitude_penalty_weight: float
    rate_penalty_weight: float
    action_smoothness_weight: float
    action_magnitude_weight: float
    illegal_contact_penalty: float
    physical_failure_penalty: float
    timeout_penalty: float
    target_forward_velocity: float
    total_min: float
    total_max: float


@dataclass(frozen=True)
class ResolvedConfig:
    schema: str
    phase: str
    raw: Mapping[str, Any]
    config_sha256: str
    model: Mapping[str, Any]
    action: ActionConfig
    reset: Mapping[str, Any]
    events: EventConfig
    apex: ApexConfig
    physical_limits: PhysicalLimits
    reward: RewardConfig
    ppo: PPOConfig


def _dataclass_from(cls: type, payload: Mapping[str, Any]):
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {cls.__name__}: {exc}") from exc


def _validate_ppo(ppo: PPOConfig) -> None:
    integer_fields = (
        "num_parallel_envs",
        "episode_horizon",
        "unroll_length",
        "batch_size",
        "num_minibatches",
        "num_updates_per_batch",
        "requested_transitions",
        "num_eval_envs",
        "num_evals",
    )
    if any(int(getattr(ppo, name)) <= 0 for name in integer_fields):
        raise ValueError("PPO integer dimensions must be positive")
    sequence_count = ppo.batch_size * ppo.num_minibatches
    if sequence_count % ppo.num_parallel_envs:
        raise ValueError(
            "batch_size * num_minibatches must be divisible by num_parallel_envs"
        )
    if ppo.requested_transitions % ppo.block_transitions:
        raise ValueError("requested_transitions must contain whole PPO blocks")
    if len(set(ppo.held_out_seeds)) != len(ppo.held_out_seeds):
        raise ValueError("held_out_seeds must be unique")
    if ppo.seed in ppo.held_out_seeds:
        raise ValueError("training and held-out seeds must be disjoint")
    for name in (
        "learning_rate",
        "entropy_cost",
        "reward_scaling",
        "discounting",
        "gae_lambda",
        "clipping_epsilon",
        "max_grad_norm",
    ):
        _positive(f"ppo.{name}", getattr(ppo, name))


def load_config(path: Path) -> ResolvedConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("phase") != "propulsion_ascent":
        raise ValueError("only propulsion_ascent is implemented")
    if not math.isclose(CTRL_DT / SIM_DT, N_SUBSTEPS, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("timing constants do not produce exactly four substeps")
    if tuple(payload.get("action_order", ACTION_ORDER)) != ACTION_ORDER:
        raise ValueError("action order does not match the immutable contract")

    ppo_payload = dict(payload["ppo"])
    ppo_payload["held_out_seeds"] = tuple(int(x) for x in ppo_payload["held_out_seeds"])
    ppo = _dataclass_from(PPOConfig, ppo_payload)
    _validate_ppo(ppo)
    events = _dataclass_from(EventConfig, payload["events"])
    apex = _dataclass_from(ApexConfig, payload["apex"])
    if events.window_relative_x_min >= events.window_relative_x_max:
        raise ValueError("jump-window bounds must be increasing")
    if apex.relative_x_min >= apex.relative_x_max:
        raise ValueError("Apex relative-x bounds must be increasing")
    if int(events.airborne_confirm_ticks) <= 0:
        raise ValueError("airborne_confirm_ticks must be positive")

    return ResolvedConfig(
        schema=str(payload["schema"]),
        phase=str(payload["phase"]),
        raw=payload,
        config_sha256=canonical_sha256(payload),
        model=dict(payload["model"]),
        action=_dataclass_from(ActionConfig, payload["action"]),
        reset=dict(payload["reset"]),
        events=events,
        apex=apex,
        physical_limits=_dataclass_from(PhysicalLimits, payload["physical_limits"]),
        reward=_dataclass_from(RewardConfig, payload["reward"]),
        ppo=ppo,
    )
