"""Strict configuration loading for the JIT Phase U engineering stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
class FormalTrainingConfig:
    checkpoint_transitions: tuple[int, ...]
    fixed_evaluation_transitions: tuple[int, ...]
    resume_semantics: str
    requested_transitions: int = 998_400
    block_transitions: int = 25_600

    @property
    def formal_blocks(self) -> int:
        return self.requested_transitions // self.block_transitions


@dataclass(frozen=True)
class ActionConfig:
    base_rear_speed: float
    rear_speed_delta: float
    knee_target_delta: float | None = None
    joint_target_semantics: str = "incremental_knee"


@dataclass(frozen=True)
class EventConfig:
    jump_zone_x_min: float
    jump_zone_x_max: float
    min_ascent_velocity: float
    apex_height: float
    min_descent_velocity: float


@dataclass(frozen=True)
class ResetConfig:
    keyframe: str
    initial_forward_velocity: float
    airborne_rsi_probability: float
    airborne_rsi_x_min: float
    airborne_rsi_x_max: float
    airborne_rsi_z_min: float
    airborne_rsi_z_max: float
    airborne_rsi_vx_min: float
    airborne_rsi_vx_max: float
    airborne_rsi_vz_min: float
    airborne_rsi_vz_max: float


@dataclass(frozen=True)
class PhysicalLimits:
    max_abs_roll: float
    max_abs_pitch: float
    max_backward_distance: float


@dataclass(frozen=True)
class RewardConfig:
    roll_coeff: float
    pitch_coeff: float
    yaw_coeff: float
    speed_coeff: float
    survival_reward: float
    height_coeff: float
    desired_velocity: float
    speed_sigma: float
    jump_reward_min_height: float
    peak_reward_height: float
    max_beneficial_height: float
    action_smoothness_scale: float
    action_magnitude_scale: float
    action_coeff: float
    pitch_angular_velocity_coeff: float
    joint_energy_penalty_coeff: float
    apex_success_bonus: float
    illegal_contact_penalty: float
    physical_failure_penalty: float
    timeout_penalty: float
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
    reset: ResetConfig
    events: EventConfig
    physical_limits: PhysicalLimits
    reward: RewardConfig
    ppo: PPOConfig
    formal: FormalTrainingConfig | None


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


def _validate_formal(
    schema: str, ppo: PPOConfig, formal: FormalTrainingConfig
) -> None:
    checkpoints = formal.checkpoint_transitions
    evaluations = formal.fixed_evaluation_transitions
    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("formal checkpoints must start at zero")
    if checkpoints[-1] != ppo.requested_transitions:
        approved_target = (
            998_400
            if schema == "jit_phase_u_formal_v2"
            else 4_988_928
        )
        raise ValueError(
            "formal checkpoints must end at requested transitions; "
            f"approved target is {approved_target}"
        )
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("formal checkpoints must be strictly increasing")
    if any(step % ppo.block_transitions for step in checkpoints):
        raise ValueError("formal checkpoints must be block-aligned")
    if schema == "jit_phase_u_formal_v2":
        if ppo.requested_transitions != 998_400:
            raise ValueError("formal requested_transitions must equal 998400")
        if ppo.block_transitions != 25_600:
            raise ValueError("formal PPO block must equal 25600 transitions")
        if ppo.num_evals != 40:
            raise ValueError("formal num_evals must equal 40")
        if ppo.seed != 820101:
            raise ValueError("formal training seed must equal 820101")
        if ppo.held_out_seeds != tuple(range(920001, 920009)):
            raise ValueError("formal held-out seeds must equal 920001 through 920008")
        expected_checkpoints = (0, 102_400, 256_000, 512_000, 742_400, 998_400)
    else:
        if ppo.requested_transitions != 4_988_928:
            raise ValueError("formal requested_transitions must equal 4988928")
        if ppo.block_transitions != 24_576:
            raise ValueError("formal PPO block must equal 24576 transitions")
        if ppo.num_evals != 204:
            raise ValueError("formal num_evals must equal 204")
        expected_seed = 820301 if schema == "jit_phase_u_formal_v4" else 820201
        expected_held_out = (
            tuple(range(940001, 940009))
            if schema == "jit_phase_u_formal_v4"
            else tuple(range(930001, 930009))
        )
        if ppo.seed != expected_seed:
            raise ValueError(f"formal training seed must equal {expected_seed}")
        if ppo.held_out_seeds != expected_held_out:
            raise ValueError("formal held-out seeds do not match the approved namespace")
        expected_checkpoints = (
            0,
            245_760,
            983_040,
            2_506_752,
            3_981_312,
            4_988_928,
        )
    if checkpoints != expected_checkpoints:
        raise ValueError("formal config must use the exact checkpoint schedule")
    if tuple(sorted(set(evaluations))) != evaluations:
        raise ValueError("formal evaluations must be strictly increasing")
    if evaluations != checkpoints[1:]:
        raise ValueError("formal evaluation must run at every nonzero checkpoint")
    if formal.resume_semantics != "parameter_warm_start_optimizer_reset":
        raise ValueError("formal resume_semantics must declare optimizer reset")


def _validate_approved_v2_method(
    schema: str,
    *,
    model: Mapping[str, Any],
    action: ActionConfig,
    reset: ResetConfig,
    events: EventConfig,
    physical_limits: PhysicalLimits,
    reward: RewardConfig,
    ppo: PPOConfig,
) -> None:
    expected_model = {
        "xml_path": "assets/orange_bike_4kg_horizontal.xml",
        "xml_sha256": "e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192",
        "reference_path": "data/reference_jump.csv",
        "reference_sha256": "612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f",
        "mjx_impl": "warp",
        "naconmax": 4096,
        "njmax": 256,
    }
    expected_action = ActionConfig(12.0, 12.0, 0.2)
    expected_reset = ResetConfig(
        keyframe="initial_state",
        initial_forward_velocity=2.0,
        airborne_rsi_probability=0.05,
        airborne_rsi_x_min=2.7,
        airborne_rsi_x_max=2.9,
        airborne_rsi_z_min=1.8,
        airborne_rsi_z_max=2.2,
        airborne_rsi_vx_min=1.8,
        airborne_rsi_vx_max=2.2,
        airborne_rsi_vz_min=0.8,
        airborne_rsi_vz_max=1.2,
    )
    expected_events = EventConfig(2.5, 3.1, 0.05, 0.5, 0.05)
    expected_limits = PhysicalLimits(
        max_abs_roll=0.6108652381980153,
        max_abs_pitch=1.3089969389957472,
        max_backward_distance=1.0,
    )
    expected_reward = RewardConfig(
        roll_coeff=3.0,
        pitch_coeff=1.0,
        yaw_coeff=0.3,
        speed_coeff=0.2,
        survival_reward=1.5,
        height_coeff=20.0,
        desired_velocity=3.5,
        speed_sigma=0.5,
        jump_reward_min_height=0.35,
        peak_reward_height=0.5,
        max_beneficial_height=0.8,
        action_smoothness_scale=0.0001,
        action_magnitude_scale=0.1,
        action_coeff=1.5,
        pitch_angular_velocity_coeff=0.15,
        joint_energy_penalty_coeff=2.0,
        apex_success_bonus=50.0,
        illegal_contact_penalty=30.0,
        physical_failure_penalty=30.0,
        timeout_penalty=10.0,
        total_min=-50.0,
        total_max=50.0,
    )
    common_ppo = dict(
        num_parallel_envs=1024,
        episode_horizon=200,
        unroll_length=25,
        batch_size=128,
        num_minibatches=8,
        num_updates_per_batch=1,
        num_eval_envs=8,
        learning_rate=0.0001,
        entropy_cost=0.001,
        reward_scaling=0.1,
        discounting=0.995,
        gae_lambda=0.97,
        clipping_epsilon=0.1,
        max_grad_norm=0.75,
        held_out_seeds=tuple(range(920001, 920009)),
    )
    if schema == "jit_phase_u_formal_v2":
        expected_ppo = PPOConfig(
            **common_ppo,
            requested_transitions=998_400,
            num_evals=40,
            seed=820101,
        )
    else:
        expected_ppo = PPOConfig(
            **common_ppo,
            requested_transitions=25_600,
            num_evals=1,
            seed=820001,
        )
    approved = {
        "model": (dict(model), expected_model),
        "action": (action, expected_action),
        "reset": (reset, expected_reset),
        "events": (events, expected_events),
        "physical_limits": (physical_limits, expected_limits),
        "reward": (reward, expected_reward),
        "ppo": (ppo, expected_ppo),
    }
    for section, (actual, expected) in approved.items():
        if actual != expected:
            raise ValueError(f"approved v2 {section} method contract drift")


def _validate_approved_absolute_method(
    schema: str,
    *,
    model: Mapping[str, Any],
    action: ActionConfig,
    reset: ResetConfig,
    events: EventConfig,
    physical_limits: PhysicalLimits,
    reward: RewardConfig,
    ppo: PPOConfig,
) -> None:
    expected_model = {
        "xml_path": "assets/orange_bike_4kg_horizontal.xml",
        "xml_sha256": "e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192",
        "reference_path": "data/reference_jump.csv",
        "reference_sha256": "612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f",
        "mjx_impl": "warp",
        "naconmax": 4096,
        "njmax": 256,
    }
    expected_action = ActionConfig(
        base_rear_speed=12.0,
        rear_speed_delta=12.0,
        knee_target_delta=None,
        joint_target_semantics="keyframe_centered_absolute",
    )
    expected_reset = ResetConfig(
        keyframe="initial_state",
        initial_forward_velocity=2.0,
        airborne_rsi_probability=0.05,
        airborne_rsi_x_min=2.7,
        airborne_rsi_x_max=2.9,
        airborne_rsi_z_min=1.8,
        airborne_rsi_z_max=2.2,
        airborne_rsi_vx_min=1.8,
        airborne_rsi_vx_max=2.2,
        airborne_rsi_vz_min=0.8,
        airborne_rsi_vz_max=1.2,
    )
    expected_events = EventConfig(2.5, 3.1, 0.05, 0.5, 0.05)
    expected_limits = PhysicalLimits(
        max_abs_roll=0.6108652381980153,
        max_abs_pitch=1.3089969389957472,
        max_backward_distance=1.0,
    )
    expected_reward = RewardConfig(
        roll_coeff=3.0,
        pitch_coeff=1.0,
        yaw_coeff=0.3,
        speed_coeff=0.2,
        survival_reward=1.5,
        height_coeff=20.0,
        desired_velocity=3.5,
        speed_sigma=0.5,
        jump_reward_min_height=0.35,
        peak_reward_height=0.5,
        max_beneficial_height=0.8,
        action_smoothness_scale=0.0001,
        action_magnitude_scale=0.1,
        action_coeff=1.5,
        pitch_angular_velocity_coeff=0.15,
        joint_energy_penalty_coeff=2.0,
        apex_success_bonus=50.0,
        illegal_contact_penalty=30.0,
        physical_failure_penalty=30.0,
        timeout_penalty=10.0,
        total_min=-50.0,
        total_max=50.0,
    )
    is_v4 = schema.endswith("_v4")
    held_out_seeds = (
        tuple(range(940001, 940009))
        if is_v4
        else tuple(range(930001, 930009))
    )
    common_ppo = dict(
        num_parallel_envs=384,
        episode_horizon=200,
        unroll_length=64,
        batch_size=16,
        num_minibatches=24,
        num_updates_per_batch=8,
        num_eval_envs=8,
        learning_rate=0.0001,
        entropy_cost=0.01,
        reward_scaling=0.1,
        discounting=0.99,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        max_grad_norm=0.5,
        held_out_seeds=held_out_seeds,
    )
    if schema in {"jit_phase_u_formal_v3", "jit_phase_u_formal_v4"}:
        expected_ppo = PPOConfig(
            **common_ppo,
            requested_transitions=4_988_928,
            num_evals=204,
            seed=820301 if is_v4 else 820201,
        )
    else:
        expected_ppo = PPOConfig(
            **common_ppo,
            requested_transitions=24_576,
            num_evals=1,
            seed=820300 if is_v4 else 820200,
        )
    approved = {
        "model": (dict(model), expected_model),
        "action": (action, expected_action),
        "reset": (reset, expected_reset),
        "events": (events, expected_events),
        "physical_limits": (physical_limits, expected_limits),
        "reward": (reward, expected_reward),
        "ppo": (ppo, expected_ppo),
    }
    for section, (actual, expected) in approved.items():
        if actual != expected:
            version = "v4" if is_v4 else "v3"
            raise ValueError(f"approved {version} {section} method contract drift")


def resolve_config_payload(payload: Mapping[str, Any]) -> ResolvedConfig:
    payload = dict(payload)
    schema = str(payload.get("schema", ""))
    if schema not in {
        "jit_phase_u_engineering_smoke_v2",
        "jit_phase_u_formal_v2",
        "jit_phase_u_engineering_smoke_v3",
        "jit_phase_u_formal_v3",
        "jit_phase_u_engineering_smoke_v4",
        "jit_phase_u_formal_v4",
    }:
        raise ValueError("unsupported JIT config schema")
    if payload.get("phase") != "propulsion_ascent":
        raise ValueError("only propulsion_ascent is implemented")
    if schema.endswith("_v4") and payload.get("training_wrapper") != {
        "full_reset": True
    }:
        raise ValueError("approved v4 training wrapper must use full_reset=true")
    if not math.isclose(CTRL_DT / SIM_DT, N_SUBSTEPS, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("timing constants do not produce exactly four substeps")
    if tuple(payload.get("action_order", ACTION_ORDER)) != ACTION_ORDER:
        raise ValueError("action order does not match the immutable contract")

    ppo_payload = dict(payload["ppo"])
    ppo_payload["held_out_seeds"] = tuple(int(x) for x in ppo_payload["held_out_seeds"])
    ppo = _dataclass_from(PPOConfig, ppo_payload)
    _validate_ppo(ppo)
    formal: FormalTrainingConfig | None = None
    if schema in {
        "jit_phase_u_formal_v2",
        "jit_phase_u_formal_v3",
        "jit_phase_u_formal_v4",
    }:
        formal_payload = dict(payload.get("formal", {}))
        for key in ("checkpoint_transitions", "fixed_evaluation_transitions"):
            formal_payload[key] = tuple(int(x) for x in formal_payload.get(key, ()))
        formal = _dataclass_from(FormalTrainingConfig, formal_payload)
        formal = replace(
            formal,
            requested_transitions=ppo.requested_transitions,
            block_transitions=ppo.block_transitions,
        )
        _validate_formal(schema, ppo, formal)
    elif "formal" in payload:
        raise ValueError("smoke config must not contain formal settings")
    events = _dataclass_from(EventConfig, payload["events"])
    reset = _dataclass_from(ResetConfig, payload["reset"])
    if events.jump_zone_x_min >= events.jump_zone_x_max:
        raise ValueError("jump-window bounds must be increasing")
    for name in ("min_ascent_velocity", "apex_height", "min_descent_velocity"):
        _positive(f"events.{name}", getattr(events, name))
    if not 0.0 <= reset.airborne_rsi_probability <= 1.0:
        raise ValueError("reset.airborne_rsi_probability must be in [0, 1]")
    for lower, upper in (
        ("airborne_rsi_x_min", "airborne_rsi_x_max"),
        ("airborne_rsi_z_min", "airborne_rsi_z_max"),
        ("airborne_rsi_vx_min", "airborne_rsi_vx_max"),
        ("airborne_rsi_vz_min", "airborne_rsi_vz_max"),
    ):
        if getattr(reset, lower) >= getattr(reset, upper):
            raise ValueError(f"reset.{lower}/{upper} bounds must be increasing")
    reward = _dataclass_from(RewardConfig, payload["reward"])
    if not reward.total_min < reward.total_max:
        raise ValueError("reward total bounds must be increasing")
    if not (
        reward.jump_reward_min_height
        < reward.peak_reward_height
        < reward.max_beneficial_height
    ):
        raise ValueError("reward height thresholds must be increasing")

    action = _dataclass_from(ActionConfig, payload["action"])
    physical_limits = _dataclass_from(PhysicalLimits, payload["physical_limits"])
    validator = (
        _validate_approved_absolute_method
        if schema.endswith(("_v3", "_v4"))
        else _validate_approved_v2_method
    )
    validator(
        schema,
        model=payload["model"],
        action=action,
        reset=reset,
        events=events,
        physical_limits=physical_limits,
        reward=reward,
        ppo=ppo,
    )
    return ResolvedConfig(
        schema=schema,
        phase=str(payload["phase"]),
        raw=payload,
        config_sha256=canonical_sha256(payload),
        model=dict(payload["model"]),
        action=action,
        reset=reset,
        events=events,
        physical_limits=physical_limits,
        reward=reward,
        ppo=ppo,
        formal=formal,
    )


def load_config(path: Path) -> ResolvedConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return resolve_config_payload(payload)
