"""Compare frozen pi_up_star and pi_unified on the same canonical natural start.

This is a bounded diagnostic, not a statistical evaluation.  It is used only to
identify whether the Round-0 unified failure occurs before the task region that
pi_up_star can already solve from the identical physical reset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from .constants import END_REASONS
from .env import TwoPhaseBikeEnv
from .evaluation import capture_episode, save_episode_trace, summarize_phase_u
from .expert_freeze import load_frozen_manifest, verify_frozen_record
from .ppo import make_checkpoint_policy
from .unified_natural_evaluation import (
    CANONICAL_ROLLOUT_SEED,
    _physical_reset_sha256,
)
from .unified_formal import load_unified_formal_config


SCHEMA = "jit_natural_start_piup_unified_compare_v1"
POLICY_KEY = 9_420_001
_ENVIRONMENT_FIELDS = (
    "model",
    "action",
    "reset",
    "events",
    "physical_limits",
    "reward",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_environment_semantics(
    unified_up_raw: Mapping[str, Any], expert_up_raw: Mapping[str, Any]
) -> None:
    """Require identical physical/task semantics; training-only PPO may differ."""
    for field in _ENVIRONMENT_FIELDS:
        if unified_up_raw.get(field) != expert_up_raw.get(field):
            raise ValueError(f"pi_up/unified upstream {field} semantics mismatch")


def classify_comparison(
    *,
    unified_jump_zone: bool,
    unified_apex: bool,
    expert_jump_zone: bool,
    expert_apex: bool,
) -> str:
    if expert_apex and not unified_jump_zone:
        return "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL"
    if expert_apex and not unified_apex:
        return "PI_UP_APEX_UNIFIED_UPSTREAM_FAIL"
    if expert_apex and unified_apex:
        return "BOTH_REACH_APEX"
    if not expert_jump_zone and not unified_jump_zone:
        return "BOTH_PRE_JUMP_FAIL"
    return "INCONCLUSIVE_UPSTREAM_DIAGNOSTIC"


def _terminal_reason(trace) -> str:
    terminal = trace.frames[-1]
    return END_REASONS.get(int(terminal.end_code), f"unknown_{terminal.end_code}")


def run_natural_start_expert_compare(
    *,
    formal_config: Path,
    frozen_manifest: Path,
    round0_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run exactly one pi_up_star rollout from the Round-0 canonical reset."""
    formal_config = Path(formal_config)
    frozen_manifest = Path(frozen_manifest)
    round0_dir = Path(round0_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_dir}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("natural-start expert comparison requires visible JAX GPU")

    round0 = _read_json(round0_dir / "report.json")
    reset_audit = _read_json(round0_dir / "reset_diversity.json")
    if round0.get("schema") != "jit_pi_unified_round0_canonical_natural_eval_v1":
        raise ValueError("unexpected Round-0 evaluation schema")
    if round0.get("status") != "completed":
        raise ValueError("Round-0 canonical evaluation is not completed")
    if round0.get("soft_tube_reset_used") is not False:
        raise ValueError("Round-0 comparison requires a natural reset")
    if reset_audit.get("unique_physical_state_count") != 1:
        raise ValueError("comparison requires the locked single canonical natural reset")
    multiplicities = reset_audit.get("physical_state_multiplicities", {})
    if len(multiplicities) != 1:
        raise ValueError("Round-0 reset audit does not contain one physical state")
    canonical_reset_sha = next(iter(multiplicities))

    formal = load_unified_formal_config(formal_config)
    manifest = load_frozen_manifest(frozen_manifest)
    up_record = manifest["experts"]["pi_up_star"]
    expert_config, expert_payload = verify_frozen_record(up_record)

    from .config import load_config

    unified_up_config = load_config(Path(formal.up_config_path))
    validate_environment_semantics(unified_up_config.raw, expert_config.raw)
    if unified_up_config.model["xml_sha256"] != expert_config.model["xml_sha256"]:
        raise ValueError("pi_up/unified XML mismatch")

    env = TwoPhaseBikeEnv(expert_config)
    reset_state = jax.jit(env.reset_natural)(jax.random.PRNGKey(CANONICAL_ROLLOUT_SEED))
    expert_reset_sha = _physical_reset_sha256(reset_state)
    if expert_reset_sha != canonical_reset_sha:
        raise ValueError("pi_up natural reset does not match Round-0 unified physical reset")

    output_dir.mkdir(parents=True, exist_ok=False)
    declaration = {
        "schema": SCHEMA,
        "status": "predeclared",
        "purpose": "single-state causal-localization diagnostic; not a statistical performance claim",
        "formal_config": str(formal_config.resolve()),
        "frozen_manifest": str(frozen_manifest.resolve()),
        "round0_report": str((round0_dir / "report.json").resolve()),
        "canonical_natural_reset_sha256": canonical_reset_sha,
        "pi_up_actor_sha256": up_record["actor_sha256"],
        "pi_up_payload_sha256": up_record["payload_sha256"],
        "pi_up_checkpoint": up_record["checkpoint"],
        "canonical_rollout_seed": CANONICAL_ROLLOUT_SEED,
        "episode_horizon": int(expert_config.ppo.episode_horizon),
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
    }
    _write_json(output_dir / "declaration.json", declaration)

    policy = make_checkpoint_policy(env, expert_payload, deterministic=True)

    def deterministic_policy(obs):
        action, _ = policy(obs, jax.random.PRNGKey(POLICY_KEY))
        return action

    trace = capture_episode(
        env,
        deterministic_policy,
        seed=CANONICAL_ROLLOUT_SEED,
        horizon=expert_config.ppo.episode_horizon,
        reset_fn=jax.jit(env.reset_natural),
        step_fn=jax.jit(env.step),
    )
    trace_artifact = save_episode_trace(trace, output_dir / "pi_up_canonical_trace")
    phase_summary = summarize_phase_u((trace,))
    expert_jump_zone = phase_summary["jump_zone_reach_rate"] > 0.5
    expert_apex = phase_summary["apex_success_rate"] > 0.5

    unified = round0["canonical_rollout"]
    classification = classify_comparison(
        unified_jump_zone=bool(unified["jump_zone_seen"]),
        unified_apex=bool(unified["apex_seen"]),
        expert_jump_zone=expert_jump_zone,
        expert_apex=expert_apex,
    )
    recommendation = (
        "test natural-reset coverage as the sole Round-1 training variable"
        if classification == "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL"
        else "do not launch Round-1 until the upstream discrepancy is further localized"
    )

    terminal = trace.frames[-1]
    report = {
        "schema": SCHEMA,
        "status": "completed",
        "canonical_natural_reset_sha256": canonical_reset_sha,
        "environment_semantics_match": True,
        "pi_up": {
            "actor_sha256": up_record["actor_sha256"],
            "environment_interactions": int(trace.environment_transitions),
            "jump_zone_seen": bool(expert_jump_zone),
            "ascending_seen": bool(phase_summary["ascending_rate"] > 0.5),
            "height_seen": bool(phase_summary["height_reach_rate"] > 0.5),
            "apex_seen": bool(expert_apex),
            "terminal_reason": _terminal_reason(trace),
            "terminal_physical_failure": bool(terminal.physical_failure),
            "terminal_timeout": bool(terminal.timeout),
            "maximum_root_height": float(phase_summary["maximum_root_height"]),
            "maximum_abs_roll": float(phase_summary["maximum_abs_roll"]),
            "maximum_abs_pitch": float(phase_summary["maximum_abs_pitch"]),
            "trace_npz": str(trace_artifact.npz_path.resolve()),
            "trace_npz_sha256": trace_artifact.npz_sha256,
        },
        "pi_unified": {
            "jump_zone_seen": bool(unified["jump_zone_seen"]),
            "ascending_seen": bool(unified["ascending_seen"]),
            "height_seen": bool(unified["height_seen"]),
            "apex_seen": bool(unified["apex_seen"]),
            "terminal_reason": str(unified["terminal_reason"]),
            "environment_interactions": int(unified["environment_interactions"]),
            "maximum_root_height": float(unified["maximum_root_height"]),
            "maximum_abs_roll": float(unified["maximum_abs_roll"]),
            "maximum_abs_pitch": float(unified["maximum_abs_pitch"]),
        },
        "classification": classification,
        "round1_recommendation": recommendation,
        "diagnostic_environment_interactions": int(trace.environment_transitions),
        "training_transitions": 0,
        "statistical_claim": False,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
    }
    _write_json(output_dir / "report.json", report)
    return report
