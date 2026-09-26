from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax
import numpy as np

import jit_dvgc.unified_natural_evaluation as unified_evaluation
from jit_dvgc.constants import END_PITCH_LIMIT, END_RECOVERY_SUCCESS
from jit_dvgc.evaluation import EpisodeFrame, EpisodeTrace
from jit_dvgc.unified_natural_evaluation import (
    EVALUATION_SCHEMA,
    _source_training_provenance,
    audit_natural_reset_diversity,
    summarize_canonical_natural_trace,
)


def _fake_reset_state(x: float = 1.0):
    return SimpleNamespace(
        data=SimpleNamespace(
            qpos=np.asarray([x, 0.0, 0.5], dtype=np.float32),
            qvel=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        ),
        info={"expert_switching_used": False},
        metrics={"reset/source_soft_tube": 0.0},
    )


def test_natural_evaluation_schema_is_round_agnostic():
    assert EVALUATION_SCHEMA == "jit_pi_unified_canonical_natural_eval_v1"
    assert "round0" not in EVALUATION_SCHEMA
    assert "round1" not in EVALUATION_SCHEMA


def test_jump_start_evaluation_contract_does_not_claim_natural_connection():
    assert hasattr(unified_evaluation, "jump_start_evaluation_contract"), (
        "canonical evaluation must expose a jump-start-specific scientific contract"
    )
    contract = unified_evaluation.jump_start_evaluation_contract()
    assert contract == {
        "schema": "jit_pi_unified_canonical_jump_start_eval_v1",
        "start_kind": "fixed_ground_jump_start",
        "jump_start_x_m": 2.5,
        "natural_start_connected": False,
        "tube_or_rsi_reset_used": False,
    }


def test_jump_start_evaluation_accepts_an_explicit_rollout_seed():
    signature = inspect.signature(
        unified_evaluation.run_canonical_jump_start_evaluation
    )
    assert "rollout_seed" in signature.parameters
    assert (
        signature.parameters["rollout_seed"].default
        == unified_evaluation.CANONICAL_ROLLOUT_SEED
    )


def test_round1_source_training_provenance_is_preserved():
    reset_mixture = {
        "selection": "bernoulli_per_episode",
        "natural_reset_probability": 0.1,
        "soft_tube_probability": 0.9,
    }
    config = SimpleNamespace(
        raw={
            "run_declaration": {
                "run_id": "pi_unified_round1_natural10_10009600_seed821101_20260831"
            }
        },
        reset_mixture=SimpleNamespace(as_dict=lambda: reset_mixture),
    )
    provenance = _source_training_provenance(config)
    assert provenance["source_training_run_id"] == (
        "pi_unified_round1_natural10_10009600_seed821101_20260831"
    )
    assert provenance["source_training_reset_mixture"] == reset_mixture


def test_natural_reset_audit_does_not_count_seed_duplicates_as_independent():
    report = audit_natural_reset_diversity(
        object(),
        [10, 11, 12, 13],
        reset_fn=lambda _key: _fake_reset_state(),
    )
    assert report["seed_count"] == 4
    assert report["unique_physical_state_count"] == 1
    assert report["duplicate_seed_count"] == 3
    assert report["environment_interactions"] == 0


def test_natural_reset_audit_detects_real_physical_diversity():
    def reset(key):
        seed = int(np.asarray(jax.device_get(key))[1])
        return _fake_reset_state(float(seed))

    report = audit_natural_reset_diversity(
        object(), [20, 21, 22], reset_fn=reset
    )
    assert report["unique_physical_state_count"] == 3
    assert report["duplicate_seed_count"] == 0


def _frame(
    *,
    x: float,
    z: float,
    metrics: dict[str, float] | None = None,
    success: bool = False,
    physical_failure: bool = False,
    timeout: bool = False,
    end_code: int = 0,
    action=(0.0, 0.0, 0.0, 0.0),
):
    merged = {
        "signal/root_x": x,
        "signal/root_z": z,
        "signal/forward_velocity": 2.0,
        "signal/vertical_velocity": 0.0,
        "signal/roll": 0.0,
        "signal/pitch": 0.0,
        "signal/roll_rate": 0.0,
        "signal/pitch_rate": 0.0,
        "signal/angular_speed": 0.0,
    }
    merged.update(metrics or {})
    return EpisodeFrame(
        qpos=np.asarray([x, 0.0, z], dtype=np.float32),
        qvel=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        ctrl=np.zeros(4, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        reward=0.0,
        reward_components={},
        metrics=merged,
        terminated=success or physical_failure,
        truncated=timeout,
        end_code=end_code,
        success=success,
        physical_failure=physical_failure,
        timeout=timeout,
    )


def test_full_recovery_requires_complete_unified_event_chain():
    trace = EpisodeTrace(
        seed=9400001,
        frames=(
            _frame(x=0.0, z=0.5),
            _frame(x=1.0, z=0.6, metrics={"event/jump_zone_seen": 1.0}),
            _frame(
                x=2.0,
                z=1.2,
                metrics={
                    "event/ascending_seen": 1.0,
                    "event/height_seen": 1.0,
                    "event/apex_seen": 1.0,
                    "event/tube_phase_transition": 1.0,
                },
            ),
            _frame(
                x=3.0,
                z=0.7,
                metrics={"event/descent_valid_contact_seen": 1.0},
            ),
            _frame(
                x=3.5,
                z=0.5,
                metrics={"terminal/descent_success": 1.0},
                success=True,
                end_code=END_RECOVERY_SUCCESS,
            ),
        ),
        environment_transitions=4,
    )
    report = summarize_canonical_natural_trace(trace)
    assert report["full_recovery_success"] is True
    assert report["terminal_reason"] == "recovery_success"
    assert report["phase_transitioned"] is True
    assert report["valid_landing_contact_seen"] is True


def test_apex_without_recovery_is_not_full_success():
    trace = EpisodeTrace(
        seed=9400001,
        frames=(
            _frame(x=0.0, z=0.5),
            _frame(
                x=2.0,
                z=1.2,
                metrics={
                    "event/apex_seen": 1.0,
                    "event/tube_phase_transition": 1.0,
                },
            ),
            _frame(
                x=2.3,
                z=1.0,
                physical_failure=True,
                end_code=END_PITCH_LIMIT,
            ),
        ),
        environment_transitions=2,
    )
    report = summarize_canonical_natural_trace(trace)
    assert report["apex_seen"] is True
    assert report["full_recovery_success"] is False
    assert report["terminal_reason"] == "pitch_limit"


def test_valid_landing_before_later_failure_is_jump_trajectory_success():
    trace = EpisodeTrace(
        seed=9400001,
        frames=(
            _frame(x=2.5, z=0.15),
            _frame(
                x=3.2,
                z=0.45,
                metrics={
                    "event/jump_zone_seen": 1.0,
                    "event/ascending_seen": 1.0,
                    "event/height_seen": 1.0,
                },
            ),
            _frame(
                x=3.7,
                z=0.65,
                metrics={
                    "event/apex_seen": 1.0,
                    "event/tube_phase_transition": 1.0,
                },
            ),
            _frame(
                x=4.1,
                z=0.28,
                metrics={"event/descent_valid_contact_seen": 1.0},
            ),
            _frame(
                x=4.3,
                z=0.25,
                physical_failure=True,
                end_code=END_PITCH_LIMIT,
            ),
        ),
        environment_transitions=4,
    )
    report = summarize_canonical_natural_trace(trace)
    assert report["jump_trajectory_success"] is True
    assert report["full_recovery_success"] is False
    assert report["first_valid_landing_state"]["frame_index"] == 3
    assert report["terminal_reason"] == "pitch_limit"
