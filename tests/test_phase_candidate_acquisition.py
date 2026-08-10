from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dvgc.phase_candidate_acquisition import (
    AcquisitionParentSummary,
    build_continuation_branch_provenance,
    build_provisional_continuation_label,
    evaluate_candidate_acquisition_gate,
    require_candidate_acquisition_integrity,
)
from dvgc.feasibility import validate_continuation_label
from dvgc.two_phase_runtime import TwoPhaseEventState, initial_two_phase_event_state


def _parents(count: int, *, success: bool = True):
    return tuple(
        AcquisitionParentSummary(
            seed=920_000 + index,
            trajectory_hash=f"{index + 1:064x}",
            success=success,
            contract_valid=True,
            candidate_count=3,
        )
        for index in range(count)
    )


def test_acquisition_gate_requires_fixed_apex_success_and_eight_unique_successful_parents():
    """One lucky or repeated trajectory must never open candidate snapshot acquisition."""
    fixed = {"physical_metrics": {"apex_band_success_rate": 0.125}}
    report = evaluate_candidate_acquisition_gate(fixed, _parents(8))
    assert report == {
        "eligible": True,
        "fixed_apex_success": True,
        "successful_parent_count": 8,
        "successful_parent_candidate_count": 8,
        "unique_successful_seed_count": 8,
        "unique_successful_trajectory_count": 8,
        "all_parent_contracts_valid": True,
        "minimum_independent_successful_parents": 8,
        "failed": [],
    }

    lucky = evaluate_candidate_acquisition_gate(fixed, _parents(1))
    assert not lucky["eligible"]
    assert "minimum_successful_parents" in lucky["failed"]

    repeated = list(_parents(8))
    repeated[-1] = AcquisitionParentSummary(
        seed=repeated[-2].seed,
        trajectory_hash=repeated[-2].trajectory_hash,
        success=True,
        contract_valid=True,
        candidate_count=3,
    )
    duplicate = evaluate_candidate_acquisition_gate(fixed, tuple(repeated))
    assert not duplicate["eligible"]
    assert set(duplicate["failed"]) >= {
        "unique_successful_seeds",
        "unique_successful_trajectories",
    }

    missing_candidates = list(_parents(8))
    missing_candidates[-1] = AcquisitionParentSummary(
        seed=missing_candidates[-1].seed,
        trajectory_hash=missing_candidates[-1].trajectory_hash,
        success=True,
        contract_valid=True,
        candidate_count=0,
    )
    no_coverage = evaluate_candidate_acquisition_gate(
        fixed, tuple(missing_candidates)
    )
    assert not no_coverage["eligible"]
    assert "successful_parent_candidate_coverage" in no_coverage["failed"]


def test_acquisition_gate_rejects_no_fixed_success_or_any_parent_contract_failure():
    """Stochastic success cannot override held-out failure or a corrupt timing contract."""
    no_fixed = evaluate_candidate_acquisition_gate(
        {"physical_metrics": {"apex_band_success_rate": 0.0}}, _parents(8)
    )
    assert not no_fixed["eligible"]
    assert "fixed_apex_success" in no_fixed["failed"]

    parents = list(_parents(8))
    parents[3] = AcquisitionParentSummary(
        seed=parents[3].seed,
        trajectory_hash=parents[3].trajectory_hash,
        success=True,
        contract_valid=False,
        candidate_count=3,
    )
    invalid = evaluate_candidate_acquisition_gate(
        {"physical_metrics": {"apex_band_success_rate": 0.125}}, tuple(parents)
    )
    assert not invalid["eligible"]
    assert "parent_contracts" in invalid["failed"]
    with pytest.raises(RuntimeError, match="snapshot.*contract"):
        require_candidate_acquisition_integrity(invalid)


def test_acquisition_parent_identity_is_fail_closed():
    """Empty hashes, negative counts, and boolean seeds are not auditable parents."""
    with pytest.raises(ValueError):
        AcquisitionParentSummary(
            seed=True,
            trajectory_hash="x",
            success=True,
            contract_valid=True,
            candidate_count=0,
        )


def test_provisional_continuation_label_uses_closed_outcomes_and_frozen_policy_identity():
    """A screen label must count downstream completion, not mere survival."""
    label = build_provisional_continuation_label(
        (
            {"outcome": "success", "termination_reason": "apex_band_entered"},
            {"outcome": "physical_failure", "termination_reason": "roll_limit"},
            {"outcome": "timeout", "termination_reason": "continuation_horizon"},
            {"outcome": "other_failure", "termination_reason": "missed_liftoff"},
        ),
        phase="propulsion_ascent",
        source_policy_hash="a" * 64,
        protocol_hash="b" * 64,
    )
    assert label["outcome_counts"] == {
        "success": 1,
        "physical_failure": 1,
        "timeout": 1,
        "other_failure": 1,
    }
    assert label["num_rollouts"] == 4
    assert label["num_successes"] == 1
    assert label["empirical_rate"] == 0.25
    assert label["physical_failure_rate"] == 0.25
    assert label["timeout_rate"] == 0.25
    assert label["provisional"] is True
    record = {
        "two_phase_context": {"source_phase": "propulsion_ascent"},
        "continuation_label": label,
    }
    assert validate_continuation_label(record)["valid"]
    assert build_continuation_branch_provenance(
        seed=123,
        seed_namespace="phase_u_checkpoint_continuation_v1",
        source_policy_hash="a" * 64,
        protocol_hash="b" * 64,
    ) == {
        "policy_mode": "stochastic",
        "branch_seed_namespace": "phase_u_checkpoint_continuation_v1",
        "branch_seeds": [123],
        "source_policy_hash": "a" * 64,
        "label_protocol_hash": "b" * 64,
    }


def test_online_acquisition_records_the_next_action_that_parent_actually_applies(monkeypatch):
    """The v4 policy_action_t at tick t must be the parent's action at tick t+1."""
    import dvgc.feasibility as feasibility
    import dvgc.phase_candidate_acquisition as acquisition
    import dvgc.runtime as runtime

    monkeypatch.setattr(acquisition.jax, "jit", lambda function: function)
    monkeypatch.setattr(acquisition.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        feasibility, "validate_phase_snapshot", lambda record: {"valid": True}
    )
    inference_calls = []

    def build_inference(_environment, _params, *, deterministic):
        assert deterministic is False

        def inference(_obs, _key):
            value = float(len(inference_calls) + 1) / 10.0
            action = np.full((4,), value, np.float32)
            inference_calls.append(action.copy())
            return action, {}

        return inference

    monkeypatch.setattr(runtime, "build_inference", build_inference)

    class Base:
        def snapshot_record_v4(self, state, _stage, policy_action, provenance):
            return {
                "provenance": dict(provenance),
                "policy_action_t": np.asarray(policy_action).copy(),
                "tick": state.tick,
            }

    class Environment:
        def __init__(self):
            self._base_env = Base()
            self._geometry = object()
            self._thresholds = SimpleNamespace(
                apex=SimpleNamespace(
                    max_abs_roll=0.5,
                    min_clearance=0.1,
                    min_forward_velocity=1.0,
                    max_abs_com_vz=0.2,
                )
            )
            self.applied = []

        @staticmethod
        def _state(tick, done=False):
            event = initial_two_phase_event_state()
            event_values = dict(zip(TwoPhaseEventState._fields, event))
            event_values.update(
                {
                    "jump_window_entered": tick >= 1,
                    "liftoff_seen": tick >= 2,
                    "stable_airborne": tick >= 2,
                    "ascending": tick >= 2,
                    "apex_band_entered": tick >= 3,
                }
            )
            info = {
                "actor_packet_fifo_valid": 3,
                "phase_expert/physical_failure": False,
                "phase_expert/task_failure": False,
                "phase_expert/timeout": False,
                "end_code": 0,
                **{
                    f"phase_expert/event/{name}": value
                    for name, value in event_values.items()
                },
            }
            return SimpleNamespace(
                tick=tick,
                obs={"state": np.asarray([tick], np.float32)},
                data=SimpleNamespace(
                    qpos=np.asarray([tick], np.float32),
                    qvel=np.asarray([tick], np.float32),
                ),
                info=info,
                done=done,
            )

        def reset(self, _key):
            return self._state(0)

        def step(self, state, action):
            self.applied.append(np.asarray(action).copy())
            return self._state(state.tick + 1, done=state.tick + 1 >= 3)

        def _extract_signals(self, state, _geometry, _hold):
            return (
                SimpleNamespace(
                    roll=0.0,
                    clearance=0.2,
                    forward_velocity=2.0,
                    com_vz=0.1,
                ),
                None,
            )

        def _window_active(self, _state):
            return True

    environment = Environment()
    provenance = {
        "xml_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "action_mapping_version": "mapping",
        "policy_params_sha256": "c" * 64,
        "policy_config_sha256": "d" * 64,
        "policy_manifest_sha256": "e" * 64,
        "normalizer_sha256": "f" * 64,
        "source_fingerprint": "1" * 64,
    }

    result = acquisition.acquire_phase_u_candidate_parents(
        environment,
        params=object(),
        fixed_evaluation={"physical_metrics": {"apex_band_success_rate": 1.0}},
        seeds=(7,),
        horizon=4,
        provenance=provenance,
        minimum_independent_successful_parents=1,
    )

    assert result.gate["eligible"] is True
    window = next(
        record
        for record in result.records
        if record["candidate_acquisition"]["stratum"] == "window_entry"
    )
    assert np.array_equal(window["policy_action_t"], environment.applied[1])
    ids = [record["id"] for record in result.records]
    assert len(ids) == len(set(ids))
