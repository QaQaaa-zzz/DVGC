import numpy as np
import pytest

from cli.build_phase_balanced_teacher_dataset import (
    build_examples, build_frozen_policy_actions, successful_sequence_medoid,
)


def test_medoid_is_an_observed_successful_sequence_not_mean():
    sequences = [
        {"branch_index": 0, "action_sequence": [[-0.8, 0, 0, 0], [-0.8, 0, 0, 0]]},
        {"branch_index": 1, "action_sequence": [[-0.7, 0, 0, 0], [-0.7, 0, 0, 0]]},
        {"branch_index": 2, "action_sequence": [[0.9, 0, 0, 0], [0.9, 0, 0, 0]]},
    ]
    selected, audit = successful_sequence_medoid(sequences)
    assert any(np.array_equal(selected, np.asarray(row["action_sequence"], np.float32)) for row in sequences)
    assert not np.allclose(selected[:, 0], np.mean([-.8, -.7, .9]))
    assert audit["mean_action_replay_forbidden"] is True


def _record(stage, policy="policy"):
    row = {
        "id": f"phase-rsi:{stage}:one", "origin_record_id": "one",
        "phase_rsi_stage": stage, "reset_parent_id": f"{stage}:parent",
        "origin_artifact_sha256": f"hash-{stage}",
        "origin_artifact_role": "proposal_support_bank", "reset_weight": .2,
        "policy_state": {"actor_observation": np.zeros(140, np.float32)},
    }
    if stage in {"takeoff", "ascent"}:
        row["selected_controller_path"] = policy
    if stage == "apex":
        row["independent_branch_count"] = 32
        row["certified_teacher_action_evidence"] = [
            {"branch_index": index, "seed": 4 + index, "dynamics_variant": "nominal",
             "first_action": [0.1, 0.2, 0.3, 0.4],
             "action_sequence": [[0.1, 0.2, 0.3, 0.4]]}
            for index in range(32)
        ]
    return row


def test_examples_are_phase_balanced_and_apex_uses_feedback_evidence():
    records = [_record(stage) for stage in ("takeoff", "ascent", "apex", "descent", "landing")]
    zero = lambda obs: np.zeros((len(obs), 4))
    actions = {"policy": lambda obs: np.tile([[.1, .2, .3, .4]], (len(obs), 1)),
               "descent-policy": zero, "descent-policy::descent": zero,
               "landing-policy": zero, "landing-policy::landing": zero}
    examples, audits = build_examples(
        records, policy_actions=actions,
        allowed_policy_paths={"policy", "descent-policy", "landing-policy"},
    )
    assert {row["phase"] for row in examples} == {"takeoff", "ascent", "apex", "descent", "landing"}
    assert sum(row["training_weight"] for row in examples) == pytest.approx(1.0)
    apex = next(row for row in examples if row["phase"] == "apex")
    assert apex["teacher_type"] == "certified_feedback_sequence_medoid"
    assert np.allclose(apex["action"], [.1, .2, .3, .4])
    assert len(audits) == 1


def test_apex_requires_complete_unique_32_branch_teacher_evidence():
    row = _record("apex")
    row["certified_teacher_action_evidence"] = row["certified_teacher_action_evidence"][:-1]
    with pytest.raises(ValueError, match="complete unique 32-branch"):
        build_examples([row], policy_actions={}, allowed_policy_paths=set())


def test_unaudited_policy_and_invalid_observation_are_rejected():
    row = _record("takeoff", policy="unknown")
    with pytest.raises(ValueError, match="unaudited policy"):
        build_examples([row], policy_actions={}, allowed_policy_paths=set())
    row["policy_state"]["actor_observation"] = np.zeros(139)
    with pytest.raises(ValueError, match="140D"):
        build_examples([row], policy_actions={}, allowed_policy_paths=set())


def test_each_frozen_expert_uses_its_own_normalizer():
    rows = [
        {"policy_path": "a", "stage": "landing", "params_sha256": "hash-a"},
        {"policy_path": "b", "stage": "descent", "params_sha256": "hash-b"},
    ]
    bundles = {
        "a": (("normalizer-a", "actor-a", "critic-a"), {}, {}),
        "b": (("normalizer-b", "actor-b", "critic-b"), {}, {}),
    }

    def tools(_env, params):
        normalizer = params[0]
        value = 1.0 if normalizer == "normalizer-a" else 2.0
        return None, lambda _actor, obs: np.full((len(obs), 4), value), None

    actions, _ = build_frozen_policy_actions(
        rows, object(), load_policy=lambda path: bundles[path], build_tools=tools,
        hash_params=lambda path: f"hash-{path}",
    )
    obs = np.zeros((1, 140))
    assert np.all(actions["a"](obs) == 1.0)
    assert np.all(actions["b"](obs) == 2.0)
