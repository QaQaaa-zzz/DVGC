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


def test_apex_exact_replay_trajectory_expands_ticks_without_changing_phase_mass():
    records = [_record(stage) for stage in ("takeoff", "ascent", "apex", "descent", "landing")]
    zero = lambda obs: np.zeros((len(obs), 4))
    actions = {"policy": zero, "descent-policy": zero, "descent-policy::descent": zero,
               "landing-policy": zero, "landing-policy::landing": zero}
    apex_id = records[2]["id"]
    trajectory = {
        apex_id: {
            "record_id": apex_id, "local_entry_replay_verified": True,
            "exact_replay_count": 2, "branch_index": 3, "seed": 7,
            "dynamics_variant": "nominal", "entry_tick": 4,
            "samples": [
                {"tick": tick, "observation": np.full(140, tick, np.float32),
                 "action": np.full(4, tick / 10, np.float32)}
                for tick in range(4)
            ],
        }
    }
    examples, audits = build_examples(
        records, policy_actions=actions,
        allowed_policy_paths={"policy", "descent-policy", "landing-policy"},
        apex_trajectories=trajectory,
    )
    apex = [row for row in examples if row["phase"] == "apex"]
    assert len(apex) == 4
    assert sum(row["training_weight"] for row in apex) == pytest.approx(.2)
    assert all(row["teacher_type"] == "certified_feedback_trajectory_medoid" for row in apex)
    assert [row["teacher_trajectory_tick"] for row in apex] == [0, 1, 2, 3]
    assert audits[0]["exact_replay_count"] == 2


def test_apex_trajectory_requires_exact_replay_and_contiguous_ticks():
    row = _record("apex")
    trajectory = {row["id"]: {"local_entry_replay_verified": False, "samples": []}}
    with pytest.raises(ValueError, match="exact-replay Apex trajectory"):
        build_examples([row], policy_actions={}, allowed_policy_paths=set(),
                       apex_trajectories=trajectory)


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

    actions, _, identities = build_frozen_policy_actions(
        rows, object(), load_policy=lambda path: bundles[path], build_tools=tools,
        hash_params=lambda path: f"hash-{path}",
    )
    obs = np.zeros((1, 140))
    assert np.all(actions["a"](obs) == 1.0)
    assert np.all(actions["b"](obs) == 2.0)
    assert identities == {"a": "params:hash-a", "b": "params:hash-b"}


def test_declared_compact_adapter_is_applied_and_identity_is_preserved():
    rows = [{"policy_path": "descent", "stage": "descent", "params_sha256": "hash-d"}]
    bundle = (("normalizer", "actor", "critic"), {},
              {"adapter_sha256": "adapter-hash", "policy_identity_hash": "combined-id"})

    def tools(_env, _params):
        return None, lambda _actor, obs: np.zeros((len(obs), 4)), None

    actions, _, identities = build_frozen_policy_actions(
        rows, object(), load_policy=lambda _path: bundle, build_tools=tools,
        hash_params=lambda _path: "hash-d",
        load_adapter=lambda _path, _manifest, _base: (
            lambda _obs, base: base + 0.25, "combined-id"
        ),
    )
    assert np.all(actions["descent"](np.zeros((2, 140))) == .25)
    assert identities["descent"] == "adapter:combined-id"


def test_declared_adapter_without_verified_loader_is_rejected():
    rows = [{"policy_path": "descent", "stage": "descent", "params_sha256": "hash-d"}]
    bundle = (("normalizer", "actor", "critic"), {}, {"adapter_sha256": "adapter-hash"})
    with pytest.raises(ValueError, match="declares an adapter"):
        build_frozen_policy_actions(
            rows, object(), load_policy=lambda _path: bundle,
            build_tools=lambda _env, _params: (None, lambda _actor, obs: np.zeros((len(obs), 4)), None),
            hash_params=lambda _path: "hash-d",
        )


def test_descent_teacher_must_match_tube_certified_adapter_identity():
    row = _record("descent")
    row["policy_identity_hash"] = "certified-combined-id"
    action = lambda obs: np.zeros((len(obs), 4))
    with pytest.raises(ValueError, match="does not match certified Tube controller"):
        build_examples(
            [row], policy_actions={"descent-policy": action,
                                   "descent-policy::descent": action},
            allowed_policy_paths={"descent-policy"},
            policy_identities={"descent-policy": "params:base-only"},
        )
