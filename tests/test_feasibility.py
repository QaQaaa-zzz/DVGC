from __future__ import annotations

import copy
import importlib
import pickle

import numpy as np
import pytest

from dvgc import feasibility
from dvgc.snapshot_timing import SNAPSHOT_SCHEMA_NAME, SNAPSHOT_SCHEMA_VERSION


def _estimator():
    values = {
        "phase": 2,
        "estimated_phase": 2,
        "had_airborne": 1,
        "had_valid_landing": 0,
        "airborne_count": 3,
        "prelaunch_airborne_count": 0,
        "landing_bounce_count": 0,
        "invalid_wheel_count": 0,
        "recovery_count": 0,
        "contact_age": 0,
        "landing_entry_age": 0,
        "landing_phase_step": 0,
        "prev_acc_z": 0.2,
        "prev_vz": -0.1,
        "prev_front_tire_bottom_z": 0.3,
        "prev_rear_tire_bottom_z": 0.3,
        "positive_pitch_count": 0,
        "wheelie_count": 0,
        "dual_wheel_liftoff_seen": True,
        "stage_entry_ever": 0,
        "apex_seen": 1,
        "jump_signal_latched": True,
        "jump_window_start_x": 2.0,
        "jump_window_end_x": 3.0,
        "chain_ever": 0,
        "recovery_success": 0,
        "episode_step": 7,
        "end_code": 0,
    }
    return values | {"phase_probs": np.array([0, 0, 1, 0], np.float32)}


def valid_phase_snapshot():
    frames = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    fifo = np.stack(
        [frames[0:4].reshape(-1), frames[1:5].reshape(-1), frames[2:6].reshape(-1)]
    )
    hashes = {
        name: f"{name}-hash"
        for name in (
            "xml_sha256",
            "config_sha256",
            "policy_params_sha256",
            "policy_config_sha256",
            "policy_manifest_sha256",
            "normalizer_sha256",
            "source_fingerprint",
        )
    }
    hashes["action_mapping_version"] = "mapping-v2"
    action = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    return {
        "schema_name": SNAPSHOT_SCHEMA_NAME,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "physical_state_t": {
            "qpos": np.zeros(12, np.float32),
            "qvel": np.zeros(11, np.float32),
            "act": np.zeros(0, np.float32),
            "ctrl_previous": np.zeros(4, np.float32),
            "qacc_warmstart": np.zeros(11, np.float32),
            "sensordata": np.zeros(32, np.float32),
            "time": np.array(0.14, np.float32),
        },
        "obs_history_pre_t": frames[2:5],
        "current_frame_t": frames[5],
        "actor_observation_t": fifo[2],
        "obs_history_post_t": frames[3:6],
        "actor_packet_fifo_t": fifo,
        "estimator_state_pre_t": _estimator(),
        "estimator_state_post_t": _estimator(),
        "last_normalized_command_t": np.zeros(4, np.float32),
        "policy_action_t": action,
        "ctrl_applied_t": action * 2,
        "rng_state_t": np.array([1, 2], np.uint32),
        "field_ticks": {
            "physical_state_t": 7,
            "actor_observation_t": 7,
            "current_frame_t": 7,
            "policy_action_t": 7,
            "ctrl_applied_t": 7,
            "ctrl_previous": 6,
            "actor_packet_fifo_t": [5, 6, 7],
        },
        "simulation_timestamps": {"physical_state_t": 0.14, "ctrl_applied_t": 0.14},
        "provenance": hashes,
        "two_phase_context": {
            "contract_version": 1,
            "source_phase": "propulsion_ascent",
            "parent_trajectory_id": "parent-1",
            "trajectory_id": "trajectory-1",
            "time_index": 7,
            "event_names": ["stable_airborne", "apex_band_entered"],
            "event_position": "nearest",
            "terminated": False,
            "truncated": False,
            "termination_reason": "none",
            "source_policy_hash": hashes["policy_params_sha256"],
            "source_xml_hash": hashes["xml_sha256"],
            "source_config_hash": hashes["config_sha256"],
        },
    }


def _v4_validation_inputs(record):
    return {
        "frame_dim": 4,
        "expected_shapes": {
            "qpos": (12,),
            "qvel": (11,),
            "act": (0,),
            "ctrl_previous": (4,),
            "qacc_warmstart": (11,),
            "sensordata": (32,),
        },
        "expected_hashes": record["provenance"],
        "actor_action_fn": lambda _: np.array([0.1, 0.2, 0.3, 0.4], np.float32),
        "ctrl_from_action_fn": lambda action: action * 2,
        "current_frame_fn": lambda _: np.arange(20, 24, dtype=np.float32),
    }


def test_phase_snapshot_composes_with_the_real_v4_authority_contract():
    importlib.import_module("dvgc.feasibility")
    record = valid_phase_snapshot()

    result = feasibility.validate_phase_snapshot(record, **_v4_validation_inputs(record))

    assert result["valid"] is True
    assert result["checks"]["v4_snapshot"] is True
    assert result["checks"]["two_phase_context"] is True


@pytest.mark.parametrize(
    "field",
    [
        "contract_version",
        "source_phase",
        "parent_trajectory_id",
        "trajectory_id",
        "time_index",
        "event_names",
        "event_position",
        "terminated",
        "truncated",
        "termination_reason",
        "source_policy_hash",
        "source_xml_hash",
        "source_config_hash",
    ],
)
def test_phase_snapshot_rejects_each_missing_context_field(field):
    record = valid_phase_snapshot()
    del record["two_phase_context"][field]

    result = feasibility.validate_phase_snapshot(record, **_v4_validation_inputs(record))

    assert result["valid"] is False
    assert "context_required_fields" in result["failed"]


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("source_phase", "apex", "source_phase"),
        ("parent_trajectory_id", "", "lineage"),
        ("trajectory_id", "", "lineage"),
        ("time_index", -1, "time_index"),
        ("event_names", ["oracle_phase"], "event_names"),
        ("event_position", "event", "event_position"),
        ("terminated", True, "terminal_exclusive"),
    ],
)
def test_phase_snapshot_rejects_invalid_context_semantics(field, value, failed_check):
    record = valid_phase_snapshot()
    record["two_phase_context"][field] = value
    if field == "terminated":
        record["two_phase_context"]["truncated"] = True

    result = feasibility.validate_phase_snapshot(record, **_v4_validation_inputs(record))

    assert result["valid"] is False
    assert failed_check in result["failed"]


@pytest.mark.parametrize(
    ("terminated", "truncated", "termination_reason", "failed_check"),
    [
        ("false", False, "none", "terminal_flags"),
        (False, False, "timeout", "termination_reason"),
        (True, False, "none", "termination_reason"),
    ],
)
def test_phase_snapshot_rejects_nonboolean_flags_or_incoherent_terminal_reason(
    terminated, truncated, termination_reason, failed_check
):
    record = valid_phase_snapshot()
    context = record["two_phase_context"]
    context.update(
        terminated=terminated,
        truncated=truncated,
        termination_reason=termination_reason,
    )

    result = feasibility.validate_phase_snapshot(record, **_v4_validation_inputs(record))

    assert result["valid"] is False
    assert failed_check in result["failed"]


@pytest.mark.parametrize(
    ("context_field", "provenance_field"),
    [
        ("source_policy_hash", "policy_params_sha256"),
        ("source_xml_hash", "xml_sha256"),
        ("source_config_hash", "config_sha256"),
    ],
)
def test_phase_snapshot_rejects_context_provenance_mismatch(
    context_field, provenance_field
):
    record = valid_phase_snapshot()
    record["two_phase_context"][context_field] = "different"
    assert record["provenance"][provenance_field] != "different"

    result = feasibility.validate_phase_snapshot(record, **_v4_validation_inputs(record))

    assert result["valid"] is False
    assert "context_provenance" in result["failed"]


def test_non_apex_event_uses_event_position_and_validation_does_not_mutate_input():
    record = valid_phase_snapshot()
    record["two_phase_context"]["event_names"] = ["stable_airborne"]
    record["two_phase_context"]["event_position"] = "event"
    before = pickle.dumps(copy.deepcopy(record))

    result = feasibility.validate_phase_snapshot(record, **_v4_validation_inputs(record))

    assert result["valid"] is True
    assert pickle.dumps(record) == before

    record["two_phase_context"]["event_position"] = "pre"
    assert feasibility.validate_phase_snapshot(
        record, **_v4_validation_inputs(record)
    )["valid"] is True


def valid_labeled_snapshot():
    record = valid_phase_snapshot()
    record["continuation_label"] = {
        "contract_version": 1,
        "phase": "propulsion_ascent",
        "num_rollouts": 10,
        "num_successes": 4,
        "empirical_rate": 0.4,
        "outcome_counts": {
            "success": 4,
            "physical_failure": 2,
            "timeout": 1,
            "other_failure": 3,
        },
        "termination_reason_counts": {
            "apex_band": 4,
            "illegal_contact": 2,
            "timeout": 1,
            "other_failure": 3,
        },
        "physical_failure_rate": 0.2,
        "timeout_rate": 0.1,
        "label_source_policy_hash": "frozen-policy-hash",
        "label_protocol_hash": "continuation-protocol-hash",
    }
    return record


def test_continuation_label_accepts_closed_outcome_accounting():
    result = feasibility.validate_continuation_label(valid_labeled_snapshot())

    assert result["valid"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda label: label["outcome_counts"].update(other_failure=2), "outcome_total"),
        (lambda label: label.update(num_successes=5), "success_count"),
        (lambda label: label.update(empirical_rate=0.5), "empirical_rate"),
        (
            lambda label: label.update(physical_failure_rate=0.3),
            "physical_failure_rate",
        ),
        (lambda label: label.update(timeout_rate=0.2), "timeout_rate"),
        (
            lambda label: label["outcome_counts"].update(physical_failure=-1),
            "outcome_counts",
        ),
        (lambda label: label["outcome_counts"].pop("other_failure"), "outcome_counts"),
        (
            lambda label: label["termination_reason_counts"].update(other_failure=2),
            "termination_reason_total",
        ),
        (lambda label: label.update(num_rollouts=0), "num_rollouts"),
        (lambda label: label.update(phase="descent_recovery"), "phase"),
        (lambda label: label.update(label_source_policy_hash=""), "label_provenance"),
    ],
)
def test_continuation_label_rejects_each_count_rate_or_provenance_inconsistency(
    mutation, failed_check
):
    record = valid_labeled_snapshot()
    mutation(record["continuation_label"])

    result = feasibility.validate_continuation_label(record)

    assert result["valid"] is False
    assert failed_check in result["failed"]


def test_continuation_label_rejects_nonstring_termination_reason_keys():
    record = valid_labeled_snapshot()
    record["continuation_label"]["termination_reason_counts"] = {1: 10}

    result = feasibility.validate_continuation_label(record)

    assert result["valid"] is False
    assert "termination_reason_counts" in result["failed"]


def parent_records(parent_count=6, rows_per_parent=2):
    return [
        {
            "id": f"{parent}-{row}",
            "two_phase_context": {
                "parent_trajectory_id": f"parent-{parent}",
            },
        }
        for parent in range(parent_count)
        for row in range(rows_per_parent)
    ]


def _parents(rows):
    return {row["two_phase_context"]["parent_trajectory_id"] for row in rows}


def test_parent_split_is_deterministic_and_never_leaks_a_trajectory():
    records = parent_records()

    first = feasibility.split_by_parent(
        records, train_fraction=0.5, validation_fraction=0.25, seed=17
    )
    second = feasibility.split_by_parent(
        records, train_fraction=0.5, validation_fraction=0.25, seed=17
    )

    train_parents = _parents(first.train)
    validation_parents = _parents(first.validation)
    test_parents = _parents(first.test)
    assert not train_parents & validation_parents
    assert not train_parents & test_parents
    assert not validation_parents & test_parents
    assert len(train_parents) == 3
    assert len(validation_parents) == 1
    assert len(test_parents) == 2
    assert [row["id"] for row in first.train] == [row["id"] for row in second.train]
    assert first.row_counts == {"train": 6, "validation": 2, "test": 4}
    assert first.parent_counts == {"train": 3, "validation": 1, "test": 2}


def test_parent_split_rejects_too_few_parents_empty_partitions_and_missing_lineage():
    with pytest.raises(ValueError, match="at least three"):
        feasibility.split_by_parent(
            parent_records(parent_count=2),
            train_fraction=0.5,
            validation_fraction=0.25,
            seed=1,
        )
    with pytest.raises(ValueError, match="empty partition"):
        feasibility.split_by_parent(
            parent_records(parent_count=3),
            train_fraction=0.1,
            validation_fraction=0.1,
            seed=1,
        )
    records = parent_records()
    del records[0]["two_phase_context"]["parent_trajectory_id"]
    with pytest.raises(ValueError, match="parent_trajectory_id"):
        feasibility.split_by_parent(
            records, train_fraction=0.5, validation_fraction=0.25, seed=1
        )


def deployable_record(roll=0.1, pitch=-0.2, obstacle_relative_x=0.3):
    return {
        "deployable_features": {
            "roll": roll,
            "pitch": pitch,
            "obstacle_relative_x": obstacle_relative_x,
        },
        "reward": 999.0,
        "success": True,
        "source_phase": "metadata-must-not-leak",
        "two_phase_context": {
            "parent_trajectory_id": "metadata-parent",
            "time_index": 999,
            "event_names": ["metadata-event"],
        },
        "continuation_label": {
            "num_successes": 999,
            "empirical_rate": 0.999,
        },
    }


def test_feature_manifest_uses_canonical_allowlist_order_and_extracts_only_physics():
    manifest = feasibility.build_feature_manifest(
        ["obstacle_relative_x", "pitch", "roll"]
    )
    record = deployable_record()

    values = feasibility.extract_deployable_features(record, manifest)

    assert manifest.fields == ("roll", "pitch", "obstacle_relative_x")
    assert values.tolist() == pytest.approx([0.1, -0.2, 0.3])


@pytest.mark.parametrize(
    "forbidden",
    [
        "terminated",
        "truncated",
        "termination_reason",
        "event_names",
        "event_position",
        "source_phase",
        "source_policy_hash",
        "source_config_hash",
        "parent_trajectory_id",
        "trajectory_id",
        "time_index",
        "continuation_label",
        "num_successes",
        "empirical_rate",
        "reward",
        "success",
        "teacher_id",
        "controller_id",
        "reference_time",
        "reference_index",
        "oracle_phase",
        "unregistered_future_field",
    ],
)
def test_feature_manifest_rejects_every_non_allowlisted_field(forbidden):
    with pytest.raises(ValueError, match="not deployable allowlisted"):
        feasibility.build_feature_manifest(["roll", forbidden])


def test_feature_manifest_rejects_empty_or_duplicate_requests():
    with pytest.raises(ValueError, match="at least one"):
        feasibility.build_feature_manifest([])
    with pytest.raises(ValueError, match="duplicate"):
        feasibility.build_feature_manifest(["roll", "roll"])


def test_feature_manifest_constructor_enforces_the_allowlist():
    with pytest.raises(ValueError, match="not deployable allowlisted"):
        feasibility.DeployableFeatureManifest(("num_successes",))


def test_feature_extraction_revalidates_even_a_forged_manifest():
    forged = object.__new__(feasibility.DeployableFeatureManifest)
    object.__setattr__(forged, "fields", ("reward",))

    with pytest.raises(ValueError, match="not deployable allowlisted"):
        feasibility.extract_deployable_features(deployable_record(), forged)
    with pytest.raises(ValueError, match="not deployable allowlisted"):
        feasibility.build_deployable_feature_matrix([deployable_record()], forged)


def test_feature_matrix_isolated_from_metadata_and_continuation_results():
    manifest = feasibility.build_feature_manifest(["roll"])
    first = deployable_record(roll=0.25)
    second = copy.deepcopy(first)
    second["reward"] = -1000.0
    second["success"] = False
    second["continuation_label"] = {"num_successes": 0, "empirical_rate": 0.0}
    second["two_phase_context"]["time_index"] = 0

    matrix = feasibility.build_deployable_feature_matrix([first, second], manifest)

    assert matrix.shape == (2, 1)
    assert matrix[:, 0].tolist() == pytest.approx([0.25, 0.25])


def test_feature_matrix_rejects_missing_nonfinite_or_shape_drifting_physics():
    manifest = feasibility.build_feature_manifest(["actor_observation"])
    with pytest.raises(ValueError, match="missing deployable feature"):
        feasibility.build_deployable_feature_matrix([deployable_record()], manifest)
    with pytest.raises(ValueError, match="finite"):
        feasibility.build_deployable_feature_matrix(
            [{"deployable_features": {"actor_observation": [0.0, np.nan]}}], manifest
        )
    with pytest.raises(ValueError, match="stable shape"):
        feasibility.build_deployable_feature_matrix(
            [
                {"deployable_features": {"actor_observation": [1.0, 2.0]}},
                {"deployable_features": {"actor_observation": [1.0, 2.0, 3.0]}},
            ],
            manifest,
        )


def test_scorer_receives_only_numeric_matrix_and_returns_one_finite_score_per_row():
    manifest = feasibility.build_feature_manifest(["roll"])
    matrix = feasibility.build_deployable_feature_matrix(
        [deployable_record(roll=0.2), deployable_record(roll=0.4)], manifest
    )

    result = feasibility.validate_scorer_inference(
        lambda features: features[:, 0] * 2.0,
        matrix,
        expected_rows=2,
    )

    assert result["valid"] is True
    assert result["scores"].tolist() == pytest.approx([0.4, 0.8])


@pytest.mark.parametrize(
    ("scorer", "failed_check"),
    [
        (lambda features: np.array([0.5]), "score_shape"),
        (lambda features: np.array([0.5, np.nan]), "finite_scores"),
    ],
)
def test_scorer_rejects_shape_and_nonfinite_results(scorer, failed_check):
    matrix = np.array([[0.1], [0.2]], dtype=np.float64)
    result = feasibility.validate_scorer_inference(scorer, matrix, expected_rows=2)
    assert result["valid"] is False
    assert failed_check in result["failed"]


def test_scorer_input_mutation_is_detected_without_mutating_the_callers_matrix():
    matrix = np.array([[0.1], [0.2]], dtype=np.float64)

    def mutating_scorer(features):
        features[0, 0] = 99.0
        return features[:, 0]

    result = feasibility.validate_scorer_inference(
        mutating_scorer, matrix, expected_rows=2
    )

    assert result["valid"] is False
    assert "input_immutable" in result["failed"]
    assert matrix[:, 0].tolist() == pytest.approx([0.1, 0.2])


def test_feasibility_delta_is_bounded_and_mixed_potential_uses_complementary_weights():
    assert float(
        feasibility.bounded_feasibility_delta(0.2, 0.9, delta_max=0.3)
    ) == pytest.approx(0.3)
    assert float(
        feasibility.bounded_feasibility_delta(0.8, 0.1, delta_max=0.25)
    ) == pytest.approx(-0.25)
    assert float(
        feasibility.mixed_feasibility_potential(0.8, 0.2, up_weight=0.25)
    ) == pytest.approx(0.35)


def test_feasibility_shaping_rejects_invalid_bounds_or_weights():
    with pytest.raises(ValueError, match="delta_max"):
        feasibility.bounded_feasibility_delta(0.2, 0.3, delta_max=0.0)
    with pytest.raises(ValueError, match="up_weight"):
        feasibility.mixed_feasibility_potential(0.8, 0.2, up_weight=-0.1)


def valid_soft_tube_metadata():
    return feasibility.build_soft_tube_metadata(
        phase="propulsion_ascent",
        model_hash="model-hash",
        labeled_dataset_hash="labels-hash",
        parent_split_hash="split-hash",
        selection_rule="quantile_layers",
        xml_hash="xml-hash",
        config_hash="config-hash",
        action_mapping_version="mapping-v2",
        source_policy_hashes=["policy-a", "policy-b"],
        parent_count=4,
        layer_counts={"core": 3, "boundary": 2, "exploration": 1},
    )


def test_soft_tube_metadata_is_explicitly_noncertified_training_guidance():
    metadata = valid_soft_tube_metadata()
    result = feasibility.validate_soft_tube_metadata(metadata)

    assert result["valid"] is True
    assert metadata["artifact_role"] == "learned_soft_feasibility_tube"
    assert metadata["certified_safe"] is False
    assert metadata["training_guidance_only"] is True
    assert metadata["total_records"] == 6


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda metadata: metadata.update(phase="apex"), "phase"),
        (
            lambda metadata: metadata.update(artifact_role="certified_tube"),
            "artifact_role",
        ),
        (lambda metadata: metadata.update(certified_safe=True), "claim_boundary"),
        (
            lambda metadata: metadata.update(training_guidance_only=False),
            "claim_boundary",
        ),
        (lambda metadata: metadata.update(model_hash=""), "provenance"),
        (lambda metadata: metadata.update(source_policy_hashes=[]), "source_policies"),
        (lambda metadata: metadata.update(parent_count=1), "parent_diversity"),
        (
            lambda metadata: metadata.update(
                layer_counts={"core": 0, "boundary": 0, "exploration": 0},
                total_records=0,
            ),
            "nonempty_support",
        ),
        (lambda metadata: metadata.update(total_records=99), "layer_total"),
        (
            lambda metadata: metadata.update(
                layer_counts={"core": 3, "boundary": -1, "exploration": 1}
            ),
            "layer_counts",
        ),
    ],
)
def test_soft_tube_metadata_rejects_claim_provenance_or_diversity_violations(
    mutation, failed_check
):
    metadata = valid_soft_tube_metadata()
    mutation(metadata)

    result = feasibility.validate_soft_tube_metadata(metadata)

    assert result["valid"] is False
    assert failed_check in result["failed"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metadata: metadata.update(
            selection_rule="certified safe Tube admission"
        ),
        lambda metadata: metadata.update(claim="formal JCE result"),
    ],
)
def test_soft_tube_metadata_rejects_certification_claim_language(mutation):
    metadata = valid_soft_tube_metadata()
    mutation(metadata)

    result = feasibility.validate_soft_tube_metadata(metadata)

    assert result["valid"] is False
    assert "claim_language" in result["failed"]
