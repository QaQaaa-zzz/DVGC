from __future__ import annotations

from copy import deepcopy
import json

import pytest

import jit_dvgc.downstream_transition_refinement as refinement
from jit_dvgc.config import file_sha256
from jit_dvgc.downstream_transition_refinement import (
    _validate_prior_search,
    load_downstream_refinement_config,
)


def test_downstream_refinement_config_is_contiguous_and_terminal_clipped(jit_root):
    path = jit_root / "configs/envelope_iter0_downstream_refinement.json"
    config = load_downstream_refinement_config(path)

    acquisition = config["fixed_acquisition"]
    assert acquisition["phase"] == "downstream"
    assert acquisition["duration_grid"] == list(range(17, 33))
    assert acquisition["strengths"] == [0.025, 0.05, 0.1]
    assert acquisition["action_names"] == [
        "steer",
        "rear_wheel_drive",
        "hip",
        "knee",
    ]
    assert acquisition["terminal_clipping"] == {
        "enabled": True,
        "capture": "last_finite_nonterminal_state_before_terminal",
        "terminal_outcome_is_not_the_continuation_label": True,
    }
    assert config["stop_rule"]["upstream_frozen"] is True
    assert config["claim_boundary"]["upstream_transition_band_frozen"] is True


def test_downstream_refinement_binds_exact_prior_candidate_counts(jit_root):
    path = jit_root / "configs/envelope_iter0_downstream_refinement.json"
    config = load_downstream_refinement_config(path)

    upstream = config["prior_search"]["expected_upstream"]
    downstream = config["prior_search"]["expected_downstream"]
    assert upstream["candidate_count"] == 571
    assert downstream["candidate_count"] == 565
    for phase in (upstream, downstream):
        assert phase["candidate_count"] == (
            phase["positive_count"] + phase["negative_count"]
        )


def test_downstream_refinement_audit_validates_current_artifacts(jit_root):
    path = jit_root / "configs/envelope_iter0_downstream_refinement.json"

    report = refinement.audit_downstream_transition_refinement(path)

    assert report["status"] == "artifact_audit_valid"
    assert report["prior_search"]["accumulated_unique_label_count"] == 1136
    assert report["prior_search"]["readiness"]["upstream"]["candidate_count"] == 571
    assert report["prior_search"]["readiness"]["downstream"]["candidate_count"] == 565
    assert report["downstream_anchor_count"] == 5
    assert len(report["downstream_anchor_identities"]) == 5


def test_downstream_refinement_rejects_gapped_duration_grid(tmp_path, jit_root):
    source = json.loads(
        (jit_root / "configs/envelope_iter0_downstream_refinement.json").read_text()
    )
    changed = deepcopy(source)
    changed["fixed_acquisition"]["duration_grid"] = [17, 18, 20, 32]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="contiguous 17..32"):
        load_downstream_refinement_config(path)


def test_downstream_refinement_rejects_terminal_as_label_semantics(tmp_path, jit_root):
    source = json.loads(
        (jit_root / "configs/envelope_iter0_downstream_refinement.json").read_text()
    )
    changed = deepcopy(source)
    changed["fixed_acquisition"]["terminal_clipping"][
        "terminal_outcome_is_not_the_continuation_label"
    ] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="terminal-clipping contract drift"):
        load_downstream_refinement_config(path)


def test_downstream_refinement_requires_upstream_already_ready(tmp_path, jit_root):
    source = json.loads(
        (jit_root / "configs/envelope_iter0_downstream_refinement.json").read_text()
    )
    changed = deepcopy(source)
    changed["prior_search"]["expected_upstream"]["ready"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="upstream already ready"):
        load_downstream_refinement_config(path)


def test_prior_search_validation_binds_exact_completed_evidence(tmp_path, jit_root):
    source = json.loads(
        (jit_root / "configs/envelope_iter0_downstream_refinement.json").read_text()
    )
    root = tmp_path / "prior"
    root.mkdir()

    upstream = {
        "positive_count": 2,
        "negative_count": 1,
        "positive_parent_group_count": 2,
        "negative_parent_group_count": 1,
        "ready": True,
    }
    downstream = {
        "positive_count": 2,
        "negative_count": 0,
        "positive_parent_group_count": 2,
        "negative_parent_group_count": 0,
        "ready": False,
    }
    rows = [
        {
            "state_sha256": "1" * 64,
            "phase": "upstream",
            "label": 1,
            "parent_group_id": "u1",
            "split": "train",
        },
        {
            "state_sha256": "2" * 64,
            "phase": "upstream",
            "label": 0,
            "parent_group_id": "u2",
            "split": "train",
        },
        {
            "state_sha256": "3" * 64,
            "phase": "downstream",
            "label": 1,
            "parent_group_id": "d1",
            "split": "train",
        },
        {
            "state_sha256": "4" * 64,
            "phase": "downstream",
            "label": 1,
            "parent_group_id": "d2",
            "split": "train",
        },
    ]
    summary = {
        "status": "search_exhausted",
        "protocol_sha256": "a" * 64,
        "accumulated_unique_label_count": 4,
        "policy_actor_sha256": "b" * 64,
        "policy_payload_sha256": "c" * 64,
        "source_tube_manifest_sha256": "d" * 64,
        "readiness": {"upstream": upstream, "downstream": downstream},
    }
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "accumulated_train_labels.json").write_text(
        json.dumps({"entries": rows})
    )

    config = deepcopy(source)
    config["prior_search"] = {
        "root": str(root),
        "expected_status": "search_exhausted",
        "expected_protocol_sha256": "a" * 64,
        "expected_accumulated_unique_label_count": 4,
        "expected_upstream": upstream,
        "expected_downstream": downstream,
    }
    config["source_tube_manifest_sha256"] = "d" * 64
    policy_record = {"actor_sha256": "b" * 64, "payload_sha256": "c" * 64}

    loaded, identity = _validate_prior_search(config, policy_record)
    assert len(loaded) == 4
    assert identity["accumulated_unique_label_count"] == 4
    assert identity["protocol_sha256"] == "a" * 64

    summary["protocol_sha256"] = "e" * 64
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="protocol SHA-256 drift"):
        _validate_prior_search(config, policy_record)


def test_resume_accepts_completed_zero_candidate_duration(tmp_path):
    root = tmp_path / "duration_17"
    acquisition = root / "acquisition"
    acquisition.mkdir(parents=True)
    actor_sha = "a" * 64
    payload_sha = "b" * 64
    frozen_sha = "c" * 64
    protocol = {
        "schema": "jit_unified_boundary_protocol_v1",
        "status": "predeclared",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "frozen_unified_manifest_sha256": frozen_sha,
        "durations": [17],
    }
    protocol_sha = refinement._canonical_sha256(protocol)
    protocol["protocol_sha256"] = protocol_sha
    catalog = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "frozen_unified_manifest_sha256": frozen_sha,
        "protocol_sha256": protocol_sha,
        "candidate_count": 0,
        "environment_interactions": 4080,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "terminal_clipped_candidate_count": 0,
        "terminal_probe_outcomes": {},
        "exclusion_counts": {"previously_labeled_state": 120},
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": [],
    }
    summary = {key: value for key, value in catalog.items() if key != "entries"}
    duration_summary = {
        "duration": 17,
        "status": "no_candidates",
        "candidate_count": 0,
        "terminal_clipped_candidate_count": 0,
        "terminal_probe_outcomes": {},
        "exclusion_counts": {"previously_labeled_state": 120},
    }
    (acquisition / "protocol.json").write_text(json.dumps(protocol))
    (acquisition / "catalog.json").write_text(json.dumps(catalog))
    (acquisition / "summary.json").write_text(json.dumps(summary))
    (root / "duration_summary.json").write_text(json.dumps(duration_summary))
    policy_record = {
        "iteration": 0,
        "name": "pi_0",
        "actor_sha256": actor_sha,
        "payload_sha256": payload_sha,
    }

    rows, report = refinement._load_completed_duration(
        root,
        duration=17,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_sha,
    )

    assert rows == []
    assert report["status"] == "no_candidates"
    assert report["resumed"] is True
    assert report["acquisition_environment_interactions"] == 4080


def _completed_duration_fixture(tmp_path):
    root = tmp_path / "duration_17"
    acquisition = root / "acquisition"
    labels_root = root / "labels"
    acquisition.mkdir(parents=True)
    labels_root.mkdir()
    actor_sha = "a" * 64
    payload_sha = "b" * 64
    frozen_sha = "c" * 64
    acquisition_protocol = {
        "schema": "jit_unified_boundary_protocol_v1",
        "status": "predeclared",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "frozen_unified_manifest_sha256": frozen_sha,
        "durations": [17],
    }
    acquisition_sha = refinement._canonical_sha256(acquisition_protocol)
    acquisition_protocol["protocol_sha256"] = acquisition_sha
    candidate = {
        "candidate_id": "pi0_downstream_d17_000000",
        "candidate_kind": "reachable_unified_frontier_probe",
        "split": "train",
        "phase": "downstream",
        "phase_index": 1,
        "snapshot": "snapshots/candidate_000000",
        "source_bank": "boundary_bank",
        "state_sha256": "1" * 64,
        "parent_group_id": "parent-1",
        "parent_state_sha256": "2" * 64,
        "policy_iteration": 0,
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "protocol_sha256": acquisition_sha,
    }
    catalog = {
        "schema": "jit_unified_boundary_catalog_v1",
        "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "frozen_unified_manifest_sha256": frozen_sha,
        "protocol_sha256": acquisition_sha,
        "candidate_count": 1,
        "environment_interactions": 17,
        "training_transitions": 0,
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "unlabeled_acquisition_only": True,
            "tube_expansion_claim": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
        "entries": [candidate],
    }
    acquisition_catalog = acquisition / "catalog.json"
    (acquisition / "protocol.json").write_text(json.dumps(acquisition_protocol))
    acquisition_catalog.write_text(json.dumps(catalog))
    (acquisition / "summary.json").write_text(
        json.dumps({key: value for key, value in catalog.items() if key != "entries"})
    )
    label_protocol = {
        "schema": "jit_unified_continuation_labels_v1",
        "status": "predeclared",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "frozen_unified_manifest_sha256": frozen_sha,
        "candidate_catalog_file_sha256": file_sha256(acquisition_catalog),
        "candidate_catalog_protocol_sha256": acquisition_sha,
        "candidate_count": 1,
    }
    label_sha = refinement._canonical_sha256(label_protocol)
    label_protocol["protocol_sha256"] = label_sha
    (labels_root / "protocol.json").write_text(json.dumps(label_protocol))
    label_row = {
        "candidate_id": candidate["candidate_id"],
        "split": "train",
        "phase": "downstream",
        "phase_index": 1,
        "state_sha256": candidate["state_sha256"],
        "parent_group_id": candidate["parent_group_id"],
        "label": 1,
        "policy_iteration": 0,
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "acquisition_protocol_sha256": acquisition_sha,
        "label_protocol_sha256": label_sha,
    }
    (labels_root / "labels.json").write_text(json.dumps([label_row]))
    label_summary = {
        "schema": "jit_unified_continuation_labels_v1",
        "status": "completed",
        "split": "train",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": actor_sha,
        "policy_payload_sha256": payload_sha,
        "frozen_unified_manifest_sha256": frozen_sha,
        "candidate_catalog_file_sha256": file_sha256(acquisition_catalog),
        "candidate_catalog_protocol_sha256": acquisition_sha,
        "protocol_sha256": label_sha,
        "candidate_count": 1,
        "label_count": 1,
        "positive_count": 1,
        "negative_count": 0,
        "environment_interactions": 12,
    }
    (labels_root / "summary.json").write_text(json.dumps(label_summary))
    (root / "duration_summary.json").write_text(
        json.dumps(
            {
                "duration": 17,
                "status": "completed",
                "candidate_count": 1,
                "positive_count": 1,
                "negative_count": 0,
            }
        )
    )
    policy_record = {
        "iteration": 0,
        "name": "pi_0",
        "actor_sha256": actor_sha,
        "payload_sha256": payload_sha,
    }

    return root, policy_record, frozen_sha


def test_resume_rejects_completed_duration_label_protocol_drift(tmp_path):
    root, policy_record, frozen_sha = _completed_duration_fixture(tmp_path)

    rows, report = refinement._load_completed_duration(
        root,
        duration=17,
        policy_record=policy_record,
        frozen_manifest_sha256=frozen_sha,
    )
    assert len(rows) == 1
    assert report["resumed"] is True
    assert report["labeling_environment_interactions"] == 12

    label_protocol_path = root / "labels" / "protocol.json"
    label_protocol = json.loads(label_protocol_path.read_text())
    label_protocol["protocol_sha256"] = "f" * 64
    label_protocol_path.write_text(json.dumps(label_protocol))

    with pytest.raises(ValueError, match="label protocol"):
        refinement._load_completed_duration(
            root,
            duration=17,
            policy_record=policy_record,
            frozen_manifest_sha256=frozen_sha,
        )


def test_resume_rejects_completed_duration_protocol_content_drift(tmp_path):
    root, policy_record, frozen_sha = _completed_duration_fixture(tmp_path)
    protocol_path = root / "acquisition" / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["purpose"] = "tampered_after_predeclaration"
    protocol_path.write_text(json.dumps(protocol))

    with pytest.raises(ValueError, match="acquisition protocol SHA"):
        refinement._load_completed_duration(
            root,
            duration=17,
            policy_record=policy_record,
            frozen_manifest_sha256=frozen_sha,
        )


def test_resume_rejects_label_parent_group_drift(tmp_path):
    root, policy_record, frozen_sha = _completed_duration_fixture(tmp_path)
    labels_path = root / "labels" / "labels.json"
    rows = json.loads(labels_path.read_text())
    rows[0]["parent_group_id"] = "invented-readiness-group"
    labels_path.write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="candidate identity drift"):
        refinement._load_completed_duration(
            root,
            duration=17,
            policy_record=policy_record,
            frozen_manifest_sha256=frozen_sha,
        )
