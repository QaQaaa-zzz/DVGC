from __future__ import annotations

from copy import deepcopy
import json

import pytest

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
