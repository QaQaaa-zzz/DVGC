from __future__ import annotations

import json

import pytest


def test_real_parent_diversity_config_selects_checkpoint_domains(jit_root):
    from jit_dvgc.upstream_parent_diversity_train import (
        _select_source_anchors,
        enumerate_parent_diversity_attempts,
        load_upstream_parent_diversity_config,
    )

    config = load_upstream_parent_diversity_config(
        jit_root / "configs/envelope_iter0_upstream_parent_diversity_train.json"
    )
    protocol = config["protocol"]
    anchors = _select_source_anchors(protocol)
    assert len(anchors) == 10
    assert {row["parent_domain_id"] for row in anchors} == {
        "transition_7987200",
        "transition_9977856",
    }
    assert {int(row["seed"]) for row in anchors} == {
        1000001,
        1000002,
        1000003,
        1000004,
        1000005,
    }
    assert all(row["role"] == "ascending_entry" for row in anchors)
    assert all(int(row["seed"]) != 1000006 for row in anchors)
    attempts = enumerate_parent_diversity_attempts(protocol, anchors)
    assert len(attempts) == 480
    assert {row["duration"] for row in attempts} == {4, 8, 16}
    assert {row["strength"] for row in attempts} == {0.025, 0.1}
    assert {row["action_name"] for row in attempts} == {
        "steer",
        "rear_wheel_drive",
        "hip",
        "knee",
    }


def test_parent_diversity_config_rejects_validation_seed(tmp_path, jit_root):
    from jit_dvgc.iteration_train_evidence import canonical_sha256
    from jit_dvgc.upstream_parent_diversity_train import (
        load_upstream_parent_diversity_config,
    )

    source = json.loads(
        (jit_root / "configs/envelope_iter0_upstream_parent_diversity_train.json").read_text()
    )
    source["protocol"]["source_seeds"] = [
        1000001,
        1000002,
        1000003,
        1000004,
        1000006,
    ]
    source["expected_protocol_sha256"] = canonical_sha256(source["protocol"])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="TRAIN seed contract"):
        load_upstream_parent_diversity_config(path)


def test_parent_diversity_config_rejects_consumed_validation_read(tmp_path, jit_root):
    from jit_dvgc.iteration_train_evidence import canonical_sha256
    from jit_dvgc.upstream_parent_diversity_train import (
        load_upstream_parent_diversity_config,
    )

    source = json.loads(
        (jit_root / "configs/envelope_iter0_upstream_parent_diversity_train.json").read_text()
    )
    source["protocol"]["data_policy"]["consumed_validation_rows_read"] = True
    source["expected_protocol_sha256"] = canonical_sha256(source["protocol"])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="data policy"):
        load_upstream_parent_diversity_config(path)


def test_parent_diversity_attempt_schedule_is_checkpoint_balanced(jit_root):
    from collections import Counter

    from jit_dvgc.upstream_parent_diversity_train import (
        _select_source_anchors,
        enumerate_parent_diversity_attempts,
        load_upstream_parent_diversity_config,
    )

    config = load_upstream_parent_diversity_config(
        jit_root / "configs/envelope_iter0_upstream_parent_diversity_train.json"
    )
    anchors = _select_source_anchors(config["protocol"])
    attempts = enumerate_parent_diversity_attempts(config["protocol"], anchors)
    by_domain = Counter(row["parent_domain_id"] for row in attempts)
    assert by_domain == {
        "transition_7987200": 240,
        "transition_9977856": 240,
    }
    by_parent = Counter(row["parent_group_id"] for row in attempts)
    assert set(by_parent.values()) == {48}
