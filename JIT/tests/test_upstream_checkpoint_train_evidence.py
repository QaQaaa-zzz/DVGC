from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_real_checkpoint_train_freeze_config_is_locked(jit_root: Path):
    from jit_dvgc.upstream_checkpoint_train_evidence import (
        load_upstream_checkpoint_train_freeze_config,
    )

    config = load_upstream_checkpoint_train_freeze_config(
        jit_root / "configs/envelope_iter0_upstream_checkpoint_train_freeze.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "e584d09165bd281cd9763bc06ccf615594ddf485c6e6713fa6cd87a362e9b4ea"
    )
    assert protocol["expected_combined"] == {
        "candidate_count": 1051,
        "positive_count": 963,
        "negative_count": 88,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }
    assert set(protocol["required_domain_stats"]) == {
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    }
    assert protocol["data_policy"]["consumed_validation_rows_read"] is False
    assert protocol["data_policy"]["consumed_validation_predictions_read"] is False
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def test_checkpoint_domain_parser():
    from jit_dvgc.upstream_checkpoint_train_evidence import checkpoint_domain

    assert checkpoint_domain("transition_4988928__1000001") == "transition_4988928"
    assert checkpoint_domain("transition_9977856__1000005") == "transition_9977856"
    with pytest.raises(ValueError):
        checkpoint_domain("seed-1000001")


def test_freeze_config_rejects_protocol_drift(tmp_path: Path, jit_root: Path):
    source = jit_root / "configs/envelope_iter0_upstream_checkpoint_train_freeze.json"
    payload = json.loads(source.read_text())
    payload["protocol"]["expected_combined"]["negative_count"] = 87
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    from jit_dvgc.upstream_checkpoint_train_evidence import (
        load_upstream_checkpoint_train_freeze_config,
    )

    with pytest.raises(ValueError):
        load_upstream_checkpoint_train_freeze_config(path)
