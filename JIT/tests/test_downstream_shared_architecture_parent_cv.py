from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_downstream_shared_architecture_config_is_locked(jit_root: Path):
    from jit_dvgc.downstream_shared_architecture_parent_cv import (
        load_downstream_shared_architecture_parent_cv_config,
    )

    config = load_downstream_shared_architecture_parent_cv_config(
        jit_root / "configs/envelope_iter0_downstream_parent_logo_tiny_mlp.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "7a653eb6eb1a17d1d777e98a2f2d40da4b611d2c7fc6bb4ca5e0798bf3f23fef"
    )
    assert protocol["expected_downstream_train"] == {
        "candidate_count": 2619,
        "positive_count": 2589,
        "negative_count": 30,
        "parent_group_count": 5,
    }
    assert protocol["fold_design"]["kind"] == "leave_one_parent_group_out"
    assert protocol["model"]["family"] == "tiny_mlp_tanh"
    assert protocol["model"]["hidden_units"] == 8
    assert protocol["model"]["parameter_count"] == 625
    assert protocol["method_decision"]["shared_up_down_architecture_required"] is True
    assert protocol["method_decision"]["no_downstream_architecture_search"] is True
    assert protocol["data_policy"]["consumed_validation_rows_reused"] is False
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False


def test_up_down_tiny_mlp_contracts_match_except_seed(jit_root: Path):
    upstream = json.loads(
        (jit_root / "configs/envelope_iter0_upstream_support_stratified_parent_cv_tiny_mlp.json").read_text()
    )["protocol"]["model"]
    downstream = json.loads(
        (jit_root / "configs/envelope_iter0_downstream_parent_logo_tiny_mlp.json").read_text()
    )["protocol"]["model"]

    upstream = dict(upstream)
    downstream = dict(downstream)
    upstream.pop("seed_base")
    downstream.pop("seed_base")
    assert downstream == upstream


def test_downstream_config_rejects_architecture_drift(tmp_path: Path, jit_root: Path):
    from jit_dvgc.downstream_shared_architecture_parent_cv import (
        load_downstream_shared_architecture_parent_cv_config,
    )

    source = jit_root / "configs/envelope_iter0_downstream_parent_logo_tiny_mlp.json"
    payload = json.loads(source.read_text())
    payload["protocol"]["model"]["hidden_units"] = 16
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_downstream_shared_architecture_parent_cv_config(path)


def test_downstream_classifier_pass_and_fail():
    from jit_dvgc.downstream_shared_architecture_parent_cv import _classify

    predictions = []
    for index in range(2619):
        label = 0 if index < 30 else 1
        predictions.append(
            {
                "state_sha256": f"{index:064x}",
                "label": label,
                "score": 0.1 if label == 0 else 0.9,
            }
        )
    folds = [
        {"heldout_metrics": {"roc_auc": 0.9, "score_gap": 0.7}}
        for _ in range(5)
    ]
    gate = {
        "minimum_pooled_oof_roc_auc": 0.70,
        "minimum_mean_fold_roc_auc": 0.70,
        "minimum_worst_fold_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }
    passed = _classify(folds, predictions, gate)
    assert passed["downstream_parent_generalization_supported"] is True

    failed_folds = list(folds)
    failed_folds[0] = {"heldout_metrics": {"roc_auc": 0.5, "score_gap": 0.7}}
    failed = _classify(failed_folds, predictions, gate)
    assert failed["downstream_parent_generalization_supported"] is False
