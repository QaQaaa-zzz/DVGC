from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_real_checkpoint_domain_cv_config_is_locked(jit_root: Path):
    from jit_dvgc.upstream_checkpoint_domain_cv import (
        load_upstream_checkpoint_domain_cv_config,
    )

    config = load_upstream_checkpoint_domain_cv_config(
        jit_root / "configs/envelope_iter0_upstream_checkpoint_domain_logo.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "e7a2e48ca1664fd0858cc2b32cf5da7f9d95e4d668980e3e26fec910cdff8766"
    )
    assert protocol["required_domains"] == [
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    ]
    assert protocol["model"]["l2_weight"] == 0.01
    assert protocol["model"]["steps"] == 4000
    assert protocol["diagnostic_gate"] == {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }
    assert protocol["data_policy"]["consumed_validation_rows_reused"] is False
    assert protocol["data_policy"]["consumed_validation_predictions_reused"] is False
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def test_checkpoint_domain_classifier_pass_and_fail():
    from jit_dvgc.upstream_checkpoint_domain_cv import (
        classify_checkpoint_domain_folds,
    )

    gate = {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }
    folds = [
        {"heldout_metrics": {"roc_auc": 0.72, "score_gap": 0.1}},
        {"heldout_metrics": {"roc_auc": 0.80, "score_gap": 0.2}},
        {"heldout_metrics": {"roc_auc": 0.75, "score_gap": 0.3}},
    ]
    result = classify_checkpoint_domain_folds(folds, gate)
    assert result["checkpoint_domain_generalization_supported"] is True

    bad = [dict(fold) for fold in folds]
    bad[0] = {"heldout_metrics": {"roc_auc": 0.55, "score_gap": 0.1}}
    result = classify_checkpoint_domain_folds(bad, gate)
    assert result["checkpoint_domain_generalization_supported"] is False


def test_checkpoint_domain_cv_config_rejects_model_tuning(tmp_path: Path, jit_root: Path):
    source = jit_root / "configs/envelope_iter0_upstream_checkpoint_domain_logo.json"
    payload = json.loads(source.read_text())
    payload["protocol"]["model"]["l2_weight"] = 0.1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    from jit_dvgc.upstream_checkpoint_domain_cv import (
        load_upstream_checkpoint_domain_cv_config,
    )

    with pytest.raises(ValueError):
        load_upstream_checkpoint_domain_cv_config(path)
