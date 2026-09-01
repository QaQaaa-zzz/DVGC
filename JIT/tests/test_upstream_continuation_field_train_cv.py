from __future__ import annotations


def test_real_upstream_train_logo_config_is_train_only(jit_root):
    from jit_dvgc.upstream_continuation_field_train_cv import (
        load_upstream_train_logo_config,
    )

    config = load_upstream_train_logo_config(
        jit_root / "configs/envelope_iter0_upstream_train_logo_diagnostic.json"
    )
    protocol = config["protocol"]
    assert protocol["expected_upstream_train"] == {
        "candidate_count": 571,
        "positive_count": 545,
        "negative_count": 26,
        "parent_group_count": 5,
    }
    assert protocol["model"]["family"] == "linear_logistic"
    assert protocol["model"]["l2_weight"] == 0.01
    assert protocol["data_policy"]["train_rows_only"] is True
    assert protocol["data_policy"]["consumed_validation_rows_reused"] is False
    assert protocol["data_policy"]["consumed_validation_predictions_reused"] is False
    assert protocol["claim_boundary"]["fresh_validation_authorized"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def _fold(auc: float, gap: float):
    return {
        "heldout_metrics": {
            "roc_auc": auc,
            "score_gap": gap,
        }
    }


def test_logo_gate_requires_mean_worst_and_positive_gap():
    from jit_dvgc.upstream_continuation_field_train_cv import classify_logo_folds

    gate = {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.55,
        "require_positive_score_gap_in_every_fold": True,
    }
    passed = classify_logo_folds(
        [_fold(0.80, 0.2), _fold(0.75, 0.1), _fold(0.72, 0.05), _fold(0.70, 0.2), _fold(0.65, 0.1)],
        gate,
    )
    assert passed["train_group_generalization_supported"] is True

    bad_worst = classify_logo_folds(
        [_fold(0.90, 0.2), _fold(0.85, 0.1), _fold(0.80, 0.05), _fold(0.75, 0.2), _fold(0.40, 0.1)],
        gate,
    )
    assert bad_worst["train_group_generalization_supported"] is False
    assert bad_worst["gate"]["worst_logo_roc_auc_at_least_minimum"] is False

    bad_gap = classify_logo_folds(
        [_fold(0.80, 0.2), _fold(0.75, 0.1), _fold(0.72, -0.01), _fold(0.70, 0.2), _fold(0.65, 0.1)],
        gate,
    )
    assert bad_gap["train_group_generalization_supported"] is False
    assert bad_gap["gate"]["positive_score_gap_in_every_fold"] is False
