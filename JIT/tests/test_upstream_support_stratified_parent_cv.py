from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_real_support_stratified_parent_cv_config_is_locked(jit_root: Path):
    from jit_dvgc.upstream_support_stratified_parent_cv import (
        _seed_family,
        load_support_stratified_parent_cv_config,
    )

    config = load_support_stratified_parent_cv_config(
        jit_root / "configs/envelope_iter0_upstream_support_stratified_parent_cv_tiny_mlp.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "bfdf8795a98720741e6edcac0d1ff82156f4eb3ec669422330a74855e28dd89c"
    )
    assert protocol["fold_design"]["fold_count"] == 5
    assert protocol["fold_design"]["heldout_parent_groups_per_fold"] == 3
    assert protocol["fold_design"]["train_parent_groups_per_fold"] == 12
    assert protocol["model"]["family"] == "tiny_mlp_tanh"
    assert protocol["model"]["hidden_units"] == 8
    assert protocol["model"]["parameter_count"] == 625
    assert protocol["method_decision"][
        "checkpoint_domain_holdout_reclassified_as_out_of_support_extrapolation_stress_test"
    ] is True
    assert protocol["method_decision"][
        "if_pass_next_gate_is_same_architecture_downstream_train_check"
    ] is True
    assert protocol["method_decision"]["no_additional_model_family_search"] is True
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False
    assert _seed_family("transition_9977856__1000004") == "1000004"


def _prediction_rows(*, reverse_domain: str | None = None):
    rows = []
    domains = ["transition_4988928", "transition_7987200", "transition_9977856"]
    for domain in domains:
        for seed_index in range(5):
            for cell in range(48):
                label = 0 if cell < 8 else 1
                score = 0.1 + 0.8 * label + 0.0001 * cell
                if domain == reverse_domain:
                    score = 1.0 - score
                rows.append(
                    {
                        "state_sha256": f"{domain}-{seed_index}-{cell}",
                        "parent_domain_id": domain,
                        "label": label,
                        "score": score,
                    }
                )
    return rows


def _folds():
    return [
        {"heldout_metrics": {"roc_auc": 0.8, "score_gap": 0.4}}
        for _ in range(5)
    ]


def test_support_stratified_classifier_requires_every_supported_domain():
    from jit_dvgc.upstream_support_stratified_parent_cv import _classify

    domains = ["transition_4988928", "transition_7987200", "transition_9977856"]
    gate = {
        "minimum_pooled_oof_roc_auc": 0.70,
        "minimum_worst_domain_oof_roc_auc": 0.60,
        "require_positive_score_gap_in_every_domain": True,
        "minimum_mean_fold_roc_auc": 0.70,
        "minimum_worst_fold_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }
    passed = _classify(_folds(), _prediction_rows(), domains, gate)
    assert passed["support_stratified_parent_generalization_supported"] is True
    assert passed["worst_domain_oof_roc_auc"] == 1.0

    failed = _classify(
        _folds(),
        _prediction_rows(reverse_domain="transition_9977856"),
        domains,
        gate,
    )
    assert failed["support_stratified_parent_generalization_supported"] is False
    assert failed["gate"]["worst_domain_oof_roc_auc_at_least_minimum"] is False
    assert failed["gate"]["positive_score_gap_in_every_domain"] is False
