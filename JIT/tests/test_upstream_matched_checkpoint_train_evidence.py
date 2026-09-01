from __future__ import annotations

from pathlib import Path


def test_real_matched_checkpoint_train_freeze_config_is_locked(jit_root: Path):
    from jit_dvgc.upstream_matched_checkpoint_train_evidence import (
        load_matched_checkpoint_train_freeze_config,
    )

    config = load_matched_checkpoint_train_freeze_config(
        jit_root / "configs/envelope_iter0_upstream_checkpoint_train_matched_freeze.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "2df65325660ac54c914b14fca98d1d5e2f0f52a3da74bf941b4979edbc32bf1d"
    )
    assert protocol["expected_combined"] == {
        "candidate_count": 720,
        "positive_count": 639,
        "negative_count": 81,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }
    assert protocol["expected_domain_stats"]["transition_4988928"] == {
        "candidate_count": 240,
        "positive_count": 221,
        "negative_count": 19,
        "parent_group_count": 5,
    }
    assert protocol["data_policy"]["environment_interactions"] == 0
    assert protocol["data_policy"]["consumed_validation_rows_read"] is False
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def test_real_matched_checkpoint_domain_cv_config_reuses_locked_model(jit_root: Path):
    from jit_dvgc.upstream_matched_checkpoint_domain_cv import (
        load_matched_checkpoint_domain_cv_config,
    )

    config = load_matched_checkpoint_domain_cv_config(
        jit_root / "configs/envelope_iter0_upstream_checkpoint_domain_logo_matched.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "908e5b3c0e235b666dfe6368504e75343eae5a9d95dd3cab85daa1803981573d"
    )
    assert protocol["expected_combined"]["candidate_count"] == 720
    assert protocol["model"] == {
        "family": "linear_logistic",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "normalization": "fold_train_only_zscore_clip10",
        "sample_weighting": "equal_parent_label_cell_mass",
        "l2_weight": 0.01,
        "optimizer": "adam_full_batch_fixed_schedule",
        "steps": 4000,
        "learning_rate": 0.01,
        "seed_base": 846000,
    }
    assert protocol["diagnostic_gate"] == {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }
    assert protocol["interpretation"]["acquisition_family_confound_removed"] is True
    assert protocol["data_policy"]["consumed_validation_rows_reused"] is False
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False
