from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_real_tiny_mlp_checkpoint_cv_config_is_locked(jit_root: Path):
    from jit_dvgc.upstream_matched_checkpoint_domain_cv import (
        NONLINEAR_STATUS,
        _tiny_mlp_parameter_count,
        load_matched_checkpoint_domain_cv_config,
    )

    config = load_matched_checkpoint_domain_cv_config(
        jit_root / "configs/envelope_iter0_upstream_checkpoint_domain_logo_matched_tiny_mlp.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "c44908d32f254b7041b774ac1d41e1977df82319888fe0a76ce42dc4cdcf4513"
    )
    assert protocol["status"] == NONLINEAR_STATUS
    assert protocol["frozen_upstream_train_manifest_sha256"] == (
        "25160e1f198d710327b8390a53869bc72ec62afced5f3d86dcc45d1543793626"
    )
    assert protocol["prior_matched_linear_cv_summary_sha256"] == (
        "018b296243a3d162b132186fe30367eed81f141a731cff0620d77b742557d340"
    )
    assert protocol["model"]["family"] == "tiny_mlp_tanh"
    assert protocol["model"]["hidden_units"] == 8
    assert protocol["model"]["parameter_count"] == 625
    assert _tiny_mlp_parameter_count(76, 8) == 625
    assert protocol["model"]["l2_weight"] == 0.01
    assert protocol["model"]["steps"] == 4000
    assert protocol["method_decision"] == {
        "reason": "matched_linear_checkpoint_cv_failed_after_acquisition_family_confound_was_removed",
        "single_repair_model_only": True,
        "hyperparameter_grid_search": False,
        "optimization_schedule_changed_from_linear": False,
        "fresh_validation_may_be_predeclared_only_if_train_checkpoint_gate_passes": True,
        "automatic_model_escalation_if_fail": False,
    }
    assert protocol["data_policy"]["consumed_validation_rows_reused"] is False
    assert protocol["data_policy"]["consumed_validation_predictions_reused"] is False
    assert protocol["claim_boundary"]["fresh_validation_bank_predeclared"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def test_tiny_mlp_checkpoint_cv_rejects_hidden_size_tuning(tmp_path: Path, jit_root: Path):
    source = jit_root / "configs/envelope_iter0_upstream_checkpoint_domain_logo_matched_tiny_mlp.json"
    payload = json.loads(source.read_text())
    payload["protocol"]["model"]["hidden_units"] = 16
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    from jit_dvgc.upstream_matched_checkpoint_domain_cv import (
        load_matched_checkpoint_domain_cv_config,
    )

    with pytest.raises(ValueError):
        load_matched_checkpoint_domain_cv_config(path)


def test_tiny_mlp_parameter_count_is_low_capacity():
    from jit_dvgc.upstream_matched_checkpoint_domain_cv import _tiny_mlp_parameter_count

    assert _tiny_mlp_parameter_count(76, 8) == 625
    assert _tiny_mlp_parameter_count(76, 8) < 1000
