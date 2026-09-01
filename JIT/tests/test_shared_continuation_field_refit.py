from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_real_shared_continuation_refit_config_is_locked(jit_root: Path):
    from jit_dvgc.shared_continuation_field_refit import (
        _tiny_mlp_parameter_count,
        load_shared_continuation_field_refit_config,
    )

    config = load_shared_continuation_field_refit_config(
        jit_root / "configs/envelope_iter0_shared_continuation_field_refit.json"
    )
    protocol = config["protocol"]
    assert config["expected_protocol_sha256"] == (
        "163768104c459e471515a846860876b17daeaa56e262cf526348a01ae6cd9c26"
    )
    assert protocol["architecture"] == {
        "family": "tiny_mlp_tanh",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "hidden_units": 8,
        "activation": "tanh",
        "parameter_count": 625,
        "normalization": "train_only_zscore_clip10",
        "sample_weighting": "equal_parent_label_cell_mass",
        "l2_weight": 0.01,
        "optimizer": "adam_full_batch_fixed_schedule",
        "steps": 4000,
        "learning_rate": 0.01,
        "phase_specific_seeds": {"upstream": 850001, "downstream": 850002},
    }
    assert _tiny_mlp_parameter_count(76, 8) == 625
    assert protocol["expected_train"]["upstream"] == {
        "candidate_count": 720,
        "positive_count": 639,
        "negative_count": 81,
        "parent_group_count": 15,
    }
    assert protocol["expected_train"]["downstream"] == {
        "candidate_count": 2619,
        "positive_count": 2589,
        "negative_count": 30,
        "parent_group_count": 5,
    }
    assert protocol["upstream_train_gate_summary_sha256"] == (
        "093e9e5d6ed7845046cf3abb483fcfecee3107ffbcda771ce85a4880a1279541"
    )
    assert protocol["downstream_train_gate_summary_sha256"] == (
        "c2bb51f0b4432db8d78c2f17bf941521aa385dec57c387f604d5a59832a100c3"
    )
    assert protocol["method_decision"]["shared_up_down_architecture_required"] is True
    assert protocol["method_decision"]["phase_specific_weights_required"] is True
    assert protocol["method_decision"]["phase_specific_calibration_required"] is True
    assert protocol["method_decision"]["no_additional_architecture_search"] is True
    assert protocol["data_policy"]["consumed_validation_rows_reused"] is False
    assert protocol["claim_boundary"]["architecture_frozen"] is True
    assert protocol["claim_boundary"]["fields_calibrated"] is False
    assert protocol["claim_boundary"]["tube_1_constructed"] is False


def test_shared_refit_rejects_architecture_drift(tmp_path: Path, jit_root: Path):
    source = jit_root / "configs/envelope_iter0_shared_continuation_field_refit.json"
    payload = json.loads(source.read_text())
    payload["protocol"]["architecture"]["hidden_units"] = 16
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))

    from jit_dvgc.shared_continuation_field_refit import (
        load_shared_continuation_field_refit_config,
    )

    with pytest.raises(ValueError):
        load_shared_continuation_field_refit_config(path)


def test_shared_refit_uses_phase_specific_weights_not_architectures(jit_root: Path):
    from jit_dvgc.shared_continuation_field_refit import load_shared_continuation_field_refit_config

    config = load_shared_continuation_field_refit_config(
        jit_root / "configs/envelope_iter0_shared_continuation_field_refit.json"
    )
    model = config["protocol"]["architecture"]
    assert set(model["phase_specific_seeds"]) == {"upstream", "downstream"}
    assert model["phase_specific_seeds"]["upstream"] != model["phase_specific_seeds"]["downstream"]
    assert model["family"] == "tiny_mlp_tanh"
    assert model["hidden_units"] == 8
    assert model["parameter_count"] == 625
