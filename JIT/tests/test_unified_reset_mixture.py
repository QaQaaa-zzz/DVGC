from __future__ import annotations

import json
from copy import deepcopy

import pytest

from jit_dvgc.unified_formal import (
    ROUND0_FAILURE_EVIDENCE,
    ROUND1_NATURAL_RESET_PROBABILITY,
    ROUND1_SOFT_TUBE_PROBABILITY,
    load_unified_formal_config,
)
from jit_dvgc.unified_training import read_json


ROUND1_CONFIG_SHA256 = "fba8ac1975727ea77ec271a6196cc998eb63b47960aba25935656927d69f2ae1"
ROUND1_RUN_ID = "pi_unified_round1_natural10_10009600_seed821101_20260831"
ROUND0_NATURAL_DIAGNOSTIC_SHA256 = (
    "eb8ccb07e1d8229abd794283fc3c48c63943f9a0356cf313c89186458a939721"
)
ROUND0_PI_UP_UNIFIED_COMPARISON_SHA256 = (
    "53c708fd33df66c53a1ab60098d033ebf1284df3fe0c16f999a3942630bbca8f"
)
HARDENING_VALIDATED_HEAD = "f9be2c6e4b5e5ef7e654bb7704e711177ffa89d6"


def test_round0_formal_config_defaults_to_soft_tube_only(jit_root):
    config = load_unified_formal_config(jit_root / "configs/pi_unified_formal.json")
    assert config.reset_mixture.selection == "soft_tube_only"
    assert config.reset_mixture.natural_reset_probability == pytest.approx(0.0)
    assert config.reset_mixture.soft_tube_probability == pytest.approx(1.0)
    assert config.ppo.seed == 821101
    assert config.ppo.requested_transitions == 10_009_600


def test_round1_config_changes_only_reset_distribution(jit_root):
    base = read_json(jit_root / "configs/pi_unified_formal.json")
    round1 = read_json(jit_root / "configs/pi_unified_round1_natural10.json")
    config = load_unified_formal_config(
        jit_root / "configs/pi_unified_round1_natural10.json"
    )

    for key in ("inputs", "initialization", "runtime", "ppo", "formal"):
        assert round1[key] == base[key]
    assert config.reset_mixture.selection == "bernoulli_per_episode"
    assert config.reset_mixture.natural_reset_probability == pytest.approx(
        ROUND1_NATURAL_RESET_PROBABILITY
    )
    assert config.reset_mixture.soft_tube_probability == pytest.approx(
        ROUND1_SOFT_TUBE_PROBABILITY
    )
    assert config.ppo.requested_transitions == 10_009_600
    assert config.ppo.seed == 821101
    assert tuple(config.formal.checkpoint_transitions) == (
        0,
        1_024_000,
        2_508_800,
        5_017_600,
        7_500_800,
        10_009_600,
    )
    assert round1["claim_boundary"]["round0_failure_evidence"] == (
        ROUND0_FAILURE_EVIDENCE
    )
    assert round1["claim_boundary"]["test_data_used"] is False
    assert round1["claim_boundary"]["validation_data_used"] is False


def test_round1_prelaunch_provenance_and_run_declaration_are_locked(jit_root):
    path = jit_root / "configs/pi_unified_round1_natural10.json"
    raw = read_json(path)
    config = load_unified_formal_config(path)
    boundary = raw["claim_boundary"]
    declaration = raw["run_declaration"]

    assert config.config_sha256 == ROUND1_CONFIG_SHA256
    assert boundary["round0_natural_diagnostic_sha256"] == (
        ROUND0_NATURAL_DIAGNOSTIC_SHA256
    )
    assert boundary["round0_pi_up_unified_comparison_sha256"] == (
        ROUND0_PI_UP_UNIFIED_COMPARISON_SHA256
    )
    assert boundary["round0_diagnostic_artifacts_regenerated"] is False
    assert boundary["prelaunch_hardening_validated_head"] == HARDENING_VALIDATED_HEAD
    assert declaration == {
        "run_id": ROUND1_RUN_ID,
        "output_dir": f"JIT/runs/pi_unified/{ROUND1_RUN_ID}",
        "purpose": "round1_single_variable_natural10_soft_tube90_unified_training",
        "status": "predeclared_not_started",
    }


def test_round1_reset_probability_drift_is_rejected(jit_root, tmp_path):
    payload = read_json(jit_root / "configs/pi_unified_round1_natural10.json")
    drifted = deepcopy(payload)
    drifted["reset_mixture"]["natural_reset_probability"] = 0.2
    drifted["reset_mixture"]["soft_tube_probability"] = 0.8
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="reset mixture contract drift"):
        load_unified_formal_config(path)


def test_round1_requires_locked_round0_causal_evidence(jit_root, tmp_path):
    payload = read_json(jit_root / "configs/pi_unified_round1_natural10.json")
    drifted = deepcopy(payload)
    drifted["claim_boundary"]["round0_failure_evidence"] = "other"
    path = tmp_path / "wrong_evidence.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="locked Round-0 diagnosis"):
        load_unified_formal_config(path)
