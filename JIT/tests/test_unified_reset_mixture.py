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
