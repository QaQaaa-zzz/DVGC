from __future__ import annotations

from copy import deepcopy

import pytest

from jit_dvgc.unified_training import read_json
from jit_dvgc.unified_round1 import (
    ROUND1_NATURAL_PROBABILITY,
    ROUND1_SOFT_TUBE_PROBABILITY,
    load_round1_config,
    validate_round1_reset_contract,
)


def test_checked_in_round1_changes_only_reset_distribution(jit_root):
    base = read_json(jit_root / "configs/pi_unified_formal.json")
    round1 = read_json(jit_root / "configs/pi_unified_round1_natural10.json")
    config = load_round1_config(jit_root / "configs/pi_unified_round1_natural10.json")

    for key in ("inputs", "initialization", "runtime", "ppo", "formal"):
        assert round1[key] == base[key]
    assert config.ppo.requested_transitions == 10_009_600
    assert config.ppo.seed == 821101
    assert round1["reset_mixture"]["natural_reset_probability"] == pytest.approx(
        ROUND1_NATURAL_PROBABILITY
    )
    assert round1["reset_mixture"]["soft_tube_probability"] == pytest.approx(
        ROUND1_SOFT_TUBE_PROBABILITY
    )
    assert round1["reset_mixture"]["single_variable"] == "reset_distribution_only"
    assert round1["claim_boundary"]["round1_single_variable_iteration"] is True
    assert round1["claim_boundary"]["round0_failure_evidence"] == (
        "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL"
    )
    assert round1["claim_boundary"]["test_data_used"] is False
    assert round1["claim_boundary"]["validation_data_used"] is False


def test_round1_reset_contract_rejects_probability_tuning(jit_root):
    payload = read_json(jit_root / "configs/pi_unified_round1_natural10.json")
    drifted = deepcopy(payload)
    drifted["reset_mixture"]["natural_reset_probability"] = 0.2
    drifted["reset_mixture"]["soft_tube_probability"] = 0.8
    with pytest.raises(ValueError, match="contract drift"):
        validate_round1_reset_contract(drifted)


def test_round1_reset_contract_requires_locked_round0_causal_evidence(jit_root):
    payload = read_json(jit_root / "configs/pi_unified_round1_natural10.json")
    drifted = deepcopy(payload)
    drifted["claim_boundary"]["round0_failure_evidence"] = "other"
    with pytest.raises(ValueError, match="Round-0 diagnosis"):
        validate_round1_reset_contract(drifted)
