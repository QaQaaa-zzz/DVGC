from __future__ import annotations

import pytest

from jit_dvgc.natural_start_expert_compare import (
    classify_comparison,
    validate_environment_semantics,
)


def test_classify_piup_apex_unified_prejump_failure():
    assert (
        classify_comparison(
            unified_jump_zone=False,
            unified_apex=False,
            expert_jump_zone=True,
            expert_apex=True,
        )
        == "PI_UP_APEX_UNIFIED_PRE_JUMP_FAIL"
    )


def test_classify_both_reach_apex():
    assert (
        classify_comparison(
            unified_jump_zone=True,
            unified_apex=True,
            expert_jump_zone=True,
            expert_apex=True,
        )
        == "BOTH_REACH_APEX"
    )


def test_environment_semantics_allow_training_only_differences():
    base = {
        "model": {"xml": "same"},
        "action": {"a": 1},
        "reset": {"r": 2},
        "events": {"e": 3},
        "physical_limits": {"p": 4},
        "reward": {"w": 5},
        "ppo": {"requested_transitions": 1},
        "schema": "smoke",
    }
    other = {
        **base,
        "ppo": {"requested_transitions": 10_000_000},
        "schema": "formal",
    }
    validate_environment_semantics(base, other)


def test_environment_semantics_reject_reward_drift():
    base = {
        "model": {},
        "action": {},
        "reset": {},
        "events": {},
        "physical_limits": {},
        "reward": {"x": 1},
    }
    other = {**base, "reward": {"x": 2}}
    with pytest.raises(ValueError, match="reward semantics mismatch"):
        validate_environment_semantics(base, other)
