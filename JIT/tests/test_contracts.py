from __future__ import annotations

import json

import pytest

from jit_dvgc.config import load_config
from jit_dvgc.constants import (
    ACTION_ORDER,
    ACTOR_FRAME_FIELDS,
    CTRL_DT,
    N_SUBSTEPS,
    SIM_DT,
)


def test_timing_and_orders_drive_the_required_runtime_contract():
    assert SIM_DT == 0.005
    assert CTRL_DT == 0.020
    assert N_SUBSTEPS == 4
    assert ACTION_ORDER == ("steer", "rear_wheel_drive", "hip", "knee")
    assert len(ACTOR_FRAME_FIELDS) == 27


def test_smoke_config_resolves_to_one_exact_1024_environment_block(jit_root):
    cfg = load_config(jit_root / "configs" / "phase_u_smoke.json")

    assert cfg.ppo.num_parallel_envs == 1024
    assert cfg.ppo.block_transitions == 25_600
    assert cfg.ppo.requested_transitions == 25_600
    assert cfg.ppo.requested_transitions // cfg.ppo.block_transitions == 1


def test_invalid_ppo_layout_is_rejected_before_a_run_is_created(jit_root, tmp_path):
    payload = json.loads((jit_root / "configs" / "phase_u_smoke.json").read_text())
    payload["ppo"]["batch_size"] = 127
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="divisible by num_parallel_envs"):
        load_config(invalid)
