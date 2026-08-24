from __future__ import annotations

import pytest

from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv


def test_environment_owns_exact_200_hz_50_hz_contract_without_conversion(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    env = TwoPhaseBikeEnv(config, convert_model=False)

    assert env.sim_dt == pytest.approx(0.005)
    assert env.dt == pytest.approx(0.020)
    assert env.n_substeps == 4
    assert env.action_size == 4
    assert env.mj_model.opt.timestep == pytest.approx(0.005)
    assert env.mjx_model is None


def test_environment_observation_sizes_are_fixed_without_actor_leakage(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    env = TwoPhaseBikeEnv(config, convert_model=False)

    assert env.actor_observation_size == 81
    assert env.privileged_observation_size == 114
