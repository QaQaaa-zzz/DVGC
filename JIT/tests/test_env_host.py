from __future__ import annotations

import pytest

from jit_dvgc.config import load_config
from jit_dvgc import env as env_module
from jit_dvgc.env import TwoPhaseBikeEnv
from jax import numpy as jp


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

    assert env.actor_observation_size == 76
    assert env.privileged_observation_size == 106
    assert callable(env.reset_natural)


def test_stuck_progress_metric_is_finite_when_physics_position_is_nonfinite():
    progress = env_module._finite_stuck_window_progress(jp.nan, jp.nan)
    assert float(progress) == 0.0
