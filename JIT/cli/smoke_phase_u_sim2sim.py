#!/usr/bin/env python3
"""One bounded GPU reset/step check for the randomized Phase U training path."""

import argparse
import json
from pathlib import Path

import jax
from jax import numpy as jp
import numpy as np

from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.phase_u_sim2sim_randomization import make_phase_u_domain_randomization
from jit_dvgc.ppo import wrap_for_jit_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("sim2sim smoke requires GPU")
    config = load_config(args.config)
    if config.sim2sim_randomization is None:
        raise ValueError("smoke requires declared randomization")
    env = TwoPhaseBikeEnv(config)
    domain = make_phase_u_domain_randomization(env, config.sim2sim_randomization)
    keys = jax.random.split(jax.random.PRNGKey(config.ppo.seed + 100_000), 8)
    model_batch, _ = domain(env.mjx_model, keys)
    wrapped = wrap_for_jit_training(
        env, episode_length=config.ppo.episode_horizon,
        randomization_fn=lambda model: domain(model, keys),
    )
    state = jax.jit(wrapped.reset)(keys)
    next_state = jax.jit(wrapped.step)(state, jp.zeros((8, 4), jp.float32))
    qpos = np.asarray(jax.device_get(next_state.data.qpos))
    qvel = np.asarray(jax.device_get(next_state.data.qvel))
    reward = np.asarray(jax.device_get(next_state.reward))
    friction = np.asarray(jax.device_get(model_batch.pair_friction))
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(reward).all()):
        raise ValueError("nonfinite randomized rollout")
    if np.unique(friction[:, 2, 0]).size < 2 or np.unique(friction[:, 2, 1]).size < 2:
        raise ValueError("sampled physical domains did not differ")
    report = {
        "schema": "jit_phase_u_sim2sim_smoke_v1",
        "backend": jax.default_backend(),
        "environment_transitions": 8,
        "num_domains": 8,
        "wheel_pair_lateral_friction_minmax": [float(friction[:, 2, 0].min()), float(friction[:, 2, 0].max())],
        "wheel_pair_forward_friction_minmax": [float(friction[:, 2, 1].min()), float(friction[:, 2, 1].max())],
        "finite": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
