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
    low_speed_probe = None
    if config.sim2sim_low_speed_failure is not None:
        probe = jax.jit(env.reset_natural)(jax.random.PRNGKey(config.ppo.seed + 200_000))
        dof = env._bundle.model_index.root_dof_address
        probe = probe.replace(data=probe.data.replace(qvel=probe.data.qvel.at[dof].set(-1.0)))
        failed = jax.jit(env.step)(probe, jp.zeros(4, jp.float32))
        speed = float(np.asarray(jax.device_get(failed.metrics["signal/forward_velocity"])))
        final_return = float(np.asarray(jax.device_get(failed.info["episode_return"])))
        if not (speed < config.sim2sim_low_speed_failure.minimum_forward_velocity_mps
                and bool(np.asarray(jax.device_get(failed.info["low_forward_speed"])))
                and bool(np.asarray(jax.device_get(failed.done)))
                and np.isclose(final_return, config.sim2sim_low_speed_failure.failed_episode_return, atol=1e-3)):
            raise ValueError("low speed failure and terminal return smoke check failed")
        low_speed_probe = {"post_step_forward_velocity_mps": speed, "episode_return": final_return}
    report = {
        "schema": "jit_phase_u_sim2sim_smoke_v1",
        "backend": jax.default_backend(),
        "environment_transitions": 8 + int(low_speed_probe is not None),
        "num_domains": 8,
        "wheel_pair_lateral_friction_minmax": [float(friction[:, 2, 0].min()), float(friction[:, 2, 0].max())],
        "wheel_pair_forward_friction_minmax": [float(friction[:, 2, 1].min()), float(friction[:, 2, 1].max())],
        "finite": True,
    }
    if low_speed_probe is not None:
        report["low_speed_failure_probe"] = low_speed_probe
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
