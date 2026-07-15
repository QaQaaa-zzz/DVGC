"""Training wrappers that preserve metrics across full bank resets."""
from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco_playground._src import wrapper as playground_wrapper


class MetricPreservingFullResetWrapper(playground_wrapper.BraxAutoResetWrapper):
    """Resample the environment while retaining the terminal episode record."""

    def __init__(self, env: Any):
        super().__init__(env, full_reset=True)

    def step(self, state: Any, action: jax.Array) -> Any:
        rng_key=jax.vmap(jax.random.split)(state.info[f"{self._info_key}_rng"])
        reset_rng,reset_key=rng_key[...,0],rng_key[...,1]
        reset_state=self.reset(reset_key)
        if "steps" in state.info:
            state.info["steps"]=jp.where(state.done,jp.zeros_like(state.info["steps"]),state.info["steps"])
        state=state.replace(done=jp.zeros_like(state.done))
        state=self.env.step(state,action)

        def where_done(x,y):
            done=state.done
            if done.shape and done.shape[0]!=x.shape[0]: return y
            if done.shape: done=jp.reshape(done,[x.shape[0]]+[1]*(len(x.shape)-1))
            return jp.where(done,x,y)

        data=jax.tree.map(where_done,reset_state.data,state.data)
        obs=jax.tree.map(where_done,reset_state.obs,state.obs)
        next_info=jax.tree.map(where_done,reset_state.info,state.info)
        done_count_key=f"{self._info_key}_done_count"
        next_info[done_count_key]=state.info[done_count_key]+state.done.astype(int)
        next_info[f"{self._info_key}_rng"]=reset_rng
        # These fields describe the trajectory that just terminated.  All
        # task state and reset provenance deliberately come from reset_state.
        for key in ("episode_metrics","episode_done","steps","truncation"):
            if key in state.info: next_info[key]=state.info[key]
        return state.replace(data=data,obs=obs,info=next_info)


def wrap_for_training(env: Any,episode_length: int=1000,action_repeat: int=1,randomization_fn: Any=None):
    if randomization_fn is None:
        env=brax_training.VmapWrapper(env)
    else:
        env=playground_wrapper.BraxDomainRandomizationVmapWrapper(env,randomization_fn)
    env=brax_training.EpisodeWrapper(env,episode_length,action_repeat)
    return MetricPreservingFullResetWrapper(env)
