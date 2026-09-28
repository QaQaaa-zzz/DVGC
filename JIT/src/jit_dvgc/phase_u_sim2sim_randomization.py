"""Declared Phase U physics domains and reset perturbations for sim2sim training."""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jp
import mujoco

from .config import Sim2SimRandomizationConfig


def _uniform(key: jax.Array, bounds: tuple[float, float], shape=()) -> jax.Array:
    return jax.random.uniform(key, shape, minval=bounds[0], maxval=bounds[1])


def sample_domain_fields(
    *, pair_friction: jax.Array, body_mass: jax.Array,
    body_inertia: jax.Array, actuator_gainprm: jax.Array,
    actuator_biasprm: jax.Array, wheel_pair_indices: tuple[int, ...],
    rng: jax.Array, config: Sim2SimRandomizationConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample one coherent domain; only declared wheel pairs and bodies change."""
    lateral_key, forward_key, mass_key, actuator_key = jax.random.split(rng, 4)
    low, high = config.wheel_lateral_friction
    lateral = jp.exp(_uniform(lateral_key, (jp.log(low), jp.log(high))))
    forward = _uniform(forward_key, config.wheel_forward_friction)
    pairs = jp.asarray(wheel_pair_indices, dtype=jp.int32)
    friction = pair_friction.at[pairs, 0].set(lateral)
    friction = friction.at[pairs, 1].set(forward)
    mass_factor = _uniform(mass_key, config.body_mass_inertia_scale, body_mass.shape)
    mass = body_mass * mass_factor
    inertia = body_inertia * mass_factor[:, None]
    actuator_factor = _uniform(actuator_key, config.actuator_scale, (actuator_gainprm.shape[0],))
    gain = actuator_gainprm.at[:, 0].set(actuator_gainprm[:, 0] * actuator_factor)
    bias = actuator_biasprm.at[:, 1:3].set(actuator_biasprm[:, 1:3] * actuator_factor[:, None])
    return friction, mass, inertia, gain, bias


def _wheel_pair_indices(host_model: mujoco.MjModel) -> tuple[int, ...]:
    names = set()
    indices = []
    for index in range(host_model.npair):
        first = mujoco.mj_id2name(host_model, mujoco.mjtObj.mjOBJ_GEOM, int(host_model.pair_geom1[index]))
        second = mujoco.mj_id2name(host_model, mujoco.mjtObj.mjOBJ_GEOM, int(host_model.pair_geom2[index]))
        pair = frozenset((first, second))
        if len(pair & {"floor", "step"}) == 1 and len(pair & {"rearwheel_collision", "frontwheel_collision"}) == 1:
            names.add(pair)
            indices.append(index)
    expected = {frozenset((terrain, wheel)) for terrain in ("floor", "step") for wheel in ("rearwheel_collision", "frontwheel_collision")}
    if names != expected or len(indices) != 4:
        raise ValueError("expected four explicit terrain/wheel contact pairs")
    return tuple(indices)


def make_phase_u_domain_randomization(env: Any, config: Sim2SimRandomizationConfig):
    """Create the Brax/MJX per-parallel-environment model randomizer."""
    pair_indices = _wheel_pair_indices(env.mj_model)

    def randomization_fn(model: Any, rng: jax.Array):
        def sample(key):
            return sample_domain_fields(
                pair_friction=model.pair_friction,
                body_mass=model.body_mass,
                body_inertia=model.body_inertia,
                actuator_gainprm=model.actuator_gainprm,
                actuator_biasprm=model.actuator_biasprm,
                wheel_pair_indices=pair_indices,
                rng=key,
                config=config,
            )
        friction, mass, inertia, gain, bias = jax.vmap(sample)(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({
            "pair_friction": 0, "body_mass": 0, "body_inertia": 0,
            "actuator_gainprm": 0, "actuator_biasprm": 0,
        })
        randomized = model.tree_replace({
            "pair_friction": friction, "body_mass": mass,
            "body_inertia": inertia, "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        })
        return randomized, in_axes

    return randomization_fn


def _quaternion_multiply(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=a.dtype)


def apply_root_reset_perturbation(
    qpos: jax.Array, qvel: jax.Array, root_qpos_address: int,
    root_dof_address: int, rng: jax.Array,
    config: Sim2SimRandomizationConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perturb only the free-root pose and twist at each reset."""
    pos_key, rot_key, lin_key, ang_key = jax.random.split(rng, 4)
    def symmetric(key, halfwidths):
        scale = jp.asarray(halfwidths, dtype=qpos.dtype)
        return jax.random.uniform(key, (3,), minval=-1.0, maxval=1.0) * scale
    position_delta = symmetric(pos_key, config.reset_position_halfwidth_m)
    rotation_delta = symmetric(rot_key, config.reset_orientation_halfwidth_rad)
    linear_delta = symmetric(lin_key, config.reset_linear_velocity_halfwidth_mps)
    angular_delta = symmetric(ang_key, config.reset_angular_velocity_halfwidth_radps)
    roll, pitch, yaw = rotation_delta / 2.0
    qx = jp.array([jp.cos(roll), jp.sin(roll), 0., 0.], dtype=qpos.dtype)
    qy = jp.array([jp.cos(pitch), 0., jp.sin(pitch), 0.], dtype=qpos.dtype)
    qz = jp.array([jp.cos(yaw), 0., 0., jp.sin(yaw)], dtype=qpos.dtype)
    rotation = _quaternion_multiply(qz, _quaternion_multiply(qy, qx))
    old = qpos[root_qpos_address + 3:root_qpos_address + 7]
    new = _quaternion_multiply(old, rotation)
    new /= jp.linalg.norm(new)
    qpos = qpos.at[root_qpos_address:root_qpos_address + 3].add(position_delta)
    qpos = qpos.at[root_qpos_address + 3:root_qpos_address + 7].set(new)
    qvel = qvel.at[root_dof_address:root_dof_address + 3].add(linear_delta)
    qvel = qvel.at[root_dof_address + 3:root_dof_address + 6].add(angular_delta)
    return qpos, qvel
