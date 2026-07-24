"""CPU MuJoCo centroidal/contact diagnostics for immutable snapshots."""
from __future__ import annotations

import mujoco
import numpy as np


def _object_velocity(model, data, body_id):
    velocity = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
    )
    return velocity[:3], velocity[3:]


def replay_centroidal(model, qpos, qvel, ctrl=None):
    """Forward an exact state and return world-frame centroidal quantities.

    Angular momentum is computed explicitly from every moving body and checked
    against MuJoCo's subtree result.  No integration or state settling occurs.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, dtype=float)
    data.qvel[:] = np.asarray(qvel, dtype=float)
    if ctrl is not None and model.nu:
        data.ctrl[:] = np.asarray(ctrl, dtype=float)
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)

    total_mass = float(np.sum(model.body_mass[1:]))
    com = np.sum(
        model.body_mass[1:, None] * data.xipos[1:], axis=0
    ) / total_mass
    linear_momentum = np.zeros(3, dtype=float)
    angular_momentum = np.zeros(3, dtype=float)
    contributions = []
    for body_id in range(1, model.nbody):
        mass = float(model.body_mass[body_id])
        omega, velocity = _object_velocity(model, data, body_id)
        rotation = data.ximat[body_id].reshape(3, 3)
        inertia_world = (
            rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        )
        spin = inertia_world @ omega
        orbital = mass * np.cross(data.xipos[body_id] - com, velocity)
        contribution = spin + orbital
        linear_momentum += mass * velocity
        angular_momentum += contribution
        contributions.append({
            "body_id": body_id,
            "body_name": model.body(body_id).name,
            "mass": mass,
            "com": data.xipos[body_id].tolist(),
            "linear_velocity": velocity.tolist(),
            "angular_velocity": omega.tolist(),
            "spin_angular_momentum": spin.tolist(),
            "orbital_angular_momentum": orbital.tolist(),
            "centroidal_angular_momentum": contribution.tolist(),
        })

    contact_force = np.zeros(3, dtype=float)
    contact_torque = np.zeros(3, dtype=float)
    contacts = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if (body1 == 0) == (body2 == 0):
            continue
        local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_id, local)
        frame = np.asarray(contact.frame).reshape(3, 3)
        force = frame.T @ local[:3]
        torque_at_contact = frame.T @ local[3:]
        # mj_contactForce is the force on geom1; orient it onto the robot.
        sign = 1.0 if body1 != 0 else -1.0
        force *= sign
        torque_at_contact *= sign
        torque = (
            torque_at_contact
            + np.cross(np.asarray(contact.pos) - com, force)
        )
        contact_force += force
        contact_torque += torque
        contacts.append({
            "contact_id": contact_id,
            "robot_body_id": body1 if body1 != 0 else body2,
            "robot_body_name": model.body(
                body1 if body1 != 0 else body2
            ).name,
            "geom1": model.geom(contact.geom1).name,
            "geom2": model.geom(contact.geom2).name,
            "position": np.asarray(contact.pos).tolist(),
            "distance": float(contact.dist),
            "force_world": force.tolist(),
            "torque_about_system_com": torque.tolist(),
        })

    subtree = np.asarray(data.subtree_angmom[0], dtype=float).copy()
    return {
        "system_mass": total_mass,
        "system_com": com.tolist(),
        "com_velocity": (linear_momentum / total_mass).tolist(),
        "linear_momentum": linear_momentum.tolist(),
        "centroidal_angular_momentum": angular_momentum.tolist(),
        "mujoco_subtree_angular_momentum": subtree.tolist(),
        "angular_momentum_crosscheck_linf": float(
            np.max(np.abs(angular_momentum - subtree))
        ),
        "body_contributions": contributions,
        "robot_terrain_contacts": contacts,
        "net_terrain_force": contact_force.tolist(),
        "net_terrain_torque_about_com": contact_torque.tolist(),
    }
