"""V21 DVGC-Physical port for MuJoCo Playground / MJX.

The environment keeps V21's essential contract:
  * CoM is used only by the host-side candidate generator, never as actor input
    or a reward tracking target;
  * reset supports natural starts and Tube-guided snapshot RSI;
  * Recovery is the only formal end-to-end success signal.

The original V21 XML keeps its ellipsoid wheel-collision geoms, cylinder knee
motor and box step.  That geometry requires MJX-Warp.  In
``contact_mode='imu'`` the simulator still resolves these physical collisions,
but DVGC never reads ``data.contact``: landing is inferred from phase history,
platform location, pre-impact downward velocity, the local accelerometer
impulse, and post-impact stable support.  This matches the intended deployment
sensor boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from .bank import SnapshotBank
from .action_mapping import knee_position_target
from .config import ID_STAGE, STAGE_ID, SNAPSHOT_SCHEMA, default_config
from .entry import ENTRY_FEATURE_NAMES
from .signals import compute_takeoff_signals, wheel_tire_bottoms
from .rewards import (
    compute_descent_local_reward, compute_descent_rsi_pilot_reward,
    compute_stage_next_entry_reward, compute_takeoff_reward_profile,
)

# Primary terminal/event codes.  They are integers because State.info must be a
# JAX pytree; host tools map them back to readable V21-style strings.
END_NONE = 0
END_RECOVERY = 1
END_PROHIBITED_CONTACT = 2
END_INVALID_WHEEL_STEP = 3
END_ROLL_LIMIT = 4
END_PITCH_LIMIT = 5
END_BACKWARD = 6
END_PLATFORM_BACK_EDGE = 7
END_STAGE_TIMEOUT = 8
END_PRETAKEOFF_AIRBORNE = 9
END_TAKEOFF_POSITIVE_PITCH_FAILURE = 10
END_TAKEOFF_WHEELIE_FAILURE = 11
END_TAKEOFF_MISSED_LIFTOFF_DEADLINE = 12
END_TAKEOFF_MISSED_WHEEL_CLEARANCE = 13
END_CHAIN_ENTRY = 14
END_NONFINITE = 15
END_NEXT_STAGE_ENTRY = 16

END_REASON = {
    END_NONE: "horizon",
    END_RECOVERY: "recovery",
    END_PROHIBITED_CONTACT: "prohibited_contact",
    END_INVALID_WHEEL_STEP: "invalid_wheel_step_contact",
    END_ROLL_LIMIT: "roll_limit",
    END_PITCH_LIMIT: "pitch_limit",
    END_BACKWARD: "backward",
    END_PLATFORM_BACK_EDGE: "platform_back_edge_exit",
    END_STAGE_TIMEOUT: "stage_timeout",
    END_PRETAKEOFF_AIRBORNE: "prelaunch_airborne",
    END_TAKEOFF_POSITIVE_PITCH_FAILURE: "takeoff_positive_pitch_failure",
    END_TAKEOFF_WHEELIE_FAILURE: "takeoff_wheelie_failure",
    END_TAKEOFF_MISSED_LIFTOFF_DEADLINE: "takeoff_missed_liftoff_deadline",
    END_TAKEOFF_MISSED_WHEEL_CLEARANCE: "takeoff_missed_wheel_clearance_deadline",
    END_CHAIN_ENTRY: "chain_entry",
    END_NONFINITE: "nonfinite",
    END_NEXT_STAGE_ENTRY: "next_stage_entry",
}

RESET_SOURCE = {
    "flight_curriculum": 0,
    "canonical_entry_rehearsal": 1,
    "landing_tube_rehearsal": 2,
    "natural": 3,
}

DESCENT_BOOTSTRAP_GROUP = {"provisional_safe": 0, "boundary": 1, "successful_anchor": 2}
DESCENT_LAYER = {"late": 0, "middle": 1, "early": 2}
REACHABILITY_OBJECTIVE = {
    "": 0,
    "takeoff_to_ascent": 1,
    "ascent_to_apex": 2,
    "apex_to_descent": 3,
    "descent_to_landing": 4,
}
PHASE_RSI_OBJECTIVE = {
    "takeoff": REACHABILITY_OBJECTIVE["takeoff_to_ascent"],
    "ascent": REACHABILITY_OBJECTIVE["ascent_to_apex"],
    "apex": REACHABILITY_OBJECTIVE["apex_to_descent"],
    "descent": REACHABILITY_OBJECTIVE["descent_to_landing"],
    "landing": REACHABILITY_OBJECTIVE[""],
}


def _deg2rad(x: float) -> float:
    return float(x) * jp.pi / 180.0


def stable_task_distance_to_front(x: jax.Array, step_front_x: float) -> jax.Array:
    """Bit-stable float32 scaling shared by online and v4 reconstruction."""
    return (
        jp.asarray(step_front_x, jp.float32) - jp.asarray(x, jp.float32)
    ) * jp.asarray(1.0 / 3.0, jp.float32)


def _quat_from_euler_xyz(roll: jax.Array, pitch: jax.Array, yaw: jax.Array) -> jax.Array:
    cr, sr = jp.cos(roll * 0.5), jp.sin(roll * 0.5)
    cp, sp = jp.cos(pitch * 0.5), jp.sin(pitch * 0.5)
    cy, sy = jp.cos(yaw * 0.5), jp.sin(yaw * 0.5)
    return jp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _euler_from_quat(q: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    qw, qx, qy, qz = q
    roll = jp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = jp.arcsin(jp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = jp.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def _rotate_world_to_body(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate world vector v by inverse of MuJoCo wxyz quaternion q."""
    qw, qxyz = q[0], q[1:]
    # inverse rotation: use conjugate q=(w,-xyz)
    xyz = -qxyz
    t = 2.0 * jp.cross(xyz, v)
    return v + qw * t + jp.cross(xyz, t)


class OrangeBikeDVGC(mjx_env.MjxEnv):
    """MJX implementation of the V21 DVGC-Physical control contract."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, float, list[Any]]]] = None,
        snapshot_bank: Optional[SnapshotBank] = None,
        cert_bank: Optional[SnapshotBank] = None,
        stage_support_bank: Optional[SnapshotBank] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        self._stage_name = str(self._config.training_stage).lower()
        if self._stage_name not in STAGE_ID and self._stage_name != "full":
            raise ValueError(f"Unsupported training_stage={self._stage_name!r}")
        self._reachability_objective = str(getattr(self._config,"stage_reachability_objective","")).lower()
        allowed={*REACHABILITY_OBJECTIVE, "phase_balanced_rsi"}
        if self._reachability_objective not in allowed:
            raise ValueError(f"Unsupported stage_reachability_objective={self._reachability_objective!r}")
        expected={"takeoff_to_ascent":"takeoff","ascent_to_apex":"flight","apex_to_descent":"flight","descent_to_landing":"flight"}
        if (self._reachability_objective in expected
                and self._stage_name != expected[self._reachability_objective]):
            raise ValueError("Reachability objective and canonical training_stage disagree")
        if self._reachability_objective == "phase_balanced_rsi" and self._stage_name != "flight":
            raise ValueError("phase_balanced_rsi requires canonical training_stage='flight'")
        self._phase_balanced_reachability = self._reachability_objective == "phase_balanced_rsi"
        self._default_reachability_objective_id = REACHABILITY_OBJECTIVE.get(
            self._reachability_objective, REACHABILITY_OBJECTIVE[""]
        )
        self._stage_id = STAGE_ID.get(self._stage_name, STAGE_ID["approach"])
        self._contact_mode = str(self._config.contact_mode).lower()
        if self._contact_mode not in ("mjx", "imu_fallback", "imu", "imu_only"):
            raise ValueError(f"Unsupported contact_mode={self._contact_mode!r}")
        self._imu_only = self._contact_mode in ("imu", "imu_only")
        self._use_imu_fallback = self._contact_mode in ("imu_fallback", "imu", "imu_only")
        self._landing_reward_profile = str(self._config.landing_reward_profile).lower()
        if self._landing_reward_profile not in ("legacy_v21", "imu_recovery_strict"):
            raise ValueError(
                "Unsupported landing_reward_profile="
                f"{self._landing_reward_profile!r}; expected 'legacy_v21' or 'imu_recovery_strict'."
            )
        self._strict_landing_reward = self._landing_reward_profile == "imu_recovery_strict"
        if str(self._config.impl).lower() == "warp" and not self._imu_only:
            raise ValueError(
                "MJX-Warp keeps contact buffers private; use contact_mode='imu' "
                "for this original OrangeBike XML."
            )

        # The supplied XML is the single authoritative model.  It is loaded
        # directly from disk so MuJoCo resolves its original meshdir and STL
        # files.  No runtime XML, geometry conversion, XML sanitation, or
        # alternate model path is used anywhere in this project.
        self._xml_path = str(Path(self._config.xml_path).resolve())
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        solver_name = str(getattr(self._config, "mjx_solver", "XML")).upper()
        solver_map = {
            "XML": None,
            "CG": mujoco.mjtSolver.mjSOL_CG,
            "NEWTON": mujoco.mjtSolver.mjSOL_NEWTON,
        }
        if solver_name not in solver_map:
            raise ValueError(
                f"Unsupported mjx_solver={solver_name!r}; expected XML, CG, or NEWTON"
            )
        self._xml_solver = int(self._mj_model.opt.solver)
        if solver_map[solver_name] is not None:
            self._mj_model.opt.solver = solver_map[solver_name]
        self._effective_mjx_solver = solver_name
        # Real dynamics variants used by branch certification.  These are
        # applied before conversion to MJX, so every branch environment has a
        # genuinely different physical model rather than only action noise.
        self._mj_model.body_mass[1:] *= float(self._config.mass_scale)
        self._mj_model.geom_friction[:, 0] *= float(self._config.friction_scale)
        self._mj_model.actuator_forcerange[:] *= float(self._config.actuator_force_scale)
        self._mj_model.opt.gravity[:] *= float(self._config.gravity_scale)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._cert_bank = cert_bank
        self._cert_bank_explicit = cert_bank is not None
        self._stage_support_bank = stage_support_bank
        self._post_init(snapshot_bank or SnapshotBank())

    # ------------------------------------------------------------------
    # Model lookup and reset-pool construction
    # ------------------------------------------------------------------
    def _id(self, kind: str, name: str) -> int:
        try:
            return int(getattr(self._mj_model, kind)(name).id)
        except (KeyError, ValueError):
            return -1

    def _post_init(self, snapshot_bank: SnapshotBank) -> None:
        free = [j for j in range(self._mj_model.njnt) if int(self._mj_model.jnt_type[j]) == 0]
        if not free:
            raise ValueError("OrangeBikeDVGC requires a free-joint base")
        self._root_jid = int(free[0])
        self._qpos0 = int(self._mj_model.jnt_qposadr[self._root_jid])
        self._qvel0 = int(self._mj_model.jnt_dofadr[self._root_jid])
        self._root_body_id = self._id("body", self._config.root_body_name)
        if self._root_body_id < 0:
            self._root_body_id = 1
        self._frontwheel_body_id = self._id("body", "frontwheel")
        self._rearwheel_body_id = self._id("body", "rearwheel")
        self._joint_qpos = {name: int(self._mj_model.jnt_qposadr[self._id("joint", name)]) for name in ("steering_joint", "rearwheel_joint", "frontwheel_joint", "hip_joint", "knee_joint")}
        self._joint_qvel = {name: int(self._mj_model.jnt_dofadr[self._id("joint", name)]) for name in ("steering_joint", "rearwheel_joint", "frontwheel_joint", "hip_joint", "knee_joint")}
        self._act = {name: self._id("actuator", name) for name in ("cmd_steering_v", "cmd_rearwheel_f", "cmd_hip_f", "cmd_knee_f")}
        self._geom = {name: self._id("geom", name) for name in ("floor", "step", "frontwheel_collision", "rearwheel_collision", "base_collision", "steer_collision", "downarm_collision", "knee_motor_collision", "uparm_collision")}
        self._body_collision_ids = tuple(v for k, v in self._geom.items() if k not in ("floor", "step", "frontwheel_collision", "rearwheel_collision") and v >= 0)
        self._sensor = {}
        for name in ("acc_local", "gyro_local", "steering_joint_pos_sensor", "steering_joint_vel_sensor", "rearwheel_joint_vel_sensor", "frontwheel_joint_vel_sensor", "hip_joint_pos_sensor", "knee_joint_pos_sensor", "hip_joint_vel_sensor", "knee_joint_vel_sensor"):
            sid = self._id("sensor", name)
            if sid >= 0:
                self._sensor[name] = (int(self._mj_model.sensor_adr[sid]), int(self._mj_model.sensor_dim[sid]))

        if self._mj_model.nkey:
            self._init_qpos = jp.asarray(self._mj_model.key_qpos[0])
        else:
            self._init_qpos = jp.zeros((self._mj_model.nq,), dtype=jp.float32)
        self._init_qvel = jp.zeros((self._mj_model.nv,), dtype=jp.float32)
        self._neutral_action = jp.zeros((self._mj_model.nu,), dtype=jp.float32)
        self._actor_frame_dim = 35
        self._actor_history_steps = max(1, int(self._config.actor_history_steps))
        self._actor_history_len = self._actor_history_steps - 1
        self._actor_obs_dim = self._actor_history_steps * self._actor_frame_dim
        self._privileged_obs_dim = 29

        # JAX arrays are stored on the environment object as static constants.
        reset_protocol = snapshot_bank.metadata.get("reset_source_protocol")
        if reset_protocol:
            records = [r for r in snapshot_bank.records if r.get("reset_source") in RESET_SOURCE]
            weights = np.asarray([float(r["reset_weight"]) for r in records], np.float32)
            if records and (not np.all(np.isfinite(weights)) or np.any(weights <= 0)):
                raise ValueError("Multi-source reset weights must be finite and positive")
            if len(weights):
                weights /= weights.sum()
        else:
            records, weights = (
                snapshot_bank.reset_distribution(
                self._stage_name,
                safe_mass=float(self._config.rsi_safe_mass),
                boundary_mass=float(self._config.rsi_boundary_mass),
                aux_mass=float(self._config.rsi_aux_mass),
                rehearsal_mass=float(self._config.downstream_rehearsal_mass),
                tube_activation_min_safe=int(self._config.tube_activation_min_safe),
            )
                if self._stage_name != "full" else ([], np.empty((0,), np.float32))
            )
        self._bank_count = len(records)
        nq, nv, nu = self._mj_model.nq, self._mj_model.nv, self._mj_model.nu
        if records:
            def ps(record: Dict[str, Any]) -> Dict[str, Any]:
                return dict(record.get("policy_state", {}))
            self._bank_qpos = jp.asarray(np.stack([r["qpos"] for r in records]))
            self._bank_qvel = jp.asarray(np.stack([r["qvel"] for r in records]))
            self._bank_ctrl = jp.asarray(np.stack([r["ctrl"] for r in records]))
            self._bank_qacc_warmstart = jp.asarray(np.stack([r["qacc_warmstart"] for r in records]))
            self._bank_phase = jp.asarray([int(r.get("oracle_phase", STAGE_ID.get(self._stage_name, 0))) for r in records], jp.int32)
            self._bank_estimated_phase = jp.asarray([int(ps(r).get("filter_phase", r.get("oracle_phase", self._stage_id))) for r in records], jp.int32)
            self._bank_phase_probs = jp.asarray(np.stack([
                np.asarray(
                    ps(r).get(
                        "phase_probs",
                        np.eye(4, dtype=np.float32)[int(ps(r).get("filter_phase", r.get("oracle_phase", self._stage_id)))],
                    ),
                    np.float32,
                )
                for r in records
            ]))
            self._bank_airborne = jp.asarray([int(r.get("had_airborne", 0)) for r in records], jp.int32)
            self._bank_landed = jp.asarray([int(r.get("had_valid_landing", 0)) for r in records], jp.int32)
            self._bank_age = jp.asarray([int(r.get("contact_age", 0)) for r in records], jp.int32)
            self._bank_recovery_count = jp.asarray([int(r.get("recovery_count", 0)) for r in records], jp.int32)
            self._bank_bounce_count = jp.asarray([int(r.get("landing_bounce_count", 0)) for r in records], jp.int32)
            self._bank_airborne_count = jp.asarray([int(r.get("airborne_count", 0)) for r in records], jp.int32)
            self._bank_prelaunch_airborne_count = jp.asarray([int(r.get("prelaunch_airborne_count", 0)) for r in records], jp.int32)
            self._bank_invalid_wheel_count = jp.asarray([int(r.get("invalid_wheel_count", 0)) for r in records], jp.int32)
            self._bank_prev_acc_z = jp.asarray([float(ps(r).get("prev_acc_z", np.nan)) for r in records], jp.float32)
            self._bank_prev_vz = jp.asarray([float(ps(r).get("prev_vz", np.nan)) for r in records], jp.float32)
            self._bank_apex_seen = jp.asarray([
                int(r.get("apex_seen", int(r.get("reference_index", r.get("source_index", -1))) >= 220))
                for r in records
            ], jp.int32)
            self._bank_last_action = jp.asarray(np.stack([np.asarray(ps(r).get("last_action", np.zeros(nu)), np.float32) for r in records]))
            default_history = np.zeros((self._actor_history_len, self._actor_frame_dim), np.float32)
            histories, valid = [], []
            for r in records:
                h = np.asarray(ps(r).get("obs_history", default_history), np.float32)
                ok = h.shape == default_history.shape
                histories.append(h if ok else default_history)
                valid.append(ok)
            self._bank_obs_history = jp.asarray(np.stack(histories), jp.float32)
            self._bank_obs_history_valid = jp.asarray(valid, bool)
            default_actor_observation = np.zeros((self._actor_obs_dim,), np.float32)
            actor_observations, actor_observation_valid = [], []
            for r in records:
                value = np.asarray(ps(r).get("actor_observation", default_actor_observation), np.float32)
                ok = value.shape == default_actor_observation.shape
                actor_observations.append(value if ok else default_actor_observation)
                actor_observation_valid.append(ok)
            self._bank_actor_observation = jp.asarray(np.stack(actor_observations), jp.float32)
            self._bank_actor_observation_valid = jp.asarray(actor_observation_valid, bool)
            self._bank_reset_source = jp.asarray([
                RESET_SOURCE[r.get("reset_source", "flight_curriculum")] for r in records
            ], jp.int32)
            self._bank_bootstrap_group = jp.asarray([
                DESCENT_BOOTSTRAP_GROUP.get(r.get("bootstrap_group"), -1) for r in records
            ], jp.int32)
            self._bank_descent_layer = jp.asarray([
                DESCENT_LAYER.get(r.get("descent_layer"), -1) for r in records
            ], jp.int32)
            self._reset_parent_ids = tuple(sorted({str(r["reset_parent_id"]) for r in records if r.get("reset_parent_id")}))
            parent_index = {name: index for index, name in enumerate(self._reset_parent_ids)}
            self._bank_reset_parent = jp.asarray([
                parent_index.get(str(r.get("reset_parent_id", "")), -1) for r in records
            ], jp.int32)
            if self._phase_balanced_reachability:
                invalid = sorted({str(r.get("phase_rsi_stage")) for r in records}
                                 - set(PHASE_RSI_OBJECTIVE))
                if invalid:
                    raise ValueError(
                        f"phase_balanced_rsi records have invalid phase_rsi_stage values: {invalid}"
                    )
                objective_ids = [PHASE_RSI_OBJECTIVE[str(r["phase_rsi_stage"])] for r in records]
            else:
                objective_ids = [self._default_reachability_objective_id] * len(records)
            self._bank_reachability_objective = jp.asarray(objective_ids, jp.int32)
            self._bank_logw = jp.log(jp.asarray(weights) + 1e-12)
        else:
            self._bank_qpos = jp.zeros((1, nq), jp.float32); self._bank_qvel = jp.zeros((1, nv), jp.float32)
            self._bank_ctrl = jp.zeros((1, nu), jp.float32); self._bank_phase = jp.zeros((1,), jp.int32)
            self._bank_qacc_warmstart = jp.zeros((1, nv), jp.float32)
            self._bank_estimated_phase = jp.zeros((1,), jp.int32)
            self._bank_phase_probs = jp.zeros((1, 4), jp.float32)
            self._bank_airborne = jp.zeros((1,), jp.int32); self._bank_landed = jp.zeros((1,), jp.int32)
            self._bank_age = jp.zeros((1,), jp.int32); self._bank_recovery_count = jp.zeros((1,), jp.int32)
            self._bank_bounce_count = jp.zeros((1,), jp.int32); self._bank_airborne_count = jp.zeros((1,), jp.int32)
            self._bank_prelaunch_airborne_count = jp.zeros((1,), jp.int32); self._bank_invalid_wheel_count = jp.zeros((1,), jp.int32)
            self._bank_prev_acc_z = jp.full((1,), jp.nan, jp.float32); self._bank_prev_vz = jp.full((1,), jp.nan, jp.float32)
            self._bank_apex_seen = jp.zeros((1,), jp.int32)
            self._bank_last_action = jp.zeros((1, nu), jp.float32)
            self._bank_obs_history = jp.zeros((1, self._actor_history_len, self._actor_frame_dim), jp.float32)
            self._bank_obs_history_valid = jp.zeros((1,), bool); self._bank_logw = jp.zeros((1,), jp.float32)
            self._bank_actor_observation = jp.zeros((1, self._actor_obs_dim), jp.float32)
            self._bank_actor_observation_valid = jp.zeros((1,), bool)
            self._bank_reset_source = jp.zeros((1,), jp.int32)
            self._bank_bootstrap_group = -jp.ones((1,), jp.int32)
            self._bank_descent_layer = -jp.ones((1,), jp.int32)
            self._bank_reset_parent = -jp.ones((1,), jp.int32)
            self._bank_reachability_objective = jp.full(
                (1,), self._default_reachability_objective_id, jp.int32
            )
            self._reset_parent_ids = ()

        # Recursive bootstrap entry set comes from the next phase's independently
        # certified Final-safe core.  Matching uses robust standardized features,
        # never a raw mixed-unit Euclidean norm.
        next_stage = {"approach": "takeoff", "takeoff": "flight", "flight": "landing"}.get(self._stage_name)
        self._chain_target_stage = next_stage
        chain_bank = self._cert_bank if self._cert_bank is not None else snapshot_bank
        safe = chain_bank.records_for_phase(next_stage, final_labels=["safe"], include_training_only=False) if next_stage else []
        self._safe_count = len(safe)
        matcher = chain_bank.metadata.get("entry_matcher") if next_stage == "landing" else None
        self._entry_matcher = bool(matcher)
        if safe:
            host = np.asarray([r["entry_feature"] if self._entry_matcher else r["physical_feature"] for r in safe], np.float32)
            if self._entry_matcher:
                center=np.asarray(matcher["center"],np.float32); scale=np.asarray(matcher["scale"],np.float32); self._safe_radius=float(matcher["radius"])
            else:
                center=np.median(host,axis=0); mad=np.median(np.abs(host-center[None,:]),axis=0)*1.4826; std=host.std(axis=0); scale=np.maximum(np.maximum(mad,.25*std),1e-4); self._safe_radius=float(self._config.tube_match_radius_z)
            self._safe_features = jp.asarray((host - center) / scale)
            self._safe_center = jp.asarray(center)
            self._safe_scale = jp.asarray(scale)
        else:
            dim=len(ENTRY_FEATURE_NAMES) if self._entry_matcher else 16
            self._safe_features=jp.zeros((1,dim),jp.float32); self._safe_center=jp.zeros((dim,),jp.float32); self._safe_scale=jp.ones((dim,),jp.float32); self._safe_radius=float(self._config.tube_match_radius_z)

        # Local stage support is separate from the recursively certified C_L.
        # It may terminate Apex->Descent discovery, but can never set Chain.
        stage_matcher = (stage_support_bank.metadata.get("stage_entry_matcher", {})
                         if (stage_support_bank := self._stage_support_bank) is not None else {})
        self._stage_support_enabled = bool(stage_matcher and stage_support_bank.records)
        if self._stage_support_enabled:
            raw = np.asarray([r["physical_feature"] for r in stage_support_bank.records], np.float32)
            center = np.asarray(stage_matcher["center"], np.float32); scale = np.asarray(stage_matcher["scale"], np.float32)
            self._stage_support_features = jp.asarray((raw-center)/scale)
            self._stage_support_center = jp.asarray(center); self._stage_support_scale = jp.asarray(scale)
            radii = stage_matcher.get("radii")
            if radii is None:
                radii = np.full((len(raw),), float(stage_matcher["radius"]), np.float32)
            radii = np.asarray(radii, np.float32)
            if radii.shape != (len(raw),) or np.any(radii <= 0.0):
                raise ValueError("Per-anchor Descent-support radii are invalid")
            self._stage_support_radii = jp.asarray(radii)
            self._stage_support_radius = float(stage_matcher.get("radius", np.max(radii)))
            envelope = stage_matcher["reference_envelope"]
            self._stage_support_x_bounds = (float(envelope["x"]["min"])-float(stage_matcher.get("envelope_tolerance_x",0.0)),
                                            float(envelope["x"]["max"])+float(stage_matcher.get("envelope_tolerance_x",0.0)))
            self._stage_support_z_bounds = (float(envelope["z"]["min"])-float(stage_matcher.get("envelope_tolerance_z",0.0)),
                                            float(envelope["z"]["max"])+float(stage_matcher.get("envelope_tolerance_z",0.0)))
            self._stage_descent_vz_bounds = (float(stage_matcher.get("descent_vz_min",-2.5)),float(stage_matcher.get("descent_vz_max",-.05)))
            self._stage_roll_rate_max = float(stage_matcher.get("max_abs_roll_rate",self._config.recovery_max_angvel))
            self._stage_pitch_rate_max = float(stage_matcher.get("max_abs_pitch_rate",self._config.recovery_max_angvel))
        else:
            self._stage_support_features=jp.zeros((1,16),jp.float32);self._stage_support_center=jp.zeros(16,jp.float32);self._stage_support_scale=jp.ones(16,jp.float32)
            self._stage_support_radius=0.0;self._stage_support_radii=jp.ones((1,),jp.float32);self._stage_support_x_bounds=(-jp.inf,jp.inf);self._stage_support_z_bounds=(-jp.inf,jp.inf)
            self._stage_descent_vz_bounds=(-2.5,-.05);self._stage_roll_rate_max=float(self._config.recovery_max_angvel);self._stage_pitch_rate_max=float(self._config.recovery_max_angvel)

    @property
    def action_size(self) -> int:
        return int(self._mj_model.nu)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return self._xml_path

    # ------------------------------------------------------------------
    # Low-level state / sensing / contacts
    # ------------------------------------------------------------------
    def _make_data(self, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array) -> mjx.Data:
        # New Playground releases expose naccdmax; the supplied mjx_test-style
        # API may be older and accept only naconmax/njmax.  Keep one source tree
        # usable in both by selecting the compatible public signature at trace
        # construction time.
        try:
            data = mjx_env.make_data(
                self.mj_model,
                qpos=qpos,
                qvel=qvel,
                impl=self._config.impl,
                naconmax=int(self._config.naconmax),
                naccdmax=int(self._config.naccdmax),
                njmax=int(self._config.njmax),
            )
        except TypeError:
            data = mjx_env.make_data(
                self.mj_model,
                qpos=qpos,
                qvel=qvel,
                impl=self._config.impl,
                naconmax=int(self._config.naconmax),
                njmax=int(self._config.njmax),
            )
        data = data.replace(ctrl=ctrl)
        return mjx.forward(self.mjx_model, data)

    def _sensor_vec(self, data: mjx.Data, name: str, size: int) -> jax.Array:
        adr_dim = self._sensor.get(name)
        if adr_dim is None:
            return jp.zeros((size,), dtype=jp.float32)
        adr, dim = adr_dim
        raw = data.sensordata[adr: adr + dim]
        return jp.pad(raw, (0, max(0, size - dim)))[:size]

    def _root_state(self, data: mjx.Data) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        qpos = data.qpos[self._qpos0:self._qpos0 + 7]
        qvel = data.qvel[self._qvel0:self._qvel0 + 6]
        roll, pitch, yaw = _euler_from_quat(qpos[3:7])
        return qpos, qvel, roll, pitch, yaw

    def _physical_feature(self, data: mjx.Data) -> jax.Array:
        qpos, qvel, roll, pitch, yaw = self._root_state(data)
        return jp.concatenate([
            qpos[:3], jp.asarray([roll, pitch, yaw]), qvel,
            jp.asarray([
                data.qpos[self._joint_qpos["steering_joint"]],
                data.qpos[self._joint_qpos["hip_joint"]],
                data.qpos[self._joint_qpos["knee_joint"]],
                data.qvel[self._joint_qvel["rearwheel_joint"]],
            ]),
        ])

    def _landing_entry_feature(self, data, had_valid_landing, support, entry_age):
        f=self._physical_feature(data); cfg=self._config
        progress=(f[0]-(cfg.step_front_x+cfg.valid_landing_min_past_edge))/max(float(cfg.step_back_x-cfg.valid_landing_back_margin-(cfg.step_front_x+cfg.valid_landing_min_past_edge)),1e-6)
        return jp.concatenate([jp.asarray([f[0]-cfg.step_front_x,f[1],f[2]-cfg.step_top_z]),f[3:16],jp.asarray([had_valid_landing.astype(jp.float32),support.astype(jp.float32),jp.minimum(entry_age.astype(jp.float32),float(cfg.landing_entry_window_steps))/float(cfg.landing_entry_window_steps),progress])])

    def _contact_masks(self, data: mjx.Data) -> Dict[str, jax.Array]:
        """Return JAX contact masks, or all-false in deployment IMU mode.

        ``imu`` deliberately performs no access to ``data.contact``.  This is
        required for MJX-Warp, and mirrors a physical robot where collision
        pairs are not observable.  It does *not* disable MuJoCo collision
        constraints: the simulator still produces support/impact forces.
        """
        if self._imu_only:
            false = jp.asarray(False)
            return {
                "front_top": false, "rear_top": false,
                "front_floor": false, "rear_floor": false,
                "wheel_top": false, "wheel_any": false,
                "wheel_invalid": false, "prohibited": false,
                "contact_active": false,
            }
        c = data.contact
        g1, g2 = c.geom1, c.geom2
        # In current MJX unused fixed contact slots have a negative EFC
        # address.  Some older MJX builds expose only geom ids; retain a safe
        # compatibility guard rather than falling back to a height proxy.
        if hasattr(c, "efc_address"):
            active = (c.efc_address >= 0) & (g1 >= 0) & (g2 >= 0)
        else:
            active = (g1 >= 0) & (g2 >= 0)

        def pair(a: int, b: int) -> jax.Array:
            if a < 0 or b < 0:
                return jp.zeros_like(active, dtype=bool)
            return active & (((g1 == a) & (g2 == b)) | ((g1 == b) & (g2 == a)))

        step, floor = self._geom["step"], self._geom["floor"]
        fw, rw = self._geom["frontwheel_collision"], self._geom["rearwheel_collision"]
        front_step = pair(fw, step); rear_step = pair(rw, step)
        front_floor = pair(fw, floor); rear_floor = pair(rw, floor)
        pos = c.pos
        frame = c.frame.reshape((-1, 3, 3))
        top = (
            (jp.abs(pos[:, 2] - float(self._config.step_top_z)) <= float(self._config.step_top_z_tol))
            & (pos[:, 0] >= float(self._config.step_front_x + self._config.step_top_xy_margin))
            & (pos[:, 0] <= float(self._config.step_back_x - self._config.step_top_xy_margin))
            & (jp.abs(pos[:, 1]) <= float(self._config.step_half_width - self._config.step_top_xy_margin))
            & (jp.abs(frame[:, 0, 2]) >= float(self._config.step_top_normal_z_min))
        )
        front_top = jp.any(front_step & top); rear_top = jp.any(rear_step & top)
        wheel_invalid = jp.any((front_step | rear_step) & ~top)
        prohibited = jp.asarray(False)
        for body_geom in self._body_collision_ids:
            prohibited = prohibited | jp.any(pair(body_geom, step) | pair(body_geom, floor))
        return {
            "front_top": front_top, "rear_top": rear_top,
            "front_floor": jp.any(front_floor), "rear_floor": jp.any(rear_floor),
            "wheel_top": front_top | rear_top,
            "wheel_any": jp.any(front_step | rear_step | front_floor | rear_floor),
            "wheel_invalid": wheel_invalid,
            "prohibited": prohibited,
            "contact_active": jp.any(active),
        }

    def _knee_action_to_target(
        self, knee_action: jax.Array, knee_position: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Maps action_3 to an incremental knee position target.

        The supplied XML starts the knee at its upper joint limit: q=2.5 rad.
        In the old residual mapping, positive action tried to increase q beyond
        2.5 rad, so every positive command was clipped to the same target and
        the positive half-axis was dead.

        The revised mapping changes only the action convention, not the XML:

            q_target = clip(q_current - action_3 * delta_q, q_min, q_max)

        Therefore positive action decreases the knee angle, which is the
        launch-extension direction from the authoritative compressed key;
        negative action increases it back toward the compressed upper limit,
        and zero action holds the current knee position.  Both signs remain
        proportional until a real XML joint limit is reached.
        """
        cfg = self._config
        return knee_position_target(
            knee_position,
            knee_action,
            target_delta=float(cfg.knee_action_target_delta),
            knee_min=float(cfg.knee_min),
            knee_max=float(cfg.knee_max),
            xp=jp,
        )

    def _action_to_ctrl(self, action: jax.Array, knee_position: jax.Array) -> jax.Array:
        """Maps [steer, drive, hip, knee] to the original XML actuators."""
        a = jp.clip(action, -1.0, 1.0)
        cfg = self._config
        steer = a[0] * float(cfg.steer_action_scale)
        rear = jp.clip(float(cfg.target_forward_speed / cfg.wheel_roll_radius) + a[1] * float(cfg.rear_wheel_action_delta), -5.0, 40.0)
        hip = jp.where(a[2] >= 0.0, float(cfg.hip_initial) + a[2] * (float(cfg.hip_max) - float(cfg.hip_initial)), float(cfg.hip_initial) + a[2] * (float(cfg.hip_initial) - float(cfg.hip_min)))
        _, knee, _ = self._knee_action_to_target(a[3], knee_position)
        ctrl = jp.zeros((self.action_size,), dtype=jp.float32)
        ctrl = ctrl.at[self._act["cmd_steering_v"]].set(steer)
        ctrl = ctrl.at[self._act["cmd_rearwheel_f"]].set(rear)
        ctrl = ctrl.at[self._act["cmd_hip_f"]].set(hip)
        ctrl = ctrl.at[self._act["cmd_knee_f"]].set(knee)
        return ctrl

    def _platform_gate(self, qpos: jax.Array) -> jax.Array:
        """Known-map platform interior used by the deployable IMU filter."""
        x, y = qpos[0], qpos[1]
        return (
            (x >= self._config.step_front_x + self._config.valid_landing_min_past_edge)
            & (x <= self._config.step_back_x - self._config.valid_landing_back_margin)
            & (jp.abs(y) <= self._config.step_half_width - self._config.landing_side_margin)
        )

    def _imu_support_estimate(
        self, qpos: jax.Array, qvel: jax.Array, had_valid_landing: jax.Array
    ) -> jax.Array:
        """Support estimate available to both deployment Actor and environment."""
        expected_base_z = self._config.step_top_z + self._config.nominal_base_z_ground
        height_ok = jp.abs(qpos[2] - expected_base_z) <= self._config.imu_support_height_tol
        vz_ok = jp.abs(qvel[2]) <= self._config.imu_support_max_abs_vz
        return (had_valid_landing > 0) & self._platform_gate(qpos) & height_ok & vz_ok

    def build_actor_current_frame_v4(
        self,
        data: mjx.Data,
        last_action: jax.Array,
        phase_probs: jax.Array,
        had_valid_landing: jax.Array,
        contact_age: jax.Array,
        recovery_count: jax.Array,
        prev_acc_z: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        """One deployment-valid sensor frame; it deliberately excludes contacts."""
        cfg = self._config
        qpos, qvel, _, _, _ = self._root_state(data)
        gyro = self._sensor_vec(data, "gyro_local", 3)
        acc = self._sensor_vec(data, "acc_local", 3)
        gravity = _rotate_world_to_body(qpos[3:7], jp.asarray([0.0, 0.0, -1.0]))
        joint_pos = jp.asarray([
            self._sensor_vec(data, "steering_joint_pos_sensor", 1)[0],
            self._sensor_vec(data, "hip_joint_pos_sensor", 1)[0],
            self._sensor_vec(data, "knee_joint_pos_sensor", 1)[0],
            self._sensor_vec(data, "rearwheel_joint_vel_sensor", 1)[0],
        ])
        joint_vel = jp.asarray([
            self._sensor_vec(data, "steering_joint_vel_sensor", 1)[0],
            self._sensor_vec(data, "hip_joint_vel_sensor", 1)[0],
            self._sensor_vec(data, "knee_joint_vel_sensor", 1)[0],
            self._sensor_vec(data, "frontwheel_joint_vel_sensor", 1)[0],
        ])
        x, z = qpos[0], qpos[2]
        local_terrain = jp.where((x >= cfg.step_front_x) & (x <= cfg.step_back_x), cfg.step_top_z, 0.0)
        # Keep this scale as an explicit float32 multiply.  XLA lowered the
        # equivalent division context-dependently inside the fused online step
        # versus standalone snapshot reconstruction, producing a one-ULP
        # mismatch for otherwise identical materialized qpos values.
        distance_to_front = stable_task_distance_to_front(x, cfg.step_front_x)
        task = jp.asarray([
            distance_to_front,
            (cfg.step_back_x - x) / 4.0,
            cfg.step_top_z / 0.30,
            cfg.target_forward_speed / 3.0,
            (z - local_terrain) / 0.50,
        ])
        support = self._imu_support_estimate(qpos, qvel, had_valid_landing).astype(jp.float32)
        phase_onehot = jp.asarray(phase_probs, dtype=jp.float32)
        wheel_v_est = self._sensor_vec(data, "rearwheel_joint_vel_sensor", 1)[0] * cfg.wheel_roll_radius
        contact_age_norm = jp.minimum(contact_age.astype(jp.float32), float(cfg.recovery_hold_steps)) / float(cfg.recovery_hold_steps)
        recovery_norm = jp.minimum(recovery_count.astype(jp.float32), float(cfg.recovery_hold_steps)) / float(cfg.recovery_hold_steps)
        frame = jp.concatenate([
            joint_pos, joint_vel, gyro, gravity, last_action, task,
            acc, jp.asarray([acc[2] - prev_acc_z, wheel_v_est, support]),
            phase_onehot, jp.asarray([contact_age_norm, recovery_norm]),
        ])
        if bool(cfg.obs_noise_enable):
            r1, r2, r3, r4, r5 = jax.random.split(rng, 5)
            noise = jp.concatenate([
                jax.random.normal(r1, (4,)) * cfg.joint_pos_noise_std,
                jax.random.normal(r2, (4,)) * cfg.joint_vel_noise_std,
                jax.random.normal(r3, (3,)) * cfg.gyro_noise_std,
                jax.random.normal(r4, (3,)) * cfg.gravity_noise_std,
                jp.zeros((9,), dtype=jp.float32),
                jax.random.normal(r5, (3,)) * cfg.accel_noise_std,
                jp.zeros((9,), dtype=jp.float32),
            ])
            frame = frame + noise
        return frame.astype(jp.float32)

    def _actor_frame(self, *args, **kwargs) -> jax.Array:
        """Deprecated compatibility alias for the canonical v4 producer."""
        return self.build_actor_current_frame_v4(*args, **kwargs)

    def _stack_actor_history(self, history: jax.Array, frame: jax.Array) -> jax.Array:
        return jp.concatenate([history, frame[None, :]], axis=0).reshape((-1,)).astype(jp.float32)

    def _advance_actor_history(self, history: jax.Array, frame: jax.Array) -> jax.Array:
        if self._actor_history_len == 0:
            return history
        return jp.concatenate([history[1:], frame[None, :]], axis=0)

    def _initial_actor_history(
        self, frame: jax.Array, saved_history: Optional[jax.Array], saved_valid: Optional[jax.Array]
    ) -> jax.Array:
        if self._actor_history_len == 0:
            return jp.zeros((0, self._actor_frame_dim), dtype=jp.float32)
        default = jp.broadcast_to(frame[None, :], (self._actor_history_len, self._actor_frame_dim))
        if saved_history is None or saved_valid is None:
            return default
        return jp.where(saved_valid, saved_history, default)

    def _privileged_obs(
        self,
        data: mjx.Data,
        phase: jax.Array,
        had_airborne: jax.Array,
        had_valid_landing: jax.Array,
        support: jax.Array,
        contact_age: jax.Array,
        recovery_count: jax.Array,
        last_action: jax.Array,
    ) -> jax.Array:
        """Training-only critic state; never routed to the Actor network."""
        feature = self._physical_feature(data)
        phase_onehot = jax.nn.one_hot(jp.clip(phase, 0, 3), 4, dtype=jp.float32)
        extras = jp.asarray([
            had_airborne.astype(jp.float32),
            had_valid_landing.astype(jp.float32),
            support.astype(jp.float32),
            jp.minimum(contact_age.astype(jp.float32), float(self._config.recovery_hold_steps)) / float(self._config.recovery_hold_steps),
            jp.minimum(recovery_count.astype(jp.float32), float(self._config.recovery_hold_steps)) / float(self._config.recovery_hold_steps),
        ])
        return jp.concatenate([feature, phase_onehot, extras, last_action]).astype(jp.float32)

    def _empty_metrics(self) -> Dict[str, jax.Array]:
        # Brax EvalWrapper injects a scalar metric named ``reward`` at reset.
        # Keep this key in the raw environment's static metric pytree as well;
        # otherwise the first wrapped env.step would remove it and JAX scan
        # would reject the changing dictionary structure.
        keys = (
            "reward", "success",
            "reward/total", "reward/roll", "reward/pitch", "reward/speed", "reward/recovery_shaping",
            "reward/recovery_quality", "reward/recovery_streak", "reward/recovery_event",
            "reward/alive", "reward/progress", "reward/platform_support", "reward/failure_penalty",
            "reward/hard_failure_penalty", "reward/timeout_penalty", "reward/instability_penalty",
            "reward/roll_excess_penalty", "reward/pitch_excess_penalty", "reward/gyro_excess_penalty",
            "reward/lateral_penalty", "reward/yaw_rate_penalty", "reward/yaw_angle_penalty",
            "reward/action_mag_penalty", "reward/action_saturation_penalty", "reward/downrange_penalty",
            "reward/time_penalty", "reward/action_smooth_penalty", "reward/chain_event",
            "reward/descent_local_shaping", "reward/descent_local_pitch_posture",
            "reward/descent_local_pitch_rate_penalty", "reward/descent_local_pitch_soft_penalty",
            "reward/descent_local_roll_score", "reward/descent_local_roll_penalty",
            "reward/descent_local_speed_score", "reward/descent_local_progress",
            "reward/descent_local_survival", "reward/descent_local_action_difference_penalty",
            "reward/descent_local_action_magnitude_penalty", "reward/descent_local_failure_penalty",
            "reward/descent_rsi_pilot_shaping", "reward/descent_rsi_pilot_survival",
            "reward/descent_rsi_pilot_roll", "reward/descent_rsi_pilot_pitch",
            "reward/descent_rsi_pilot_angular", "reward/descent_rsi_pilot_vz",
            "reward/descent_rsi_pilot_action_smooth_penalty",
            "reward/descent_rsi_pilot_action_magnitude_penalty",
            "reward/descent_rsi_pilot_failure_penalty",
            "diag/legacy_recovery_gate_fraction", "diag/recovery_quality_gate",
            "event/takeoff", "event/landing", "event/chain", "event/chain_ever", "event/recovery",
            "phase/approach", "phase/takeoff", "phase/flight", "phase/landing",
            "contact/front_top", "contact/rear_top", "contact/wheel_invalid", "contact/prohibited", "contact/imu_impact",
            "imu/acc_z", "imu/prev_acc_z", "imu/delta_z", "imu/abs_delta_z", "imu/threshold",
            "gate/airborne_history", "gate/in_platform_x", "gate/in_platform_y", "gate/in_platform",
            "gate/descending", "gate/imu_impulse", "gate/impact_phase_flight",
            "gate/impact_had_airborne", "gate/impact_not_landed", "gate/impact_base",
            "gate/support_height", "gate/support_vz",
            "gate/support", "gate/airborne", "gate/recovery_now", "gate/recovery_roll",
            "gate/recovery_pitch", "gate/recovery_gyro", "gate/recovery_vx", "gate/recovery_no_fault",
            "state/x", "state/y", "state/z", "state/vx", "state/vy", "state/vz", "state/yaw_deg",
            "state/expected_base_z", "state/support_height_error", "state/gyro_norm",
            "state/roll_excess_norm", "state/pitch_excess_norm", "state/gyro_excess_norm",
            "state/roll_deg", "state/pitch_deg", "state/phase_before", "state/phase",
            "state/estimated_phase", "state/tube_distance_z", "state/recovery_count", "state/contact_age",
            "state/pitch_signed_deg", "state/frontwheel_center_z", "state/rearwheel_center_z",
            "state/front_tire_bottom_z", "state/rear_tire_bottom_z", "state/front_tire_vz", "state/rear_tire_vz",
            "state/min_tire_bottom_z", "state/wheel_height_diff", "state/frontmost_x", "state/distance_to_obstacle",
            "state/knee_target", "state/knee_pos", "state/knee_vel", "state/knee_target_delta",
            "diag/dual_wheel_liftoff", "diag/wheelie_detected", "diag/positive_pitch_bad",
            "diag/knee_moved", "diag/base_z_success_disabled", "diag/cert_bank_safe_count", "diag/cert_bank_explicit",
            "diag/nonfinite_transition",
            "event/dual_wheel_liftoff",
            "reward/takeoff_total", "reward/takeoff_dual_wheel_vz", "reward/takeoff_dual_wheel_height",
            "reward/takeoff_min_tire_clearance", "reward/takeoff_knee_useful_motion",
            "reward/takeoff_positive_pitch_penalty", "reward/takeoff_wheelie_penalty",
            "reward/takeoff_wheel_unsync_penalty", "reward/takeoff_action_mag_penalty",
            "reward/stage_entry_total", "reward/stage_entry_shaping", "reward/stage_entry_event",
            "reward/stage_entry_handoff_quality", "reward/stage_entry_handoff_bonus",
            "reward/stage_entry_failure_penalty", "reward/stage_entry_progress",
            "event/stage_entry", "event/stage_entry_ever",
            "reset/episode/flight_curriculum", "reset/episode/canonical_entry_rehearsal",
            "reset/episode/landing_tube_rehearsal", "reset/episode/natural",
            "reset/transition/flight_curriculum", "reset/transition/canonical_entry_rehearsal",
            "reset/transition/landing_tube_rehearsal", "reset/transition/natural",
            "reset/episode/group/provisional_safe", "reset/episode/group/boundary",
            "reset/episode/group/successful_anchor", "reset/transition/group/provisional_safe",
            "reset/transition/group/boundary", "reset/transition/group/successful_anchor",
            "reset/episode/layer/late", "reset/episode/layer/middle", "reset/episode/layer/early",
            "reset/transition/layer/late", "reset/transition/layer/middle", "reset/transition/layer/early",
            "reset/episode/objective/takeoff_to_ascent",
            "reset/episode/objective/ascent_to_apex",
            "reset/episode/objective/apex_to_descent",
            "reset/episode/objective/descent_to_landing",
            "reset/episode/objective/landing_recovery",
            "reset/transition/objective/takeoff_to_ascent",
            "reset/transition/objective/ascent_to_apex",
            "reset/transition/objective/apex_to_descent",
            "reset/transition/objective/descent_to_landing",
            "reset/transition/objective/landing_recovery",
            "done/code",
            "end/recovery", "end/prohibited_contact", "end/invalid_wheel_step_contact",
            "end/roll_limit", "end/pitch_limit", "end/backward", "end/platform_back_edge_exit",
            "end/stage_timeout", "end/prelaunch_airborne",
            "end/takeoff_positive_pitch_failure", "end/takeoff_wheelie_failure",
            "end/takeoff_missed_liftoff_deadline", "end/takeoff_missed_wheel_clearance_deadline",
            "end/chain_entry",
            "end/next_stage_entry",
            "end/nonfinite",
        )
        dynamic = tuple(
            f"reset/{kind}/parent/p{index:03d}"
            for kind in ("episode", "transition") for index in range(len(self._reset_parent_ids))
        )
        return {k: jp.zeros((), dtype=jp.float32) for k in keys + dynamic}

    def _stage_timeout_steps(self) -> int:
        name = self._stage_name
        seconds = float(getattr(self._config, f"stage_timeout_{name}" if name != "full" else "stage_timeout_full"))
        return max(1, int(round(seconds / float(self._config.ctrl_dt))))

    def _natural_prob(self) -> float:
        return float(getattr(self._config, f"natural_prob_{self._stage_name}" if self._stage_name != "full" else "natural_prob_full"))

    # ------------------------------------------------------------------
    # Reset paths: natural, Tube-bank, explicit snapshot, and CoM seeds
    # ------------------------------------------------------------------
    def _initial_info(
        self, rng: jax.Array, qpos: jax.Array, last_action: jax.Array,
        phase: jax.Array, had_airborne: jax.Array, had_valid_landing: jax.Array,
        contact_age: jax.Array, accel_z: jax.Array, vz: jax.Array,
        estimated_phase: Optional[jax.Array] = None,
        phase_probs: Optional[jax.Array] = None,
        airborne_count: Optional[jax.Array] = None,
        prelaunch_airborne_count: Optional[jax.Array] = None,
        landing_bounce_count: Optional[jax.Array] = None,
        invalid_wheel_count: Optional[jax.Array] = None,
        recovery_count: Optional[jax.Array] = None,
        prev_acc_z: Optional[jax.Array] = None,
        prev_vz: Optional[jax.Array] = None,
        prev_front_tire_bottom_z: Optional[jax.Array] = None,
        prev_rear_tire_bottom_z: Optional[jax.Array] = None,
        positive_pitch_count: Optional[jax.Array] = None,
        wheelie_count: Optional[jax.Array] = None,
        dual_wheel_liftoff_seen: Optional[jax.Array] = None,
        stage_entry_ever: Optional[jax.Array] = None,
        apex_seen: Optional[jax.Array] = None,
        jump_signal_latched: Optional[jax.Array] = None,
        jump_window_start_x: Optional[jax.Array] = None,
        jump_window_end_x: Optional[jax.Array] = None,
        obs_history: Optional[jax.Array] = None,
        reset_source: Optional[jax.Array] = None,
        bootstrap_group: Optional[jax.Array] = None,
        descent_layer: Optional[jax.Array] = None,
        reset_parent: Optional[jax.Array] = None,
        reachability_objective_id: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        """Initialise physical state plus the IMU phase-filter state.

        The optional values are restored from a DVGC snapshot when available.
        Older banks omit them and therefore fall back to the freshly-forwarded
        sensor state, avoiding an artificial derivative impulse at reset.
        """
        def _i(value: Optional[jax.Array]) -> jax.Array:
            return jp.zeros((), jp.int32) if value is None else jp.asarray(value, jp.int32)

        def _f_or_current(value: Optional[jax.Array], current: jax.Array) -> jax.Array:
            if value is None:
                return current
            value = jp.asarray(value, jp.float32)
            return jp.where(jp.isfinite(value), value, current)

        curriculum=float(self._config.stage_curriculum_scale)
        jump_window_length=float(self._config.stage_jump_window_length_easy)+(float(self._config.stage_jump_window_length_hard)-float(self._config.stage_jump_window_length_easy))*curriculum

        return {
            "rng": rng,
            "phase": phase.astype(jp.int32),
            "estimated_phase": (phase if estimated_phase is None else estimated_phase).astype(jp.int32),
            "phase_probs": (
                jax.nn.one_hot(jp.clip(phase if estimated_phase is None else estimated_phase, 0, 3), 4, dtype=jp.float32)
                if phase_probs is None else jp.asarray(phase_probs, jp.float32)
            ),
            "had_airborne": had_airborne.astype(jp.int32),
            "had_valid_landing": had_valid_landing.astype(jp.int32),
            "airborne_count": _i(airborne_count),
            "prelaunch_airborne_count": _i(prelaunch_airborne_count),
            "landing_bounce_count": _i(landing_bounce_count),
            "invalid_wheel_count": _i(invalid_wheel_count),
            "recovery_count": _i(recovery_count),
            "recovery_success": jp.zeros((), jp.int32),
            "stage_entry_ever": _i(stage_entry_ever),
            "apex_seen": _i(apex_seen),
            "jump_signal_latched": ((had_airborne>0) | (phase==STAGE_ID["takeoff"])) if jump_signal_latched is None else jp.asarray(jump_signal_latched,bool),
            "jump_window_start_x": qpos[self._qpos0+0] if jump_window_start_x is None else jp.asarray(jump_window_start_x,jp.float32),
            "jump_window_end_x": qpos[self._qpos0+0]+jump_window_length if jump_window_end_x is None else jp.asarray(jump_window_end_x,jp.float32),
            "chain_success": jp.zeros((), jp.int32),
            "chain_ever": jp.zeros((), jp.int32),
            "landing_entry_age": jp.where(had_valid_landing>0,jp.maximum(contact_age,1),0).astype(jp.int32),
            "landing_phase_step": jp.zeros((), jp.int32),
            "reset_source": jp.asarray(
                RESET_SOURCE["natural"] if reset_source is None else reset_source, jp.int32
            ),
            "bootstrap_group": jp.asarray(-1 if bootstrap_group is None else bootstrap_group, jp.int32),
            "descent_layer": jp.asarray(-1 if descent_layer is None else descent_layer, jp.int32),
            "reset_parent": jp.asarray(-1 if reset_parent is None else reset_parent, jp.int32),
            "reachability_objective_id": jp.asarray(
                self._default_reachability_objective_id
                if reachability_objective_id is None else reachability_objective_id,
                jp.int32,
            ),
            "descent_local_episode": (
                jp.asarray(bool(self._config.descent_local_reward_enable))
                & ((jp.asarray(-1 if bootstrap_group is None else bootstrap_group, jp.int32) >= 0)
                   | (phase == STAGE_ID["flight"]))
            ),
            "terminated": jp.zeros((), jp.int32),
            "truncated": jp.zeros((), jp.int32),
            "episode_step": jp.zeros((), jp.int32),
            "start_x": qpos[self._qpos0 + 0],
            "last_action": last_action,
            "prev_acc_z": _f_or_current(prev_acc_z, accel_z),
            "prev_vz": _f_or_current(prev_vz, vz),
            "prev_front_tire_bottom_z": _f_or_current(prev_front_tire_bottom_z, jp.asarray(0.0, jp.float32)),
            "prev_rear_tire_bottom_z": _f_or_current(prev_rear_tire_bottom_z, jp.asarray(0.0, jp.float32)),
            "positive_pitch_count": _i(positive_pitch_count),
            "wheelie_count": _i(wheelie_count),
            "dual_wheel_liftoff_seen": _i(dual_wheel_liftoff_seen).astype(bool),
            "obs_history": (
                jp.zeros((self._actor_history_len, self._actor_frame_dim), dtype=jp.float32)
                if obs_history is None else jp.asarray(obs_history, dtype=jp.float32)
            ),
            "contact_age": contact_age.astype(jp.int32),
            "end_code": jp.asarray(END_NONE, jp.int32),
        }

    def _state_from_values(
        self, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array,
        rng: jax.Array, phase: jax.Array, had_airborne: jax.Array,
        had_valid_landing: jax.Array, contact_age: jax.Array,
        last_action: Optional[jax.Array] = None,
        estimated_phase: Optional[jax.Array] = None,
        phase_probs: Optional[jax.Array] = None,
        airborne_count: Optional[jax.Array] = None,
        prelaunch_airborne_count: Optional[jax.Array] = None,
        landing_bounce_count: Optional[jax.Array] = None,
        invalid_wheel_count: Optional[jax.Array] = None,
        recovery_count: Optional[jax.Array] = None,
        prev_acc_z: Optional[jax.Array] = None,
        prev_vz: Optional[jax.Array] = None,
        obs_history: Optional[jax.Array] = None,
        obs_history_valid: Optional[jax.Array] = None,
        continuation_obs_history: Optional[jax.Array] = None,
        actor_packet_fifo: Optional[jax.Array] = None,
        actor_packet_fifo_valid: Optional[jax.Array] = None,
        actor_observation: Optional[jax.Array] = None,
        actor_observation_valid: Optional[jax.Array] = None,
        observation_rng: Optional[jax.Array] = None,
        data_act: Optional[jax.Array] = None,
        data_sensordata: Optional[jax.Array] = None,
        data_time: Optional[jax.Array] = None,
        qacc_warmstart: Optional[jax.Array] = None,
        reset_source: Optional[jax.Array] = None,
        bootstrap_group: Optional[jax.Array] = None,
        descent_layer: Optional[jax.Array] = None,
        reset_parent: Optional[jax.Array] = None,
        reachability_objective_id: Optional[jax.Array] = None,
        stage_entry_ever: Optional[jax.Array] = None,
        apex_seen: Optional[jax.Array] = None,
        jump_signal_latched: Optional[jax.Array] = None,
        jump_window_start_x: Optional[jax.Array] = None,
        jump_window_end_x: Optional[jax.Array] = None,
    ) -> mjx_env.State:
        data = self._make_data(qpos, qvel, ctrl)
        data_replacements = {}
        if qacc_warmstart is not None:
            data_replacements["qacc_warmstart"] = jp.asarray(qacc_warmstart, jp.float32)
        if data_act is not None:
            data_replacements["act"] = jp.asarray(data_act, jp.float32)
        if data_sensordata is not None:
            data_replacements["sensordata"] = jp.asarray(data_sensordata, jp.float32)
        if data_time is not None:
            data_replacements["time"] = jp.asarray(data_time, jp.float32)
        if data_replacements:
            data = data.replace(**data_replacements)
        last_action = self._neutral_action if last_action is None else last_action
        accel_z = self._sensor_vec(data, "acc_local", 3)[2]
        qpos_root, qvel_root, _, _, _ = self._root_state(data)
        _, _, init_front_tire_z, init_rear_tire_z = wheel_tire_bottoms(
            data,
            frontwheel_body_id=self._frontwheel_body_id,
            rearwheel_body_id=self._rearwheel_body_id,
            wheel_radius=float(self._config.takeoff_wheel_radius),
            fallback_base_z=qpos_root[2],
        )
        r_obs, r_info = jax.random.split(rng)
        if observation_rng is not None:
            r_obs = jp.asarray(observation_rng)
        prev_acc_eff = accel_z if prev_acc_z is None else jp.where(jp.isfinite(jp.asarray(prev_acc_z)), jp.asarray(prev_acc_z), accel_z)
        rec_count = jp.zeros((), jp.int32) if recovery_count is None else jp.asarray(recovery_count, jp.int32)
        estimated_phase_eff = phase if estimated_phase is None else jp.asarray(estimated_phase, jp.int32)
        phase_probs_eff = (
            jax.nn.one_hot(jp.clip(estimated_phase_eff, 0, 3), 4, dtype=jp.float32)
            if phase_probs is None else jp.asarray(phase_probs, jp.float32)
        )
        frame = self.build_actor_current_frame_v4(
            data, last_action, phase_probs_eff, had_valid_landing, contact_age, rec_count, prev_acc_eff, r_obs
        )
        history = self._initial_actor_history(frame, obs_history, obs_history_valid)
        info = self._initial_info(
            r_info, qpos, last_action, phase, had_airborne, had_valid_landing,
            contact_age, accel_z, qvel_root[2],
            estimated_phase=estimated_phase_eff, phase_probs=phase_probs_eff,
            airborne_count=airborne_count,
            prelaunch_airborne_count=prelaunch_airborne_count,
            landing_bounce_count=landing_bounce_count,
            invalid_wheel_count=invalid_wheel_count,
            recovery_count=recovery_count,
            prev_acc_z=prev_acc_z,
            prev_vz=prev_vz,
            prev_front_tire_bottom_z=init_front_tire_z,
            prev_rear_tire_bottom_z=init_rear_tire_z,
            obs_history=history,
            reset_source=reset_source,
            bootstrap_group=bootstrap_group,
            descent_layer=descent_layer,
            reset_parent=reset_parent,
            reachability_objective_id=reachability_objective_id,
            stage_entry_ever=stage_entry_ever,
            apex_seen=apex_seen,
            jump_signal_latched=jump_signal_latched,
            jump_window_start_x=jump_window_start_x,
            jump_window_end_x=jump_window_end_x,
        )
        support = self._imu_support_estimate(qpos, qvel_root, had_valid_landing)
        privileged_obs = self._privileged_obs(
            data, phase, had_airborne, had_valid_landing, support, contact_age, rec_count, last_action
        )
        rebuilt_obs = self._stack_actor_history(history, frame)
        obs = rebuilt_obs
        if actor_observation is not None and actor_observation_valid is not None:
            obs = jp.where(actor_observation_valid, jp.asarray(actor_observation, jp.float32), obs)
        # The online actor input and the continuation history are distinct
        # timing objects.  Legacy records may provide a logged actor tensor and
        # already-advanced history; v4 records provide both explicitly.
        frames = obs.reshape((self._actor_history_steps, self._actor_frame_dim))
        pre_history = frames[:-1]
        current_frame = frames[-1]
        post_history = self._advance_actor_history(pre_history, current_frame)
        continuation = post_history if continuation_obs_history is None else jp.asarray(continuation_obs_history, jp.float32)
        info["obs_history"] = continuation
        empty_fifo = jp.zeros((3, self._actor_obs_dim), jp.float32)
        initial_fifo = empty_fifo.at[2].set(obs)
        fifo = initial_fifo if actor_packet_fifo is None else jp.asarray(actor_packet_fifo, jp.float32)
        fifo_valid = jp.asarray(1 if actor_packet_fifo_valid is None else actor_packet_fifo_valid, jp.int32)
        info.update({
            "actor_obs_history_pre": pre_history,
            "actor_current_frame": current_frame,
            "actor_obs_history_post": post_history,
            "actor_packet_fifo": fifo,
            "actor_packet_fifo_valid": fifo_valid,
            "actor_observation_rng": r_obs,
            "actor_frame_prev_acc_z": prev_acc_eff,
        })
        return mjx_env.State(
            data, {"state": obs, "privileged_state": privileged_obs},
            jp.zeros(()), jp.zeros(()), self._empty_metrics(), info
        )

    def _natural_values(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        cfg = self._config
        rng, rr, rp, ry, r_y, rv = jax.random.split(rng, 6)
        qpos = self._init_qpos
        qvel = self._init_qvel
        roll = jax.random.uniform(rr, (), minval=-_deg2rad(cfg.init_roll_deg), maxval=_deg2rad(cfg.init_roll_deg)) if cfg.domain_randomization else jp.zeros(())
        pitch = jax.random.uniform(rp, (), minval=-_deg2rad(cfg.init_pitch_deg), maxval=_deg2rad(cfg.init_pitch_deg)) if cfg.domain_randomization else jp.zeros(())
        yaw = jax.random.uniform(ry, (), minval=-_deg2rad(cfg.init_yaw_deg), maxval=_deg2rad(cfg.init_yaw_deg)) if cfg.domain_randomization else jp.zeros(())
        y = jax.random.normal(r_y) * cfg.init_pos_y_std if cfg.domain_randomization else jp.zeros(())
        vx_noise = jax.random.normal(rv) * cfg.init_vel_x_std if cfg.domain_randomization else jp.zeros(())
        qpos = qpos.at[self._qpos0 + 0].set(cfg.ground_spawn_x)
        qpos = qpos.at[self._qpos0 + 1].set(y)
        qpos = qpos.at[self._qpos0 + 2].set(cfg.nominal_base_z_ground)
        qpos = qpos.at[self._qpos0 + 3:self._qpos0 + 7].set(_quat_from_euler_xyz(roll, pitch, yaw))
        qpos = qpos.at[self._joint_qpos["hip_joint"]].set(cfg.hip_initial)
        qpos = qpos.at[self._joint_qpos["knee_joint"]].set(cfg.knee_initial)
        vx = cfg.initial_velocity + vx_noise
        qvel = qvel.at[self._qvel0 + 0].set(vx)
        qvel = qvel.at[self._joint_qvel["rearwheel_joint"]].set(vx / cfg.wheel_roll_radius)
        qvel = qvel.at[self._joint_qvel["frontwheel_joint"]].set(vx / cfg.wheel_roll_radius)
        ctrl = self._action_to_ctrl(self._neutral_action, qpos[self._joint_qpos["knee_joint"]])
        return rng, qpos, qvel, ctrl

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, choose_rng, bank_rng = jax.random.split(rng, 3)
        rng, qnat, vnat, cnat = self._natural_values(rng)
        if self._bank_count > 0 and bool(self._config.use_bank_resets) and self._stage_name != "full":
            idx = jax.random.categorical(bank_rng, self._bank_logw)
            use_bank = jax.random.uniform(choose_rng) >= float(self._natural_prob())
            qpos = jp.where(use_bank, self._bank_qpos[idx], qnat)
            qvel = jp.where(use_bank, self._bank_qvel[idx], vnat)
            ctrl = jp.where(use_bank, self._bank_ctrl[idx], cnat)
            qacc_warmstart = jp.where(use_bank, self._bank_qacc_warmstart[idx], jp.zeros_like(vnat))
            phase = jp.where(use_bank, self._bank_phase[idx], jp.asarray(STAGE_ID["approach"], jp.int32))
            estimated_phase = jp.where(use_bank, self._bank_estimated_phase[idx], jp.asarray(STAGE_ID["approach"], jp.int32))
            phase_probs = jp.where(
                use_bank,
                self._bank_phase_probs[idx],
                jax.nn.one_hot(STAGE_ID["approach"], 4, dtype=jp.float32),
            )
            airborne = jp.where(use_bank, self._bank_airborne[idx], jp.asarray(0, jp.int32))
            landed = jp.where(use_bank, self._bank_landed[idx], jp.asarray(0, jp.int32))
            age = jp.where(use_bank, self._bank_age[idx], jp.asarray(0, jp.int32))
            recovery_count = jp.where(use_bank, self._bank_recovery_count[idx], jp.asarray(0, jp.int32))
            bounce_count = jp.where(use_bank, self._bank_bounce_count[idx], jp.asarray(0, jp.int32))
            airborne_count = jp.where(use_bank, self._bank_airborne_count[idx], jp.asarray(0, jp.int32))
            prelaunch_count = jp.where(use_bank, self._bank_prelaunch_airborne_count[idx], jp.asarray(0, jp.int32))
            invalid_count = jp.where(use_bank, self._bank_invalid_wheel_count[idx], jp.asarray(0, jp.int32))
            prev_acc_z = jp.where(use_bank, self._bank_prev_acc_z[idx], jp.asarray(jp.nan, jp.float32))
            prev_vz = jp.where(use_bank, self._bank_prev_vz[idx], jp.asarray(jp.nan, jp.float32))
            apex_seen = jp.where(use_bank, self._bank_apex_seen[idx], jp.asarray(0, jp.int32))
            last_action = jp.where(use_bank, self._bank_last_action[idx], self._neutral_action)
            obs_history = self._bank_obs_history[idx]
            obs_history_valid = use_bank & self._bank_obs_history_valid[idx]
            actor_observation = self._bank_actor_observation[idx]
            actor_observation_valid = use_bank & self._bank_actor_observation_valid[idx]
            reset_source = jp.where(
                use_bank, self._bank_reset_source[idx], jp.asarray(RESET_SOURCE["natural"], jp.int32)
            )
            bootstrap_group = jp.where(use_bank, self._bank_bootstrap_group[idx], jp.asarray(-1, jp.int32))
            descent_layer = jp.where(use_bank, self._bank_descent_layer[idx], jp.asarray(-1, jp.int32))
            reset_parent = jp.where(use_bank, self._bank_reset_parent[idx], jp.asarray(-1, jp.int32))
            reachability_objective_id = jp.where(
                use_bank,
                self._bank_reachability_objective[idx],
                jp.asarray(self._default_reachability_objective_id, jp.int32),
            )
            return self._state_from_values(
                qpos, qvel, ctrl, rng, phase, airborne, landed, age, last_action,
                estimated_phase=estimated_phase, phase_probs=phase_probs,
                airborne_count=airborne_count, prelaunch_airborne_count=prelaunch_count,
                landing_bounce_count=bounce_count, invalid_wheel_count=invalid_count,
                recovery_count=recovery_count, prev_acc_z=prev_acc_z, prev_vz=prev_vz,
                qacc_warmstart=qacc_warmstart,
                obs_history=obs_history, obs_history_valid=obs_history_valid,
                actor_observation=actor_observation,
                actor_observation_valid=actor_observation_valid,
                reset_source=reset_source,
                bootstrap_group=bootstrap_group,
                descent_layer=descent_layer,
                reset_parent=reset_parent,
                reachability_objective_id=reachability_objective_id,
                apex_seen=apex_seen,
            )
        return self._state_from_values(qnat, vnat, cnat, rng, jp.asarray(STAGE_ID["approach"], jp.int32), jp.zeros((), jp.int32), jp.zeros((), jp.int32), jp.zeros((), jp.int32))

    def reset_from_snapshot(
        self, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array, rng: jax.Array,
        phase: jax.Array, had_airborne: jax.Array, had_valid_landing: jax.Array,
        contact_age: jax.Array, last_action: Optional[jax.Array] = None,
        *,
        estimated_phase: Optional[jax.Array] = None,
        phase_probs: Optional[jax.Array] = None,
        airborne_count: Optional[jax.Array] = None,
        prelaunch_airborne_count: Optional[jax.Array] = None,
        landing_bounce_count: Optional[jax.Array] = None,
        invalid_wheel_count: Optional[jax.Array] = None,
        recovery_count: Optional[jax.Array] = None,
        prev_acc_z: Optional[jax.Array] = None,
        prev_vz: Optional[jax.Array] = None,
        obs_history: Optional[jax.Array] = None,
        obs_history_valid: Optional[jax.Array] = None,
        continuation_obs_history: Optional[jax.Array] = None,
        actor_packet_fifo: Optional[jax.Array] = None,
        actor_packet_fifo_valid: Optional[jax.Array] = None,
        actor_observation: Optional[jax.Array] = None,
        actor_observation_valid: Optional[jax.Array] = None,
        observation_rng: Optional[jax.Array] = None,
        data_act: Optional[jax.Array] = None,
        data_sensordata: Optional[jax.Array] = None,
        data_time: Optional[jax.Array] = None,
        qacc_warmstart: Optional[jax.Array] = None,
        reset_source: Optional[jax.Array] = None,
        bootstrap_group: Optional[jax.Array] = None,
        descent_layer: Optional[jax.Array] = None,
        reset_parent: Optional[jax.Array] = None,
        stage_entry_ever: Optional[jax.Array] = None,
        apex_seen: Optional[jax.Array] = None,
        jump_signal_latched: Optional[jax.Array] = None,
        jump_window_start_x: Optional[jax.Array] = None,
        jump_window_end_x: Optional[jax.Array] = None,
    ) -> mjx_env.State:
        """Restore physical and IMU phase-filter state for branch certification.

        All keyword fields are optional for compatibility with existing banks.
        New snapshots persist them so a branch rollout starts from the same
        landing/recovery-filter history that produced the candidate.
        """
        return self._state_from_values(
            qpos, qvel, ctrl, rng, phase, had_airborne, had_valid_landing, contact_age, last_action,
            estimated_phase=estimated_phase, phase_probs=phase_probs,
            airborne_count=airborne_count, prelaunch_airborne_count=prelaunch_airborne_count,
            landing_bounce_count=landing_bounce_count, invalid_wheel_count=invalid_wheel_count,
            recovery_count=recovery_count, prev_acc_z=prev_acc_z, prev_vz=prev_vz,
            obs_history=obs_history, obs_history_valid=obs_history_valid,
            continuation_obs_history=continuation_obs_history,
            actor_packet_fifo=actor_packet_fifo,
            actor_packet_fifo_valid=actor_packet_fifo_valid,
            actor_observation=actor_observation,
            actor_observation_valid=actor_observation_valid,
            observation_rng=observation_rng,
            data_act=data_act,
            data_sensordata=data_sensordata,
            data_time=data_time,
            qacc_warmstart=qacc_warmstart,
            reset_source=reset_source,
            bootstrap_group=bootstrap_group,
            descent_layer=descent_layer,
            reset_parent=reset_parent,
            stage_entry_ever=stage_entry_ever,
            apex_seen=apex_seen,
            jump_signal_latched=jump_signal_latched,
            jump_window_start_x=jump_window_start_x,
            jump_window_end_x=jump_window_end_x,
        )

    def reset_from_com_seed(self, seed: Dict[str, Any], rng: jax.Array) -> mjx_env.State:
        """Host-facing CoM proposal -> full MJX state conversion.

        The provisional posture is forwarded first.  Its actual mass-weighted
        root-body subtree CoM offset is then used to translate the free base,
        preserving the V21 rule that ballistic proposal is expressed in system
        CoM coordinates rather than blindly treating base position as CoM.
        """
        cfg = self._config
        qpos = self._init_qpos
        qvel = self._init_qvel
        e = jp.asarray(seed["euler"], dtype=jp.float32)
        qpos = qpos.at[self._qpos0 + 3:self._qpos0 + 7].set(_quat_from_euler_xyz(e[0], e[1], e[2]))
        qpos = qpos.at[self._joint_qpos["steering_joint"]].set(float(seed["steer"]))
        qpos = qpos.at[self._joint_qpos["hip_joint"]].set(float(seed["hip"]))
        qpos = qpos.at[self._joint_qpos["knee_joint"]].set(float(seed["knee"]))
        if seed["seed_type"] == "system_com":
            provisional = self._make_data(qpos, qvel, self._action_to_ctrl(self._neutral_action, qpos[self._joint_qpos["knee_joint"]]))
            base = qpos[self._qpos0:self._qpos0 + 3]
            com = provisional.subtree_com[self._root_body_id]
            desired = jp.asarray(seed["desired_com"], dtype=jp.float32)
            qpos = qpos.at[self._qpos0:self._qpos0 + 3].set(desired - (com - base))
            phase = jp.asarray(STAGE_ID["flight"], jp.int32)
            had_airborne = jp.ones((), jp.int32)
        else:
            qpos = qpos.at[self._qpos0:self._qpos0 + 3].set(jp.asarray(seed["base_pos"], dtype=jp.float32))
            phase = jp.asarray(STAGE_ID["takeoff"], jp.int32)
            had_airborne = jp.zeros((), jp.int32)
        lin = jp.asarray(seed["linear_velocity"], dtype=jp.float32)
        ang = jp.asarray(seed["angular_velocity"], dtype=jp.float32)
        qvel = qvel.at[self._qvel0:self._qvel0 + 3].set(lin)
        qvel = qvel.at[self._qvel0 + 3:self._qvel0 + 6].set(ang)
        qvel = qvel.at[self._joint_qvel["rearwheel_joint"]].set(lin[0] / cfg.wheel_roll_radius)
        qvel = qvel.at[self._joint_qvel["frontwheel_joint"]].set(lin[0] / cfg.wheel_roll_radius)
        if "hip_velocity" in seed:
            qvel = qvel.at[self._joint_qvel["hip_joint"]].set(float(seed["hip_velocity"]))
        if "knee_velocity" in seed:
            qvel = qvel.at[self._joint_qvel["knee_joint"]].set(float(seed["knee_velocity"]))
        ctrl = self._action_to_ctrl(self._neutral_action, qpos[self._joint_qpos["knee_joint"]])
        return self._state_from_values(qpos, qvel, ctrl, rng, phase, had_airborne, jp.zeros((), jp.int32), jp.zeros((), jp.int32))

    def snapshot_record(self, state: mjx_env.State, source_stage: str) -> Dict[str, Any]:
        """Serialize a deprecated v3-compatible record.

        Authority-sensitive callers must use :meth:`snapshot_record_v4`,
        which requires the current policy action and provenance explicitly.
        """
        data = jax.device_get(state.data)
        info = jax.device_get(state.info)
        feature = np.asarray(jax.device_get(self._physical_feature(state.data)), np.float32)
        estimated_phase = int(info["estimated_phase"])
        phase_probs = np.asarray(info["phase_probs"], np.float32)
        # Phase progress is event-anchored and intentionally coarse; it is a
        # policy/viability feature, not a trajectory-tracking target.
        x = float(feature[0])
        if estimated_phase == STAGE_ID["approach"]:
            progress = (x - float(self._config.ground_spawn_x)) / max(float(self._config.step_front_x - self._config.takeoff_window_far - self._config.ground_spawn_x), 1e-6)
        elif estimated_phase == STAGE_ID["takeoff"]:
            progress = (x - float(self._config.step_front_x - self._config.takeoff_window_far)) / max(float(self._config.takeoff_window_far - self._config.takeoff_window_near), 1e-6)
        elif estimated_phase == STAGE_ID["flight"]:
            progress = (x - float(self._config.step_front_x - self._config.takeoff_window_near)) / max(float(self._config.valid_landing_min_past_edge + self._config.takeoff_window_near), 1e-6)
        else:
            progress = float(info["recovery_count"]) / max(float(self._config.recovery_hold_steps), 1.0)
        return {
            "schema_name": "dvgc_physical_policy_state_v3_warmstart",
            "schema_version": 3,
            "qpos": np.asarray(data.qpos, np.float32),
            "qvel": np.asarray(data.qvel, np.float32),
            "ctrl": np.asarray(data.ctrl, np.float32),
            "qacc_warmstart": np.asarray(data.qacc_warmstart, np.float32),
            "physical_feature": feature,
            "source_phase": str(source_stage),
            "oracle_phase": int(info["phase"]),
            "had_airborne": int(info["had_airborne"]),
            "had_valid_landing": int(info["had_valid_landing"]),
            "contact_age": int(info["contact_age"]),
            "landing_entry_age": int(info["landing_entry_age"]),
            "airborne_count": int(info["airborne_count"]),
            "prelaunch_airborne_count": int(info["prelaunch_airborne_count"]),
            "landing_bounce_count": int(info["landing_bounce_count"]),
            "invalid_wheel_count": int(info["invalid_wheel_count"]),
            "recovery_count": int(info["recovery_count"]),
            "stage_entry_ever": int(info["stage_entry_ever"]),
            "apex_seen": int(info["apex_seen"]),
            "jump_signal_latched": bool(info["jump_signal_latched"]),
            "jump_window_start_x": float(info["jump_window_start_x"]),
            "jump_window_end_x": float(info["jump_window_end_x"]),
            "policy_state": {
                "last_action": np.asarray(info["last_action"], np.float32),
                "obs_history": np.asarray(info["obs_history"], np.float32),
                "actor_observation": np.asarray(jax.device_get(state.obs["state"]), np.float32),
                "filter_phase": estimated_phase,
                "phase_probs": phase_probs,
                "contact_probs": np.asarray([1.0 - float(info["had_airborne"]), float(info["had_valid_landing"])], np.float32),
                "phase_progress": float(np.clip(progress, 0.0, 1.0)),
                "phase_confidence": float(np.max(phase_probs)),
                "estimator_hidden": np.zeros((0,), np.float32),
                "delay_buffer": phase_probs[None, :],
                "prev_acc_z": float(info["prev_acc_z"]),
                "prev_vz": float(info["prev_vz"]),
            },
        }

    def snapshot_record_v4(
        self, state: mjx_env.State, source_stage: str,
        policy_action: jax.Array, provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture the timing-explicit policy-evaluation boundary at tick t."""
        data = jax.device_get(state.data)
        info = jax.device_get(state.info)
        if int(info["actor_packet_fifo_valid"]) != 3:
            raise ValueError("v4 capture requires three real actor packets; synthetic padding is forbidden")
        required_hashes = {
            "xml_sha256", "config_sha256", "action_mapping_version",
            "policy_params_sha256", "policy_config_sha256",
            "policy_manifest_sha256", "normalizer_sha256",
            "source_fingerprint",
        }
        if not required_hashes.issubset(provenance):
            raise ValueError(f"incomplete v4 provenance: {sorted(required_hashes-set(provenance))}")
        action = np.asarray(jax.device_get(jp.clip(policy_action, -1.0, 1.0)), np.float32)
        ctrl_applied = np.asarray(jax.device_get(self._action_to_ctrl(
            jp.asarray(action), state.data.qpos[self._joint_qpos["knee_joint"]]
        )), np.float32)
        legacy = self.snapshot_record(state, source_stage)
        estimator_post = {
            name: np.asarray(info[name]).copy() if np.asarray(info[name]).ndim else np.asarray(info[name]).item()
            for name in (
                "phase", "estimated_phase", "phase_probs", "had_airborne",
                "had_valid_landing", "airborne_count", "prelaunch_airborne_count",
                "landing_bounce_count", "invalid_wheel_count", "recovery_count",
                "contact_age", "landing_entry_age", "landing_phase_step",
                "prev_acc_z", "prev_vz", "prev_front_tire_bottom_z",
                "prev_rear_tire_bottom_z", "positive_pitch_count", "wheelie_count",
                "dual_wheel_liftoff_seen", "stage_entry_ever", "apex_seen",
                "jump_signal_latched", "jump_window_start_x", "jump_window_end_x",
                "chain_ever", "recovery_success", "episode_step", "end_code",
            )
        }
        estimator_pre = dict(estimator_post)
        estimator_pre["prev_acc_z"] = float(info["actor_frame_prev_acc_z"])
        tick = int(info["episode_step"])
        simulation_time = float(np.asarray(data.time))
        record = {
            **legacy,
            "schema_name": SNAPSHOT_SCHEMA,
            "schema_version": 4,
            "physical_state_t": {
                "qpos": np.asarray(data.qpos, np.float32),
                "qvel": np.asarray(data.qvel, np.float32),
                "act": np.asarray(data.act, np.float32),
                "ctrl_previous": np.asarray(data.ctrl, np.float32),
                "qacc_warmstart": np.asarray(data.qacc_warmstart, np.float32),
                "sensordata": np.asarray(data.sensordata, np.float32),
                "time": np.asarray(data.time, np.float32),
            },
            "obs_history_pre_t": np.asarray(info["actor_obs_history_pre"], np.float32),
            "current_frame_t": np.asarray(info["actor_current_frame"], np.float32),
            "actor_observation_t": np.asarray(jax.device_get(state.obs["state"]), np.float32),
            "obs_history_post_t": np.asarray(info["actor_obs_history_post"], np.float32),
            "actor_packet_fifo_t": np.asarray(info["actor_packet_fifo"], np.float32),
            "actor_packet_fifo_valid": int(info["actor_packet_fifo_valid"]),
            "estimator_state_pre_t": estimator_pre,
            "estimator_state_post_t": estimator_post,
            "last_normalized_command_t": np.asarray(info["last_action"], np.float32),
            "policy_action_t": action,
            "ctrl_applied_t": ctrl_applied,
            "rng_state_t": np.asarray(info["rng"]),
            "observation_rng_t": np.asarray(info["actor_observation_rng"]),
            "field_ticks": {
                "physical_state_t": tick,
                "actor_observation_t": tick,
                "current_frame_t": tick,
                "policy_action_t": tick,
                "ctrl_applied_t": tick,
                "ctrl_previous": tick - 1,
                "actor_packet_fifo_t": [tick - 2, tick - 1, tick],
                "estimator_state_pre_t": tick,
                "estimator_state_post_t": tick,
            },
            "simulation_timestamps": {
                "physical_state_t": simulation_time,
                "actor_observation_t": simulation_time,
                "policy_action_t": simulation_time,
                "ctrl_applied_t": simulation_time,
            },
            "provenance": dict(provenance),
        }
        # Keep the compatibility policy state aligned with the explicit v4
        # semantics instead of exposing post-history under the old name.
        record["policy_state"]["obs_history"] = record["obs_history_pre_t"]
        record["policy_state"]["actor_observation"] = record["actor_observation_t"]
        return record

    # ------------------------------------------------------------------
    # Transition, reward, and termination
    # ------------------------------------------------------------------
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        action = jp.clip(action, -1.0, 1.0)
        rng, obs_rng = jax.random.split(state.info["rng"])
        knee_position0 = state.data.qpos[self._joint_qpos["knee_joint"]]
        ctrl = self._action_to_ctrl(action, knee_position0)
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
        nonfinite = ~(
            jp.all(jp.isfinite(action))
            & jp.all(jp.isfinite(data.qpos))
            & jp.all(jp.isfinite(data.qvel))
            & jp.all(jp.isfinite(data.qacc_warmstart))
        )
        # A rare divergent solver transition must terminate as explicit
        # physical evidence without leaking NaNs into PPO observations or
        # aggregate metrics.  Preserve the last finite simulator state only
        # for this terminal transition; normal dynamics remain untouched.
        data = data.replace(
            qpos=jp.where(nonfinite, state.data.qpos, data.qpos),
            qvel=jp.where(nonfinite, state.data.qvel, data.qvel),
            ctrl=jp.where(nonfinite, state.data.ctrl, data.ctrl),
            qacc_warmstart=jp.where(nonfinite, state.data.qacc_warmstart, data.qacc_warmstart),
        )
        action = jp.where(nonfinite, state.info["last_action"], action)
        qpos, qvel, roll, pitch, yaw = self._root_state(data)
        x, y, z = qpos[0], qpos[1], qpos[2]
        vx, vy, vz = qvel[0], qvel[1], qvel[2]
        gyro = self._sensor_vec(data, "gyro_local", 3)
        acc_z = self._sensor_vec(data, "acc_local", 3)[2]
        contact = self._contact_masks(data)
        phase0 = state.info["phase"]
        _, knee_target, _ = self._knee_action_to_target(action[3], knee_position0)
        knee_pos = data.qpos[self._joint_qpos["knee_joint"]]
        knee_vel = data.qvel[self._joint_qvel["knee_joint"]]
        takeoff_signals = compute_takeoff_signals(
            cfg=cfg,
            data=data,
            qpos=qpos,
            qvel=qvel,
            roll=roll,
            pitch=pitch,
            action=action,
            frontwheel_body_id=self._frontwheel_body_id,
            rearwheel_body_id=self._rearwheel_body_id,
            prev_front_tire_bottom_z=state.info["prev_front_tire_bottom_z"],
            prev_rear_tire_bottom_z=state.info["prev_rear_tire_bottom_z"],
            knee_target=knee_target,
            knee_pos=knee_pos,
            knee_vel=knee_vel,
        )

        in_takeoff_window = (x >= cfg.step_front_x - cfg.takeoff_window_far) & (x <= cfg.step_front_x - cfg.takeoff_window_near)
        in_platform_x = (x >= cfg.step_front_x + cfg.valid_landing_min_past_edge) & (x <= cfg.step_back_x - cfg.valid_landing_back_margin)
        in_platform_y = jp.abs(y) <= (cfg.step_half_width - cfg.landing_side_margin)
        in_platform = self._platform_gate(qpos)
        platform_geom = (
            (x >= cfg.step_front_x + cfg.step_top_xy_margin)
            & (x <= cfg.step_back_x - cfg.step_top_xy_margin)
            & (jp.abs(y) <= cfg.step_half_width - cfg.step_top_xy_margin)
        )
        terrain_z = jp.where(platform_geom, cfg.step_top_z, 0.0)
        expected_base_z = cfg.step_top_z + cfg.nominal_base_z_ground
        support_height_error = z - expected_base_z
        imu_delta_z = acc_z - state.info["prev_acc_z"]
        imu_abs_delta_z = jp.abs(imu_delta_z)
        descending_gate = state.info["prev_vz"] <= cfg.imu_impact_min_descend_vz
        imu_impulse_gate = (
            imu_abs_delta_z >= cfg.imu_impact_delta_threshold
            if bool(cfg.imu_impact_use_abs_delta)
            else float(cfg.imu_impact_sign) * imu_delta_z >= cfg.imu_impact_delta_threshold
        )
        # Contact event available to a physical robot: previously descending
        # after confirmed Flight, inside the known platform interior, with a
        # sharp local-IMU impulse.  It makes no use of collision pair IDs.
        # ``impact_base_gate`` contains every gate *except* the acceleration
        # threshold.  Logging it lets calibration compare event amplitudes only
        # against physically eligible non-events, rather than unrelated shocks
        # during takeoff, airborne motion, or after Landing.
        impact_phase_flight = state.info["estimated_phase"] == STAGE_ID["flight"]
        impact_had_airborne = state.info["had_airborne"] > 0
        impact_not_landed = state.info["had_valid_landing"] == 0
        impact_base_gate = (
            impact_phase_flight
            & impact_had_airborne
            & impact_not_landed
            & in_platform
            & descending_gate
        )
        imu_impact = impact_base_gate & imu_impulse_gate
        support_height_gate = jp.abs(support_height_error) <= cfg.imu_support_height_tol
        support_vz_gate = jp.abs(vz) <= cfg.imu_support_max_abs_vz
        # Support is evaluated after updating the landing flag below so the
        # first physically valid landing step is not discarded by an artificial
        # one-control-tick delay.
        # For Approach/Takeoff/Flight phase bookkeeping only: infer airborne
        # from body height and vertical motion, not from a collision buffer.
        above_terrain = z > (terrain_z + cfg.nominal_base_z_ground + cfg.imu_airborne_height_margin)
        moving_vertically = jp.abs(vz) >= cfg.imu_airborne_min_abs_vz
        imu_airborne = above_terrain | (moving_vertically & (z > terrain_z + 0.5 * cfg.nominal_base_z_ground))

        valid_landing = jp.where(jp.asarray(self._imu_only), imu_impact, contact["wheel_top"] | (jp.asarray(self._use_imu_fallback) & imu_impact))
        wheel_any = jp.where(jp.asarray(self._imu_only), ~imu_airborne, contact["wheel_any"])

        takeoff_event = (phase0 == STAGE_ID["approach"]) & in_takeoff_window & wheel_any & (vx >= 0.90)
        jump_signal_latched=(state.info["jump_signal_latched"]>0) | takeoff_event
        jump_window_start_x=jp.where(takeoff_event,x,state.info["jump_window_start_x"])
        curriculum=jp.clip(float(cfg.stage_curriculum_scale),0.0,1.0)
        jump_window_length=float(cfg.stage_jump_window_length_easy)+(float(cfg.stage_jump_window_length_hard)-float(cfg.stage_jump_window_length_easy))*curriculum
        jump_window_end_x=jp.where(takeoff_event,x+jump_window_length,state.info["jump_window_end_x"])
        jump_window_active=jump_signal_latched & (x<=jump_window_end_x)
        raw_airborne = imu_airborne if self._imu_only else ~wheel_any
        takeoff_airborne = jp.where(
            (phase0 == STAGE_ID["takeoff"]) & bool(cfg.takeoff_use_wheel_clearance),
            takeoff_signals["dual_wheel_liftoff"],
            raw_airborne,
        )
        airborne_count = jp.where(takeoff_airborne, state.info["airborne_count"] + 1, 0)
        confirmed_airborne = airborne_count >= int(cfg.airborne_confirm_steps)
        phase1 = jp.where(takeoff_event, jp.asarray(STAGE_ID["takeoff"], jp.int32), phase0)
        phase1 = jp.where((phase1 == STAGE_ID["takeoff"]) & confirmed_airborne, jp.asarray(STAGE_ID["flight"], jp.int32), phase1)
        phase1 = jp.where((phase1 == STAGE_ID["flight"]) & valid_landing, jp.asarray(STAGE_ID["landing"], jp.int32), phase1)
        had_airborne = jp.where(confirmed_airborne, 1, state.info["had_airborne"])
        had_landing = jp.where(valid_landing, 1, state.info["had_valid_landing"])

        # Deployable phase estimate: a separate event filter derived from
        # proprioception/IMU and known obstacle geometry.  The Actor receives
        # only these soft probabilities, never the oracle phase above.
        estimated0 = state.info["estimated_phase"]
        estimated1 = jp.where(
            (estimated0 == STAGE_ID["approach"]) & in_takeoff_window & (vx >= 0.90),
            jp.asarray(STAGE_ID["takeoff"], jp.int32), estimated0,
        )
        estimated1 = jp.where(
            (estimated1 == STAGE_ID["takeoff"]) & confirmed_airborne,
            jp.asarray(STAGE_ID["flight"], jp.int32), estimated1,
        )
        estimated1 = jp.where(
            (estimated1 == STAGE_ID["flight"]) & imu_impact,
            jp.asarray(STAGE_ID["landing"], jp.int32), estimated1,
        )
        phase_probs1 = 0.25 * state.info["phase_probs"] + 0.75 * jax.nn.one_hot(
            jp.clip(estimated1, 0, 3), 4, dtype=jp.float32
        )
        phase_probs1 = phase_probs1 / jp.maximum(jp.sum(phase_probs1), 1e-6)
        imu_support = self._imu_support_estimate(qpos, qvel, had_landing)
        support = jp.where(
            jp.asarray(self._imu_only), imu_support,
            contact["wheel_top"] | (jp.asarray(self._use_imu_fallback) & imu_support),
        )
        prelaunch_airborne = jp.where((phase0 == STAGE_ID["approach"]) & imu_airborne & (~in_takeoff_window), state.info["prelaunch_airborne_count"] + 1, 0)

        invalid_count = jp.where(contact["wheel_invalid"], state.info["invalid_wheel_count"] + 1, 0)
        invalid_fail = invalid_count >= int(cfg.invalid_wheel_confirm_steps)
        bounce = jp.where((had_landing > 0) & (~support) & (~contact["prohibited"]) & (~invalid_fail), state.info["landing_bounce_count"] + 1, 0)
        contact_age = jp.where(support, state.info["contact_age"] + 1, jp.where(bounce <= int(cfg.landing_bounce_grace_steps), state.info["contact_age"], 0))

        gyro_norm = jp.linalg.norm(gyro)
        recovery_roll_gate = jp.abs(roll) <= _deg2rad(cfg.recovery_max_roll_deg)
        recovery_pitch_gate = jp.abs(pitch) <= _deg2rad(cfg.recovery_max_pitch_deg)
        recovery_gyro_gate = gyro_norm <= cfg.recovery_max_angvel
        recovery_vx_gate = vx >= cfg.recovery_min_vx
        recovery_no_fault_gate = (~contact["prohibited"]) & (~invalid_fail)
        recovery_now = (
            (phase1 == STAGE_ID["landing"])
            & support & in_platform
            & recovery_roll_gate
            & recovery_pitch_gate
            & recovery_gyro_gate
            & recovery_vx_gate
            & recovery_no_fault_gate
        )
        recovery_count = jp.where(recovery_now, state.info["recovery_count"] + 1, 0)
        recovery = recovery_count >= int(cfg.recovery_hold_steps)

        previous_feature = self._landing_entry_feature(
            state.data,
            state.info["had_valid_landing"] > 0,
            state.info["contact_age"] > 0,
            state.info["landing_entry_age"],
        ) if self._entry_matcher else self._physical_feature(state.data)
        landing_entry_age=jp.where(valid_landing & (state.info["had_valid_landing"]==0),1,jp.where(had_landing>0,state.info["landing_entry_age"]+1,0))
        feature = self._landing_entry_feature(data,had_landing,support,landing_entry_age) if self._entry_matcher else self._physical_feature(data)
        if self._safe_count:
            previous_feature_z = (previous_feature - self._safe_center) / self._safe_scale
            previous_safe_dist = jp.min(jp.linalg.norm(self._safe_features - previous_feature_z[None, :], axis=1))
            feature_z = (feature - self._safe_center) / self._safe_scale
            safe_dist = jp.min(jp.linalg.norm(self._safe_features - feature_z[None, :], axis=1))
            safe_entry = safe_dist <= self._safe_radius
        else:
            previous_safe_dist = jp.asarray(float(self._safe_radius) + 1.0, dtype=jp.float32)
            safe_dist = jp.asarray(jp.inf, dtype=jp.float32)
            safe_entry = jp.asarray(False)
        if self._stage_name == "landing":
            chain = recovery
        elif self._stage_name == "flight":
            chain = (landing_entry_age>=1) & (landing_entry_age<=int(cfg.landing_entry_window_steps)) & safe_entry
        elif self._stage_name == "takeoff":
            chain = confirmed_airborne & safe_entry
        elif self._stage_name == "approach":
            chain = takeoff_event & safe_entry
        else:
            chain = recovery
        chain_ever = (state.info["chain_ever"] > 0) | chain
        # No downstream certified Tube exists during Landing bootstrap.  Keep
        # the matching sentinel finite so Brax episode aggregation never turns
        # an inapplicable diagnostic into NaN.
        safe_dist_metric = jp.where(
            jp.isfinite(safe_dist),
            safe_dist,
            jp.asarray(float(self._safe_radius) + 1.0, jp.float32),
        )

        step_no = state.info["episode_step"] + 1
        landing_phase_step = jp.where(
            phase1 == STAGE_ID["landing"], state.info["landing_phase_step"] + 1, 0
        )
        takeoff_profile = compute_takeoff_reward_profile(
            cfg=cfg,
            signals=takeoff_signals,
            action=action,
            phase0=phase0,
            phase1=phase1,
            dual_wheel_liftoff_seen=state.info["dual_wheel_liftoff_seen"],
            positive_pitch_count=state.info["positive_pitch_count"],
            wheelie_count=state.info["wheelie_count"],
        )
        takeoff_task_failure = (
            takeoff_profile["positive_pitch_failure"]
            | takeoff_profile["wheelie_failure"]
            | takeoff_profile["missed_liftoff_deadline"]
            | takeoff_profile["missed_wheel_clearance_deadline"]
        )
        landing_timeout_steps = max(1, int(round(float(cfg.stage_timeout_landing) / float(cfg.ctrl_dt))))
        timeout = jp.where(
            phase1 == STAGE_ID["landing"],
            landing_phase_step >= landing_timeout_steps,
            step_no >= self._stage_timeout_steps(),
        )
        roll_bad = jp.abs(roll) > _deg2rad(cfg.max_roll_deg)
        pitch_bad = jp.abs(pitch) > _deg2rad(cfg.max_pitch_deg)
        backward = x < state.info["start_x"] - cfg.max_backward_distance
        back_edge = (had_landing > 0) & (x >= cfg.step_back_x - cfg.platform_back_edge_diagnostic_margin)
        prelaunch_fail = prelaunch_airborne >= int(cfg.pretakeoff_airborne_fail_steps)
        hard_failure = contact["prohibited"] | invalid_fail | roll_bad | pitch_bad | backward | back_edge | prelaunch_fail | takeoff_task_failure | nonfinite
        entry_pose_ok=(jp.abs(roll)<=_deg2rad(cfg.max_roll_deg)) & (jp.abs(pitch)<=_deg2rad(min(float(cfg.max_pitch_deg),35.0))) & (gyro_norm<=float(cfg.recovery_max_angvel))
        current_physical=self._physical_feature(data);previous_physical=self._physical_feature(state.data)
        current_support_z=(current_physical-self._stage_support_center)/self._stage_support_scale
        previous_support_z=(previous_physical-self._stage_support_center)/self._stage_support_scale
        current_stage_support_distance=jp.min(jp.linalg.norm(self._stage_support_features-current_support_z[None,:],axis=1)/self._stage_support_radii)
        previous_stage_support_distance=jp.min(jp.linalg.norm(self._stage_support_features-previous_support_z[None,:],axis=1)/self._stage_support_radii)
        stage_support_near=jp.asarray(self._stage_support_enabled) & (current_stage_support_distance<=1.0)
        apex_seen=(state.info["apex_seen"]>0) | ((state.info["prev_vz"]>0.0)&(vz<=0.0))
        objective_id = state.info["reachability_objective_id"]
        takeoff_to_ascent_entry = confirmed_airborne & (vz>=float(cfg.takeoff_liftoff_vz)) & entry_pose_ok
        ascent_to_apex_entry = ((phase1==STAGE_ID["flight"]) & (z>=0.4015475838251305)
                               & (z<=0.7015475838251305) & (jp.abs(vz)<=0.25) & entry_pose_ok)
        apex_to_descent_entry = (phase1==STAGE_ID["flight"]) & apex_seen & raw_airborne & \
                (vz>=self._stage_descent_vz_bounds[0]) & (vz<=self._stage_descent_vz_bounds[1]) & \
                (x>=self._stage_support_x_bounds[0]) & (x<=self._stage_support_x_bounds[1]) & \
                (z>=self._stage_support_z_bounds[0]) & (z<=self._stage_support_z_bounds[1]) & \
                (jp.abs(gyro[0])<=self._stage_roll_rate_max) & (jp.abs(gyro[1])<=self._stage_pitch_rate_max) & \
                stage_support_near & entry_pose_ok
        descent_to_landing_entry = ((phase1==STAGE_ID["landing"]) & valid_landing & support
                                    & in_platform & (jp.abs(vz)<=0.75) & entry_pose_ok)
        stage_entry_raw = jp.where(
            objective_id == REACHABILITY_OBJECTIVE["takeoff_to_ascent"],
            takeoff_to_ascent_entry,
            jp.where(
                objective_id == REACHABILITY_OBJECTIVE["ascent_to_apex"],
                ascent_to_apex_entry,
                jp.where(
                    objective_id == REACHABILITY_OBJECTIVE["apex_to_descent"],
                    apex_to_descent_entry,
                    jp.where(
                        objective_id == REACHABILITY_OBJECTIVE["descent_to_landing"],
                        descent_to_landing_entry,
                        jp.asarray(False),
                    ),
                ),
            ),
        )
        stage_entry=stage_entry_raw & (~hard_failure)
        stage_entry_ever=(state.info["stage_entry_ever"]>0) | stage_entry
        chain_terminal = jp.asarray(bool(cfg.expert_chain_termination)) & chain & (self._stage_name != "landing")
        stage_terminal=(objective_id > REACHABILITY_OBJECTIVE[""]) & stage_entry
        terminated = recovery | hard_failure | chain_terminal | stage_terminal
        truncated = timeout & (~terminated)
        done = terminated | truncated
        code = jp.where(recovery, END_RECOVERY, END_NONE)
        code = jp.where(timeout & ~recovery, END_STAGE_TIMEOUT, code)
        code = jp.where(chain_terminal & ~recovery, END_CHAIN_ENTRY, code)
        code = jp.where(stage_terminal & ~recovery, END_NEXT_STAGE_ENTRY, code)
        code = jp.where(takeoff_profile["missed_wheel_clearance_deadline"] & ~recovery, END_TAKEOFF_MISSED_WHEEL_CLEARANCE, code)
        code = jp.where(takeoff_profile["missed_liftoff_deadline"] & ~recovery, END_TAKEOFF_MISSED_LIFTOFF_DEADLINE, code)
        code = jp.where(takeoff_profile["wheelie_failure"] & ~recovery, END_TAKEOFF_WHEELIE_FAILURE, code)
        code = jp.where(takeoff_profile["positive_pitch_failure"] & ~recovery, END_TAKEOFF_POSITIVE_PITCH_FAILURE, code)
        code = jp.where(prelaunch_fail & ~recovery, END_PRETAKEOFF_AIRBORNE, code)
        code = jp.where(back_edge & ~recovery, END_PLATFORM_BACK_EDGE, code)
        code = jp.where(backward & ~recovery, END_BACKWARD, code)
        code = jp.where(pitch_bad & ~recovery, END_PITCH_LIMIT, code)
        code = jp.where(roll_bad & ~recovery, END_ROLL_LIMIT, code)
        code = jp.where(invalid_fail & ~recovery, END_INVALID_WHEEL_STEP, code)
        code = jp.where(contact["prohibited"] & ~recovery, END_PROHIBITED_CONTACT, code)
        code = jp.where(nonfinite & ~recovery, END_NONFINITE, code)

        # Landing reward: no CoM tracking and no fixed-height target.  The
        # strict profile corrects a V21 legacy issue where an ever-growing
        # recovery-streak fraction was paid again on every step, allowing a
        # zero-action rollout to accrue a high return before rolling over.
        r_roll = jp.exp(-((roll / _deg2rad(cfg.roll_sigma_deg)) ** 2))
        r_pitch = jp.exp(-((pitch / _deg2rad(cfg.pitch_sigma_deg)) ** 2))
        r_speed_landing = jp.exp(-(((vx - cfg.landing_target_forward_speed) / cfg.landing_forward_sigma) ** 2))
        r_speed = jp.exp(-(((vx - cfg.target_forward_speed) / cfg.forward_sigma) ** 2))
        r_gyro = jp.exp(-((gyro_norm / max(float(cfg.recovery_max_angvel), 1e-6)) ** 2))
        legacy_recovery_score = jp.mean(jp.asarray([
            support & in_platform,
            recovery_roll_gate,
            recovery_pitch_gate,
            recovery_gyro_gate,
            recovery_vx_gate,
            recovery_no_fault_gate,
        ], dtype=jp.float32))
        recovery_quality_gate = (
            recovery_roll_gate
            & recovery_pitch_gate
            & recovery_gyro_gate
            & recovery_vx_gate
            & recovery_no_fault_gate
        )
        recovery_quality = (
            (support & in_platform).astype(jp.float32)
            * r_roll * r_pitch * r_gyro * r_speed_landing
        )
        # Strict Landing reward must not call a state "recovery-shaped" after
        # it has already left the posture/velocity envelope required for actual
        # Recovery.  The legacy diagnostic remains available below but is never
        # used as a strict-mode reward component.
        if bool(cfg.landing_quality_requires_recovery_gates):
            strict_recovery_quality = recovery_quality * recovery_quality_gate.astype(jp.float32)
        else:
            strict_recovery_quality = recovery_quality
        recovery_score = strict_recovery_quality if self._strict_landing_reward else legacy_recovery_score
        rec_frac = jp.minimum(recovery_count.astype(jp.float32), float(cfg.recovery_hold_steps)) / float(cfg.recovery_hold_steps)
        prev_rec_frac = jp.minimum(state.info["recovery_count"].astype(jp.float32), float(cfg.recovery_hold_steps)) / float(cfg.recovery_hold_steps)
        rec_frac_delta = jp.maximum(0.0, rec_frac - prev_rec_frac)
        recovery_streak_reward = cfg.landing_coeff_recovery_streak * (rec_frac_delta if self._strict_landing_reward else rec_frac)
        action_delta = jp.mean((action - state.info["last_action"]) ** 2)
        action_mag = jp.mean(action ** 2)
        action_sat = jp.mean((jp.abs(action) > 0.85).astype(jp.float32))
        downrange = jp.maximum(0.0, x - cfg.landing_target_x)
        yaw_rate = gyro[2]
        stage_elapsed = step_no.astype(jp.float32) * cfg.ctrl_dt
        lateral_penalty = cfg.coeff_lateral_vel * (vy ** 2)
        yaw_rate_penalty = cfg.coeff_yaw_rate * (yaw_rate ** 2)
        yaw_angle_penalty = cfg.coeff_yaw_angle * ((yaw / _deg2rad(25.0)) ** 2)
        action_mag_penalty = cfg.coeff_action_mag * action_mag
        action_saturation_penalty = cfg.coeff_action_saturation * action_sat
        downrange_penalty = cfg.coeff_downrange * (downrange ** 2)
        time_penalty = cfg.landing_stage_timeout_step_penalty * stage_elapsed
        action_smooth_penalty = cfg.coeff_action_smooth * action_delta
        # Progressive instability starts at the strict Recovery boundary, not
        # only at the much larger hard-failure angle.  Normalisation keeps the
        # three terms dimensionless and comparable across threshold edits.
        roll_excess_norm = jp.maximum(0.0, jp.abs(roll) - _deg2rad(cfg.recovery_max_roll_deg)) / jp.maximum(_deg2rad(cfg.recovery_max_roll_deg), 1e-6)
        pitch_excess_norm = jp.maximum(0.0, jp.abs(pitch) - _deg2rad(cfg.recovery_max_pitch_deg)) / jp.maximum(_deg2rad(cfg.recovery_max_pitch_deg), 1e-6)
        gyro_excess_norm = jp.maximum(0.0, gyro_norm - cfg.recovery_max_angvel) / max(float(cfg.recovery_max_angvel), 1e-6)
        roll_excess_penalty = cfg.landing_roll_excess_penalty * (roll_excess_norm ** 2)
        pitch_excess_penalty = cfg.landing_pitch_excess_penalty * (pitch_excess_norm ** 2)
        gyro_excess_penalty = cfg.landing_gyro_excess_penalty * (gyro_excess_norm ** 2)
        instability_penalty = roll_excess_penalty + pitch_excess_penalty + gyro_excess_penalty
        hard_failure_penalty = cfg.landing_terminal_hard_failure_penalty * hard_failure.astype(jp.float32)
        timeout_penalty = cfg.landing_terminal_timeout_penalty * (timeout & ~recovery & ~hard_failure).astype(jp.float32)
        failure_penalty = (hard_failure_penalty + timeout_penalty) if self._strict_landing_reward else jp.zeros((), jp.float32)

        stable_support = (support & in_platform).astype(jp.float32)
        reward_alive = cfg.coeff_alive
        reward_forward = cfg.landing_coeff_forward_speed * r_speed_landing * stable_support
        reward_progress = cfg.landing_coeff_progress * jp.clip(vx, -3.0, 3.0) * stable_support
        reward_roll = cfg.coeff_roll * r_roll * stable_support
        reward_pitch = cfg.coeff_pitch * r_pitch * stable_support
        reward_platform = cfg.landing_coeff_platform_contact * stable_support
        reward_recovery_shape = cfg.landing_coeff_recovery_shaping * recovery_score
        reward_success = cfg.coeff_recovery_success * recovery.astype(jp.float32)
        landing_reward = (
            reward_alive + reward_forward + reward_progress + reward_roll + reward_pitch
            + reward_platform + reward_recovery_shape + recovery_streak_reward + reward_success
            - lateral_penalty - yaw_rate_penalty - yaw_angle_penalty
            - action_mag_penalty - action_saturation_penalty - downrange_penalty
            - time_penalty - action_smooth_penalty - instability_penalty - failure_penalty
        )
        unified_reward = (
            cfg.coeff_alive + cfg.coeff_forward_speed * r_speed + cfg.coeff_progress * jp.clip(vx, -3.0, 3.0)
            + cfg.coeff_roll * r_roll + cfg.coeff_pitch * r_pitch
            + cfg.coeff_takeoff_event * takeoff_event.astype(jp.float32)
            + cfg.coeff_valid_landing_event * valid_landing.astype(jp.float32)
            + cfg.coeff_platform_contact * support.astype(jp.float32)
            + cfg.coeff_recovery_shaping * legacy_recovery_score
            + cfg.coeff_chain_event * chain.astype(jp.float32)
            + cfg.coeff_recovery_success * recovery.astype(jp.float32)
            - cfg.coeff_action_smooth * action_delta
        )
        descent_local = compute_descent_local_reward(
            cfg=cfg,
            pitch=pitch,
            pitch_rate=gyro[1],
            roll=roll,
            roll_rate=gyro[0],
            vx=vx,
            previous_distance=jp.where(jp.isfinite(previous_safe_dist), previous_safe_dist, safe_dist_metric),
            current_distance=safe_dist_metric,
            action=action,
            previous_action=state.info["last_action"],
            chain=chain,
            hard_failure=hard_failure,
        )
        descent_rsi_pilot = compute_descent_rsi_pilot_reward(
            cfg=cfg, roll=roll, pitch=pitch, gyro=gyro, vz=vz, action=action,
            previous_action=state.info["last_action"], hard_failure=hard_failure,
        )
        zero=jp.asarray(0.0,jp.float32)
        stage_reward={"reward":zero,"shaping":zero,"event":zero,"handoff_quality":zero,
                      "handoff_bonus":zero,"failure_penalty":zero,"progress":zero,
                      "pose":zero,"speed":zero,"angular_penalty":zero,"yaw_score":zero,"bounded_height":zero,
                      "joint_energy_penalty":zero,"action_smooth_penalty":zero,"action_magnitude_penalty":zero}
        if self._reachability_objective:
            joint_energy=(jp.abs(data.ctrl[2]*data.qvel[self._joint_qvel["hip_joint"]])+jp.abs(data.ctrl[3]*data.qvel[self._joint_qvel["knee_joint"]]))/100.0
            objective_names = tuple(name for name in REACHABILITY_OBJECTIVE if name)
            computed = {
                name: compute_stage_next_entry_reward(
                    cfg=cfg, objective=name, feature=current_physical,
                    previous_feature=previous_physical, action=action,
                    previous_action=state.info["last_action"], next_entry=stage_entry,
                    hard_failure=hard_failure, jump_latched=jump_signal_latched,
                    window_active=jump_window_active, joint_energy=joint_energy,
                    current_support_distance=current_stage_support_distance,
                    previous_support_distance=previous_stage_support_distance,
                )
                for name in objective_names
            }
            stage_reward = {
                key: sum(
                    jp.where(objective_id == REACHABILITY_OBJECTIVE[name], values[key], zero)
                    for name, values in computed.items()
                )
                for key in stage_reward
            }
        landing_reward_active = phase1 == STAGE_ID["landing"]
        reward = jp.where(landing_reward_active, landing_reward, unified_reward)
        # Candidate-guided descent episodes retain the local C_L objective even
        # after a missed entry contact; otherwise the much larger Landing
        # Recovery bonus would reward bypassing the fixed C_L matcher.
        descent_local_active = state.info["descent_local_episode"]
        reward = jp.where(descent_local_active, descent_local["reward"], reward)
        reward = jp.where(
            jp.asarray(bool(cfg.descent_rsi_pilot_reward_enable)),
            descent_rsi_pilot["reward"], reward,
        )
        # Stage-C must not lose its objective immediately after liftoff.  Local
        # launch shaping is used only while still in Takeoff; after transition
        # the same Actor receives continuation/chain/final reward through Flight
        # and Landing.  This also rehearses downstream skills and limits forgetting.
        takeoff_local = (phase0 == STAGE_ID["takeoff"]) & (phase1 == STAGE_ID["takeoff"])
        takeoff_total = jp.where(takeoff_local, takeoff_profile["reward"], reward)
        reward = jp.where(self._stage_name == "takeoff", takeoff_total, reward)
        if self._reachability_objective:
            reward=jp.where(objective_id > REACHABILITY_OBJECTIVE[""], stage_reward["reward"], reward)
        reward = jp.where(nonfinite, -jp.asarray(float(cfg.coeff_hard_failure), jp.float32), reward)

        metrics = self._empty_metrics()
        metrics.update({
            # ``reward`` is required by Brax EvalWrapper's static metric
            # contract; ``reward/total`` remains the DVGC diagnostic name.
            "reward": reward, "success": (recovery | chain_terminal | stage_terminal).astype(jp.float32), "reward/total": reward,
            "reward/roll": reward_roll, "reward/pitch": reward_pitch,
            "reward/speed": jp.where(landing_reward_active,reward_forward,cfg.coeff_forward_speed * r_speed),
            "reward/recovery_shaping": reward_recovery_shape,
            "reward/recovery_quality": strict_recovery_quality if self._strict_landing_reward else recovery_quality,
            "reward/recovery_streak": recovery_streak_reward, "reward/recovery_event": recovery.astype(jp.float32),
            "reward/alive": reward_alive, "reward/progress": reward_progress, "reward/platform_support": reward_platform,
            "reward/failure_penalty": failure_penalty, "reward/hard_failure_penalty": hard_failure_penalty,
            "reward/timeout_penalty": timeout_penalty, "reward/instability_penalty": instability_penalty,
            "reward/roll_excess_penalty": roll_excess_penalty, "reward/pitch_excess_penalty": pitch_excess_penalty,
            "reward/gyro_excess_penalty": gyro_excess_penalty, "reward/lateral_penalty": lateral_penalty,
            "reward/yaw_rate_penalty": yaw_rate_penalty, "reward/yaw_angle_penalty": yaw_angle_penalty,
            "reward/action_mag_penalty": action_mag_penalty, "reward/action_saturation_penalty": action_saturation_penalty,
            "reward/downrange_penalty": downrange_penalty, "reward/time_penalty": time_penalty,
            "reward/action_smooth_penalty": action_smooth_penalty,
            "reward/chain_event": cfg.coeff_chain_event * chain.astype(jp.float32),
            "reward/descent_local_shaping": descent_local["shaping"],
            "reward/descent_local_pitch_posture": descent_local["pitch_posture"],
            "reward/descent_local_pitch_rate_penalty": descent_local["pitch_rate_penalty"],
            "reward/descent_local_pitch_soft_penalty": descent_local["pitch_soft_penalty"],
            "reward/descent_local_roll_score": descent_local["roll_score"],
            "reward/descent_local_roll_penalty": descent_local["roll_penalty"],
            "reward/descent_local_speed_score": descent_local["speed_score"],
            "reward/descent_local_progress": descent_local["progress"],
            "reward/descent_local_survival": descent_local["survival"],
            "reward/descent_local_action_difference_penalty": descent_local["action_difference_penalty"],
            "reward/descent_local_action_magnitude_penalty": descent_local["action_magnitude_penalty"],
            "reward/descent_local_failure_penalty": descent_local["failure_penalty"],
            "reward/descent_rsi_pilot_shaping": descent_rsi_pilot["shaping"],
            "reward/descent_rsi_pilot_survival": descent_rsi_pilot["survival"],
            "reward/descent_rsi_pilot_roll": descent_rsi_pilot["roll_score"],
            "reward/descent_rsi_pilot_pitch": descent_rsi_pilot["pitch_score"],
            "reward/descent_rsi_pilot_angular": descent_rsi_pilot["angular_score"],
            "reward/descent_rsi_pilot_vz": descent_rsi_pilot["descent_speed_score"],
            "reward/descent_rsi_pilot_action_smooth_penalty": descent_rsi_pilot["action_smooth_penalty"],
            "reward/descent_rsi_pilot_action_magnitude_penalty": descent_rsi_pilot["action_magnitude_penalty"],
            "reward/descent_rsi_pilot_failure_penalty": descent_rsi_pilot["failure_penalty"],
            "reward/takeoff_total": takeoff_profile["reward"],
            "reward/takeoff_dual_wheel_vz": takeoff_profile["terms"]["dual_wheel_vz"],
            "reward/takeoff_dual_wheel_height": takeoff_profile["terms"]["dual_wheel_height"],
            "reward/takeoff_min_tire_clearance": takeoff_profile["terms"]["min_tire_clearance"],
            "reward/takeoff_knee_useful_motion": takeoff_profile["terms"]["knee_useful_motion"],
            "reward/takeoff_positive_pitch_penalty": takeoff_profile["terms"]["positive_pitch_penalty"],
            "reward/takeoff_wheelie_penalty": takeoff_profile["terms"]["wheelie_penalty"],
            "reward/takeoff_wheel_unsync_penalty": takeoff_profile["terms"]["wheel_unsync_penalty"],
            "reward/takeoff_action_mag_penalty": takeoff_profile["terms"]["action_mag_penalty"],
            "reward/stage_entry_total": stage_reward["reward"],
            "reward/stage_entry_shaping": stage_reward["shaping"],
            "reward/stage_entry_event": stage_reward["event"],
            "reward/stage_entry_handoff_quality": stage_reward["handoff_quality"],
            "reward/stage_entry_handoff_bonus": stage_reward["handoff_bonus"],
            "reward/stage_entry_failure_penalty": stage_reward["failure_penalty"],
            "reward/stage_entry_progress": stage_reward["progress"],
            "diag/legacy_recovery_gate_fraction": legacy_recovery_score,
            "diag/recovery_quality_gate": recovery_quality_gate.astype(jp.float32),
            "event/takeoff": takeoff_event.astype(jp.float32), "event/landing": valid_landing.astype(jp.float32),
            "event/chain": chain.astype(jp.float32), "event/chain_ever": chain_ever.astype(jp.float32), "event/recovery": recovery.astype(jp.float32),
            "contact/front_top": contact["front_top"].astype(jp.float32), "contact/rear_top": contact["rear_top"].astype(jp.float32),
            "contact/wheel_invalid": contact["wheel_invalid"].astype(jp.float32), "contact/prohibited": contact["prohibited"].astype(jp.float32),
            "contact/imu_impact": imu_impact.astype(jp.float32),
            "imu/acc_z": acc_z, "imu/prev_acc_z": state.info["prev_acc_z"],
            "imu/delta_z": imu_delta_z, "imu/abs_delta_z": imu_abs_delta_z,
            "imu/threshold": jp.asarray(cfg.imu_impact_delta_threshold, jp.float32),
            "gate/airborne_history": (state.info["had_airborne"] > 0).astype(jp.float32),
            "gate/in_platform_x": in_platform_x.astype(jp.float32), "gate/in_platform_y": in_platform_y.astype(jp.float32),
            "gate/in_platform": in_platform.astype(jp.float32), "gate/descending": descending_gate.astype(jp.float32),
            "gate/imu_impulse": imu_impulse_gate.astype(jp.float32),
            "gate/impact_phase_flight": impact_phase_flight.astype(jp.float32),
            "gate/impact_had_airborne": impact_had_airborne.astype(jp.float32),
            "gate/impact_not_landed": impact_not_landed.astype(jp.float32),
            "gate/impact_base": impact_base_gate.astype(jp.float32),
            "gate/support_height": support_height_gate.astype(jp.float32), "gate/support_vz": support_vz_gate.astype(jp.float32),
            "gate/support": support.astype(jp.float32), "gate/airborne": imu_airborne.astype(jp.float32),
            "gate/recovery_now": recovery_now.astype(jp.float32),
            "gate/recovery_roll": recovery_roll_gate.astype(jp.float32),
            "gate/recovery_pitch": recovery_pitch_gate.astype(jp.float32),
            "gate/recovery_gyro": recovery_gyro_gate.astype(jp.float32),
            "gate/recovery_vx": recovery_vx_gate.astype(jp.float32),
            "gate/recovery_no_fault": recovery_no_fault_gate.astype(jp.float32),
            "state/x": x, "state/y": y, "state/z": z, "state/vx": vx, "state/vy": vy, "state/vz": vz,
            "state/yaw_deg": yaw * 180.0 / jp.pi, "state/expected_base_z": expected_base_z,
            "state/support_height_error": support_height_error, "state/gyro_norm": gyro_norm,
            "state/roll_excess_norm": roll_excess_norm, "state/pitch_excess_norm": pitch_excess_norm,
            "state/gyro_excess_norm": gyro_excess_norm,
            "state/roll_deg": jp.abs(roll) * 180.0 / jp.pi, "state/pitch_deg": jp.abs(pitch) * 180.0 / jp.pi,
            "state/pitch_signed_deg": takeoff_signals["pitch_signed_deg"],
            "state/frontwheel_center_z": takeoff_signals["frontwheel_center_z"],
            "state/rearwheel_center_z": takeoff_signals["rearwheel_center_z"],
            "state/front_tire_bottom_z": takeoff_signals["front_tire_bottom_z"],
            "state/rear_tire_bottom_z": takeoff_signals["rear_tire_bottom_z"],
            "state/front_tire_vz": takeoff_signals["front_tire_vz"],
            "state/rear_tire_vz": takeoff_signals["rear_tire_vz"],
            "state/min_tire_bottom_z": takeoff_signals["min_tire_bottom_z"],
            "state/wheel_height_diff": takeoff_signals["wheel_height_diff"],
            "state/frontmost_x": takeoff_signals["frontmost_x"],
            "state/distance_to_obstacle": takeoff_signals["distance_to_obstacle"],
            "state/knee_target": takeoff_signals["knee_target"],
            "state/knee_pos": takeoff_signals["knee_pos"],
            "state/knee_vel": takeoff_signals["knee_vel"],
            "state/knee_target_delta": takeoff_signals["knee_target_delta"],
            "diag/dual_wheel_liftoff": takeoff_signals["dual_wheel_liftoff"].astype(jp.float32),
            "diag/wheelie_detected": takeoff_signals["wheelie_detected"].astype(jp.float32),
            "diag/positive_pitch_bad": takeoff_signals["positive_pitch_bad"].astype(jp.float32),
            "diag/knee_moved": takeoff_signals["knee_moved"].astype(jp.float32),
            "diag/base_z_success_disabled": takeoff_signals["base_z_success_disabled"],
            "diag/cert_bank_safe_count": jp.asarray(float(self._safe_count), jp.float32),
            "diag/cert_bank_explicit": jp.asarray(float(self._cert_bank_explicit), jp.float32),
            "diag/nonfinite_transition": nonfinite.astype(jp.float32),
            "event/dual_wheel_liftoff": takeoff_signals["dual_wheel_liftoff"].astype(jp.float32),
            "event/stage_entry": stage_entry.astype(jp.float32),
            "event/stage_entry_ever": stage_entry_ever.astype(jp.float32),
            **{
                f"reset/episode/{name}": (
                    (state.info["reset_source"] == source_id) & (step_no == 1)
                ).astype(jp.float32)
                for name, source_id in RESET_SOURCE.items()
            },
            **{
                f"reset/transition/{name}": (
                    state.info["reset_source"] == source_id
                ).astype(jp.float32)
                for name, source_id in RESET_SOURCE.items()
            },
            **{
                f"reset/episode/objective/{name}": (
                    (objective_id == objective_value) & (step_no == 1)
                ).astype(jp.float32)
                for name, objective_value in {
                    **{name: value for name, value in REACHABILITY_OBJECTIVE.items() if name},
                    "landing_recovery": REACHABILITY_OBJECTIVE[""],
                }.items()
            },
            **{
                f"reset/transition/objective/{name}": (
                    objective_id == objective_value
                ).astype(jp.float32)
                for name, objective_value in {
                    **{name: value for name, value in REACHABILITY_OBJECTIVE.items() if name},
                    "landing_recovery": REACHABILITY_OBJECTIVE[""],
                }.items()
            },
            **{
                f"reset/episode/group/{name}": (
                    (state.info["bootstrap_group"] == group_id) & (step_no == 1)
                ).astype(jp.float32)
                for name, group_id in DESCENT_BOOTSTRAP_GROUP.items()
            },
            **{
                f"reset/transition/group/{name}": (
                    state.info["bootstrap_group"] == group_id
                ).astype(jp.float32)
                for name, group_id in DESCENT_BOOTSTRAP_GROUP.items()
            },
            **{
                f"reset/episode/layer/{name}": (
                    (state.info["descent_layer"] == layer_id) & (step_no == 1)
                ).astype(jp.float32)
                for name, layer_id in DESCENT_LAYER.items()
            },
            **{
                f"reset/transition/layer/{name}": (
                    state.info["descent_layer"] == layer_id
                ).astype(jp.float32)
                for name, layer_id in DESCENT_LAYER.items()
            },
            **{
                f"reset/episode/parent/p{index:03d}": (
                    (state.info["reset_parent"] == index) & (step_no == 1)
                ).astype(jp.float32)
                for index in range(len(self._reset_parent_ids))
            },
            **{
                f"reset/transition/parent/p{index:03d}": (
                    state.info["reset_parent"] == index
                ).astype(jp.float32)
                for index in range(len(self._reset_parent_ids))
            },
            "phase/approach": (phase1 == STAGE_ID["approach"]).astype(jp.float32),
            "phase/takeoff": (phase1 == STAGE_ID["takeoff"]).astype(jp.float32),
            "phase/flight": (phase1 == STAGE_ID["flight"]).astype(jp.float32),
            "phase/landing": (phase1 == STAGE_ID["landing"]).astype(jp.float32),
            "state/phase_before": phase0.astype(jp.float32), "state/phase": phase1.astype(jp.float32),
            "state/estimated_phase": estimated1.astype(jp.float32), "state/tube_distance_z": safe_dist_metric,
            "state/recovery_count": recovery_count.astype(jp.float32),
            "state/contact_age": contact_age.astype(jp.float32), "done/code": code.astype(jp.float32),
            "end/recovery": (code == END_RECOVERY).astype(jp.float32),
            "end/prohibited_contact": (code == END_PROHIBITED_CONTACT).astype(jp.float32),
            "end/invalid_wheel_step_contact": (code == END_INVALID_WHEEL_STEP).astype(jp.float32),
            "end/roll_limit": (code == END_ROLL_LIMIT).astype(jp.float32),
            "end/pitch_limit": (code == END_PITCH_LIMIT).astype(jp.float32),
            "end/backward": (code == END_BACKWARD).astype(jp.float32),
            "end/platform_back_edge_exit": (code == END_PLATFORM_BACK_EDGE).astype(jp.float32),
            "end/stage_timeout": (code == END_STAGE_TIMEOUT).astype(jp.float32),
            "end/prelaunch_airborne": (code == END_PRETAKEOFF_AIRBORNE).astype(jp.float32),
            "end/takeoff_positive_pitch_failure": (code == END_TAKEOFF_POSITIVE_PITCH_FAILURE).astype(jp.float32),
            "end/takeoff_wheelie_failure": (code == END_TAKEOFF_WHEELIE_FAILURE).astype(jp.float32),
            "end/takeoff_missed_liftoff_deadline": (code == END_TAKEOFF_MISSED_LIFTOFF_DEADLINE).astype(jp.float32),
            "end/takeoff_missed_wheel_clearance_deadline": (code == END_TAKEOFF_MISSED_WHEEL_CLEARANCE).astype(jp.float32),
            "end/chain_entry": (code == END_CHAIN_ENTRY).astype(jp.float32),
            "end/next_stage_entry": (code == END_NEXT_STAGE_ENTRY).astype(jp.float32),
            "end/nonfinite": (code == END_NONFINITE).astype(jp.float32),
        })
        metrics = {key: jp.nan_to_num(value) for key, value in metrics.items()}
        info = dict(state.info)
        frame = self.build_actor_current_frame_v4(
            data, action, phase_probs1, had_landing, contact_age, recovery_count,
            state.info["prev_acc_z"], obs_rng,
        )
        frame = jp.nan_to_num(frame)
        next_history = self._advance_actor_history(state.info["obs_history"], frame)
        privileged_obs = self._privileged_obs(
            data, phase1, had_airborne, had_landing, support, contact_age, recovery_count, action
        )
        privileged_obs = jp.nan_to_num(privileged_obs)
        info.update({
            "rng": rng, "phase": phase1.astype(jp.int32), "had_airborne": had_airborne.astype(jp.int32),
            "had_valid_landing": had_landing.astype(jp.int32), "airborne_count": airborne_count.astype(jp.int32),
            "prelaunch_airborne_count": prelaunch_airborne.astype(jp.int32), "landing_bounce_count": bounce.astype(jp.int32),
            "invalid_wheel_count": invalid_count.astype(jp.int32), "recovery_count": recovery_count.astype(jp.int32),
            "recovery_success": recovery.astype(jp.int32), "chain_success": chain.astype(jp.int32),
            "chain_ever": chain_ever.astype(jp.int32),
            "stage_entry_ever": stage_entry_ever.astype(jp.int32),
            "apex_seen": apex_seen.astype(jp.int32),
            "jump_signal_latched": jump_signal_latched.astype(bool),
            "jump_window_start_x": jump_window_start_x,
            "jump_window_end_x": jump_window_end_x,
            "landing_entry_age": landing_entry_age.astype(jp.int32),
            "landing_phase_step": landing_phase_step.astype(jp.int32),
            "estimated_phase": estimated1.astype(jp.int32), "phase_probs": phase_probs1,
            "terminated": terminated.astype(jp.int32), "truncated": truncated.astype(jp.int32),
            "episode_step": step_no.astype(jp.int32), "last_action": action, "prev_acc_z": acc_z,
            "prev_vz": vz,
            "prev_front_tire_bottom_z": takeoff_signals["front_tire_bottom_z"],
            "prev_rear_tire_bottom_z": takeoff_signals["rear_tire_bottom_z"],
            "positive_pitch_count": takeoff_profile["positive_pitch_count"].astype(jp.int32),
            "wheelie_count": takeoff_profile["wheelie_count"].astype(jp.int32),
            "dual_wheel_liftoff_seen": takeoff_profile["dual_wheel_liftoff_seen"].astype(bool),
            "obs_history": next_history,
            "contact_age": contact_age.astype(jp.int32), "end_code": code.astype(jp.int32),
        })
        obs = self._stack_actor_history(state.info["obs_history"], frame)
        packet_fifo = jp.concatenate(
            [state.info["actor_packet_fifo"][1:], obs[None, :]], axis=0
        )
        info.update({
            "actor_obs_history_pre": state.info["obs_history"],
            "actor_current_frame": frame,
            "actor_obs_history_post": next_history,
            "actor_packet_fifo": packet_fifo,
            "actor_packet_fifo_valid": jp.minimum(
                state.info["actor_packet_fifo_valid"] + 1, 3
            ).astype(jp.int32),
            "actor_observation_rng": obs_rng,
            "actor_frame_prev_acc_z": state.info["prev_acc_z"],
        })
        return mjx_env.State(
            data, {"state": obs, "privileged_state": privileged_obs},
            reward, done.astype(jp.float32), metrics, info
        )
