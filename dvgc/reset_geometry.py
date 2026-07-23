"""MuJoCo-native grounded reset placement and contact audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class GroundSupportResult:
    accepted: bool
    reason: str
    qpos: np.ndarray
    root_z_shift_m: float
    wheel_clearance_min_m: float
    wheel_clearance_max_m: float
    nonwheel_clearance_min_m: float
    minimum_penetration_m: float
    wheel_terrain_contacts: int
    body_terrain_contacts: int
    contact_pairs: tuple[str, ...]

    def summary(self) -> dict:
        payload = asdict(self)
        payload.pop("qpos")
        payload["contact_pairs"] = list(self.contact_pairs)
        return payload


class GroundSupportSolver:
    """Place a free-root proposal on terrain without changing its posture."""

    def __init__(
        self,
        xml_path: str,
        *,
        max_vertical_correction: float = 0.08,
        target_wheel_penetration: float = -2e-4,
        deep_penetration: float = -0.002,
        maximum_other_wheel_gap: float = 0.025,
        tolerance: float = 1e-6,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.max_vertical_correction = float(max_vertical_correction)
        self.target_wheel_penetration = float(target_wheel_penetration)
        self.deep_penetration = float(deep_penetration)
        self.maximum_other_wheel_gap = float(maximum_other_wheel_gap)
        self.tolerance = float(tolerance)
        root = int(self.model.joint("floating_base_joint").id)
        self.root_z = int(self.model.jnt_qposadr[root]) + 2
        self.terrain = tuple(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == 0 and self._enabled(g)
        )
        wheel_bodies = {int(self.model.body(name).id) for name in ("frontwheel", "rearwheel")}
        self.wheels = tuple(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) in wheel_bodies and self._enabled(g)
        )
        self.body = tuple(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) != 0
            and int(self.model.geom_bodyid[g]) not in wheel_bodies
            and self._enabled(g)
        )

    def _enabled(self, geom: int) -> bool:
        return bool(int(self.model.geom_contype[geom]) or int(self.model.geom_conaffinity[geom]))

    def _distance(self, robot: int, terrain: int) -> float:
        return float(mujoco.mj_geomDistance(self.model, self.data, robot, terrain, 2.0, None))

    def measure(self, qpos, qvel=None, ctrl=None) -> dict:
        self.data.qpos[:] = np.asarray(qpos)
        self.data.qvel[:] = 0 if qvel is None else np.asarray(qvel)
        if self.model.nu:
            self.data.ctrl[:] = 0 if ctrl is None else np.asarray(ctrl)
        mujoco.mj_forward(self.model, self.data)
        wheel_distances = [
            min(self._distance(wheel, terrain) for terrain in self.terrain)
            for wheel in self.wheels
        ]
        nonwheel = min(
            self._distance(body, terrain) for body in self.body for terrain in self.terrain
        )
        wheel_set, body_set, terrain_set = set(self.wheels), set(self.body), set(self.terrain)
        wheel_contacts = body_contacts = 0
        minimum_penetration = min(wheel_distances + [nonwheel])
        pairs = []
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if not ((a in terrain_set and b in wheel_set | body_set)
                    or (b in terrain_set and a in wheel_set | body_set)):
                continue
            robot = b if a in terrain_set else a
            wheel_contacts += int(robot in wheel_set)
            body_contacts += int(robot in body_set)
            minimum_penetration = min(minimum_penetration, float(contact.dist))
            pairs.append(f"{self.model.geom(a).name}:{self.model.geom(b).name}")
        return {
            "wheel_min": float(min(wheel_distances)),
            "wheel_max": float(max(wheel_distances)),
            "nonwheel": float(nonwheel),
            "minimum_penetration": float(minimum_penetration),
            "wheel_contacts": int(wheel_contacts),
            "body_contacts": int(body_contacts),
            "pairs": tuple(sorted(pairs)),
        }

    def solve(self, qpos, qvel=None, ctrl=None) -> GroundSupportResult:
        qpos = np.asarray(qpos, np.float64).copy()
        qvel = None if qvel is None else np.asarray(qvel, np.float64)
        ctrl = None if ctrl is None else np.asarray(ctrl, np.float64)
        values = [qpos] + ([] if qvel is None else [qvel]) + ([] if ctrl is None else [ctrl])
        if not all(np.isfinite(value).all() for value in values):
            return self._result(False, "nonfinite", qpos, 0.0, {
                "wheel_min": np.nan, "wheel_max": np.nan, "nonwheel": np.nan,
                "minimum_penetration": np.nan, "wheel_contacts": 0,
                "body_contacts": 0, "pairs": (),
            })
        low, high = -self.max_vertical_correction, self.max_vertical_correction
        low_qpos, high_qpos = qpos.copy(), qpos.copy()
        low_qpos[self.root_z] += low
        high_qpos[self.root_z] += high
        low_measure = self.measure(low_qpos, qvel, ctrl)
        high_measure = self.measure(high_qpos, qvel, ctrl)
        target = self.target_wheel_penetration
        if not (low_measure["wheel_min"] <= target <= high_measure["wheel_min"]):
            measured = low_measure if abs(low_measure["wheel_min"] - target) < abs(high_measure["wheel_min"] - target) else high_measure
            return self._result(False, "support_height_unbracketed", qpos, 0.0, measured)
        for _ in range(40):
            if high - low <= self.tolerance:
                break
            mid = 0.5 * (low + high)
            trial = qpos.copy()
            trial[self.root_z] += mid
            measured = self.measure(trial, qvel, ctrl)
            if measured["wheel_min"] < target:
                low = mid
            else:
                high = mid
        shift = 0.5 * (low + high)
        result_qpos = qpos.copy()
        result_qpos[self.root_z] += shift
        measured = self.measure(result_qpos, qvel, ctrl)
        reason = "placed"
        accepted = True
        if measured["minimum_penetration"] < self.deep_penetration:
            accepted, reason = False, "deep_penetration"
        elif measured["body_contacts"]:
            accepted, reason = False, "body_terrain_contact"
        elif measured["wheel_max"] > self.maximum_other_wheel_gap:
            accepted, reason = False, "unsupported_other_wheel"
        elif measured["wheel_contacts"] < 1 and measured["wheel_min"] > 5e-4:
            accepted, reason = False, "no_wheel_support"
        return self._result(accepted, reason, result_qpos, shift, measured)

    @staticmethod
    def _result(accepted, reason, qpos, shift, measured) -> GroundSupportResult:
        return GroundSupportResult(
            bool(accepted), str(reason), np.asarray(qpos, np.float64).copy(), float(shift),
            float(measured["wheel_min"]), float(measured["wheel_max"]),
            float(measured["nonwheel"]), float(measured["minimum_penetration"]),
            int(measured["wheel_contacts"]), int(measured["body_contacts"]),
            tuple(measured["pairs"]),
        )
