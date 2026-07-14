"""MuJoCo-native terrain clearance for reference-state proposals."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ClearanceResult:
    accepted: bool
    reason: str
    qpos: np.ndarray
    root_z_shift: float
    original_com: np.ndarray
    corrected_com: np.ndarray
    clearance: float
    wheel_clearance: float
    nonwheel_clearance: float
    robot_terrain_contacts: int


class TerrainClearanceSolver:
    """Find the least upward free-root translation satisfying real geom clearance."""

    def __init__(self, xml_path: str, *, margin: float, max_correction: float,
                 tolerance: float = 1e-5, max_iterations: int = 32):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.margin = float(margin)
        self.max_correction = float(max_correction)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        if self.margin < 0 or self.max_correction <= 0 or self.tolerance <= 0:
            raise ValueError("Invalid Flight clearance solver parameters")

        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"Expected one robot free joint, found {len(free)}")
        joint = free[0]
        self.root_z_index = int(self.model.jnt_qposadr[joint]) + 2
        self.root_body_id = int(self.model.jnt_bodyid[joint])
        self.terrain_geoms = tuple(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == 0 and self._collision_enabled(g)
        )
        robot = tuple(
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) != 0 and self._collision_enabled(g)
        )
        wheel_bodies = {self.model.body(name).id for name in ("frontwheel", "rearwheel")}
        self.wheel_geoms = tuple(g for g in robot if int(self.model.geom_bodyid[g]) in wheel_bodies)
        self.nonwheel_geoms = tuple(g for g in robot if g not in self.wheel_geoms)
        self.robot_geoms = robot
        if not self.terrain_geoms or not self.wheel_geoms or not self.nonwheel_geoms:
            raise ValueError("Authoritative XML lacks required collision-enabled terrain/robot geoms")

    def _collision_enabled(self, geom: int) -> bool:
        return bool(int(self.model.geom_contype[geom]) or int(self.model.geom_conaffinity[geom]))

    def _forward(self, qpos: np.ndarray, qvel: np.ndarray | None, ctrl: np.ndarray | None) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0 if qvel is None else qvel
        if self.model.nu:
            self.data.ctrl[:] = 0 if ctrl is None else ctrl
        mujoco.mj_forward(self.model, self.data)

    def _minimum_distance(self, geoms: tuple[int, ...]) -> float:
        distmax = self.max_correction + self.margin + 2.0
        return min(
            float(mujoco.mj_geomDistance(self.model, self.data, robot, terrain, distmax, None))
            for robot in geoms for terrain in self.terrain_geoms
        )

    def _measure(self, qpos: np.ndarray, qvel: np.ndarray | None, ctrl: np.ndarray | None) -> dict:
        self._forward(qpos, qvel, ctrl)
        robot = set(self.robot_geoms); terrain = set(self.terrain_geoms)
        contacts = sum(
            1 for con in self.data.contact
            if ((int(con.geom1) in robot and int(con.geom2) in terrain)
                or (int(con.geom2) in robot and int(con.geom1) in terrain))
        )
        wheel = self._minimum_distance(self.wheel_geoms)
        nonwheel = self._minimum_distance(self.nonwheel_geoms)
        return {
            "clearance": min(wheel, nonwheel), "wheel": wheel, "nonwheel": nonwheel,
            "contacts": contacts,
            "com": np.asarray(self.data.subtree_com[self.root_body_id], np.float64).copy(),
        }

    def solve(self, qpos, qvel=None, ctrl=None, *, com_z_tolerance: float | None = None) -> ClearanceResult:
        qpos = np.asarray(qpos, np.float64).copy()
        qvel = None if qvel is None else np.asarray(qvel, np.float64)
        ctrl = None if ctrl is None else np.asarray(ctrl, np.float64)
        if not all(np.isfinite(x).all() for x in (qpos, qvel if qvel is not None else qpos[:0], ctrl if ctrl is not None else qpos[:0])):
            nan = np.full(3, np.nan)
            return ClearanceResult(False, "nonfinite", qpos, 0.0, nan, nan, np.nan, np.nan, np.nan, 0)
        original = self._measure(qpos, qvel, ctrl)

        def legal(measure):
            return measure["contacts"] == 0 and measure["wheel"] >= self.margin and measure["nonwheel"] >= self.margin

        if legal(original):
            return self._result(True, "already_clear", qpos, 0.0, original, original)

        low, high = 0.0, min(self.max_correction, max(self.tolerance, self.margin - original["clearance"]))
        corrected = None
        while True:
            trial = qpos.copy(); trial[self.root_z_index] += high
            measured = self._measure(trial, qvel, ctrl)
            if legal(measured):
                corrected = measured
                break
            if high >= self.max_correction:
                return self._result(False, "max_correction_exceeded", trial, high, original, measured)
            low = high
            high = min(self.max_correction, max(high * 2.0, high + self.tolerance))

        for _ in range(self.max_iterations):
            if high - low <= self.tolerance:
                break
            mid = 0.5 * (low + high)
            trial = qpos.copy(); trial[self.root_z_index] += mid
            measured = self._measure(trial, qvel, ctrl)
            if legal(measured):
                high, corrected = mid, measured
            else:
                low = mid
        result_qpos = qpos.copy(); result_qpos[self.root_z_index] += high
        corrected = self._measure(result_qpos, qvel, ctrl)
        if not legal(corrected):
            return self._result(False, "clearance_unsatisfied", result_qpos, high, original, corrected)
        if com_z_tolerance is not None and abs(float(corrected["com"][2] - original["com"][2])) > float(com_z_tolerance):
            return self._result(False, "reference_envelope_exceeded", result_qpos, high, original, corrected)
        return self._result(True, "corrected", result_qpos, high, original, corrected)

    @staticmethod
    def _result(accepted, reason, qpos, shift, original, corrected):
        return ClearanceResult(
            bool(accepted), str(reason), np.asarray(qpos, np.float64).copy(), float(shift),
            np.asarray(original["com"], np.float64), np.asarray(corrected["com"], np.float64),
            float(corrected["clearance"]), float(corrected["wheel"]), float(corrected["nonwheel"]),
            int(corrected["contacts"]),
        )
