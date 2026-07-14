"""Reference-trajectory audit and broad phase envelope extraction.

The trajectory is a proposal/diagnostic source only.  No pointwise trajectory
tracking reward is produced by this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "time", "pos_x", "pos_z", "vel_x", "vel_z", "roll_angle", "pitch_angle", "yaw_angle",
    "hip_position", "knee_position", "hip_velocity", "knee_velocity",
    "action_hip", "action_knee", "action_steering", "action_rearwheel",
}


@dataclass(frozen=True)
class ReferenceAnchors:
    approach_end: int
    takeoff_end: int
    apex: int
    landing_start: int
    recovery_start: int
    recovery_end: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


class ReferenceTrajectory:
    def __init__(self, frame: pd.DataFrame):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")
        self.df = frame.copy().reset_index(drop=True)
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceTrajectory":
        return cls(pd.read_csv(path))

    def _validate(self) -> None:
        if len(self.df) < 20:
            raise ValueError("Reference trajectory is too short")
        t = self.df["time"].to_numpy(float)
        dt = np.diff(t)
        if not np.all(dt > 0):
            raise ValueError("Reference time must be strictly increasing")
        # Supplied data uses degrees.  Radian data would stay far below 2*pi.
        max_angle = float(self.df[["roll_angle", "pitch_angle", "yaw_angle"]].abs().to_numpy().max())
        self.angle_unit = "degree" if max_angle > 2 * np.pi else "radian"
        self.dt_median = float(np.median(dt))
        self.dt_jitter = float(np.max(np.abs(dt - self.dt_median)))
        if float(self.df["hip_position"].min()) < -1.35 or float(self.df["hip_position"].max()) > 0.55:
            raise ValueError("Reference hip range conflicts with the supplied XML beyond solver tolerance")
        if float(self.df["knee_position"].min()) < -1.51 or float(self.df["knee_position"].max()) > 2.60:
            raise ValueError("Reference knee range conflicts with the supplied XML")

    def anchors(self, *, step_front_x: float = 3.60, step_top_z: float = 0.16, base_ground_z: float = 0.25) -> ReferenceAnchors:
        d = self.df
        n = len(d)
        # Approach ends at the first clear upward launch response.  This is
        # intentionally more conservative than merely entering the x-window.
        launch_candidates = np.where(
            (d["pos_x"].to_numpy() >= step_front_x - 1.30)
            & (d["vel_z"].to_numpy() >= 0.45)
        )[0]
        approach_end = int(launch_candidates[0]) if len(launch_candidates) else int(np.argmax(d["vel_z"].to_numpy()))
        # Takeoff ends once the body has gained clear height and positive vz;
        # this approximates dual-wheel airborne because the CSV has no contacts.
        takeoff_candidates = np.where(
            (np.arange(n) >= approach_end)
            & (d["pos_z"].to_numpy() >= base_ground_z + 0.055)
            & (d["vel_z"].to_numpy() >= 0.25)
        )[0]
        takeoff_end = int(takeoff_candidates[0]) if len(takeoff_candidates) else approach_end
        apex = int(d["pos_z"].idxmax())
        # First stable platform-height segment after descent.  Require several
        # consecutive low-|vz| samples to avoid labeling an airborne crossing.
        expected_platform_z = base_ground_z + step_top_z
        landing_start = min(n - 1, max(apex + 1, takeoff_end + 1))
        stable = (
            (d["pos_x"].to_numpy() >= step_front_x + 0.20)
            & (np.abs(d["pos_z"].to_numpy() - expected_platform_z) <= 0.09)
            & (np.abs(d["vel_z"].to_numpy()) <= 0.35)
        )
        width = max(3, int(round(0.020 / self.dt_median)))
        for i in range(landing_start, n - width):
            if bool(np.all(stable[i:i + width])):
                landing_start = i
                break
        recovery_start = min(n - 1, landing_start + max(1, int(round(0.10 / self.dt_median))))
        # Stop the usable reference before the later roll failure.  This keeps
        # the successful jump/recovery envelope but does not pretend the final
        # fall is a desired reference.
        roll = np.abs(d["roll_angle"].to_numpy())
        fail = np.where((np.arange(n) > recovery_start) & (roll >= 14.0))[0]
        recovery_end = int(fail[0] - 1) if len(fail) else n - 1
        return ReferenceAnchors(approach_end, takeoff_end, apex, landing_start, recovery_start, recovery_end)

    def phase_slices(self, anchors: ReferenceAnchors) -> dict[str, slice]:
        a = anchors
        return {
            "approach": slice(0, a.approach_end + 1),
            "takeoff": slice(a.approach_end, a.takeoff_end + 1),
            "flight_ascent": slice(a.takeoff_end, a.apex + 1),
            "flight_descent": slice(a.apex, a.landing_start + 1),
            "landing": slice(a.landing_start, a.recovery_start + 1),
            "recovery": slice(a.recovery_start, a.recovery_end + 1),
        }

    def envelope(self, anchors: ReferenceAnchors) -> pd.DataFrame:
        columns = [
            "pos_x", "pos_z", "vel_x", "vel_z", "roll_angle", "pitch_angle", "yaw_angle",
            "hip_position", "knee_position", "hip_velocity", "knee_velocity",
            "action_hip", "action_knee", "action_steering", "action_rearwheel",
        ]
        rows: list[dict[str, Any]] = []
        for phase, sl in self.phase_slices(anchors).items():
            part = self.df.iloc[sl]
            row: dict[str, Any] = {"phase": phase, "start_index": int(part.index.min()), "end_index": int(part.index.max()), "count": len(part)}
            for col in columns:
                values = part[col].to_numpy(float)
                row[f"{col}_p05"] = float(np.quantile(values, 0.05))
                row[f"{col}_p50"] = float(np.quantile(values, 0.50))
                row[f"{col}_p95"] = float(np.quantile(values, 0.95))
                row[f"{col}_min"] = float(values.min())
                row[f"{col}_max"] = float(values.max())
            rows.append(row)
        return pd.DataFrame(rows)

    def report(self, anchors: ReferenceAnchors) -> dict[str, Any]:
        d = self.df
        return {
            "rows": len(d),
            "duration_s": float(d["time"].iloc[-1] - d["time"].iloc[0]),
            "median_dt_s": self.dt_median,
            "max_dt_jitter_s": self.dt_jitter,
            "angle_unit": self.angle_unit,
            "anchors": anchors.as_dict(),
            "anchor_times_s": {k: float(d.loc[v, "time"]) for k, v in anchors.as_dict().items()},
            "anchor_x_m": {k: float(d.loc[v, "pos_x"]) for k, v in anchors.as_dict().items()},
            "max_z_m": float(d["pos_z"].max()),
            "max_vz_mps": float(d["vel_z"].max()),
            "hip_range": [float(d["hip_position"].min()), float(d["hip_position"].max())],
            "knee_range": [float(d["knee_position"].min()), float(d["knee_position"].max())],
            "note": "The post-recovery roll failure is excluded from the desired envelope.",
        }

    def save_analysis(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        anchors = self.anchors()
        report_path = out / "reference_report.json"
        envelope_path = out / "reference_phase_envelopes.csv"
        report_path.write_text(json.dumps(self.report(anchors), indent=2), encoding="utf-8")
        self.envelope(anchors).to_csv(envelope_path, index=False)
        return report_path, envelope_path
