"""Single-lineage MJX stage controller switching and trajectory I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from .config import STAGE_ID


CONTINUOUS_STAGES = (
    "approach", "takeoff", "ascent", "apex", "descent", "landing"
)


def _scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)))


@dataclass
class ContinuousPhaseTracker:
    """Monotonic event tracker; it never writes or restores simulator state."""

    stage: str = "approach"
    previous_vz: float = 0.0
    ascent_positive_seen: bool = False
    switches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def index(self) -> int:
        return CONTINUOUS_STAGES.index(self.stage)

    def observe(
        self, state: Any, *, descent_entry: bool, tick: int,
        physical_descent: bool = False,
    ) -> str:
        phase = int(_scalar(state.info["phase"]))
        had_airborne = bool(_scalar(state.info["had_airborne"]))
        had_landing = bool(_scalar(state.info["had_valid_landing"]))
        vz = _scalar(state.data.qvel[2])
        target = self.stage
        if self.stage == "approach" and phase >= STAGE_ID["takeoff"]:
            target = "takeoff"
        elif (
            self.stage == "takeoff"
            and phase >= STAGE_ID["flight"]
            and had_airborne
        ):
            target = "ascent"
            self.ascent_positive_seen = vz > 0.0
        elif self.stage == "ascent":
            self.ascent_positive_seen |= vz > 0.0
            if self.ascent_positive_seen and self.previous_vz > 0.0 and vz <= 0.0:
                target = "apex"
        elif self.stage == "apex" and (descent_entry or physical_descent):
            target = "descent"
        elif (
            self.stage == "descent"
            and phase >= STAGE_ID["landing"]
            and had_landing
        ):
            target = "landing"
        if CONTINUOUS_STAGES.index(target) < self.index:
            raise RuntimeError(f"illegal continuous phase regression {self.stage}->{target}")
        if target != self.stage:
            self.switches.append({
                "tick": int(tick),
                "from": self.stage,
                "to": target,
                "env_phase": phase,
                "vz": vz,
                "descent_support_entry": bool(descent_entry),
                "physical_descent_event": bool(physical_descent),
            })
            self.stage = target
        self.previous_vz = vz
        return self.stage


class DescentSupportMatcher:
    """Current immutable stage-support matcher without environment mutation."""

    def __init__(self, env: Any):
        self.enabled = bool(env._stage_support_enabled)
        self.features = np.asarray(env._stage_support_features)
        self.center = np.asarray(env._stage_support_center)
        self.scale = np.asarray(env._stage_support_scale)
        self.radius = float(env._stage_support_radius)
        self.x_bounds = tuple(map(float, env._stage_support_x_bounds))
        self.z_bounds = tuple(map(float, env._stage_support_z_bounds))
        self.vz_bounds = tuple(map(float, env._stage_descent_vz_bounds))
        self.roll_rate_max = float(env._stage_roll_rate_max)
        self.pitch_rate_max = float(env._stage_pitch_rate_max)
        self.env = env

    def evaluate(self, state: Any, *, apex_crossed: bool) -> tuple[bool, float]:
        feature = np.asarray(jax.device_get(
            self.env._physical_feature(state.data)
        ), np.float32)
        distance = float(np.min(np.linalg.norm(
            self.features - (feature - self.center)[None, :] / self.scale[None, :],
            axis=1,
        )))
        x, z, vz = float(feature[0]), float(feature[2]), float(feature[8])
        wx, wy = abs(float(feature[9])), abs(float(feature[10]))
        valid = (
            self.enabled
            and apex_crossed
            and self.x_bounds[0] <= x <= self.x_bounds[1]
            and self.z_bounds[0] <= z <= self.z_bounds[1]
            and self.vz_bounds[0] <= vz <= self.vz_bounds[1]
            and wx <= self.roll_rate_max
            and wy <= self.pitch_rate_max
            and distance <= self.radius
        )
        return bool(valid), distance


def trajectory_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_trajectory(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(target, **normalized)
    return trajectory_sha256(normalized)


def load_trajectory(path: str | Path, expected_sha256: str | None = None):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if not arrays or len({value.shape[0] for value in arrays.values()}) != 1:
        raise ValueError("continuous trajectory fields have inconsistent lengths")
    digest = trajectory_sha256(arrays)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("continuous trajectory hash mismatch")
    return arrays, digest
