"""Canonical next-stage reachability semantics for the RA-L pipeline.

Ascent, Apex, and Descent are event-aligned substages of the existing Flight
oracle phase.  This module deliberately does not mutate the environment phase
machine and does not use rollout success labels to tune entry thresholds.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

STAGES = ("takeoff", "ascent", "apex", "descent", "landing", "stable")
NEXT_STAGE = dict(zip(STAGES[:-1], STAGES[1:]))
CANONICAL_PHASE = {
    "takeoff": "takeoff", "ascent": "flight", "apex": "flight",
    "descent": "flight", "landing": "landing", "stable": "landing",
}
PROTOCOL_VERSION = "stage_next_entry_v1"
LABEL_SCHEMA_VERSION = "stage_reachability_label_v1"


@dataclass(frozen=True)
class StageEntryThresholds:
    """Frozen physical thresholds and their declared provenance."""

    takeoff_min_vz: float
    max_roll_rad: float
    max_pitch_rad: float
    max_angular_speed: float
    apex_min_height: float
    apex_max_height: float
    apex_max_abs_vz: float
    descent_max_vz: float
    landing_min_x: float
    landing_max_x: float
    landing_max_abs_y: float
    landing_max_abs_vz: float
    recovery_hold_steps: int
    recovery_min_vx: float
    source: str = "authoritative XML + fixed termination gates + reference envelope v1"


def thresholds_from_config(cfg: Any) -> StageEntryThresholds:
    # Apex height is the fixed reference maximum 0.55155 m with a symmetric
    # 0.15 m physical window.  This is declared before reachability labels are
    # inspected and is intentionally broader than point tracking.
    reference_apex_z = 0.5515475838251305
    return StageEntryThresholds(
        takeoff_min_vz=float(cfg.takeoff_liftoff_vz),
        max_roll_rad=math.radians(float(cfg.max_roll_deg)),
        max_pitch_rad=math.radians(min(float(cfg.max_pitch_deg), 35.0)),
        max_angular_speed=float(cfg.recovery_max_angvel),
        apex_min_height=reference_apex_z - 0.15,
        apex_max_height=reference_apex_z + 0.15,
        apex_max_abs_vz=0.25,
        descent_max_vz=-0.05,
        landing_min_x=float(cfg.step_front_x + cfg.valid_landing_min_past_edge),
        landing_max_x=float(cfg.step_back_x - cfg.valid_landing_back_margin),
        landing_max_abs_y=float(cfg.step_half_width - cfg.landing_side_margin),
        landing_max_abs_vz=0.75,
        recovery_hold_steps=int(cfg.recovery_hold_steps),
        recovery_min_vx=float(cfg.recovery_min_vx),
    )


def protocol_payload(cfg: Any) -> dict[str, Any]:
    payload = {
        "version": PROTOCOL_VERSION,
        "stages": list(STAGES),
        "next_stage": NEXT_STAGE,
        "canonical_phase": CANONICAL_PHASE,
        "thresholds": asdict(thresholds_from_config(cfg)),
        "rules": {
            "takeoff_to_ascent": "confirmed dual-wheel airborne; upward velocity; bounded pose/rates; no illegal contact",
            "ascent_to_apex": "Flight; reference apex height window; |vz| near zero; bounded pose/rates; no illegal contact",
            "apex_to_descent": "positive-to-negative vz crossing; legal airborne state; bounded pose/rates",
            "descent_to_landing": "first valid wheel landing/support in platform support region; no body/deep/invalid contact or immediate failure",
            "landing_to_stable": "existing Landing recovery gates held for recovery_hold_steps",
        },
        "success_definition": "reach valid next-stage entry before physical failure and within the stage horizon",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["protocol_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _finite(sample: Mapping[str, Any]) -> bool:
    values = sample.get("physical_feature", sample.get("feature", []))
    return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))


def evaluate_entry(stage: str, sample: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    """Evaluate one event-aligned sample without consulting reward or policy labels."""
    stage = str(stage).lower()
    if stage not in NEXT_STAGE:
        raise ValueError(f"No successor entry for stage {stage!r}")
    t = thresholds_from_config(cfg)
    f = np.asarray(sample.get("physical_feature", sample.get("feature", [])), np.float64)
    if f.shape != (16,):
        return {"valid": False, "stage": stage, "next_stage": NEXT_STAGE[stage], "reasons": ["feature_shape"]}
    x, y, z, roll, pitch, _yaw, vx, _vy, vz, wx, wy, wz = f[:12]
    phase = str(sample.get("canonical_phase", sample.get("source_phase", ""))).lower()
    illegal = bool(sample.get("prohibited_contact", False) or sample.get("body_terrain_contact", False)
                   or sample.get("deep_penetration", False) or sample.get("invalid_wheel_contact", False))
    physical_failure = bool(sample.get("physical_failure", False) or sample.get("nonfinite", False))
    bounded = abs(roll) <= t.max_roll_rad and abs(pitch) <= t.max_pitch_rad and np.linalg.norm([wx, wy, wz]) <= t.max_angular_speed
    common = _finite(sample) and not illegal and not physical_failure
    quality: dict[str, bool] = {"finite": _finite(sample), "no_illegal_contact": not illegal,
                                "no_physical_failure": not physical_failure, "pose_rates_bounded": bool(bounded)}
    if stage == "takeoff":
        quality.update(dual_wheel_airborne=bool(sample.get("dual_wheel_airborne", False)), upward=bool(vz >= t.takeoff_min_vz),
                       phase_ok=phase in ("takeoff", "flight"))
    elif stage == "ascent":
        quality.update(phase_ok=phase == "flight", apex_height=bool(t.apex_min_height <= z <= t.apex_max_height),
                       vertical_speed=bool(abs(vz) <= t.apex_max_abs_vz))
    elif stage == "apex":
        quality.update(phase_ok=phase == "flight", airborne=bool(sample.get("dual_wheel_airborne", True)),
                       vz_crossing=bool(float(sample.get("previous_vz", np.inf)) > 0.0 and vz <= t.descent_max_vz),
                       legal_height=bool(z >= t.apex_min_height))
    elif stage == "descent":
        first_valid = bool(sample.get("first_valid_landing", sample.get("had_valid_landing", False)))
        support = bool(sample.get("support", first_valid))
        quality.update(phase_ok=phase == "landing", first_valid_landing=first_valid, support=support,
                       platform_x=bool(t.landing_min_x <= x <= t.landing_max_x),
                       platform_y=bool(abs(y) <= t.landing_max_abs_y), impact_speed=bool(abs(vz) <= t.landing_max_abs_vz),
                       no_immediate_failure=not bool(sample.get("immediate_physical_failure", False)))
    else:  # Landing -> Stable
        quality.update(phase_ok=phase == "landing", support=bool(sample.get("support", False)),
                       recovery_hold=bool(int(sample.get("recovery_count", 0)) >= t.recovery_hold_steps),
                       forward_speed=bool(vx >= t.recovery_min_vx))
    valid = bool(common and bounded and all(quality.values()))
    return {"valid": valid, "stage": stage, "next_stage": NEXT_STAGE[stage],
            "entry_quality": quality, "reasons": [k for k, value in quality.items() if not value]}


def reachability_label(*, stage: str, successes: int, branches: int, branch_records: list[dict[str, Any]],
                       controller_bank_exhausted: bool = False) -> dict[str, Any]:
    """Create a soft probabilistic label; controller failure is not impossibility."""
    n, s = int(branches), int(successes)
    if n < 0 or s < 0 or s > n:
        raise ValueError("Require 0 <= successes <= branches")
    # Beta(1,1), with a normal approximation interval retained for a SciPy-free
    # portable schema. Formal Tube certification continues to use exact Beta.
    mean = (s + 1.0) / (n + 2.0)
    variance = (s + 1.0) * (n - s + 1.0) / ((n + 2.0) ** 2 * (n + 3.0))
    half = 1.645 * math.sqrt(variance)
    lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    if n < 4:
        label = "unknown"
    elif lo >= 0.70:
        label = "high_confidence_positive"
    elif hi <= 0.20 or (s == 0 and n >= 8):
        label = "negative_under_current_controller_bank" if controller_bank_exhausted else "unknown"
    elif hi - lo <= 0.45:
        label = "boundary"
    else:
        label = "unknown"
    return {"schema_version": LABEL_SCHEMA_VERSION, "stage": stage, "next_stage": NEXT_STAGE[stage],
            "n": n, "s": s, "p_next": (s / n if n else None),
            "posterior": {"alpha": s + 1, "beta": n - s + 1, "mean": mean, "lower_05_approx": lo, "upper_95_approx": hi},
            "label": label, "branches": branch_records}


def next_branch_budget(label: Mapping[str, Any]) -> int:
    """Adaptive 4 -> 8 -> 16 -> 32 evidence funnel."""
    n = int(label.get("n", 0)); p = label.get("p_next")
    if n < 4:
        return 4
    if n < 8 and (p is None or 0.0 < float(p) < 1.0):
        return 8
    if n < 16 and (p is None or 0.15 < float(p) < 0.85):
        return 16
    if n < 32 and (p is None or 0.30 < float(p) < 0.70):
        return 32
    return n
