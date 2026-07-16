"""Takeoff reward profiles for OrangeBike DVGC.

The environment core computes state transitions and signals; this module turns
those signals into a reward and task-specific terminal gates.  It intentionally
contains only clean, named task profiles instead of historical experiment code.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jp


def compute_descent_local_reward(
    *, cfg, pitch, pitch_rate, roll, roll_rate, vx, previous_distance,
    current_distance, action, previous_action, chain, hard_failure,
) -> Dict[str, jp.ndarray]:
    """Low-budget potential shaping for pre-Tube local descent bootstrap."""
    pitch_z = (pitch - float(cfg.descent_local_pitch_center)) / float(cfg.descent_local_pitch_scale)
    pitch_rate_z = (pitch_rate - float(cfg.descent_local_pitch_rate_center)) / float(cfg.descent_local_pitch_rate_scale)
    roll_z = (roll - float(cfg.descent_local_roll_center)) / float(cfg.descent_local_roll_scale)
    roll_rate_z = (roll_rate - float(cfg.descent_local_roll_rate_center)) / float(cfg.descent_local_roll_rate_scale)
    speed_z = (vx - float(cfg.descent_local_vx_center)) / float(cfg.descent_local_vx_scale)
    pitch_posture = 0.025 * jp.exp(-0.5 * pitch_z ** 2)
    pitch_rate_penalty = 0.055 * jp.minimum(pitch_rate_z ** 2, 4.0)
    soft = float(cfg.descent_local_pitch_soft_limit_deg) * jp.pi / 180.0
    hard = float(cfg.max_pitch_deg) * jp.pi / 180.0
    pitch_excess = jp.maximum(0.0, jp.abs(pitch) - soft) / jp.maximum(hard - soft, 1e-6)
    pitch_soft_penalty = 0.12 * pitch_excess ** 2
    roll_score = 0.0075 * jp.exp(-0.5 * roll_z ** 2)
    roll_penalty = 0.0125 * jp.minimum(jp.maximum(jp.abs(roll_z) - 2.0, 0.0) ** 2 + 0.25 * roll_rate_z ** 2, 4.0)
    speed_score = 0.0075 * jp.exp(-0.5 * speed_z ** 2)
    progress = 0.10 * jp.clip(
        (previous_distance - current_distance) / float(cfg.descent_local_progress_scale), -1.0, 1.0
    )
    survival = jp.asarray(float(cfg.descent_local_survival_reward), jp.float32)
    action_difference_penalty = 0.008 * jp.mean((action - previous_action) ** 2)
    action_magnitude_penalty = 0.004 * jp.mean(jp.abs(action) ** 1.5)
    shaping = jp.clip(
        survival + pitch_posture + roll_score + speed_score + progress
        - pitch_rate_penalty - pitch_soft_penalty - roll_penalty
        - action_difference_penalty - action_magnitude_penalty,
        float(cfg.descent_local_shaping_clip_min),
        float(cfg.descent_local_shaping_clip_max),
    )
    chain_reward = float(cfg.coeff_chain_event) * chain.astype(jp.float32)
    failure_penalty = float(cfg.descent_local_terminal_failure_penalty) * hard_failure.astype(jp.float32)
    return {
        "reward": shaping + chain_reward - failure_penalty,
        "shaping": shaping,
        "pitch_posture": pitch_posture,
        "pitch_rate_penalty": pitch_rate_penalty,
        "pitch_soft_penalty": pitch_soft_penalty,
        "roll_score": roll_score,
        "roll_penalty": roll_penalty,
        "speed_score": speed_score,
        "progress": progress,
        "survival": survival,
        "action_difference_penalty": action_difference_penalty,
        "action_magnitude_penalty": action_magnitude_penalty,
        "chain": chain_reward,
        "failure_penalty": failure_penalty,
    }


def _clip01(x: jp.ndarray) -> jp.ndarray:
    return jp.clip(x, 0.0, 1.0)


def compute_takeoff_reward_profile(
    *,
    cfg,
    signals: Dict[str, jp.ndarray],
    action: jp.ndarray,
    phase0: jp.ndarray,
    phase1: jp.ndarray,
    dual_wheel_liftoff_seen: jp.ndarray,
    positive_pitch_count: jp.ndarray,
    wheelie_count: jp.ndarray,
) -> Dict[str, jp.ndarray]:
    """Returns reward terms and clean takeoff terminal gates."""
    profile = str(getattr(cfg, "takeoff_reward_profile", "minimal_dual_wheel")).lower()
    takeoff_active = (phase0 == 1) | (phase1 == 1)  # STAGE_ID["takeoff"] is static in dvgc_common.
    takeoff_gate = takeoff_active.astype(jp.float32)

    if profile == "debug_no_reward":
        zero = jp.asarray(0.0, dtype=jp.float32)
        return {
            "reward": zero,
            "dual_wheel_liftoff_seen": dual_wheel_liftoff_seen,
            "positive_pitch_count": positive_pitch_count,
            "wheelie_count": wheelie_count,
            "positive_pitch_failure": jp.asarray(False),
            "wheelie_failure": jp.asarray(False),
            "missed_liftoff_deadline": jp.asarray(False),
            "missed_wheel_clearance_deadline": jp.asarray(False),
            "terms": {
                "dual_wheel_vz": zero,
                "dual_wheel_height": zero,
                "min_tire_clearance": zero,
                "knee_useful_motion": zero,
                "positive_pitch_penalty": zero,
                "wheelie_penalty": zero,
                "wheel_unsync_penalty": zero,
                "action_mag_penalty": zero,
            },
        }
    if profile not in {"minimal_dual_wheel", "com_guided", "front_follow_dual_wheel"}:
        raise ValueError(
            f"Unsupported takeoff_reward_profile={profile!r}; "
            "expected minimal_dual_wheel, front_follow_dual_wheel, com_guided, or debug_no_reward."
        )

    front_vz_score = _clip01(signals["front_tire_vz"] / 1.20)
    rear_vz_score = _clip01(signals["rear_tire_vz"] / 1.20)
    dual_vz_score = jp.minimum(front_vz_score, rear_vz_score)

    front_h_score = _clip01(signals["front_tire_bottom_z"] / 0.18)
    rear_h_score = _clip01(signals["rear_tire_bottom_z"] / 0.18)
    dual_h_score = jp.minimum(front_h_score, rear_h_score)

    clearance_score = _clip01((signals["min_tire_bottom_z"] - 0.0) / (float(cfg.step_top_z) + float(cfg.takeoff_step_clearance_margin)))
    no_bad_pitch = (~signals["positive_pitch_bad"]).astype(jp.float32)
    wheel_progress = jp.maximum(dual_vz_score, dual_h_score)
    knee_motion_score = _clip01(jp.abs(signals["knee_vel"]) / 2.0)

    r_dual_vz = float(cfg.takeoff_reward_scale_dual_wheel_vz) * takeoff_gate * no_bad_pitch * dual_vz_score
    r_dual_height = float(cfg.takeoff_reward_scale_dual_wheel_height) * takeoff_gate * no_bad_pitch * dual_h_score
    r_clearance = float(cfg.takeoff_reward_scale_dual_wheel_clearance) * takeoff_gate * no_bad_pitch * clearance_score
    r_knee = float(cfg.takeoff_reward_scale_knee_useful_motion) * takeoff_gate * no_bad_pitch * knee_motion_score * wheel_progress

    positive_excess = jp.maximum(0.0, signals["pitch_signed_deg"] - float(cfg.takeoff_positive_pitch_soft_deg))
    p_positive_pitch = float(cfg.takeoff_penalty_scale_positive_pitch) * takeoff_gate * (positive_excess / 10.0) ** 2
    p_wheelie = float(cfg.takeoff_penalty_scale_wheelie) * takeoff_gate * signals["wheelie_detected"].astype(jp.float32)
    wheel_unsync = jp.maximum(0.0, signals["wheel_height_diff"] - float(cfg.takeoff_sync_height_diff_max))
    p_wheel_unsync = float(cfg.takeoff_penalty_scale_wheel_unsync) * takeoff_gate * wheel_unsync ** 2
    p_action_mag = float(cfg.coeff_action_mag) * takeoff_gate * jp.mean(action * action)

    dual_wheel_liftoff_seen = dual_wheel_liftoff_seen | signals["dual_wheel_liftoff"]
    positive_pitch_count = jp.where(takeoff_active & signals["positive_pitch_hard"], positive_pitch_count + 1, 0)
    wheelie_count = jp.where(takeoff_active & signals["wheelie_detected"], wheelie_count + 1, 0)

    positive_pitch_failure = positive_pitch_count >= int(cfg.takeoff_positive_pitch_terminate_ticks)
    wheelie_failure = wheelie_count >= int(cfg.takeoff_wheelie_terminate_ticks)
    missed_liftoff_deadline = (
        takeoff_active
        & (signals["frontmost_x"] >= signals["liftoff_deadline_x"])
        & (~dual_wheel_liftoff_seen)
    )
    missed_wheel_clearance_deadline = (
        takeoff_active
        & (signals["frontmost_x"] >= signals["clearance_deadline_x"])
        & (~signals["wheel_clearance_ready"])
    )

    # ── front_follow_dual_wheel extra terms ──
    if profile == "front_follow_dual_wheel":
        ff_front_follow_coeff = float(getattr(cfg, "takeoff_front_follow_reward_coeff", 3.0))
        ff_rear_over_penalty_coeff = float(getattr(cfg, "takeoff_rear_over_front_penalty_coeff", 6.0))
        ff_height_diff_max = float(getattr(cfg, "takeoff_front_follow_height_diff_max", 0.08))
        ff_min_tire_coeff = float(getattr(cfg, "takeoff_front_follow_min_tire_coeff", 2.0))
        front_follow_score = _clip01(signals["front_tire_bottom_z"] / jp.maximum(signals["rear_tire_bottom_z"], 1e-6))
        r_front_follow = ff_front_follow_coeff * takeoff_gate * no_bad_pitch * front_follow_score
        rear_over_front = jp.maximum(0.0, signals["rear_tire_bottom_z"] - signals["front_tire_bottom_z"] - ff_height_diff_max)
        p_rear_over_front = ff_rear_over_penalty_coeff * takeoff_gate * rear_over_front ** 2
        min_tire_score = _clip01(signals["min_tire_bottom_z"] / 0.18)
        r_min_tire_bonus = ff_min_tire_coeff * takeoff_gate * no_bad_pitch * min_tire_score
    else:
        r_front_follow = jp.asarray(0.0, dtype=jp.float32)
        p_rear_over_front = jp.asarray(0.0, dtype=jp.float32)
        r_min_tire_bonus = jp.asarray(0.0, dtype=jp.float32)

    terminal_penalty = float(getattr(cfg, "takeoff_terminal_failure_penalty", 25.0)) * (
        positive_pitch_failure
        | wheelie_failure
        | missed_liftoff_deadline
        | missed_wheel_clearance_deadline
    ).astype(jp.float32)

    reward = (
        r_dual_vz
        + r_dual_height
        + r_clearance
        + r_knee
        + r_front_follow
        + r_min_tire_bonus
        - p_positive_pitch
        - p_wheelie
        - p_wheel_unsync
        - p_rear_over_front
        - p_action_mag
        - terminal_penalty
    )
    return {
        "reward": reward,
        "dual_wheel_liftoff_seen": dual_wheel_liftoff_seen,
        "positive_pitch_count": positive_pitch_count,
        "wheelie_count": wheelie_count,
        "positive_pitch_failure": positive_pitch_failure,
        "wheelie_failure": wheelie_failure,
        "missed_liftoff_deadline": missed_liftoff_deadline,
        "missed_wheel_clearance_deadline": missed_wheel_clearance_deadline,
        "terms": {
            "dual_wheel_vz": r_dual_vz,
            "dual_wheel_height": r_dual_height,
            "min_tire_clearance": r_clearance,
            "knee_useful_motion": r_knee,
            "front_follow": r_front_follow,
            "min_tire_bonus": r_min_tire_bonus,
            "positive_pitch_penalty": p_positive_pitch,
            "wheelie_penalty": p_wheelie,
            "wheel_unsync_penalty": p_wheel_unsync,
            "rear_over_front_penalty": p_rear_over_front,
            "action_mag_penalty": p_action_mag,
        },
    }
