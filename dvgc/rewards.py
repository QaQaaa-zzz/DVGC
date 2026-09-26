"""Takeoff reward profiles for OrangeBike DVGC.

The environment core computes state transitions and signals; this module turns
those signals into a reward and task-specific terminal gates.  It intentionally
contains only clean, named task profiles instead of historical experiment code.
"""
from __future__ import annotations

from typing import Dict

import jax.numpy as jp


def compute_stage_next_entry_reward(
    *, cfg, objective: str, feature: jp.ndarray, previous_feature: jp.ndarray,
    action: jp.ndarray, previous_action: jp.ndarray, next_entry: jp.ndarray,
    hard_failure: jp.ndarray, jump_latched: jp.ndarray, window_active: jp.ndarray,
    joint_energy: jp.ndarray, current_support_distance: jp.ndarray | None = None,
    previous_support_distance: jp.ndarray | None = None,
) -> Dict[str, jp.ndarray]:
    """Bounded phase objective inspired by the supplied jump reward design.

    The environment owns every latch/event; this pure function has no mutable
    Python state.  Entry success is the dominant term and all dense shaping is
    clipped so survival/height cannot substitute for the successor event.
    """
    f=feature;prev=previous_feature
    roll,pitch,yaw=f[3],f[4],f[5];vx,vz=f[6],f[8];gyro=f[9:12]
    pose=jp.exp(-((roll/jp.deg2rad(12.0))**2+(pitch/jp.deg2rad(22.0))**2))
    yaw_score=jp.exp(-((yaw/jp.deg2rad(25.0))**2))
    target_speed=jp.where(jump_latched,float(cfg.stage_postjump_target_speed),float(cfg.stage_prejump_target_speed))
    speed=jp.exp(-(((vx-target_speed)/float(cfg.forward_sigma))**2))
    angular=jp.minimum(jp.sum(gyro*gyro)/(float(cfg.recovery_max_angvel)**2),4.0)
    curriculum=jp.clip(float(cfg.stage_curriculum_scale),0.0,1.0)
    height_target=float(cfg.stage_height_target_easy)+(float(cfg.stage_height_target_hard)-float(cfg.stage_height_target_easy))*curriculum
    bounded_height=jp.clip(f[2]/jp.maximum(height_target,1e-6),0.0,1.0)
    if objective=="takeoff_to_ascent":
        progress=jp.clip(vz/max(float(cfg.takeoff_liftoff_vz),1e-6),-1.0,1.0)*window_active.astype(jp.float32)
    elif objective=="ascent_to_apex":
        progress=0.65*jp.clip((f[2]-prev[2])/0.05,-1.0,1.0)+0.35*bounded_height
    elif objective=="apex_to_descent":
        # Potential shaping cannot pay the Actor for hovering indefinitely at
        # the apex.  The frozen Descent-support matcher supplies the preferred
        # distance; this kinematic fallback is only for pure reward preflight.
        current_distance=(jp.abs(vz+0.30)/0.30+jp.abs(f[2]-float(cfg.stage_apex_target_height))/0.15
                          if current_support_distance is None else current_support_distance)
        previous_distance=(jp.abs(prev[8]+0.30)/0.30+jp.abs(prev[2]-float(cfg.stage_apex_target_height))/0.15
                           if previous_support_distance is None else previous_support_distance)
        # Tight certified entry regions can put fresh upstream states hundreds
        # of matcher radii away.  Clipping both potentials at eight erased all
        # approach/departure information there and created a positive stall
        # term.  Relative decrease is bounded, scale-aware, and exactly zero
        # when the state makes no support progress; it does not alter the gate.
        relative_decrease=(previous_distance-current_distance)/jp.maximum(previous_distance,1.0)
        progress=jp.clip(float(cfg.stage_support_relative_progress_gain)*relative_decrease,-1.0,1.0)
    elif objective=="descent_to_landing":
        progress=jp.exp(-(((vz+0.45)/0.35)**2))
    else:
        raise ValueError(f"Unsupported stage reachability objective {objective!r}")
    smooth=jp.mean((action-previous_action)**2);magnitude=jp.mean(action*action)
    dense=(float(cfg.stage_entry_survival_reward)+float(cfg.stage_entry_progress_coeff)*progress+
           float(cfg.stage_entry_pose_coeff)*pose+float(cfg.stage_entry_speed_coeff)*speed-
           float(cfg.stage_entry_yaw_coeff)*(1.0-yaw_score)-
           float(cfg.stage_entry_angular_penalty_coeff)*angular-
           float(cfg.stage_entry_action_smooth_coeff)*smooth-
           float(cfg.stage_entry_action_magnitude_coeff)*magnitude-
           float(cfg.stage_entry_joint_energy_coeff)*joint_energy)
    shaping=jp.clip(dense,float(cfg.stage_entry_shaping_clip_min),float(cfg.stage_entry_shaping_clip_max))
    event=float(cfg.stage_entry_event_reward)*next_entry.astype(jp.float32)
    angular_quality=jp.exp(-jp.sum(gyro*gyro)/(float(cfg.recovery_max_angvel)**2))
    handoff_quality=0.5*pose+0.5*angular_quality
    handoff_bonus=jp.where(
        objective=="takeoff_to_ascent",
        float(cfg.stage_takeoff_handoff_quality_coeff)*next_entry.astype(jp.float32)*handoff_quality,
        jp.asarray(0.0,jp.float32),
    )
    failure=float(cfg.stage_entry_failure_penalty)*hard_failure.astype(jp.float32)
    return {"reward":shaping+event+handoff_bonus-failure,"shaping":shaping,"event":event,
            "handoff_quality":handoff_quality,"handoff_bonus":handoff_bonus,"failure_penalty":failure,
            "progress":progress,"pose":pose,"speed":speed,"angular_penalty":angular,
            "yaw_score":yaw_score,"bounded_height":bounded_height,"joint_energy_penalty":joint_energy,
            "action_smooth_penalty":smooth,"action_magnitude_penalty":magnitude}


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


def compute_descent_rsi_pilot_reward(
    *, cfg, roll, pitch, gyro, vz, action, previous_action, hard_failure,
) -> Dict[str, jp.ndarray]:
    """Matcher-free bounded reward for the one-shot Descent RSI pilot.

    This adapter measures local physical control only.  It deliberately has
    no old-support distance, Chain, Landing, Final-Recovery, or Tube term.
    """
    roll_score = float(cfg.descent_rsi_pilot_roll) * jp.exp(
        -((roll / jp.deg2rad(15.0)) ** 2)
    )
    pitch_score = float(cfg.descent_rsi_pilot_pitch) * jp.exp(
        -((pitch / jp.deg2rad(25.0)) ** 2)
    )
    angular_score = float(cfg.descent_rsi_pilot_angular) * jp.exp(
        -((jp.linalg.norm(gyro) / max(float(cfg.recovery_max_angvel), 1e-6)) ** 2)
    )
    descent_speed_score = float(cfg.descent_rsi_pilot_vz) * jp.exp(
        -(((vz + 0.50) / 0.50) ** 2)
    )
    survival = jp.asarray(float(cfg.descent_rsi_pilot_survival), jp.float32)
    action_smooth_penalty = float(cfg.descent_rsi_pilot_action_smooth) * jp.mean(
        (action - previous_action) ** 2
    )
    action_magnitude_penalty = float(cfg.descent_rsi_pilot_action_magnitude) * jp.mean(
        action ** 2
    )
    shaping = jp.clip(
        survival + roll_score + pitch_score + angular_score + descent_speed_score
        - action_smooth_penalty - action_magnitude_penalty,
        float(cfg.descent_rsi_pilot_shaping_clip_min),
        float(cfg.descent_rsi_pilot_shaping_clip_max),
    )
    failure_penalty = float(cfg.descent_rsi_pilot_failure_penalty) * hard_failure.astype(jp.float32)
    return {
        "reward": shaping - failure_penalty,
        "shaping": shaping,
        "survival": survival,
        "roll_score": roll_score,
        "pitch_score": pitch_score,
        "angular_score": angular_score,
        "descent_speed_score": descent_speed_score,
        "action_smooth_penalty": action_smooth_penalty,
        "action_magnitude_penalty": action_magnitude_penalty,
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
    jump_latched: jp.ndarray,
    dual_wheel_liftoff_seen: jp.ndarray,
    positive_pitch_count: jp.ndarray,
    wheelie_count: jp.ndarray,
) -> Dict[str, jp.ndarray]:
    """Returns reward terms and clean takeoff terminal gates."""
    profile = str(getattr(cfg, "takeoff_reward_profile", "minimal_dual_wheel")).lower()
    takeoff_active = ((phase0 == 1) | (phase1 == 1)) & jp.asarray(
        jump_latched, dtype=bool
    )  # STAGE_ID["takeoff"] is static in dvgc_common.
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

    dual_wheel_liftoff_seen = dual_wheel_liftoff_seen | (
        takeoff_active & signals["dual_wheel_liftoff"]
    )
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
