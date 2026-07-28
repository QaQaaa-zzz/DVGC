"""Single source of truth for the cleaned DVGC project configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from ml_collections import config_dict
except ModuleNotFoundError:  # lightweight fallback for offline audits/tests
    class _Config(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
        def __setattr__(self, name, value):
            self[name] = value
        def to_dict(self):
            return dict(self)
    class _ConfigModule:
        ConfigDict = _Config
        @staticmethod
        def create(**kwargs):
            return _Config(kwargs)
    config_dict = _ConfigModule()

PHASES = ("approach", "takeoff", "flight", "landing")
STAGE_ID = {name: i for i, name in enumerate(PHASES)}
ID_STAGE = {i: name for name, i in STAGE_ID.items()}
ACTION_MAPPING_VERSION = "steer_drive_hip_knee.incremental_positive_flexion.v2"
POLICY_NETWORK_VERSION = "bounded_neutral_tanh.v2"
SNAPSHOT_SCHEMA = "dvgc_physical_policy_state_v4_timing_explicit"
AUTHORITATIVE_XML_PATH = "assets/orange_bike_4kg_horizontal.xml"
AUTHORITATIVE_XML_SHA256 = "d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c"


def default_config() -> config_dict.ConfigDict:
    """Returns the only supported environment configuration.

    The reference trajectory is used to set broad candidate envelopes only.  It
    is never used as a pointwise tracking target or as the formal success label.
    """
    return config_dict.create(
        # Simulation and files.
        xml_path=AUTHORITATIVE_XML_PATH,
        ctrl_dt=0.020,
        sim_dt=0.002,
        episode_length=750,
        action_repeat=1,
        impl="warp",
        # MJX-Warp's tiled Newton factorization is not repeatable for the
        # contact-sensitive single-world lineage used by the continuous
        # pipeline.  CG is supported by the same MJX-Warp runtime and keeps
        # identical source/action/seed replays bit-exact.
        mjx_solver="CG",
        contact_mode="imu",
        naconmax=1024,
        naccdmax=1024,
        njmax=512,
        root_body_name="base_link",
        # Current backward-bootstrap stage.
        training_stage="landing",
        use_bank_resets=True,
        downstream_bank_path="",
        # Stage experts terminate successfully at their immutable downstream
        # canonical entry.  Composite Final-Recovery evaluation keeps this off.
        expert_chain_termination=False,
        descent_entry_minimum_calibration_precision=0.95,
        descent_entry_scale_floors=[0.10,0.05,0.05,0.03,0.03,0.03,0.20,0.10,0.20,0.20,0.20,0.20,0.03,0.05,0.05,0.50],
        descent_local_candidate_seed=7400000,
        descent_local_max_children_per_parent=4,
        descent_local_target_children=60,
        descent_local_proposal_budget=1200,
        descent_local_validation_steps=5,
        descent_local_normalized_dedup_distance=0.15,
        descent_local_safe_covariance_scale=0.08,
        descent_local_boundary_covariance_scale=0.18,
        descent_local_anchor_covariance_scale=0.12,
        descent_local_reset_safe_mass=0.35,
        descent_local_reset_boundary_mass=0.45,
        descent_local_reset_anchor_mass=0.20,
        stable_construction_stage_branches=32,
        stable_construction_adaptive_max_branches=32,
        stable_construction_near_safe_lcb_margin=0.05,
        stable_construction_min_safe_states=4,
        stable_construction_min_safe_parents=2,
        descent_acquisition_max_rounds=2,
        descent_acquisition_max_proposals=64,
        descent_acquisition_middle_mass=0.45,
        descent_acquisition_late_mass=0.35,
        descent_acquisition_early_mass=0.20,
        discrete_tube_rsi_safe_mass=0.70,
        discrete_tube_rsi_boundary_mass=0.30,
        trajectory_mining_branches_per_state=2,
        trajectory_mining_target_snapshots=64,
        trajectory_mining_middle_mass=0.40,
        trajectory_mining_late_mass=0.40,
        trajectory_mining_early_mass=0.20,
        roll_controllability_action_delta=0.15,
        roll_controllability_horizon=10,
        roll_controllability_pulse_steps=3,
        roll_controllability_min_roll_sensitivity=0.01,
        roll_controllability_min_roll_rate_sensitivity=0.05,
        roll_controllability_min_beneficial_fraction=0.25,
        roll_controllability_max_pitch_side_effect=0.10,
        descent_local_reward_enable=False,
        # One-shot learnability adapter for the provisional Descent RSI bank.
        # It is matcher/landing independent and is never the formal JEL reward.
        descent_rsi_pilot_reward_enable=False,
        descent_rsi_pilot_survival=0.010,
        descent_rsi_pilot_roll=0.040,
        descent_rsi_pilot_pitch=0.040,
        descent_rsi_pilot_angular=0.020,
        descent_rsi_pilot_vz=0.020,
        descent_rsi_pilot_action_smooth=0.005,
        descent_rsi_pilot_action_magnitude=0.003,
        descent_rsi_pilot_failure_penalty=2.0,
        descent_rsi_pilot_shaping_clip_min=-0.25,
        descent_rsi_pilot_shaping_clip_max=0.20,
        # Event-aligned stage reachability controller pilots.  The event is
        # intentionally much larger than the clipped sum of dense terms.
        stage_reachability_objective="",
        stage_entry_event_reward=8.0,
        stage_entry_survival_reward=0.002,
        stage_entry_progress_coeff=0.10,
        stage_entry_pose_coeff=0.04,
        stage_entry_speed_coeff=0.02,
        stage_entry_angular_penalty_coeff=0.02,
        stage_takeoff_handoff_quality_coeff=1.0,
        stage_entry_action_smooth_coeff=0.005,
        stage_entry_action_magnitude_coeff=0.003,
        stage_entry_shaping_clip_min=-0.25,
        stage_entry_shaping_clip_max=0.20,
        stage_entry_failure_penalty=2.0,
        stage_jump_window_length_easy=0.80,
        stage_jump_window_length_hard=1.40,
        stage_height_target_easy=0.40,
        stage_height_target_hard=0.5515475838251305,
        stage_apex_target_height=0.5515475838251305,
        stage_prejump_target_speed=2.0,
        stage_postjump_target_speed=3.5,
        stage_entry_yaw_coeff=0.01,
        stage_entry_joint_energy_coeff=0.002,
        stage_curriculum_scale=0.0,
        # Robust centers/scales calibrated from the 11 immutable successful or
        # provisional-safe descent anchors plus 99 canonical C_L safe entries.
        descent_local_roll_center=0.07294333,
        descent_local_roll_scale=0.05,
        descent_local_pitch_center=-0.05370666,
        descent_local_pitch_scale=0.08,
        descent_local_vx_center=3.28729534,
        descent_local_vx_scale=0.30,
        descent_local_roll_rate_center=0.05529662,
        descent_local_roll_rate_scale=0.30,
        descent_local_pitch_rate_center=0.21333298,
        descent_local_pitch_rate_scale=0.93591303,
        descent_local_progress_scale=0.50,
        descent_local_survival_reward=0.005,
        descent_local_pitch_soft_limit_deg=35.0,
        descent_local_shaping_clip_min=-0.35,
        descent_local_shaping_clip_max=0.25,
        descent_local_terminal_failure_penalty=2.0,
        # Terrain, copied from the supplied XML.
        step_front_x=3.60,
        step_back_x=7.60,
        step_top_z=0.16,
        step_half_width=1.00,
        step_top_xy_margin=0.02,
        valid_landing_min_past_edge=0.25,
        valid_landing_back_margin=0.50,
        # Landing candidate construction.  The reference supplies a broad
        # pose/velocity envelope, but its vertical coordinate is not the
        # authoritative XML's collision height.  Proposals are therefore
        # placed by the actual wheel geometry, then advanced through a real
        # low-height impact before their snapshots become bootstrap states.
        landing_candidate_clearance_min=0.002,
        landing_candidate_clearance_max=0.010,
        landing_candidate_descend_vz_min=0.18,
        landing_candidate_descend_vz_max=0.38,
        landing_candidate_x_jitter=0.040,
        landing_candidate_y_jitter=0.015,
        landing_candidate_roll_jitter_deg=1.0,
        landing_candidate_pitch_jitter_deg=1.5,
        landing_candidate_yaw_jitter_deg=1.0,
        landing_candidate_vx_jitter=0.12,
        landing_candidate_vy_jitter=0.05,
        landing_candidate_hip_jitter=0.015,
        landing_candidate_knee_jitter=0.030,
        landing_candidate_impact_horizon=12,
        landing_candidate_relaxation_steps=5,
        # A candidate bank must include learnable Boundary/Dead states; it is
        # not a pre-certified Safe tube.  Immediate explosions remain banned,
        # while a small zero-action 0.5 s failure tail is reported and bounded.
        landing_candidate_max_short_horizon_failure_rate=0.10,
        flight_candidate_clearance_margin=0.002,
        flight_candidate_max_root_z_correction=0.20,
        flight_candidate_com_z_envelope_tolerance=0.20,
        flight_candidate_clearance_tolerance=1e-5,
        flight_candidate_clearance_iterations=32,
        flight_candidate_validation_steps=25,
        flight_candidate_apex_window_steps=12,
        flight_candidate_min_ascent_fraction=0.20,
        flight_candidate_min_apex_fraction=0.08,
        flight_candidate_min_descent_fraction=0.20,
        flight_augmentation_seed=17001,
        flight_augmentation_proposal_budget=3000,
        flight_augmentation_covariance_scale=0.30,
        flight_augmentation_interpolation_min=0.15,
        flight_augmentation_interpolation_max=0.85,
        flight_augmentation_max_children_per_parent=3,
        flight_augmentation_normalized_dedup_distance=0.15,
        flight_augmentation_envelope_sigma=0.50,
        landing_entry_window_steps=3,
        landing_entry_dedup_distance_z=0.15,
        landing_entry_minimum_calibration_precision=0.95,
        landing_entry_construction_seed=4100000,
        landing_entry_certification_seed=4200000,
        landing_entry_audit_seed=5200000,
        landing_entry_scale_floors=[0.10,0.05,0.03,0.0873,0.0873,0.0873,0.25,0.10,0.20,0.50,0.50,0.50,0.05,0.05,0.08,1.0,1.0,1.0,0.3333,0.10],
        # Robot/action convention.  Action order is fixed and stored in every
        # policy manifest: [steer, rear-wheel drive, hip, knee].
        action_mapping_version=ACTION_MAPPING_VERSION,
        ground_spawn_x=1.50,
        nominal_base_z_ground=0.120,
        initial_velocity=2.00,
        target_forward_speed=2.00,
        landing_target_forward_speed=2.00,
        wheel_roll_radius=0.070,
        hip_initial=-1.20,
        knee_initial=2.50,
        hip_min=-1.30,
        hip_max=0.50,
        knee_min=-1.50,
        knee_max=2.50,
        steer_action_scale=0.80,
        rear_wheel_action_min=-5.0,
        rear_wheel_action_max=40.0,
        # Event anchors and deployable phase filter.
        takeoff_window_far=1.25,
        takeoff_window_near=0.18,
        takeoff_liftoff_before_step=0.55,
        takeoff_clearance_before_step=0.20,
        takeoff_liftoff_height=0.015,
        takeoff_liftoff_vz=0.20,
        airborne_confirm_steps=2,
        landing_confirm_steps=2,
        landing_bounce_grace_steps=20,
        invalid_wheel_confirm_steps=2,
        phase_output_delay_steps=0,
        phase_delay_random_max=2,
        phase_progress_clip=True,
        # Recovery and failures.
        recovery_hold_steps=25,
        recovery_min_vx=0.55,
        recovery_max_roll_deg=14.0,
        recovery_max_pitch_deg=22.0,
        recovery_max_angvel=4.0,
        max_roll_deg=35.0,
        max_pitch_deg=75.0,
        max_backward_distance=1.0,
        stage_timeout_landing=3.0,
        stage_timeout_flight=4.0,
        stage_timeout_takeoff=5.0,
        stage_timeout_approach=7.0,
        stage_timeout_full=15.0,
        # Tube matching is phase-conditioned and standardized by the certified
        # downstream bank.  Radius is in standardized feature space.
        tube_match_radius_z=2.75,
        tube_match_k=1,
        tube_activation_min_safe=4,
        # RSI mixture.  Final-safe states form the core; boundary states keep
        # learning near the decision surface.  Auxiliary velocity seeds are
        # training-only and can never enter certification/audit.
        rsi_safe_mass=0.70,
        rsi_boundary_mass=0.15,
        rsi_aux_mass=0.05,
        downstream_rehearsal_mass=0.10,
        natural_prob_landing=0.00,
        natural_prob_flight=0.05,
        natural_prob_takeoff=0.10,
        natural_prob_approach=0.25,
        natural_prob_full=1.00,
        # Observation and policy state.
        actor_history_steps=4,
        obs_noise_enable=True,
        joint_pos_noise_std=0.002,
        joint_vel_noise_std=0.020,
        gyro_noise_std=0.10,
        gravity_noise_std=0.015,
        accel_noise_std=0.20,
        # Initial-state randomization.
        domain_randomization=True,
        init_roll_deg=3.0,
        init_pitch_deg=3.0,
        init_yaw_deg=3.0,
        init_pos_y_std=0.010,
        init_vel_x_std=0.10,
        # Contact/IMU adapter and exact-label compatibility.
        landing_side_margin=0.18,
        step_top_z_tol=0.030,
        step_top_normal_z_min=0.65,
        pretakeoff_airborne_fail_steps=4,
        platform_back_edge_diagnostic_margin=0.20,
        imu_impact_delta_threshold=6.0,
        imu_impact_use_abs_delta=True,
        imu_impact_sign=1.0,
        imu_impact_min_descend_vz=-0.15,
        imu_support_height_tol=0.095,
        imu_support_max_abs_vz=1.20,
        imu_airborne_height_margin=0.055,
        imu_airborne_min_abs_vz=0.45,
        takeoff_use_wheel_clearance=True,
        takeoff_wheel_radius=0.070,
        # Incremental knee target mapping.  Positive action decreases the XML
        # knee angle (flexion); negative action increases it (extension).
        # The XML joint range and actuator parameters remain unchanged.
        knee_action_target_delta=0.20,
        rear_wheel_action_delta=15.0,
        # Clean takeoff signal/reward protocol.
        takeoff_frontmost_offset=0.235,
        takeoff_disable_base_z_success=True,
        takeoff_positive_pitch_soft_deg=12.0,
        takeoff_positive_pitch_hard_deg=28.0,
        takeoff_positive_pitch_terminate_ticks=5,
        takeoff_wheelie_pitch_deg=16.0,
        takeoff_wheelie_terminate_ticks=5,
        takeoff_sync_height_diff_max=0.10,
        takeoff_sync_vz_diff_max=1.50,
        takeoff_step_clearance_margin=0.025,
        takeoff_reward_profile="minimal_dual_wheel",
        takeoff_reward_scale_dual_wheel_vz=1.20,
        takeoff_reward_scale_dual_wheel_height=1.00,
        takeoff_reward_scale_dual_wheel_clearance=0.80,
        takeoff_reward_scale_knee_useful_motion=0.12,
        takeoff_penalty_scale_positive_pitch=0.65,
        takeoff_penalty_scale_wheelie=1.50,
        takeoff_penalty_scale_wheel_unsync=0.60,
        takeoff_terminal_failure_penalty=12.0,
        # Certification may instantiate several actual dynamics variants.
        mass_scale=1.0,
        friction_scale=1.0,
        actuator_force_scale=1.0,
        gravity_scale=1.0,
        tube_activation_min_safe_records=4,
        certified_match_radius=2.75,
        # Recovery-centred legacy-compatible shaping terms retained by env.py.
        landing_reward_profile="imu_recovery_strict",
        forward_sigma=0.60,
        roll_sigma_deg=12.0,
        pitch_sigma_deg=26.0,
        coeff_action_mag=0.03,
        coeff_action_saturation=0.10,
        coeff_lateral_vel=0.45,
        coeff_yaw_rate=0.08,
        coeff_yaw_angle=0.15,
        coeff_downrange=0.00,
        landing_forward_sigma=0.60,
        landing_coeff_forward_speed=0.20,
        landing_coeff_progress=0.00,
        landing_coeff_platform_contact=0.15,
        landing_coeff_recovery_shaping=1.20,
        landing_coeff_recovery_streak=8.00,
        landing_terminal_hard_failure_penalty=30.00,
        landing_terminal_timeout_penalty=3.00,
        landing_roll_excess_penalty=2.00,
        landing_pitch_excess_penalty=0.75,
        landing_gyro_excess_penalty=0.25,
        landing_quality_requires_recovery_gates=True,
        landing_target_x=4.15,
        landing_stage_timeout_step_penalty=0.0,
        # Unified stage-continuation reward.  Chain is a bootstrap event; Final
        # Recovery remains the formal Tube label and receives the largest bonus.
        coeff_alive=0.05,
        coeff_forward_speed=0.45,
        coeff_progress=0.18,
        coeff_roll=0.40,
        coeff_pitch=0.30,
        coeff_action_smooth=0.02,
        coeff_takeoff_vz=1.20,
        coeff_takeoff_height=1.00,
        coeff_takeoff_posture=0.45,
        coeff_takeoff_joint_motion=0.15,
        coeff_takeoff_event=2.0,
        coeff_valid_landing_event=3.0,
        coeff_platform_contact=0.80,
        coeff_recovery_shaping=1.00,
        coeff_chain_event=8.0,
        coeff_recovery_success=30.0,
        coeff_hard_failure=12.0,
        coeff_timeout=2.0,
        # Certification protocol.
        beta_alpha0=1.0,
        beta_beta0=1.0,
        posterior_q_low=0.05,
        posterior_q_high=0.95,
        safe_threshold=0.70,
        dead_threshold=0.30,
        boundary_max_width=0.35,
        min_branches=8,
        max_branches=32,
        branch_horizon=750,
        action_noise_std=0.03,
        # Re-label/audit trigger metadata.
        max_label_age=4,
        policy_kl_trigger=0.08,
        eval_drop_trigger=0.08,
        relabel_budget=64,
        audit_seed_namespace="audit",
        build_seed_namespace="build",
    )


def apply_overrides(cfg: config_dict.ConfigDict, values: Mapping[str, Any]) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict(cfg.to_dict())
    for key, value in values.items():
        if key not in cfg:
            raise KeyError(f"Unknown DVGC config key: {key}")
        cfg[key] = value
    return cfg


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> config_dict.ConfigDict:
    cfg = default_config()
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = apply_overrides(cfg, payload)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def config_hash(cfg: config_dict.ConfigDict) -> str:
    raw = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_config(cfg: config_dict.ConfigDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
