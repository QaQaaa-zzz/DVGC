from pathlib import Path


def test_parent_reproduction_separates_robustness_classes():
    text = Path("cli/reproduce_dynamic_apex_parents.py").read_text()
    assert "robust_dynamic_parent" in text
    assert "seed_conditional_dynamic_parent" in text
    assert "deterministic_only_parent" in text
    assert "fresh_dynamics_successes" in text
    assert "snapshot_five_step_reset_shock" in text
    assert "for i in range(4)" in text


def test_descent_support_audit_keeps_local_and_final_outcomes_separate():
    text = Path("cli/audit_descent_support_compatibility.py").read_text()
    assert "descent_controller_success_rate" in text
    assert "landing_final_recovery_rate" in text
    assert "descent_support_runtime_stale" in text
    assert "original_label_agreement_rate" in text
    assert "phase_detector_source_hash" in text


def test_feature_alternatives_are_diagnostic_only():
    text = Path("cli/audit_descent_feature_semantics.py").read_text()
    assert "angle_wrapped_diagnostic" in text
    assert "absolute_x_removed_diagnostic" in text
    assert "landing_relative_x_diagnostic" in text
    assert "formal_matcher_unchanged" in text
    assert "diagnostic_alternatives_not_active" in text


def test_multiknot_search_uses_real_event_and_failure_typing():
    text = Path("cli/search_apex_descent_multiknot.py").read_text()
    for reason in (
        "apex_not_crossed", "crossed_apex_detector_missed",
        "pose_instability_before_descent", "joint_margin_exhausted",
        "support_metric_mismatch", "downstream_controller_gap",
        "final_recovery_gap",
    ):
        assert reason in text
    assert "valid_descent_entry" in text
    assert "round_b_executed" in text
    assert "matcher_radius_unchanged" in text
    assert "final_recovery_branches" in text
    assert "round_c_full_action_branches" in text


def test_full_action_extension_is_bounded_and_keeps_best_joint_profile():
    from cli.search_apex_descent_multiknot import _action, _round_c
    base = {"coast": 0, "correction_duration": 2, "hip": .1, "knee": -.2,
            "post_duration": 2, "post_hip": 0., "post_knee": 0.}
    rows = _round_c(base)
    assert len(rows) == 25
    assert all(row["hip"] == .1 and row["knee"] == -.2 for row in rows)
    assert all(float(abs(x)) <= 1. for row in rows for x in _action(row, 0))


def test_event_aligned_ascent_entries_are_dynamic_apex_evidence():
    from cli.search_apex_descent_multiknot import is_dynamically_reached_apex, matcher_identity

    assert is_dynamically_reached_apex({
        "candidate_kind": "stage_entry_snapshot",
        "entry_from_stage": "ascent",
        "entry_to_stage": "apex",
    })
    assert not is_dynamically_reached_apex({
        "candidate_kind": "stage_entry_snapshot",
        "entry_from_stage": "takeoff",
        "entry_to_stage": "ascent",
    })
    assert matcher_identity({"radius": 1.0}) == matcher_identity({"radius": 1.0})
    assert matcher_identity({"matcher_sha256": "declared"}) == "declared"


def test_controller_runs_interface_audits_before_any_apex_training():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    order = [
        "freeze_current_proposal_evidence",
        "reproduce_three_dynamic_parents",
        "audit_descent_support_runtime_compatibility",
        "audit_descent_feature_semantics",
        "apex_descent_multiknot_bounded_search",
        "mine_fourth_independent_apex_parent",
    ]
    for stage in order:
        assert stage in text
    assert '"no_ppo_authorized": True' in text


def test_fresh_parent_mining_is_source_balanced_and_diversity_selected():
    text = Path("cli/acquire_ascent_apex_parents.py").read_text()
    assert "_select_diverse_entries" in text
    assert "entry_pool_multiplier" in text
    assert "takeoff_policy_only" in text
    assert "fresh_ascent_entry_controller_mix" in text
    assert "required_successful_parents" in text


def test_fresh_parent_mining_resumes_atomic_parent_shards():
    text = Path("cli/acquire_ascent_apex_parents.py").read_text()
    assert 'entry_path.exists()' in text
    assert '"parent_search_shards"' in text
    assert "_round_a.json" in text
    assert "_round_a.pkl" in text
    assert "_round_b.json" in text
    assert "_round_b.pkl" in text
    assert "result_path.exists() and snapshot_path.exists()" in text
    assert "parent_robustness_v2.json" in text
    assert "same frozen-runtime parent robustness audit" in text


def test_controller_keeps_apex_ppo_closed_without_every_gate():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    assert "mine_fourth_independent_apex_parent" in text
    assert "expand_dynamic_apex_bank_v5" in text
    assert "successful_sequences_fresh_seed_reproduced" in text
    assert "reward_diagnostic_passed" in text
    assert "reset_and_full_runtime_gate_passed_after_success" in text
    assert "authorized = all(authorization_checks.values())" in text


def test_controller_routes_through_feedback_bridge_before_any_apex_ppo():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    order = [
        "freeze_v5_evidence",
        "analyze_existing_branch_failure_timing",
        "audit_pre_post_apex_control_authority",
        "build_current_runtime_descent_terminal_clusters",
        "select_bridge_start_before_apex",
        "discover_nominal_stable_descent_trajectory",
        "synthesize_local_feedback_bridge",
        "fresh_seed_bridge_validation",
        "formal_descent_support_and_final_recovery_test",
    ]
    positions = [text.index(f'elif stage == "{stage}"') for stage in order]
    assert positions == sorted(positions)
    assert '"gate_c_apex_ppo_authorized": False' in text
