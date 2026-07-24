from pathlib import Path


def test_parent_reproduction_separates_robustness_classes():
    text = Path("cli/reproduce_dynamic_apex_parents.py").read_text()
    assert "robust_dynamic_parent" in text
    assert "seed_conditional_dynamic_parent" in text
    assert "deterministic_only_parent" in text
    assert "fresh_dynamics_successes" in text
    assert "snapshot_five_step_reset_shock" in text


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
