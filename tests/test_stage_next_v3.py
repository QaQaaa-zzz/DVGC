from pathlib import Path


def test_v3_eval_contract_is_balanced_and_training_disjoint():
    text = Path("cli/prepare_stage_next_v3.py").read_text()
    assert '"canonical_compressed": 12' in text
    assert '"reference_aligned_compressed": 12' in text
    assert "excluded.update" in text
    assert "accepted_and_frozen_reset_protocol" in text
    assert "parent_count: int = 4" in text


def test_takeoff_controller_eval_reports_strata_union_and_reward_breakdown():
    text = Path("cli/evaluate_takeoff_controllers.py").read_text()
    assert "canonical_compressed" in text
    assert "reference_aligned_compressed" in text
    assert "union_of_controllers" in text
    assert "reward_breakdown" in text
    assert "time_to_ascent" in text


def test_specialist_curriculum_excludes_eval_parents_and_does_not_interpolate():
    text = Path("cli/build_takeoff_specialist_curriculum.py").read_text()
    assert "eval_parents" in text
    assert "reference_parent_overlap_with_eval" in text
    assert '"state_mutation": False' in text


def test_stage_search_keeps_local_failure_semantics():
    text = Path("cli/search_stage_support.py").read_text()
    assert "negative_under_bounded_controller_bank_only" in text
    assert "minimum_support_distance" in text
    assert "successful_parent_count" in text


def test_apex_v3_requires_dynamic_parents_and_keeps_reset_valid_separate():
    text = Path("cli/build_apex_reset_bank_v3.py").read_text()
    assert "apex_reference_anchor_reset_valid" in text
    assert "apex_dynamically_reached" in text
    assert "parent_count >= 4" in text
    assert "five_step_reset_shock" in text
    assert "reset_valid_is_not_reachability_evidence" in text
    assert "noise_scale = (0., .005, .015)" in text
    assert "continuation_action = jp.asarray(action_history[-1]" in text


def test_ascent_reverse_bank_is_reference_aligned_and_stratified():
    text = Path("cli/build_ascent_reverse_bank.py").read_text()
    assert "aligned_reference_anchors" in text
    assert "late_ascent" in text and "early_ascent" in text
    assert "reference_after_joint_limit_violation_used" in text


def test_reward_diagnostic_compares_success_to_missed_liftoff():
    text = Path("cli/analyze_takeoff_reward_diagnostic.py").read_text()
    assert "success_return_dominates" in text
    assert "missed_liftoff_return" in text
    assert "success_event_present" in text


def test_v3_controller_uses_best_stratified_checkpoint_and_local_blockers():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    assert "min(canonical, aligned)" in text
    assert "takeoff_canonical_stagnant_blocks" in text
    assert "bounded_controller_support_gap" in text
    assert "research_gate_valid=False" in text
    assert "resume_missing_apex_bounded_support_search_r3" in text


def test_pre_apex_horizon_audit_is_bounded_and_has_no_ppo():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    assert '"minimal_pre_apex_horizon_audit"' in text
    assert '"prediction_horizons": [3, 6, 9, 12]' in text
    assert '"ppo_steps": 0' in text
    assert '"apex_centroidal_contact_audit"' in text


def test_takeoff_tail_authority_screen_is_bounded_and_no_ppo():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    assert '"audit_takeoff_tail_window_authority"' in text
    assert '"windows": [4, 8, 12, "full"]' in text
    assert '"ppo_steps": 0' in text


def test_v3_start_is_nonblocking_systemd_controller():
    text = Path("scripts/start_stage_next_v3_controller.sh").read_text()
    assert "systemd-run --user" in text
    assert "RuntimeMaxSec=infinity" in text


def test_frozen_takeoff_labels_use_all_policy_controllers_and_check_source_mix():
    controller = Path("cli/stage_next_v3_controller.py").read_text()
    analysis = Path("cli/analyze_takeoff_frozen_labels.py").read_text()
    assert "frozen_label_pilot_120x4x3" in controller
    assert controller.count('"--flight-policy"') >= 3
    assert "both_strata_contain_success_and_failure_branches" in analysis
    assert "negative_under_frozen_controller_bank" in analysis
    assert "train_source_stratified_takeoff_reachability" in controller


def test_ascent_parent_acquisition_uses_independent_lineage_and_bounded_rounds():
    text = Path("cli/acquire_ascent_apex_parents.py").read_text()
    assert '"upstream_source_parent_id"' in text
    assert '"reset_protocol_hash"' in text
    assert '"initial_state_id"' in text
    assert '"dynamics_seed"' in text
    assert '"round": "A"' in text and '"round": "B"' in text
    assert "ascent_multi_parent_controller_gap" in text
    assert "late_ascent_training_authorized" in text
    assert "certified_tube" in text and "safe_claim_allowed" in text


def test_ascent_parent_acquisition_can_stop_before_downstream_search():
    text = Path("cli/acquire_ascent_apex_parents.py").read_text()
    assert '"--entries-only"' in text
    assert '"search_executed": False' in text


def test_v3_controller_does_not_launch_ascent_ppo_without_two_parents():
    text = Path("cli/stage_next_v3_controller.py").read_text()
    assert "mine_independent_ascent_apex_parents" in text
    assert "if parent_count >= 2" in text
    assert "stage_local_gate_no_unbounded_ppo" in text


def test_late_ascent_discovery_is_bc_initialized_and_bounded():
    bc = Path("cli/behavior_clone_ascent_sequences.py").read_text()
    curriculum = Path("cli/build_late_ascent_curriculum.py").read_text()
    controller = Path("cli/stage_next_v3_controller.py").read_text()
    assert "len(successful) < 2" in bc
    assert "proposal_controller_initialization" in bc
    assert "successful_parent_ids" in curriculum
    assert '"--timesteps", "25600"' in controller
    assert "block >= 4 or stagnant >= 2" in controller
    assert "late_ascent_best_score" in controller


def test_dynamic_apex_bank_separates_reference_dynamic_and_descent_positive():
    text = Path("cli/assemble_dynamic_apex_bank.py").read_text()
    assert "reference_reset_valid" in text
    assert "dynamically_reached" in text
    assert '"descent_positive": 0' in text
    assert "reference_valid_is_not_dynamic_support" in text
    assert "dynamic_parent_count" in text
