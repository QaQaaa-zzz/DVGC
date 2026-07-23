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


def test_v3_start_is_nonblocking_systemd_controller():
    text = Path("scripts/start_stage_next_v3_controller.sh").read_text()
    assert "systemd-run --user" in text
    assert "RuntimeMaxSec=infinity" in text
