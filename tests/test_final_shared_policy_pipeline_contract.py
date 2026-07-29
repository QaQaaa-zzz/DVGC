from pathlib import Path


def test_follow_on_controller_is_bounded_and_does_not_touch_apex_worker():
    text = Path("scripts/run_final_shared_policy_pipeline.sh").read_text()
    assert "sleep 120" in text
    assert "systemctl --user is-active" in text
    assert "systemctl --user stop" not in text
    assert "systemctl --user restart" not in text
    assert "build_phase_balanced_teacher_dataset" in text
    assert "train_phase_balanced_distillation" in text
    assert "preflight_phase_balanced_unified_rsi" in text
    assert "train_phase_balanced_unified_rsi_pilot" in text
    assert "--steps 500 --learning-rate 3e-5" in text
    assert "pilot_5120_seed0" in text


def test_pipeline_preserves_atomic_nonoverwrite_outputs():
    text = Path("scripts/run_final_shared_policy_pipeline.sh").read_text()
    assert "partial teacher artifact; refusing overwrite" in text
    assert "partial distillation artifact; refusing overwrite" in text
    assert "repair_distillation_without_PPO" in text
    assert "without_budget_increase" in text
