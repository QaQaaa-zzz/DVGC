from pathlib import Path


def test_follow_on_controller_is_bounded_and_does_not_touch_apex_worker():
    text = Path("scripts/run_final_shared_policy_pipeline.sh").read_text()
    assert "sleep 120" in text
    assert "systemctl --user is-active" in text
    assert "systemctl --user stop" not in text
    assert "systemctl --user restart" not in text
    assert "build_phase_balanced_teacher_dataset" in text
    assert "build_phase_balanced_tube_rsi_bank" in text
    assert "phase_balanced_tube_rsi_v2" in text
    assert "descent_tube_v6.pkl" in text
    assert "verification_v2.json" in text
    assert 'verification.get("tube_sha256") == actual' in text
    assert 'normalization.get("output_bank_sha256") == actual' in text
    assert 'verification.get("policy_identity_hash") == normalization.get("policy_identity_hash")' in text
    assert "train_phase_balanced_distillation" in text
    assert "preflight_phase_balanced_unified_rsi" in text
    assert "train_phase_balanced_unified_rsi_pilot" in text
    assert "--steps 500 --learning-rate 3e-5" in text
    assert "pilot_4096_seed0_frozen_normalizer_v2" in text
    assert "normalize_until_count=0" in Path(
        "cli/train_phase_balanced_unified_rsi_pilot.py"
    ).read_text()


def test_pipeline_preserves_atomic_nonoverwrite_outputs():
    text = Path("scripts/run_final_shared_policy_pipeline.sh").read_text()
    assert "partial teacher artifact; refusing overwrite" in text
    assert "partial distillation artifact; refusing overwrite" in text
    assert "partial phase-balanced v2 bank/report; refusing overwrite" in text
    assert "repair_distillation_without_PPO" in text
    assert "distillation_downstream_fidelity" in text
    assert "without_budget_increase" in text
    assert '"controller_unit": "dvgc-final-shared-jel-audit-v2.service"' in text
    assert '"run_path": sys.argv[1]' in text


def test_v2_follow_on_start_is_nonblocking_and_updates_active_pointer():
    text = Path("scripts/start_final_shared_v2_followons.sh").read_text()
    assert "systemd-run --user" in text
    assert "systemctl --user --no-block start" in text
    assert "dvgc-final-shared-policy-v2.service" in text
    assert "dvgc-final-shared-jel-audit-v2.service" in text
    assert "runs/ACTIVE_PIPELINE.json" in text
    assert "run_final_shared_policy_pipeline.sh" in text
    assert "run_final_shared_jel_audit.sh" in text
    assert "rm " not in text
