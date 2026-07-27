from pathlib import Path


def test_builder_uses_provisional_not_formal_semantics():
    text = Path("cli/build_descent_candidate_bank.py").read_text()
    assert '"old_bridge_gate": "superseded_as_training_gate"' in text
    assert '"formal_tube_member": False' in text
    assert '"formal_jel_member": False' in text
    assert '"old_support_entry_required": False' in text
    assert '"landing_or_final_required": False' in text
    assert '"ppo_authorization": False' in text
    assert "legacy_support_distance" in text
    assert "legacy_matcher_member" in text


def test_builder_replays_actions_to_capture_complete_snapshot():
    text = Path("cli/build_descent_candidate_bank.py").read_text()
    assert "env.snapshot_record(state, \"flight\")" in text
    assert "trajectory replay mismatch" in text
    assert "reset_replay" in text
    assert "short_horizon_trainability" in text
    assert "local_rsi_perturbation" in text
    assert "phase_detector_restore_mismatch" in text
    assert "phase_detector_regression" in text
    assert "_assign_layers(natural_eligible)" in text
