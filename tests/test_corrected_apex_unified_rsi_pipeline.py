from pathlib import Path


def test_corrected_apex_pipeline_is_one_fresh_bounded_block():
    text = Path("scripts/run_corrected_apex_unified_rsi_pipeline.sh").read_text()
    assert "corrected_apex_contract_pilot_4096_seed2" in text
    assert '--initial-policy "$ANCHOR" --anchor-policy "$ANCHOR"' in text
    assert '--run "$PILOT" --seed 2' in text
    assert "--continuation-report" not in text
    assert "train_phase_balanced_unified_rsi_pilot" in text
    assert "PASS_PROMOTE" in text
    assert "diagnose_fixed_final_without_more_budget" in text
    assert "systemctl --user --no-block start" in text


def test_corrected_apex_start_is_nonblocking_and_audit_is_parameterized():
    start = Path("scripts/start_corrected_apex_unified_rsi_followons.sh").read_text()
    audit = Path("scripts/run_final_shared_jel_audit.sh").read_text()
    assert "systemd-run --user" in start
    assert "dvgc-final-shared-policy-v3.service" in start
    assert "dvgc-final-shared-jel-audit-v3.service" in start
    assert "FINAL_SHARED_PILOT=" in start
    assert "FINAL_SHARED_POLICY_UNIT=" in start
    assert "FINAL_SHARED_PILOT:-" in audit
    assert "FINAL_SHARED_AUDIT:-" in audit
    assert 'is-active --quiet "$POLICY_UNIT"' in audit
    assert "rm " not in start
