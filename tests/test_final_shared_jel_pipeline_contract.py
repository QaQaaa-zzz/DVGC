from pathlib import Path


def test_final_jel_controller_uses_exact_disjoint_funnel_and_one_shared_policy():
    text = Path("scripts/run_final_shared_jel_audit.sh").read_text()
    assert "sleep 120" in text
    assert "--required-branches 4 --next-branches 8" in text
    assert "--required-branches 8 --next-branches 32" in text
    assert text.count("cli.audit_final_shared_policy_candidates") == 4
    assert "final-shared-construction-32" in text
    assert "final-shared-independent-32" in text
    assert "cli.freeze_final_shared_policy_jel" in text
    assert "--policy \"$POLICY\"" in text
    assert "rm " not in text


def test_final_audit_never_starts_for_unpromoted_pilot():
    text = Path("scripts/run_final_shared_jel_audit.sh").read_text()
    assert '[[ "$pilot_status" != "PASS_PROMOTE" ]]' in text
    assert "diagnose_unified_pilot_without_formal_audit" in text
    assert "diagnose_shared_actor_zero_exact_final_support" in text
    assert "diagnose_shared_actor_independent_recertification_failure" in text
    assert "write_cost" in text
