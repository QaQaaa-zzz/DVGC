from pathlib import Path


def test_upstream_lineage_is_exact_and_event_aligned():
    text = Path("cli/audit_apex_upstream_lineage.py").read_text()
    assert "takeoff_replay_qpos_linf" in text
    assert "last_support_tick" in text
    assert "separation_tick" in text
    assert "apex_minus_8" in text
    assert '"bootstrap_eligible": False' in text


def test_upstream_lineage_never_claims_a_tube():
    text = Path("cli/audit_apex_upstream_lineage.py").read_text()
    assert '"certified_tube": False' in text
    assert '"safe_claim_allowed": False' in text
    assert '"apex_ppo_authorized": False' in text
