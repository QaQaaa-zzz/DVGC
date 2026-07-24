from pathlib import Path


def test_sensitivity_audit_is_event_local_and_bounded():
    text = Path("cli/audit_descent_bridge_sensitivity.py").read_text()
    assert "for window in (4, 8, 12)" in text
    assert "for channel in range(4)" in text
    assert '"domain_randomization": False' in text
    assert '"ppo_authorization": False' in text
    assert "restore_snapshot" not in text
