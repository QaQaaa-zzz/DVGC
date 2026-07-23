from pathlib import Path

from cli.audit_takeoff_key import main


def test_key_audit_uses_named_lookup_and_records_mapping():
    text = Path("cli/audit_takeoff_key.py").read_text()
    assert 'model.joint(name)' in text
    assert 'model.actuator(name)' in text
    assert "qpos_address" in text and "qvel_address" in text
    assert "full_key_qpos" in text
    assert "actor_action_order" in text
