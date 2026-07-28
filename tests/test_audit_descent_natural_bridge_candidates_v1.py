from pathlib import Path


def test_targeted_audit_accepts_explicit_nonoverwriting_sources():
    text=Path('cli/audit_descent_natural_bridge_candidates_v1.py').read_text()
    assert "--source" in text
    assert "--base" in text
    assert "source_path" in text and "base_path" in text
