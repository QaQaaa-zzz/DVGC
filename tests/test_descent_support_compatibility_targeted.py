from pathlib import Path


def test_runtime_compatibility_supports_bounded_index_selection():
    text = Path("cli/audit_descent_support_compatibility.py").read_text()
    assert '"--indices"' in text
    assert '"--historical-audit"' in text
    assert '"descent_asset_runtime_check"' in text
    assert '"maximum_stable_descent_ticks"' in text
