from pathlib import Path


def test_feedback_labeler_can_complete_fresh_branches_after_nominal_failure():
    text = Path("cli/discover_apex_feedback_bridge.py").read_text()
    assert '"--fresh-regardless-of-nominal"' in text
    assert "deterministic_stable or a.fresh_regardless_of_nominal" in text
