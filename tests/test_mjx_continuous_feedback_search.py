from pathlib import Path


def test_feedback_search_is_bounded_fixed_variant_and_no_restore():
    text = Path("cli/search_mjx_continuous_feedback.py").read_text()
    assert '"domain_randomization": False' in text
    assert "itertools.product" in text
    assert "restore_snapshot" not in text
    assert '"ppo_authorization": False' in text
    assert "_natural_takeoff" in text
