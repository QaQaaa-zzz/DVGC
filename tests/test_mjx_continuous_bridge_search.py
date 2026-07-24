from pathlib import Path


def test_search_is_single_engine_fixed_variant_and_no_ppo():
    text = Path("cli/search_mjx_continuous_bridge.py").read_text()
    assert '"domain_randomization": False' in text
    assert "restore_snapshot" not in text
    assert "reset_from_snapshot" not in text
    assert '"ppo_authorization": False' in text
    assert "_natural_takeoff_state" in text


def test_search_is_resumable_and_strictly_replays_best():
    text = Path("cli/search_mjx_continuous_bridge.py").read_text()
    assert '"progress.json"' in text
    assert '"strict_natural_replay"' in text
    assert "sequences[0] = mean" in text


def test_search_exposes_non_saturating_selection_weights():
    text = Path("cli/search_mjx_continuous_bridge.py").read_text()
    assert "--support-distance-weight" in text
    assert "--post-apex-weight" in text
    assert "--forward-weight" in text
    assert "--failure-penalty" in text
    assert "--pulse-knots" in text
    assert "--pulse-action" in text
    assert "--initial-prefix-knots" in text
    assert "--approach-drive" in text
