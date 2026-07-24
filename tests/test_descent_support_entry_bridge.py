from pathlib import Path


def test_bridge_search_is_low_dimensional_and_natural_lineage():
    text = Path("cli/search_descent_support_entry_bridge.py").read_text()
    assert "params = np.zeros(args.segments * 2" in text
    assert "for radius in (0.16, 0.08, 0.04, 0.02)" in text
    assert "restore_snapshot" not in text
    assert '"domain_randomization": False' in text
    assert "for repeat in range(20)" in text
    assert '"ppo_authorization": False' in text
    assert "bridge_window_complete" in text
    assert 'int(outcome["termination_reason"] != "pitch_limit")' not in text
    assert 'parser.add_argument("--window-start"' in text
    assert 'parser.add_argument("--residual-bound"' in text
    assert '"--screen-delays"' in text
    assert "control_tick = local_tick - launch_delay" in text
    assert 'choices=(2, 3)' in text
    assert '"squared_contribution"' in text
