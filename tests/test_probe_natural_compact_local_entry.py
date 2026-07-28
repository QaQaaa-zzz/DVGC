from pathlib import Path


def test_natural_probe_accepts_a_frozen_local_entry_bank():
    text = Path("cli/probe_natural_compact_descent_bridge_v1.py").read_text()
    assert 'parser.add_argument("--tube"' in text
    assert 'evaluate_entry("apex"' in text
    assert '"formal_entries"' in text
