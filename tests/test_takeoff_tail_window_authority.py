import numpy as np

from cli.audit_takeoff_tail_window_authority import _rank, _window_start


def test_window_start_never_fabricates_missing_history():
    assert _window_start(2, "4") == (0, 2, False)
    assert _window_start(22, "4") == (18, 4, True)
    assert _window_start(22, "8") == (14, 8, True)
    assert _window_start(22, "12") == (10, 12, True)
    assert _window_start(22, "full") == (0, 22, True)


def test_normalized_rank_detects_independent_columns():
    matrix = np.asarray([[1., 0.], [0., 1.], [0., 0.]])
    singular, rank = _rank(matrix, np.ones(3))
    assert rank == 2
    assert len(singular) == 2


def test_authority_audit_keeps_diagnostic_semantics():
    text = open("cli/audit_takeoff_tail_window_authority.py").read()
    assert '"diagnostic_only": True' in text
    assert '"ppo_authorization": False' in text
    assert "external_impulse_between_unrelated_snapshots_assumed" in text
    assert "complete_requested_history" in text
    assert "regardless of its section label" in text
    assert 'row["section"] == "takeoff"' not in text
    assert 'source_entry["dynamics_seed"]' in text
    assert "continuous_from_takeoff_source_without_intermediate_restore" in text
    assert "historical_composite_used_as_authority_baseline" in text
