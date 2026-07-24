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
