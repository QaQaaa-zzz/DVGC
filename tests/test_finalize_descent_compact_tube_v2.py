from types import SimpleNamespace

from cli.finalize_descent_compact_tube_v2 import certified_outcome


def test_certified_outcome_preserves_counts_and_safe_label():
    cfg=SimpleNamespace(min_branches=16,safe_threshold=.8,dead_threshold=.2,boundary_max_width=.5)
    result=certified_outcome(32,32,cfg)
    assert result["successes"]==32 and result["failures"]==0 and result["label"]=="safe"
