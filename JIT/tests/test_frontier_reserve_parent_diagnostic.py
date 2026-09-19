from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "analyze_frontier_reserve_parents.py"
    spec = importlib.util.spec_from_file_location("analyze_frontier_reserve_parents", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_domain_uses_parent_prefix() -> None:
    cli = _load_cli()
    assert cli.checkpoint_domain("transition_4988928__1000001") == "transition_4988928"
    assert cli.checkpoint_domain("plain") == "plain"


def test_ordered_parent_unique_preserves_lowest_score_group_and_state_uniqueness() -> None:
    cli = _load_cli()
    rows = [
        (5, 5, {"value_score": 0.3, "parent_group_id": "g1", "state_sha256": "s1", "sampling_weight": 1.0}),
        (2, 2, {"value_score": 0.1, "parent_group_id": "g1", "state_sha256": "s2", "sampling_weight": 1.0}),
        (3, 3, {"value_score": 0.2, "parent_group_id": "g2", "state_sha256": "s3", "sampling_weight": 2.0}),
        (4, 4, {"value_score": 0.25, "parent_group_id": "g3", "state_sha256": "s3", "sampling_weight": 3.0}),
    ]
    result = cli.ordered_parent_unique(rows)
    assert [row["parent_group_id"] for row in result] == ["g1", "g2"]
    assert [row["entry_index"] for row in result] == [2, 3]
