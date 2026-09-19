from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "analyze_parent_generation_root_capacity.py"
    spec = importlib.util.spec_from_file_location("analyze_parent_generation_root_capacity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_family_prefers_explicit_provenance() -> None:
    cli = _load_cli()
    assert cli.source_family({"source_name": "up_nominal", "role": "fallback"}) == "up_nominal"
    assert cli.source_family({"source_kind": "retained"}) == "retained"
    assert cli.source_family({}) == "unspecified"


def test_summarize_phase_excludes_current_frontier_groups_and_keeps_layers() -> None:
    cli = _load_cli()
    rows = [
        {
            "global_index": 0,
            "entry_index": 0,
            "layer": "retained_core",
            "parent_group_id": "old_domain__1",
            "state_sha256": "a" * 64,
            "value_score": 0.4,
            "source_name": "old",
        },
        {
            "global_index": 1,
            "entry_index": 1,
            "layer": "retained_core",
            "parent_group_id": "fresh_domain__2",
            "state_sha256": "b" * 64,
            "value_score": 0.2,
            "source_name": "old",
        },
        {
            "global_index": 10,
            "entry_index": 2,
            "layer": "newest_shell",
            "parent_group_id": "used_domain__3",
            "state_sha256": "c" * 64,
            "value_score": 0.1,
            "source_name": "new",
        },
    ]
    result = cli.summarize_phase(
        rows,
        used_groups={"old_domain__1", "used_domain__3"},
    )
    assert result["parent_group_count"] == 3
    assert result["current_frontier_used_parent_group_count"] == 2
    assert result["unused_for_current_frontier_parent_group_count"] == 1
    assert result["unused_for_current_frontier_physical_state_count"] == 1
    assert result["unused_parent_groups_by_layer"] == {
        "retained_core": 1,
        "newest_shell": 0,
    }
    assert result["unused_parent_groups_by_checkpoint_domain"] == {"fresh_domain": 1}
    assert result["representative_unused_roots"][0]["parent_group_id"] == "fresh_domain__2"
