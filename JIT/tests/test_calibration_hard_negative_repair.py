from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "calibration_hard_negative_repair.py"
    spec = importlib.util.spec_from_file_location("calibration_hard_negative_repair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3b_panel_is_fixed_sparse_two_axis_historical_family() -> None:
    cli = _load_cli()
    assert cli.UPSTREAM_PANEL == {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.15, 0.30, 0.50],
        "durations": [2, 4, 8],
        "active_action_dimensions": 2,
    }
    assert cli.MIN_SUPPLEMENT_NEGATIVES == 5
    assert cli.MIN_SUPPLEMENT_NEGATIVE_PARENT_GROUPS == 2


def test_phase_counts_tracks_negative_parent_support() -> None:
    cli = _load_cli()
    rows = [
        {"phase": "upstream", "label": 1, "parent_group_id": "a"},
        {"phase": "upstream", "label": 0, "parent_group_id": "a"},
        {"phase": "upstream", "label": 0, "parent_group_id": "b"},
        {"phase": "downstream", "label": 0, "parent_group_id": "d"},
    ]
    assert cli._phase_counts(rows, "upstream") == {
        "candidate_count": 3,
        "positive_count": 1,
        "negative_count": 2,
        "parent_group_count": 2,
        "negative_parent_group_count": 2,
    }
