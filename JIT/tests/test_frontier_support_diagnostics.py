from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "cli" / "analyze_frontier_support.py"
    spec = importlib.util.spec_from_file_location("analyze_frontier_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frontier_support_diagnostics_detects_all_positive_max_cell(tmp_path: Path) -> None:
    cli = _load_cli()
    root = tmp_path / "frontier_train"
    acquisition = root / "acquisition"
    labels = root / "labels"
    acquisition.mkdir(parents=True)
    labels.mkdir(parents=True)

    protocol = {
        "status": "predeclared",
        "strengths": [0.05, 0.10],
        "durations": [1, 8],
    }
    candidates = [
        {
            "candidate_id": "u0",
            "phase": "upstream",
            "parent_group_id": "ug",
            "perturbation": {"action_name": "hip", "sign": 1, "strength": 0.10, "duration": 8},
        },
        {
            "candidate_id": "d0",
            "phase": "downstream",
            "parent_group_id": "dg",
            "perturbation": {"action_name": "knee", "sign": -1, "strength": 0.10, "duration": 8},
        },
        {
            "candidate_id": "d1",
            "phase": "downstream",
            "parent_group_id": "dg2",
            "perturbation": {"action_name": "knee", "sign": -1, "strength": 0.05, "duration": 1},
        },
    ]
    catalog = {
        "status": "completed",
        "candidate_count": len(candidates),
        "iteration": 1,
        "policy_actor_sha256": "a" * 64,
        "policy_payload_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "entries": candidates,
    }
    label_rows = [
        {"candidate_id": "u0", "label": 0, "outcome_class": "physical_failure"},
        {"candidate_id": "d0", "label": 1, "outcome_class": "success"},
        {"candidate_id": "d1", "label": 1, "outcome_class": "success"},
    ]
    label_summary = {
        "status": "completed",
        "candidate_count": len(label_rows),
        "protocol_sha256": "d" * 64,
    }

    (acquisition / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (acquisition / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (labels / "labels.json").write_text(json.dumps(label_rows), encoding="utf-8")
    (labels / "summary.json").write_text(json.dumps(label_summary), encoding="utf-8")

    result = cli.analyze(root)
    downstream = result["by_phase"]["downstream"]
    assert downstream["candidate_count"] == 2
    assert downstream["positive_count"] == 2
    assert downstream["negative_count"] == 0
    assert downstream["max_strength_max_duration"] == {
        "candidate_count": 1,
        "positive_count": 1,
        "negative_count": 0,
        "parent_group_count": 1,
    }
    assert downstream["all_positive_at_max_strength_max_duration"] is True
    assert "does not bracket" in downstream["interpretation"]
