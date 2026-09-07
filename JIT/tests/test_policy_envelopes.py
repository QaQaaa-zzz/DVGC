from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jit_dvgc.analysis.policy_envelopes import project_row, summarize, render_comparison
from jit_dvgc.analysis.capability_tube import FULL_PHYSICAL_FIELDS


def panel():
    points = []
    for i, (phase, x, z) in enumerate((("upstream",2.5,.2), ("upstream",2.5,.2),
                                    ("upstream",2.6,.3), ("downstream",2.6,.3), ("downstream",2.7,.2))):
        coords = dict.fromkeys(FULL_PHYSICAL_FIELDS, 0.)
        coords.update(root_x_m=x, root_z_m=z, root_vx_mps=2., root_vz_mps=1. if phase=="upstream" else -1.)
        row = {"candidate_id": f"c{i}", "state_sha256": f"s{i}", "parent_group_id": "g", "phase": phase}
        points.append(project_row(row, coords, f"context{i}"))
    # A unique exact context need not be a unique physical cell (c0 and c1).
    outcomes = {"pi_0": [1,0,1,0,0], "pi_1": [0,1,1,0,0], "pi_2": [0,0,0,1,0], "pi_3": [0,0,0,0,0]}
    labels = {name: [{"candidate_id":p["candidate_id"], "state_sha256":p["state_sha256"], "phase":p["phase"],
                       "label": ok, "success_criterion":"first_valid_landing", "environment_interactions":3}
                      for p, ok in zip(points, values)] for name, values in outcomes.items()}
    return points, labels


def test_unique_context_and_physical_contribution_are_distinct():
    points, labels = panel()
    result = summarize(points, labels, role="train")
    metrics = {r["policy"]:r for r in result["metrics"]}
    assert metrics["pi_0"]["unique_contexts"] == 1
    assert metrics["pi_0"]["unique_root_cells"] == 0
    assert metrics["pi_2"]["unique_root_cells"] == 1
    assert metrics["union"]["successful_candidates"] == 4
    assert metrics["union"]["root_cells"] == 3
    assert metrics["union"]["observed_contexts"] == 4
    assert result["jaccard"]["root_cell"][0][1] == 1.
    assert result["jaccard"]["root_cell"][3][3] is None
    assert all(row["root_z_m_min"] is None for row in result["x_slices"] if row["policy"] == "pi_3")


@pytest.mark.parametrize("change", ["missing", "order", "endpoint", "phase", "outcome", "duplicate"])
def test_comparison_refuses_noncomparable_or_incomplete_panels(change):
    points, labels = panel()
    if change == "missing": labels["pi_1"].pop()
    elif change == "order": labels["pi_1"].reverse()
    elif change == "endpoint": labels["pi_1"][0]["success_criterion"] = "stable_recovery"
    elif change == "phase": labels["pi_1"][0]["phase"] = "downstream"
    elif change == "outcome": labels["pi_1"][0]["label"] = None
    elif change == "duplicate": points[1] = points[0]
    with pytest.raises(ValueError): summarize(points, labels, role="train")


def test_figure_exports_handle_empty_policy_and_identical_coordinates(tmp_path):
    points, labels = panel()
    result = summarize(points, labels, role="calibration")
    render_comparison(points, result, tmp_path, title_prefix="SYNTHETIC TEST ONLY | ")
    for name in ["all_policy_projections", "pi_0_projections", "pi_3_projections", "union_projections",
                 "cross_section_occupancy", "cross_section_spans", "policy_contributions", "policy_overlap"]:
        for ext in ("pdf", "svg", "png"):
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 500
    assert len(json.loads((tmp_path / "summary.json").read_text())["resolution_sensitivity"]) == 15
    assert "successful_candidates" in (tmp_path / "policy_metrics.csv").read_text()
