from __future__ import annotations

from jit_dvgc.acquisition.resolution_frontier import select_resolution_distinct_anchors


def _row(phase: str, group: str, state: str, score: float) -> dict:
    return {
        "phase": phase,
        "parent_group_id": group,
        "state_sha256": state,
        "value_score": score,
        "sampling_weight": 0.5,
    }


def _projection(index: int, phase: str, state: str, cell: str, x: float, vz: float) -> dict:
    x_bin = int(round(x / 0.1))
    return {
        "global_index": index,
        "phase": phase,
        "state_sha256": state,
        "root_geometry_cell_id": cell,
        "root_geometry_bins": {"root_x_m": x_bin},
        "x_bin": x_bin,
        "x_center_m": x_bin * 0.1,
        "coordinates": {
            "root_x_m": x,
            "root_vz_mps": vz,
        },
    }


def _centerline() -> dict:
    points = []
    for x in (2.5, 2.6, 2.7, 2.8, 2.9):
        points.append({"x_target_m": x, "phase_semantics": "upstream"})
    points.append({"x_target_m": 3.0, "phase_semantics": "apex"})
    for x in (3.1, 3.2, 3.3, 3.4, 3.5):
        points.append({"x_target_m": x, "phase_semantics": "downstream"})
    return {
        "effective_centerline_max_x_m": 3.5,
        "points": points,
    }


def test_frontier_parent_selection_is_cell_unique_x_balanced_and_semantic() -> None:
    tube = [
        _row("upstream", "core-u", "core-u", 0.9),
        _row("downstream", "core-d", "core-d", 0.9),
    ]
    projections = [
        _projection(0, "upstream", "core-u", "core-cell-u", 2.5, 0.2),
        _projection(1, "downstream", "core-d", "core-cell-d", 3.2, -0.2),
    ]

    # Six upstream shell rows across five x bins; first two share one physical cell.
    for i, x in enumerate((2.5, 2.5, 2.6, 2.7, 2.8, 2.9)):
        state = f"u-{i}"
        tube.append(_row("upstream", f"u-g-{i}", state, 0.1 + i * 0.01))
        cell = "u-cell-0" if i < 2 else f"u-cell-{i}"
        projections.append(_projection(len(projections), "upstream", state, cell, x, 0.3))

    # Five valid descending downstream rows in different x bins.
    for i, x in enumerate((3.1, 3.2, 3.3, 3.4, 3.5)):
        state = f"d-{i}"
        tube.append(_row("downstream", f"d-g-{i}", state, 0.1 + i * 0.01))
        projections.append(_projection(len(projections), "downstream", state, f"d-cell-{i}", x, -0.2))

    # These must never become Jump-Tube frontier parents.
    tube.append(_row("downstream", "bad-upward", "bad-upward", 0.0))
    projections.append(_projection(len(projections), "downstream", "bad-upward", "bad-upward-cell", 3.3, +0.2))
    tube.append(_row("downstream", "bad-late", "bad-late", 0.0))
    projections.append(_projection(len(projections), "downstream", "bad-late", "bad-late-cell", 4.5, -0.2))

    anchors, audit = select_resolution_distinct_anchors(
        tube_entries=tube,
        projected_entries=projections,
        core_retained_count=2,
        max_parent_cells_per_phase=5,
        nominal_centerline=_centerline(),
    )
    for phase in ("upstream", "downstream"):
        selected = [row for row in anchors if row["phase"] == phase]
        cells = [row["root_geometry_cell_id"] for row in selected]
        assert len(selected) == 5
        assert len(cells) == len(set(cells))
        assert audit[phase]["selected_x_bin_count"] >= 4

    downstream = [row for row in anchors if row["phase"] == "downstream"]
    assert all(row["root_vz_mps"] < 0.0 for row in downstream)
    assert all(row["x_center_m"] <= 3.5 + 1.0e-9 for row in downstream)
    assert audit["downstream"]["semantic_rejected_count"] == 2
    assert audit["upstream"]["excluded_same_root_geometry_cell_count"] >= 1
