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


def _projection(index: int, phase: str, state: str, cell: str) -> dict:
    return {
        "global_index": index,
        "phase": phase,
        "state_sha256": state,
        "root_geometry_cell_id": cell,
        "root_geometry_bins": {"root_x_m": index},
    }


def test_frontier_parent_selection_deduplicates_nearby_states_by_geometry_cell() -> None:
    tube = [
        _row("upstream", "core-u", "core-u", 0.9),
        _row("downstream", "core-d", "core-d", 0.9),
    ]
    projections = [
        _projection(0, "upstream", "core-u", "core-cell-u"),
        _projection(1, "downstream", "core-d", "core-cell-d"),
    ]
    for phase, prefix in (("upstream", "u"), ("downstream", "d")):
        for i in range(6):
            state = f"{prefix}-{i}"
            tube.append(_row(phase, f"{prefix}-g-{i}", state, 0.1 + i * 0.01))
            # The first two exact states intentionally occupy the same physical cell.
            cell = f"{prefix}-cell-{0 if i < 2 else i}"
            projections.append(_projection(len(projections), phase, state, cell))

    anchors, audit = select_resolution_distinct_anchors(
        tube_entries=tube,
        projected_entries=projections,
        core_retained_count=2,
        max_parent_cells_per_phase=5,
    )
    for phase in ("upstream", "downstream"):
        selected = [row for row in anchors if row["phase"] == phase]
        cells = [row["root_geometry_cell_id"] for row in selected]
        assert len(selected) == 5
        assert len(cells) == len(set(cells))
        assert audit[phase]["excluded_same_root_geometry_cell_count"] >= 1
