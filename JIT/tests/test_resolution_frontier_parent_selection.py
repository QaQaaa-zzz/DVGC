from __future__ import annotations

from collections import Counter

from jit_dvgc.acquisition.resolution_frontier import (
    ROLE_PATTERN,
    build_centerline_slice_anchors,
    select_resolution_distinct_anchors,
)


def _row(phase: str, group: str, state: str) -> dict:
    return {
        "phase": phase,
        "parent_group_id": group,
        "state_sha256": state,
        "value_score": 0.5,
        "sampling_weight": 0.5,
    }


def _projection(index: int, phase: str, state: str, x: float) -> dict:
    x_bin = int(round(x / 0.1))
    return {
        "global_index": index,
        "phase": phase,
        "state_sha256": state,
        "root_geometry_cell_id": f"cell-{phase}-{index}",
        "root_geometry_bins": {"root_x_m": x_bin},
        "x_bin": x_bin,
        "x_center_m": x_bin * 0.1,
        "coordinates": {"root_x_m": x, "root_vz_mps": -0.2 if phase == "downstream" else 0.2},
    }


def _centerline() -> dict:
    points = []
    frame = 10
    for x in (2.5, 2.6, 2.7, 2.8, 2.9):
        points.append(
            {
                "x_target_m": x,
                "phase_semantics": "upstream",
                "physical_state_sha256": f"{frame:064x}",
                "frame_index": frame,
                "root_vz_mps": 0.3,
            }
        )
        frame += 1
    points.append(
        {
            "x_target_m": 3.0,
            "phase_semantics": "apex",
            "physical_state_sha256": f"{frame:064x}",
            "frame_index": frame,
            "root_vz_mps": 0.0,
        }
    )
    frame += 1
    for x in (3.1, 3.2, 3.3, 3.4, 3.5):
        points.append(
            {
                "x_target_m": x,
                "phase_semantics": "downstream",
                "physical_state_sha256": f"{frame:064x}",
                "frame_index": frame,
                "root_vz_mps": -0.3,
            }
        )
        frame += 1
    return {
        "effective_centerline_max_x_m": 3.5,
        "points": points,
    }


def test_causal_frontier_uses_every_centerline_slice_and_never_tube_reset_indices() -> None:
    tube = [
        _row("upstream", "u0", "u0"),
        _row("downstream", "d0", "d0"),
        _row("downstream", "late", "late"),
    ]
    projected = [
        _projection(0, "upstream", "u0", 2.6),
        _projection(1, "downstream", "d0", 3.2),
        # Deliberately late source occupancy: it must not create a proposal slice.
        _projection(2, "downstream", "late", 4.5),
    ]
    centerline = _centerline()

    anchors, audit = select_resolution_distinct_anchors(
        tube_entries=tube,
        projected_entries=projected,
        core_retained_count=2,
        nominal_centerline=centerline,
    )

    expected_up_slices = 6  # upstream + exact Apex
    expected_down_slices = 5
    assert audit["centerline_slice_counts"] == {
        "upstream": expected_up_slices,
        "downstream": expected_down_slices,
    }
    assert len(anchors) == (expected_up_slices + expected_down_slices) * len(ROLE_PATTERN)
    assert all(anchor["entry_index"] == -1 for anchor in anchors)
    assert all(anchor["global_index"] == -1 for anchor in anchors)
    assert all(anchor["proposal_anchor_is_physical_reset"] is False for anchor in anchors)
    assert all(anchor["x_target_m"] <= 3.5 for anchor in anchors)
    assert not any(abs(anchor["x_target_m"] - 4.5) < 1e-9 for anchor in anchors)

    by_slice = Counter((anchor["phase"], anchor["x_bin"]) for anchor in anchors)
    assert set(by_slice.values()) == {len(ROLE_PATTERN)}
    role_counts = Counter(anchor["role"] for anchor in anchors)
    assert role_counts["train"] == 3 * (expected_up_slices + expected_down_slices)
    assert role_counts["calibration"] == expected_up_slices + expected_down_slices
    assert role_counts["acceptance"] == expected_up_slices + expected_down_slices


def test_build_centerline_slice_anchors_rejects_upward_downstream_centerline() -> None:
    centerline = _centerline()
    bad = dict(centerline)
    bad["points"] = [dict(point) for point in centerline["points"]]
    for point in bad["points"]:
        if point["phase_semantics"] == "downstream":
            point["root_vz_mps"] = +0.1
            break
    try:
        build_centerline_slice_anchors(centerline=bad, projected_entries=[])
    except ValueError as exc:
        assert "non-descending" in str(exc)
    else:
        raise AssertionError("upward downstream centerline was accepted")
