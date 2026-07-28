import numpy as np

from cli.acquire_terminal_aligned_apex_parents_v1 import (
    pilot_specs, select_nearest_supported_entries, select_parent_entries,
    terminal_distance,
)


def test_pilot_specs_are_bounded_unique_five_percent_grid():
    specs = pilot_specs()
    assert len(specs) == 24
    assert len({tuple(sorted(row.items())) for row in specs}) == 24
    assert len(specs) * 4 / (24 * 81) < 0.05
    assert max(row["hip_amplitude"] for row in specs) <= 0.85


def test_parent_selection_balances_sources_and_is_disjoint():
    rows = [{"trajectory_parent_id": f"p{i}", "id": f"i{i}", "upstream_source_kind": "a" if i < 3 else "b"}
            for i in range(6)]
    selected = select_parent_entries(rows, 4)
    assert len({row["trajectory_parent_id"] for row in selected}) == 4
    assert {row["upstream_source_kind"] for row in selected} == {"a", "b"}


def test_terminal_distance_is_zero_at_target():
    feature = np.arange(16, dtype=float)
    query = np.asarray([feature[index] for index in (0, 2, 3, 4, 6, 8, 9, 10, 11, 13, 14)])
    assert terminal_distance(feature, query[None, :], np.zeros(11), np.ones(11)) == 0


def test_nearest_supported_selection_is_disjoint_and_excludes_seen():
    records = []
    for index, value in enumerate((0., .1, .2, 1., 2.)):
        feature = [0.] * 16
        feature[0] = value
        records.append({"id": f"id-{index}", "trajectory_parent_id": f"p-{index}",
                        "physical_feature": feature})
    selected, distances = select_nearest_supported_entries(
        records, 2, {"p-0"}, {"p-1"},
    )
    assert [row["trajectory_parent_id"] for row in selected] == ["p-2", "p-3"]
    assert distances[0] < distances[1]
