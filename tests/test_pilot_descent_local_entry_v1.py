from cli.pilot_descent_local_entry_v1 import select_pilot_anchors


def test_pilot_anchor_selection_covers_regions_and_extension():
    rows = [
        {"id": "e", "descent_region": "early", "final": {"successes": 4, "branches": 4}},
        {"id": "m0", "descent_region": "middle", "final": {"successes": 3, "branches": 4}},
        {"id": "m1", "descent_region": "middle", "final": {"successes": 4, "branches": 4}},
        {"id": "l", "descent_region": "late", "final": {"successes": 4, "branches": 4}},
        {"id": "x", "descent_region": None, "target_distance": 1., "final": {"successes": 4, "branches": 4}},
    ]
    assert [row["id"] for row in select_pilot_anchors(rows)] == ["e", "m1", "l", "x"]
