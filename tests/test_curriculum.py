import pytest

from dvgc.curriculum import select_flight_reset_records


def _rows():
    return [
        {"id": f"a{i}", "flight_subinterval": "ascent", "source_index": i} for i in range(2)
    ] + [
        {"id": "p", "flight_subinterval": "apex", "source_index": 2},
        {"id": "d3", "flight_subinterval": "descent", "source_index": 3},
        {"id": "d4", "flight_subinterval": "descent", "source_index": 4},
        {"id": "d5", "flight_subinterval": "descent", "source_index": 5},
    ]


def test_flight_curriculum_is_nested_and_late_descent_is_near_entry():
    rows=_rows()
    supports=[{row["id"] for row in select_flight_reset_records(rows,stage)} for stage in ("late_descent","descent","apex_descent","full")]
    assert supports[0]=={"d4","d5"}
    assert all(left < right for left,right in zip(supports,supports[1:]))


def test_unknown_flight_curriculum_stage_is_rejected():
    with pytest.raises(ValueError): select_flight_reset_records(_rows(),"unknown")


def test_stage_expert_schedule_has_explicit_apex_and_ascent_support():
    rows=_rows()
    assert {r["id"] for r in select_flight_reset_records(rows,"apex")}=={"p"}
    assert {r["id"] for r in select_flight_reset_records(rows,"ascent")}=={"a0","a1"}


def test_apex_bridge_pairs_apex_with_nearest_early_descent():
    rows = _rows()
    assert [r["id"] for r in select_flight_reset_records(rows, "apex_bridge")] == ["p", "d3"]
