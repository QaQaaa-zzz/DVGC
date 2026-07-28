import pytest

from cli.select_apex_bridge_starts import choose_start


def test_forced_event_offset_overrides_latest_apex_selection():
    rows = [{"relative_to_apex": value, "id": str(value)} for value in (-8, -4, 0)]
    chosen, reason = choose_start(rows, "apex_local_response_detected", {"latest_effective_relative_to_apex": 0}, -4)
    assert chosen["relative_to_apex"] == -4
    assert "forced" in reason


def test_missing_forced_event_offset_is_rejected():
    with pytest.raises(ValueError, match="expected one"):
        choose_start([{"relative_to_apex": 0}], "apex_local_response_detected", {}, -4)
