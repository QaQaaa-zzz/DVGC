from cli.probe_natural_compact_descent_bridge_v1 import handoff_ticks


def test_handoff_ticks_are_fixed_bounded_and_nonnegative():
    assert handoff_ticks(5)==[0,1,3,5,7,9]
    assert handoff_ticks(20)==[12,14,16,18,20,22,24]
