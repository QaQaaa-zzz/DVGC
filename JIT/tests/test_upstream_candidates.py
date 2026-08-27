from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Events:
    jump_zone_seen: bool = False
    ascending_seen: bool = False
    height_seen: bool = False
    apex_seen: bool = False


@dataclass
class _State:
    tick: int
    events: _Events

    @property
    def info(self):
        return {
            "events": self.events,
            "terminated": False,
            "truncated": False,
        }


class _Collector:
    def __init__(self):
        self.transitions = 0
        self.rows = []

    def add(self, snapshot, *, seed, role, parent_trajectory):
        self.rows.append(
            {
                "snapshot": snapshot,
                "seed": seed,
                "role": role,
                "parent_trajectory": parent_trajectory,
            }
        )
        return True


def test_upstream_streaming_landmarks_are_pre_apex_and_bounded():
    from jit_dvgc.upstream_candidates import collect_upstream_streaming_rollout

    states = {
        tick: _State(
            tick,
            _Events(
                jump_zone_seen=tick >= 1,
                ascending_seen=tick >= 2,
                height_seen=tick >= 3,
                apex_seen=tick >= 13,
            ),
        )
        for tick in range(14)
    }
    collector = _Collector()
    added = collect_upstream_streaming_rollout(
        collector,
        states[0],
        seed=1000001,
        step=lambda _state, tick: states[tick + 1],
        capture=lambda state, _parent, tick: (state.tick, tick),
        parent_trajectory="transition_1__1000001",
        max_ticks=20,
    )

    assert added == 7
    assert collector.transitions == 13
    assert [(row["role"], row["snapshot"][1]) for row in collector.rows] == [
        ("pre_jump_zone", 0),
        ("jump_zone_entry", 1),
        ("ascending_entry", 2),
        ("height_entry", 3),
        ("pre_apex_10", 3),
        ("pre_apex_5", 8),
        ("pre_apex_1", 12),
    ]
    assert all(row["snapshot"][1] < 13 for row in collector.rows)


def test_upstream_streaming_requires_positive_horizon():
    import pytest
    from jit_dvgc.upstream_candidates import collect_upstream_streaming_rollout

    collector = _Collector()
    with pytest.raises(ValueError, match="max_ticks must be positive"):
        collect_upstream_streaming_rollout(
            collector,
            _State(0, _Events()),
            seed=1,
            step=lambda state, _tick: state,
            capture=lambda state, _parent, tick: (state.tick, tick),
            parent_trajectory="parent",
            max_ticks=0,
        )
