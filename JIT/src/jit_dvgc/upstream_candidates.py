"""Streaming nominal candidate collection for V_up continuation labels."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np

from .handoff_snapshot import HandoffSnapshot


UPSTREAM_SELECTION_ROLES = (
    "pre_jump_zone",
    "jump_zone_entry",
    "ascending_entry",
    "height_entry",
    "pre_apex_10",
    "pre_apex_5",
    "pre_apex_1",
)


def _truth(value: Any) -> bool:
    return bool(np.asarray(value))


def _history_state(history: deque[tuple[int, Any]], tick: int) -> Any | None:
    return next((state for saved_tick, state in history if saved_tick == tick), None)


def collect_upstream_streaming_rollout(
    collector: Any,
    initial_state: Any,
    *,
    seed: int,
    step: Callable[[Any, int], Any],
    capture: Callable[[Any, str, int], HandoffSnapshot],
    parent_trajectory: str,
    max_ticks: int,
) -> int:
    """Capture semantic pre-Apex landmarks without retaining a full MJX trace."""
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    history: deque[tuple[int, Any]] = deque(maxlen=12)
    state = initial_state
    previous = {
        "jump_zone_seen": False,
        "ascending_seen": False,
        "height_seen": False,
        "apex_seen": False,
    }
    added = 0

    def add_state(saved_state: Any, tick: int, role: str) -> None:
        nonlocal added
        snapshot = capture(saved_state, parent_trajectory, tick)
        added += int(
            collector.add(
                snapshot,
                seed=seed,
                role=role,
                parent_trajectory=parent_trajectory,
            )
        )

    for tick in range(max_ticks + 1):
        history.append((tick, state))
        events = state.info["events"]
        current = {
            "jump_zone_seen": _truth(events.jump_zone_seen),
            "ascending_seen": _truth(events.ascending_seen),
            "height_seen": _truth(events.height_seen),
            "apex_seen": _truth(events.apex_seen),
        }

        if current["jump_zone_seen"] and not previous["jump_zone_seen"]:
            if len(history) >= 2:
                prior_tick, prior_state = history[-2]
                add_state(prior_state, prior_tick, "pre_jump_zone")
            add_state(state, tick, "jump_zone_entry")

        if current["ascending_seen"] and not previous["ascending_seen"]:
            add_state(state, tick, "ascending_entry")

        if current["height_seen"] and not previous["height_seen"]:
            add_state(state, tick, "height_entry")

        if current["apex_seen"] and not previous["apex_seen"]:
            for offset, role in (
                (-10, "pre_apex_10"),
                (-5, "pre_apex_5"),
                (-1, "pre_apex_1"),
            ):
                target_tick = tick + offset
                if target_tick < 0:
                    continue
                saved_state = _history_state(history, target_tick)
                if saved_state is not None:
                    add_state(saved_state, target_tick, role)
            break

        terminated = _truth(state.info.get("terminated", False))
        truncated = _truth(state.info.get("truncated", False))
        if terminated or truncated or tick == max_ticks:
            break

        previous = current
        state = step(state, tick)
        collector.transitions += 1

    return added
