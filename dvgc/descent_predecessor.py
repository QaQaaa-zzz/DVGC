"""Pure contracts for physical Descent predecessor harvesting."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def expand_residual_knots(knots: Sequence[Sequence[float]] | None, *,
                          ticks_per_knot: int = 4, horizon: int = 8) -> np.ndarray:
    """Expand an existing bounded controller into its exact tick residuals."""
    if knots is None:
        return np.zeros((horizon, 4), np.float32)
    value = np.asarray(knots, np.float32)
    if value.ndim != 2 or value.shape[1] != 4 or ticks_per_knot <= 0:
        raise ValueError("residual knots must have shape (K,4)")
    expanded = np.repeat(value, int(ticks_per_knot), axis=0)
    if len(expanded) < horizon:
        expanded = np.concatenate((expanded, np.zeros((horizon-len(expanded), 4), np.float32)))
    return expanded[:horizon]


def remaining_residual_suffix(expanded: np.ndarray, source_tick: int, *, horizon: int = 8) -> np.ndarray:
    """Shift an original controller forward without inventing future actions."""
    value = np.asarray(expanded, np.float32)
    tick = int(source_tick)
    if value.shape != (horizon, 4) or tick < 0:
        raise ValueError("invalid expanded controller or source tick")
    suffix = np.zeros_like(value)
    remaining = value[tick:]
    suffix[:len(remaining)] = remaining
    return suffix


def active_predecessor_ticks(entry_tick: int, fifo_valid_by_tick: Sequence[int],
                             terminal_tick: int) -> list[int]:
    """Return only active, pre-entry ticks with a fully real v4 packet FIFO."""
    stop = min(int(entry_tick), int(terminal_tick))
    return [tick for tick in range(stop) if tick < len(fifo_valid_by_tick)
            and int(fifo_valid_by_tick[tick]) == 3]


def predecessor_priority(row: Mapping[str, Any], dominant_candidate: str,
                         existing_p1_counts: Mapping[str, int]) -> tuple:
    """Stable Source-A priority: diverse candidates and near-entry states first."""
    candidate = str(row["candidate_id"])
    relative = int(row["relative_tick_to_downstream_entry"])
    return (
        candidate == dominant_candidate,
        int(existing_p1_counts.get(candidate, 0)),
        not bool(row.get("source_was_p1", False)),
        0 if -6 <= relative <= -1 else 1,
        abs(relative), candidate, int(row["source_tick"]), str(row["physical_state_hash"]),
    )


def require_forward_lineage(row: Mapping[str, Any]) -> None:
    """Reject state-spliced or reverse-integrated predecessor proposals."""
    if row.get("construction_method") != "forward_mjx_active_prefix":
        raise ValueError("predecessor must come from uninterrupted forward MJX dynamics")
    if row.get("state_splicing") or row.get("reverse_integration"):
        raise ValueError("state splicing and reverse integration are forbidden")
    if int(row.get("source_tick", -1)) >= int(row.get("downstream_entry_tick", -1)):
        raise ValueError("proposal is not before downstream entry")
    if int(row.get("actor_packet_fifo_valid", 0)) != 3:
        raise ValueError("predecessor lacks three real actor packets")
