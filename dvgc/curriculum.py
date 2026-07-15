"""Deterministic reset-support schedules for event-anchored Flight training."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


FLIGHT_RESET_STAGES = ("late_descent", "descent", "apex", "apex_bridge", "ascent", "full", "apex_descent")


def select_flight_reset_records(records: Sequence[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    if stage not in FLIGHT_RESET_STAGES:
        raise ValueError(f"Unknown Flight reset stage: {stage}")
    rows = [row for row in records if not row.get("training_only", False)]
    if stage == "full":
        selected = rows
    elif stage == "apex_descent":
        selected = [row for row in rows if row.get("flight_subinterval") in ("apex", "descent")]
    elif stage == "apex_bridge":
        apex = [row for row in rows if row.get("flight_subinterval") == "apex"]
        descent = sorted(
            (row for row in rows if row.get("flight_subinterval") == "descent"),
            key=lambda row: (row.get("reference_index", row.get("source_index", 10**9)), row.get("id", "")),
        )
        selected = apex + descent[: len(apex)]
    elif stage in ("apex", "ascent"):
        selected = [row for row in rows if row.get("flight_subinterval") == stage]
    elif stage == "descent":
        selected = [row for row in rows if row.get("flight_subinterval") == "descent"]
    else:
        descent = [row for row in rows if row.get("flight_subinterval") == "descent"]
        indices = sorted({int(row["source_index"]) for row in descent})
        if not indices:
            selected = []
        else:
            cutoff = indices[len(indices) // 2]
            selected = [row for row in descent if int(row["source_index"]) >= cutoff]
    if not selected:
        raise ValueError(f"Flight reset stage {stage} selected no candidates")
    return selected
