"""Pure contracts for the fixed Descent feedback-teacher probe."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


TICK_REGIONS = {"early": range(0, 3), "middle": range(3, 6), "late": range(6, 8)}


def select_three_ticks(old_audits: Sequence[Mapping]) -> list[int]:
    """Select three ticks before search, maximizing accepted audit evidence.

    The historical audit has four accepted ticks for one candidate, so the
    exact three-per-candidate quota cannot contain all eleven accepted rows.
    We retain the lexicographically earliest three accepted ticks and fill
    deterministically with region representatives, then remaining ticks.
    """
    accepted = sorted(int(row["tick"]) for row in old_audits if row.get("accepted"))
    selected = accepted[:3]
    region_representatives = (1, 4, 7)
    for tick in region_representatives + tuple(range(8)):
        if len(selected) == 3:
            break
        if tick not in selected:
            selected.append(tick)
    return sorted(selected)


def tick_region(tick: int) -> str:
    return next(name for name, ticks in TICK_REGIONS.items() if int(tick) in ticks)


def physical_order(rows: Sequence[Mapping]) -> list[int]:
    """Stable lexicographic physical ranking, with effort only as tie-break."""
    return sorted(range(len(rows)), key=lambda index: (
        -int(rows[index]["survival"]),
        -float(rows[index]["minimum_margin"]),
        -float(rows[index]["terminal_margin"]),
        float(rows[index]["residual_rms"]),
        int(rows[index].get("generation", 0)),
        int(rows[index].get("sample", 0)),
    ))


def distinct_top_sequences(rows: Sequence[Mapping], count: int = 5,
                           minimum_action_distance: float = 1e-5) -> list[Mapping]:
    chosen = []
    for index in physical_order(rows):
        row = rows[index]
        action = np.asarray(row["actions"])
        if any(float(np.linalg.norm(action - np.asarray(old["actions"]))) <= minimum_action_distance
               for old in chosen):
            continue
        chosen.append(row)
        if len(chosen) == int(count):
            break
    return chosen


def local_support_gate(rows: Sequence[Mapping]) -> dict:
    passed = [row for row in rows if row["authoritative_correction"]]
    candidates = sorted({row["candidate_id"] for row in rows})
    candidate_counts = {candidate: sum(
        row["authoritative_correction"] for row in rows if row["candidate_id"] == candidate
    ) for candidate in candidates}
    supported = sum(value >= 2 for value in candidate_counts.values())
    gate = len(passed) >= 16 and supported >= 6
    return {"status": "PASS" if gate else "FAIL", "snapshot_passes": len(passed),
            "snapshot_total": len(rows), "candidate_pass_counts": candidate_counts,
            "supported_candidates": supported, "candidate_total": len(candidates),
            "gate": gate}
