from __future__ import annotations

from copy import deepcopy
import json

import pytest

from jit_dvgc.unified_transition_band_search import (
    load_transition_band_search_config,
    phase_transition_band_readiness,
    unique_label_rows,
)


def _row(*, state: str, phase: str, label: int, group: str):
    return {
        "state_sha256": state,
        "phase": phase,
        "label": label,
        "parent_group_id": group,
        "split": "train",
    }


def test_iter0_automated_search_config_is_predeclared_and_duration_only(jit_root):
    path = jit_root / "configs/envelope_iter0_transition_band_search.json"
    config = load_transition_band_search_config(path)

    assert config["iteration"] == 0
    assert config["fixed_acquisition"]["single_variable"] == "perturbation_duration_only"
    assert config["fixed_acquisition"]["strengths"] == [0.025, 0.05, 0.1]
    assert config["fixed_acquisition"]["action_names"] == [
        "steer",
        "rear_wheel_drive",
        "hip",
        "knee",
    ]
    assert [shell["durations"] for shell in config["outer_shells"]] == [
        [4, 8],
        [16, 32],
        [64, 128],
    ]
    assert config["readiness"] == {
        "minimum_positive_candidates": 20,
        "minimum_negative_candidates": 20,
        "minimum_parent_groups_with_positive": 3,
        "minimum_parent_groups_with_negative": 3,
        "phasewise_stopping": True,
    }


def test_config_rejects_posthoc_strength_change(tmp_path, jit_root):
    original = json.loads(
        (jit_root / "configs/envelope_iter0_transition_band_search.json").read_text()
    )
    changed = deepcopy(original)
    changed["fixed_acquisition"]["strengths"] = [0.05, 0.1, 0.2]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="strengths must remain"):
        load_transition_band_search_config(path)


def test_config_rejects_nonmonotonic_duration_schedule(tmp_path, jit_root):
    original = json.loads(
        (jit_root / "configs/envelope_iter0_transition_band_search.json").read_text()
    )
    changed = deepcopy(original)
    changed["outer_shells"][1]["durations"] = [8, 16]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="increase monotonically"):
        load_transition_band_search_config(path)


def test_unique_label_rows_deduplicates_same_physical_state_without_inflating_counts():
    repeated = _row(state="a" * 64, phase="upstream", label=1, group="g1")
    rows = unique_label_rows([[repeated], [dict(repeated)]])
    assert len(rows) == 1


def test_unique_label_rows_rejects_conflicting_duplicate_label():
    positive = _row(state="a" * 64, phase="upstream", label=1, group="g1")
    negative = _row(state="a" * 64, phase="upstream", label=0, group="g1")
    with pytest.raises(ValueError, match="conflicting"):
        unique_label_rows([[positive], [negative]])


def test_readiness_is_phasewise_and_requires_three_failure_groups():
    criteria = {
        "minimum_positive_candidates": 20,
        "minimum_negative_candidates": 20,
        "minimum_parent_groups_with_positive": 3,
        "minimum_parent_groups_with_negative": 3,
    }
    rows = []
    for phase in ("upstream", "downstream"):
        for i in range(20):
            rows.append(
                _row(
                    state=f"{phase[0]}p{i:062d}",
                    phase=phase,
                    label=1,
                    group=f"{phase}_pos_{i % 3}",
                )
            )
        for i in range(20):
            rows.append(
                _row(
                    state=f"{phase[0]}n{i:062d}",
                    phase=phase,
                    label=0,
                    group=f"{phase}_neg_{i % 3}",
                )
            )

    ready = phase_transition_band_readiness(rows, criteria)
    assert ready["upstream"]["ready"] is True
    assert ready["downstream"]["ready"] is True

    # Collapse downstream failures to one parent trajectory family: count alone
    # must not make the phase ready.
    for row in rows:
        if row["phase"] == "downstream" and row["label"] == 0:
            row["parent_group_id"] = "one_failure_group"
    ready = phase_transition_band_readiness(rows, criteria)
    assert ready["upstream"]["ready"] is True
    assert ready["downstream"]["ready"] is False
    assert ready["downstream"]["negative_count"] == 20
    assert ready["downstream"]["negative_parent_group_count"] == 1
