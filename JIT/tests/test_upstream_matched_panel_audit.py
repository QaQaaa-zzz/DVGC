from __future__ import annotations

import json


def test_real_matched_panel_config_loads(jit_root):
    from jit_dvgc.upstream_matched_panel_audit import (
        load_upstream_matched_panel_audit_config,
    )

    config = load_upstream_matched_panel_audit_config(
        jit_root / "configs/envelope_iter0_upstream_matched_panel_audit.json"
    )
    assert config["expected_protocol_sha256"] == (
        "7ed71a260e631b42c26a090691b67c20f09310a342dc98b66599452fb99b024c"
    )
    assert config["protocol"]["matched_panel"]["expected_total_cells"] == 240


def test_target_panel_match_is_exact():
    from jit_dvgc.upstream_matched_panel_audit import _is_target_panel

    panel = {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.025, 0.1],
        "durations": [4, 8, 16],
    }
    assert _is_target_panel(
        {"action_name": "hip", "sign": 1, "strength": 0.1, "duration": 16}, panel
    )
    assert not _is_target_panel(
        {"action_name": "hip", "sign": 1, "strength": 0.05, "duration": 16}, panel
    )
    assert not _is_target_panel(
        {"action_name": "hip", "sign": 1, "strength": 0.1, "duration": 32}, panel
    )


def test_catalog_join_uses_state_and_acquisition_protocol(tmp_path):
    from jit_dvgc.upstream_matched_panel_audit import _candidate_metadata_by_state

    state = "a" * 64
    protocol = "b" * 64
    target_rows = [
        {
            "state_sha256": state,
            "parent_group_id": "transition_4988928__1000001",
            "label": 0,
            "acquisition_protocol_sha256": protocol,
        }
    ]
    good = tmp_path / "catalog.json"
    good.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol,
                "entries": [
                    {
                        "candidate_id": "run_local_0001",
                        "phase": "upstream",
                        "state_sha256": state,
                        "parent_group_id": "transition_4988928__1000001",
                        "perturbation": {
                            "action_name": "hip",
                            "sign": 1,
                            "strength": 0.1,
                            "duration": 8,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    wrong = tmp_path / "candidate_catalog.json"
    wrong.write_text(
        json.dumps(
            {
                "protocol_sha256": "c" * 64,
                "entries": [
                    {
                        "candidate_id": "same_local_id",
                        "phase": "upstream",
                        "state_sha256": state,
                        "parent_group_id": "transition_4988928__1000001",
                        "perturbation": {
                            "action_name": "knee",
                            "sign": -1,
                            "strength": 0.025,
                            "duration": 4,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    matches, scan = _candidate_metadata_by_state(target_rows, [good, wrong])
    assert set(matches) == {state}
    assert matches[state]["action_name"] == "hip"
    assert matches[state]["duration"] == 8
    assert scan["reconstructed_by_acquisition_protocol_sha256"] == {protocol: 1}
