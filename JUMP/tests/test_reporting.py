"""Reporting must retain failed/unknown cases and the true observation extent."""

import hashlib
import json


def test_report_keeps_failure_endpoint_and_unknown_denominator(tmp_path, monkeypatch):
    from jump_planning.reporting import build_report
    import matplotlib.pyplot as plt
    import numpy as np

    closed_figures = []
    original_close = plt.close

    def capture_closed_figure(figure=None):
        if hasattr(figure, "axes"):
            closed_figures.append(figure)
        original_close(figure)

    monkeypatch.setattr(plt, "close", capture_closed_figure)

    (tmp_path / "trajectories").mkdir()
    trajectory = tmp_path / "trajectories" / "failed.jsonl"
    rows = [
        {
            "time": t, "x": t, "z": 0.2, "vx": 1.0,
            "roll": 0.05, "pitch": -0.01, "yaw": 0.5235987755982988,
            "wx": 0.1, "wy": -0.2, "wz": 1.0 + t,
            "signal": int(t >= 0.02), "action": [0.1, -0.2, 0.3, -0.4],
            "ctrl": [0.08, -8.0, 0.5, -0.7], "qpos": [], "qvel": [],
        }
        for t in [0.0, 0.02, 0.04]
    ]
    trajectory.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original_hash = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    (tmp_path / "results.json").write_text(json.dumps({
        "metadata": {"config": {"limits": {"nominal_speed": 1.0, "min_forward_speed": 0.5}}},
        "accounting": {"charged_control_steps": 2, "physics_steps": 8},
        "cases": [
            {"case_id": "failed", "policy": "frozen", "case_name": "trigger", "seed": 4,
             "status": "failure", "reason": "obstacle_contact", "qualified": True,
             "trigger_time": 0.02, "liftoff_time": None, "charged_control_steps": 2,
             "physics_steps": 8, "wall_seconds": 1.2, "trajectory": "trajectories/failed.jsonl"},
            {"case_id": "missing", "policy": "frozen", "case_name": "never", "seed": 5,
             "status": "unknown", "reason": "runtime_error", "qualified": False,
             "trigger_time": None, "liftoff_time": None, "charged_control_steps": 0,
             "physics_steps": 0, "wall_seconds": 0.1, "trajectory": None},
        ],
    }))

    report = build_report(tmp_path)

    assert report["counts"] == {"success": 0, "failure": 1, "unknown": 1}
    assert report["cases"][0]["recorded_rows"] == 3
    assert report["cases"][0]["last_recorded_time"] == 0.04
    assert report["cases"][1]["recorded_rows"] == 0
    assert report["cases"][1]["warnings"]
    for case in report["cases"]:
        assert (tmp_path / case["png"]).read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / case["pdf"]).read_bytes().startswith(b"%PDF")
    index = (tmp_path / "analysis" / "INDEX.md").read_text()
    assert "obstacle_contact" in index
    assert "runtime_error" in index
    assert "0.040000" in index
    assert "trajectories/failed.jsonl" in index
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == original_hash
    actual_series = {line.get_label(): line.get_ydata()
                     for figure in closed_figures for ax in figure.axes for line in ax.lines}
    np.testing.assert_allclose(actual_series["yaw"], [30.0, 30.0, 30.0])
    np.testing.assert_allclose(actual_series["wz"], [1.0, 1.02, 1.04])


def test_report_records_broken_trajectory_without_dropping_the_case(tmp_path):
    from jump_planning.reporting import build_report

    (tmp_path / "broken.jsonl").write_text('{"time": 0.0, "vx": 1.0}\nnot json\n{"time": 0.02}\n')
    (tmp_path / "results.json").write_text(json.dumps({
        "cases": [{"case_id": "broken", "status": "unknown", "trajectory": "broken.jsonl"}],
        "metadata": {}, "accounting": {},
    }))
    report = build_report(tmp_path)
    assert report["counts"]["unknown"] == 1
    assert report["cases"][0]["recorded_rows"] == 3
    assert any("line 2" in warning for warning in report["cases"][0]["warnings"])
    assert report["cases"][0]["last_recorded_time"] == 0.02


def test_identical_untriggered_runs_are_one_trace_without_relabeling_cases(tmp_path):
    from jump_planning.reporting import build_report

    raw = '{"time": 0.02, "vx": 1.0, "yaw": 0.1, "wz": 0.5, "signal": 0}\n'
    (tmp_path / "first.jsonl").write_text(raw)
    (tmp_path / "second.jsonl").write_text(raw)
    (tmp_path / "results.json").write_text(json.dumps({
        "metadata": {}, "accounting": {}, "cases": [
            {"case_id": "early", "case_name": "early", "status": "failure", "reason": "yaw_limit",
             "trigger_time": None, "trajectory": "first.jsonl"},
            {"case_id": "late", "case_name": "late", "status": "failure", "reason": "yaw_limit",
             "trigger_time": None, "trajectory": "second.jsonl"},
            {"case_id": "error", "status": "unknown", "reason": "runtime_error", "trajectory": None},
        ],
    }))
    report = build_report(tmp_path)
    assert report["counts"] == {"success": 0, "failure": 2, "unknown": 1}
    assert report["unique_physical_trace_count"] == 1
    assert report["recorded_trace_case_count"] == 2
    assert report["untriggered_recorded_case_count"] == 2
    assert list(report["physical_trace_groups"].values()) == [["early", "late"]]
    index = (tmp_path / "analysis" / "INDEX.md").read_text()
    assert "Unique physical traces by trajectory SHA-256: **1**" in index
    assert "cannot serve as independent trigger-timing comparisons" in index
