from __future__ import annotations

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_PRETAKEOFF_AIRBORNE, OrangeBikeDVGC
from dvgc.failure_video import (
    capture_failure_scenario,
    render_failure_archive,
    render_failure_trace,
    write_failure_video_manifest,
)
from dvgc.reference import ReferenceTrajectory
from dvgc.two_phase_runtime import TwoPhaseThresholds, build_two_phase_geometry
from dvgc.two_phase_semantics import ApexBandThresholds, RecoveryThresholds


def _runtime():
    cfg = load_config(
        "configs/default.json",
        {
            "domain_randomization": False,
            "obs_noise_enable": False,
            "use_bank_resets": False,
        },
    )
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    reference = ReferenceTrajectory.load("data/reference_jump.csv")
    geometry = build_two_phase_geometry(env.mj_model, cfg)
    thresholds = TwoPhaseThresholds(
        apex=ApexBandThresholds(
            max_abs_com_vz=0.07,
            min_clearance=0.11,
            max_abs_roll=0.08,
            max_abs_pitch=0.06,
            max_angular_speed=1.25,
            min_forward_velocity=3.5,
            relative_x_min=0.15,
            relative_x_max=0.37,
        ),
        recovery=RecoveryThresholds(
            max_abs_roll=0.28,
            max_abs_pitch=0.09,
            max_angular_speed=2.10,
            min_forward_velocity=2.58,
            required_hold_ticks=25,
        ),
    )
    return cfg, env, reference, geometry, thresholds


def test_full_guideline_failure_trace_reproduces_prelaunch_airborne():
    _, env, reference, geometry, thresholds = _runtime()

    trace = capture_failure_scenario(
        env,
        reference,
        geometry,
        thresholds,
        scenario="full_guideline_prelaunch_airborne",
        seed=44_000,
    )

    assert trace.summary["start_reference_index"] == 0
    assert trace.summary["environment_transitions"] == 12
    assert trace.summary["formal_training_transitions"] == 0
    assert trace.summary["terminal"] is True
    assert trace.summary["end_code"] == END_PRETAKEOFF_AIRBORNE
    assert trace.summary["failure_reason"] == "prelaunch_airborne"
    assert [row["action_reference_index"] for row in trace.telemetry[1:4]] == [10, 20, 30]
    assert len(trace.frames) == len(trace.telemetry) == 13


def test_launch_history_trace_shows_support_lost_before_window_entry():
    _, env, reference, geometry, thresholds = _runtime()

    trace = capture_failure_scenario(
        env,
        reference,
        geometry,
        thresholds,
        scenario="launch_history_airborne_before_window",
        seed=44_000,
    )

    assert trace.summary["start_reference_index"] == 83
    assert trace.summary["environment_transitions"] == 8
    assert trace.summary["formal_training_transitions"] == 0
    assert trace.summary["failure_reason"] == "airborne_before_jump_window_latch"
    inside = next(row for row in trace.telemetry if row["inside_jump_window"])
    assert inside["tick"] == 6
    assert inside["host_wheel_contacts"] == 1
    assert inside["deployable_wheel_support"] is False
    assert inside["jump_signal_latched"] is False
    assert [row["action_reference_index"] for row in trace.telemetry[1:4]] == [93, 103, 113]
    assert len(trace.frames) == len(trace.telemetry) == 9


def test_failure_trace_is_exactly_reproducible_for_same_seed():
    _, env, reference, geometry, thresholds = _runtime()
    kwargs = dict(
        env=env,
        reference=reference,
        geometry=geometry,
        thresholds=thresholds,
        scenario="launch_history_airborne_before_window",
        seed=44_000,
    )

    first = capture_failure_scenario(**kwargs)
    second = capture_failure_scenario(**kwargs)

    assert first.summary == second.summary
    assert first.telemetry == second.telemetry
    for left, right in zip(first.frames, second.frames):
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])


def test_render_uses_captured_states_without_advancing_environment(tmp_path, monkeypatch):
    _, env, reference, geometry, thresholds = _runtime()
    trace = capture_failure_scenario(
        env,
        reference,
        geometry,
        thresholds,
        scenario="launch_history_airborne_before_window",
        seed=44_000,
    )
    monkeypatch.setattr(
        env,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("renderer must not call env.step")
        ),
    )
    output = tmp_path / "launch_history_airborne_before_window.mp4"

    report = render_failure_trace(
        env,
        trace,
        output,
        width=320,
        height=180,
        fps=20,
    )

    assert output.is_file() and output.stat().st_size > 1_000
    assert report["video_sha256"] == file_sha256(output)
    assert report["scenario"] == trace.scenario.name
    assert report["formal_training_transitions"] == 0
    assert report["telemetry"] == list(trace.telemetry)


def test_manifest_requires_both_named_nonempty_mp4_files(tmp_path):
    videos = []
    for name, payload in (
        ("full_guideline_prelaunch_airborne", b"full-video"),
        ("launch_history_airborne_before_window", b"launch-video"),
    ):
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(payload)
        videos.append(
            {
                "scenario": name,
                "video": str(path),
                "video_sha256": file_sha256(path),
                "formal_training_transitions": 0,
                "telemetry": [],
            }
        )

    manifest = write_failure_video_manifest(
        tmp_path / "failure_video_manifest.json",
        videos,
        source_hashes={"xml": "a" * 64, "config": "b" * 64, "reference": "c" * 64},
    )

    assert manifest["status"] == "pass"
    assert manifest["formal_training_transitions"] == 0
    assert {row["scenario"] for row in manifest["videos"]} == {
        "full_guideline_prelaunch_airborne",
        "launch_history_airborne_before_window",
    }
    assert (tmp_path / "failure_video_manifest.json").is_file()


def test_archive_renders_both_real_failure_scenarios(tmp_path):
    cfg, env, reference, geometry, thresholds = _runtime()

    manifest = render_failure_archive(
        env,
        reference,
        geometry,
        thresholds,
        output_dir=tmp_path,
        seed=44_000,
        source_hashes={"xml": "a" * 64, "config": "b" * 64, "reference": "c" * 64},
        width=320,
        height=180,
        fps=20,
    )

    assert manifest["status"] == "pass"
    assert sum(row["environment_transitions"] for row in manifest["videos"]) == 20
    for scenario in (
        "full_guideline_prelaunch_airborne",
        "launch_history_airborne_before_window",
    ):
        path = tmp_path / f"{scenario}.mp4"
        assert path.is_file() and path.stat().st_size > 1_000
