from __future__ import annotations

import jax
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_PRETAKEOFF_AIRBORNE, END_ROLL_LIMIT, OrangeBikeDVGC
from dvgc.failure_video import (
    _capture_failure_scenario,
    _expected_action_indices,
    render_failure_archive,
    render_failure_trace,
    write_failure_video_manifest,
)
from dvgc.reference import ReferenceTrajectory
from dvgc.two_phase_runtime import TwoPhaseThresholds, build_two_phase_geometry
from dvgc.two_phase_semantics import ApexBandThresholds, RecoveryThresholds


_CACHED_RUNTIME = None
_CACHED_STEP = None


def _source_hashes():
    return {
        name: character * 64
        for name, character in zip(
            (
                "xml",
                "config",
                "reference",
                "environment",
                "rewards",
                "failure_video",
                "two_phase_runtime",
                "two_phase_guideline",
                "builder",
            ),
            "123456789",
        )
    }


def _runtime():
    global _CACHED_RUNTIME, _CACHED_STEP
    if _CACHED_RUNTIME is not None:
        return _CACHED_RUNTIME
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
    _CACHED_RUNTIME = (cfg, env, reference, geometry, thresholds)
    _CACHED_STEP = jax.jit(env.step)
    return _CACHED_RUNTIME


def _capture(env, reference, geometry, thresholds, *, scenario, seed):
    return _capture_failure_scenario(
        env,
        reference,
        geometry,
        thresholds,
        scenario=scenario,
        seed=seed,
        step=_CACHED_STEP,
    )


def test_full_guideline_trace_continues_then_reproduces_roll_limit():
    _, env, reference, geometry, thresholds = _runtime()

    trace = _capture(
        env,
        reference,
        geometry,
        thresholds,
        scenario="full_guideline_continuation",
        seed=44_000,
    )

    assert trace.summary["start_reference_index"] == 0
    assert trace.summary["environment_transitions"] == 17
    assert trace.summary["formal_training_transitions"] == 0
    assert trace.summary["terminal"] is True
    assert trace.summary["end_code"] == END_ROLL_LIMIT
    assert trace.summary["end_code"] != END_PRETAKEOFF_AIRBORNE
    assert trace.summary["audit_outcome"] == "roll_limit"
    assert trace.summary["terminal_reason"] == "roll_limit"
    assert trace.telemetry[-1]["jump_signal_latched"] is True
    assert trace.summary["first_event_ticks"]["jump_window_entered"] == -1
    assert [row["action_reference_index"] for row in trace.telemetry[1:4]] == [10, 20, 30]
    assert len(trace.frames) == len(trace.telemetry) == 18


def test_launch_history_trace_latches_window_despite_lost_support():
    _, env, reference, geometry, thresholds = _runtime()

    trace = _capture(
        env,
        reference,
        geometry,
        thresholds,
        scenario="launch_history_window_latch",
        seed=44_000,
    )

    assert trace.summary["start_reference_index"] == 83
    assert trace.summary["initial_action_reference_index"] == 73
    assert trace.summary["environment_transitions"] == 8
    assert trace.summary["formal_training_transitions"] == 0
    assert trace.summary["audit_outcome"] == "window_latched_after_early_airborne"
    assert trace.summary["terminal_reason"] == "horizon"
    inside = next(row for row in trace.telemetry if row["inside_jump_window"])
    assert inside["tick"] == 6
    assert inside["host_wheel_contacts"] == 1
    assert inside["deployable_wheel_support"] is False
    assert inside["jump_signal_latched"] is True
    assert trace.summary["first_event_ticks"]["jump_window_entered"] == 6
    assert [row["action_reference_index"] for row in trace.telemetry[1:4]] == [83, 93, 103]
    assert len(trace.frames) == len(trace.telemetry) == 9


def test_failure_trace_is_exactly_reproducible_for_same_seed():
    _, env, reference, geometry, thresholds = _runtime()
    kwargs = dict(
        env=env,
        reference=reference,
        geometry=geometry,
        thresholds=thresholds,
        scenario="launch_history_window_latch",
        seed=44_000,
    )

    first = _capture_failure_scenario(**kwargs, step=_CACHED_STEP)
    second = _capture_failure_scenario(**kwargs, step=_CACHED_STEP)

    assert first.summary == second.summary
    assert first.telemetry == second.telemetry
    for left, right in zip(first.frames, second.frames):
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])


def test_render_uses_captured_states_without_advancing_environment(tmp_path, monkeypatch):
    _, env, reference, geometry, thresholds = _runtime()
    trace = _capture(
        env,
        reference,
        geometry,
        thresholds,
        scenario="launch_history_window_latch",
        seed=44_000,
    )
    monkeypatch.setattr(
        env,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("renderer must not call env.step")
        ),
    )
    output = tmp_path / "launch_history_window_latch.mp4"

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
    assert report["frame_count"] == len(trace.frames)
    assert len(report["state_trace_sha256"]) == 64
    assert report["state_trace_file_sha256"] == file_sha256(report["state_trace"])
    with np.load(report["state_trace"], allow_pickle=False) as states:
        assert set(states.files) == {"qpos", "qvel", "ctrl"}
        assert states["qpos"].shape[0] == len(trace.frames)
    assert set(report["summary"]["first_event_ticks"]) == set(
        trace.telemetry[0]["events"]
    )


def test_manifest_rejects_unclosed_fake_video_reports(tmp_path):
    videos = []
    for name, payload in (
        ("full_guideline_continuation", b"full-video"),
        ("launch_history_window_latch", b"launch-video"),
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

    with np.testing.assert_raises_regex(ValueError, "audit report is incomplete"):
        write_failure_video_manifest(
            tmp_path / "failure_video_manifest.json",
            videos,
            source_hashes=_source_hashes(),
        )


def test_archive_renders_both_real_failure_scenarios(tmp_path):
    cfg, env, reference, geometry, thresholds = _runtime()

    manifest = render_failure_archive(
        env,
        reference,
        geometry,
        thresholds,
        output_dir=tmp_path,
        seed=44_000,
        source_hashes=_source_hashes(),
        width=320,
        height=180,
        fps=20,
    )

    assert manifest["status"] == "pass"
    assert sum(row["environment_transitions"] for row in manifest["videos"]) == 25
    assert manifest["environment_transitions"] == 25
    assert set(manifest["source_hashes"]) == set(_source_hashes())
    for scenario in (
        "full_guideline_continuation",
        "launch_history_window_latch",
    ):
        path = tmp_path / f"{scenario}.mp4"
        assert path.is_file() and path.stat().st_size > 1_000
        row = next(item for item in manifest["videos"] if item["scenario"] == scenario)
        assert row["frame_count"] == len(row["telemetry"])
        assert len(row["state_trace_sha256"]) == 64
        assert file_sha256(row["state_trace"]) == row["state_trace_file_sha256"]


def test_manifest_rejects_action_schedule_tampering(tmp_path):
    cfg, env, reference, geometry, thresholds = _runtime()
    manifest = render_failure_archive(
        env,
        reference,
        geometry,
        thresholds,
        output_dir=tmp_path,
        seed=44_000,
        source_hashes=_source_hashes(),
        width=320,
        height=180,
        fps=20,
    )
    videos = manifest["videos"]
    videos[1]["telemetry"][1]["action_reference_index"] += 10

    with np.testing.assert_raises_regex(ValueError, "action schedule"):
        write_failure_video_manifest(
            tmp_path / "tampered.json",
            videos,
            source_hashes=manifest["source_hashes"],
        )


def test_manifest_rejects_launch_history_outcome_tampering(tmp_path):
    _, env, reference, geometry, thresholds = _runtime()
    manifest = render_failure_archive(
        env,
        reference,
        geometry,
        thresholds,
        output_dir=tmp_path,
        seed=44_000,
        source_hashes=_source_hashes(),
        width=320,
        height=180,
        fps=20,
    )
    videos = manifest["videos"]
    launch = next(
        row for row in videos if row["scenario"] == "launch_history_window_latch"
    )
    launch["summary"].update(
        {
            "terminal": True,
            "terminated": True,
            "truncated": False,
            "end_code": END_ROLL_LIMIT,
            "terminal_reason": "roll_limit",
        }
    )

    with np.testing.assert_raises_regex(ValueError, "observed outcome"):
        write_failure_video_manifest(
            tmp_path / "tampered-outcome.json",
            videos,
            source_hashes=manifest["source_hashes"],
        )


def test_expected_action_schedule_clamps_at_reference_end():
    indices = _expected_action_indices(
        scenario="full_guideline_continuation",
        stride=10,
        frame_count=101,
        reference_row_count=822,
    )

    assert indices[:3] == [0, 10, 20]
    assert indices[82] == 820
    assert indices[83:] == [821] * 18
