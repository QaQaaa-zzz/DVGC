from __future__ import annotations

import numpy as np

from jit_dvgc.constants import REWARD_COMPONENT_KEYS
from jit_dvgc.diagnostics import save_trace_dashboard, telemetry_panel
from jit_dvgc.evaluation import EpisodeFrame, EpisodeTrace


def _trace() -> EpisodeTrace:
    frames = []
    for tick in range(4):
        components = {name: 0.0 for name in REWARD_COMPONENT_KEYS}
        components["height"] = float(2 * tick)
        components["pitch"] = float(-tick)
        clipped = float(tick)
        frames.append(
            EpisodeFrame(
                qpos=np.asarray([1.5 + tick, 0.1, 0.2 + tick, *([0.0] * 9)]),
                qvel=np.asarray([2.0, 0.0, 1.0 - tick, *([0.0] * 8)]),
                ctrl=np.asarray([0.0, 12.0, -1.0, 2.0]),
                action=np.asarray([0.0, 0.5, -0.2, 0.1]),
                reward=clipped,
                reward_components=components,
                metrics={
                    "reward": clipped,
                    "reward/unclipped": clipped + 0.5,
                    "signal/root_x": 1.5 + tick,
                    "signal/root_y": 0.1,
                    "signal/root_z": 0.2 + tick,
                    "signal/forward_velocity": 2.0,
                    "signal/lateral_velocity": 0.0,
                    "signal/vertical_velocity": 1.0 - tick,
                    "signal/roll": 0.01 * tick,
                    "signal/pitch": -0.02 * tick,
                    "signal/yaw": 0.03 * tick,
                    "signal/roll_rate": 0.1 * tick,
                    "signal/pitch_rate": 0.2 * tick,
                    "signal/yaw_rate": 0.3 * tick,
                    "signal/hip_force": -5.0,
                    "signal/knee_force": 6.0,
                    "signal/joint_power": 4.0,
                    "event/jump_signal": float(tick == 1),
                    "event/jump_zone_seen": float(tick >= 1),
                    "event/jump_zone_consumed": float(tick >= 2),
                    "event/ascending_seen": float(tick >= 1),
                    "event/height_seen": float(tick >= 1),
                    "event/apex_seen": float(tick == 3),
                },
                terminated=tick == 3,
                truncated=False,
                end_code=1 if tick == 3 else 0,
                success=tick == 3,
                physical_failure=False,
                timeout=False,
            )
        )
    return EpisodeTrace(seed=7, frames=tuple(frames), environment_transitions=3)


def test_dashboard_saves_png_and_complete_numeric_npz(tmp_path):
    report = save_trace_dashboard(
        _trace(), tmp_path / "diagnostic.png", reward_scaling=0.1, fps=50
    )

    assert report.plot.is_file() and report.plot.stat().st_size > 1000
    assert report.data.is_file() and report.data.stat().st_size > 1000
    assert len(report.plot_sha256) == 64
    assert len(report.data_sha256) == 64
    with np.load(report.data) as arrays:
        required = {
            "time_seconds",
            "reward_clipped",
            "reward_unclipped",
            "reward_scaled",
            "qpos",
            "qvel",
            "ctrl",
            "action",
            "terminal_end_code",
            "metric__signal__root_x",
            "metric__signal__root_z",
            "metric__signal__pitch",
            "metric__event__jump_signal",
            "reward_component__height",
            "reward_component__physical_failure",
        }
        assert required.issubset(arrays.files)
        np.testing.assert_allclose(arrays["reward_scaled"], [0.0, 0.1, 0.2, 0.3])
        assert arrays["qpos"].shape == (4, 12)


def test_telemetry_panel_has_requested_size_and_visible_content():
    panel = telemetry_panel(_trace().frames[-1], width=320, height=180, tick=3)

    assert panel.shape == (180, 320, 3)
    assert panel.dtype == np.uint8
    assert panel.std() > 1.0
