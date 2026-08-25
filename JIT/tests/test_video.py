from __future__ import annotations

import mediapy as media
import numpy as np

from jit_dvgc.config import load_config
from jit_dvgc.constants import REWARD_COMPONENT_KEYS
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.evaluation import EpisodeFrame, EpisodeTrace
from jit_dvgc.video import render_trace


def test_video_encodes_each_captured_state_once_without_stepping(jit_root, tmp_path):
    config = load_config(jit_root / "configs" / "phase_u_smoke.json")
    env = TwoPhaseBikeEnv(config, convert_model=False)
    key = env.mj_model.key_qpos[0]
    frames = []
    for tick in range(4):
        qpos = np.asarray(key).copy()
        qpos[0] += 0.01 * tick
        frames.append(
            EpisodeFrame(
                qpos=qpos,
                qvel=np.zeros(env.mj_model.nv),
                ctrl=np.array([0.0, 12.0, -1.2, 2.5]),
                action=np.zeros(4),
                reward=float(tick),
                reward_components={
                    name: (float(tick) if name == "height" else 0.0)
                    for name in REWARD_COMPONENT_KEYS
                },
                metrics={
                    "reward": float(tick),
                    "reward/unclipped": float(tick) + 0.25,
                    "signal/root_x": float(qpos[0]),
                    "signal/root_y": float(qpos[1]),
                    "signal/root_z": float(qpos[2]),
                    "signal/roll": 0.0,
                    "signal/pitch": 0.0,
                    "signal/yaw": 0.0,
                    "event/jump_signal": float(tick == 1),
                    "event/jump_zone_seen": float(tick >= 1),
                    "event/height_seen": 0.0,
                    "event/apex_seen": 0.0,
                },
                terminated=tick == 3,
                truncated=False,
                end_code=4 if tick == 3 else 0,
                success=False,
                physical_failure=tick == 3,
                timeout=False,
            )
        )
    trace = EpisodeTrace(seed=1, frames=tuple(frames), environment_transitions=3)
    env.step = lambda *_args: (_ for _ in ()).throw(
        AssertionError("renderer must never call env.step")
    )
    output = tmp_path / "trace.mp4"

    report = render_trace(
        env, trace, output, fps=50, width=320, height=180, reward_scaling=0.1
    )

    assert report.captured_state_count == 4
    assert report.encoded_frame_count == 4
    assert report.environment_transitions == 3
    assert output.is_file() and output.stat().st_size > 1000
    video = media.read_video(output)
    assert len(video) == 4
    assert video.shape[2] == 640
    assert report.diagnostic_plot.endswith(".png")
    assert report.diagnostic_data.endswith(".npz")
    assert len(report.video_sha256) == 64
    assert len(report.diagnostic_plot_sha256) == 64
    assert len(report.diagnostic_data_sha256) == 64
