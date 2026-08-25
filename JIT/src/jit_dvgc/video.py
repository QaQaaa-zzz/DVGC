"""Host-only rendering of saved state traces with truthful frame accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np

from .diagnostics import save_trace_dashboard, sha256_file, telemetry_panel
from .evaluation import EpisodeTrace


@dataclass(frozen=True)
class VideoReport:
    video: str
    state_trace: str
    diagnostic_plot: str
    diagnostic_data: str
    video_sha256: str
    diagnostic_plot_sha256: str
    diagnostic_data_sha256: str
    captured_state_count: int
    encoded_frame_count: int
    environment_transitions: int
    fps: int


def render_trace(
    env,
    trace: EpisodeTrace,
    path: Path,
    *,
    fps: int = 50,
    width: int = 640,
    height: int = 360,
    reward_scaling: float = 0.1,
) -> VideoReport:
    if len(trace.frames) != trace.environment_transitions + 1:
        raise ValueError("captured state count must equal transitions plus one")
    if fps <= 0:
        raise ValueError("video fps must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(env.mj_model)
    renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
    rendered: list[np.ndarray] = []
    try:
        for tick, frame in enumerate(trace.frames):
            data.qpos[:] = frame.qpos
            data.qvel[:] = frame.qvel
            data.ctrl[:] = frame.ctrl
            mujoco.mj_forward(env.mj_model, data)
            renderer.update_scene(data, camera="chasis_camera")
            physical = renderer.render().copy()
            telemetry = telemetry_panel(
                frame,
                width=width,
                height=height,
                tick=tick,
                reward_scaling=reward_scaling,
            )
            rendered.append(np.concatenate((physical, telemetry), axis=1))
    finally:
        renderer.close()
    media.write_video(output, rendered, fps=fps)
    diagnostic = save_trace_dashboard(
        trace,
        output.with_name(f"{output.stem}_diagnostic.png"),
        reward_scaling=reward_scaling,
        fps=fps,
    )
    report = VideoReport(
        video=str(output.resolve()),
        state_trace=str(diagnostic.data),
        diagnostic_plot=str(diagnostic.plot),
        diagnostic_data=str(diagnostic.data),
        video_sha256=sha256_file(output),
        diagnostic_plot_sha256=diagnostic.plot_sha256,
        diagnostic_data_sha256=diagnostic.data_sha256,
        captured_state_count=len(trace.frames),
        encoded_frame_count=len(rendered),
        environment_transitions=trace.environment_transitions,
        fps=int(fps),
    )
    output.with_suffix(".json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
