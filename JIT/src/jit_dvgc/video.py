"""Host-only rendering of saved state traces with truthful frame accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np

from .evaluation import EpisodeTrace


@dataclass(frozen=True)
class VideoReport:
    video: str
    state_trace: str
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
        for frame in trace.frames:
            data.qpos[:] = frame.qpos
            data.qvel[:] = frame.qvel
            data.ctrl[:] = frame.ctrl
            mujoco.mj_forward(env.mj_model, data)
            renderer.update_scene(data, camera="chasis_camera")
            rendered.append(renderer.render().copy())
    finally:
        renderer.close()
    media.write_video(output, rendered, fps=fps)
    state_path = output.with_suffix(".npz")
    np.savez_compressed(
        state_path,
        qpos=np.stack([frame.qpos for frame in trace.frames]),
        qvel=np.stack([frame.qvel for frame in trace.frames]),
        ctrl=np.stack([frame.ctrl for frame in trace.frames]),
        action=np.stack([frame.action for frame in trace.frames]),
        terminated=np.asarray([frame.terminated for frame in trace.frames]),
        truncated=np.asarray([frame.truncated for frame in trace.frames]),
        end_code=np.asarray([frame.end_code for frame in trace.frames]),
    )
    report = VideoReport(
        video=str(output.resolve()),
        state_trace=str(state_path.resolve()),
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
