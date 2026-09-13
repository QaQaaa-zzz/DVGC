"""Auditable numeric and visual diagnostics for saved evaluation traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from .constants import REWARD_COMPONENT_KEYS
from .evaluation import EpisodeFrame, EpisodeTrace, split_trace_at_apex


@dataclass(frozen=True)
class DiagnosticReport:
    plot: Path
    data: Path
    plot_sha256: str
    data_sha256: str
    pre_apex_data: Path
    post_apex_data: Path
    pre_apex_data_sha256: str
    post_apex_data_sha256: str
    sample_count: int
    apex_frame_index: int
    pre_apex_sample_count: int
    post_apex_sample_count: int
    pre_apex_environment_transitions: int
    post_apex_environment_transitions: int
    reward_scaling: float
    fps: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_key(name: str) -> str:
    return f"metric__{name.replace('/', '__')}"


def _component_key(name: str) -> str:
    return f"reward_component__{name}"


def trace_arrays(
    trace: EpisodeTrace, *, reward_scaling: float, fps: int
) -> dict[str, np.ndarray]:
    if not trace.frames:
        raise ValueError("diagnostic trace must contain at least one state")
    if len(trace.frames) != trace.environment_transitions + 1:
        raise ValueError("captured state count must equal transitions plus one")
    if not np.isfinite(reward_scaling) or reward_scaling <= 0.0:
        raise ValueError("reward scaling must be finite and positive")
    if fps <= 0:
        raise ValueError("diagnostic fps must be positive")

    frames = trace.frames
    metric_names = tuple(sorted({name for frame in frames for name in frame.metrics}))
    clipped = np.asarray(
        [frame.metrics.get("reward", frame.reward) for frame in frames],
        dtype=np.float64,
    )
    arrays: dict[str, np.ndarray] = {
        "time_seconds": np.arange(len(frames), dtype=np.float64) / float(fps),
        "reward_clipped": clipped,
        "reward_unclipped": np.asarray(
            [frame.metrics.get("reward/unclipped", frame.reward) for frame in frames],
            dtype=np.float64,
        ),
        "reward_scaled": clipped * float(reward_scaling),
        "qpos": np.stack([frame.qpos for frame in frames]),
        "qvel": np.stack([frame.qvel for frame in frames]),
        "ctrl": np.stack([frame.ctrl for frame in frames]),
        "action": np.stack([frame.action for frame in frames]),
        "terminal_terminated": np.asarray([frame.terminated for frame in frames]),
        "terminal_truncated": np.asarray([frame.truncated for frame in frames]),
        "terminal_end_code": np.asarray(
            [frame.end_code for frame in frames], dtype=np.int32
        ),
        "terminal_success": np.asarray([frame.success for frame in frames]),
        "terminal_physical_failure": np.asarray(
            [frame.physical_failure for frame in frames]
        ),
        "terminal_timeout": np.asarray([frame.timeout for frame in frames]),
    }
    for component in REWARD_COMPONENT_KEYS:
        arrays[_component_key(component)] = np.asarray(
            [frame.reward_components[component] for frame in frames],
            dtype=np.float64,
        )
    for name in metric_names:
        arrays[_metric_key(name)] = np.asarray(
            [frame.metrics.get(name, np.nan) for frame in frames],
            dtype=np.float64,
        )
    split = split_trace_at_apex(trace)
    indices = np.arange(len(frames), dtype=np.int32)
    arrays["apex_frame_index"] = np.asarray(
        [split.apex_frame_index], dtype=np.int32
    )
    arrays["segment_pre_apex"] = np.isin(indices, split.pre_frame_indices)
    arrays["segment_post_apex"] = np.isin(indices, split.post_frame_indices)
    return arrays


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _segment_arrays(
    arrays: dict[str, np.ndarray], indices: tuple[int, ...]
) -> dict[str, np.ndarray]:
    sample_count = int(arrays["qpos"].shape[0])
    selected = np.asarray(indices, dtype=np.int32)
    result = {
        name: value[selected]
        for name, value in arrays.items()
        if name != "apex_frame_index"
        and value.ndim >= 1
        and value.shape[0] == sample_count
    }
    result["source_frame_index"] = selected
    result["apex_frame_index"] = arrays["apex_frame_index"].copy()
    return result


def _series(
    arrays: dict[str, np.ndarray], metric: str, fallback: np.ndarray
) -> np.ndarray:
    return arrays.get(_metric_key(metric), np.asarray(fallback, dtype=np.float64))


def _draw_dashboard(arrays: dict[str, np.ndarray], path: Path) -> None:
    time = arrays["time_seconds"]
    qpos = arrays["qpos"]
    qvel = arrays["qvel"]
    figure, axes = plt.subplots(5, 2, figsize=(16, 16), constrained_layout=True)
    figure.suptitle("Phase U evaluation diagnostic", fontsize=16)

    axis = axes[0, 0]
    axis.plot(time, arrays["reward_unclipped"], label="unclipped")
    axis.plot(time, arrays["reward_clipped"], label="clipped")
    axis.plot(time, arrays["reward_scaled"], label="PPO scaled")
    axis.set_title("Reward totals")
    axis.set_ylabel("reward")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    apex_index = int(arrays["apex_frame_index"][0])
    if apex_index >= 0:
        apex_time = float(time[apex_index])
        for item in axes.flat:
            item.axvline(
                apex_time,
                color="#8e44ad",
                linestyle="--",
                linewidth=1.1,
                label="first Apex" if item is axes[0, 0] else None,
            )

    component_matrix = np.stack(
        [arrays[_component_key(name)] for name in REWARD_COMPONENT_KEYS]
    )
    axis = axes[0, 1]
    limit = max(float(np.max(np.abs(component_matrix))), 1.0)
    image = axis.imshow(
        component_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        extent=[time[0], time[-1] if len(time) > 1 else 1.0, len(REWARD_COMPONENT_KEYS) - 0.5, -0.5],
    )
    axis.set_yticks(range(len(REWARD_COMPONENT_KEYS)))
    axis.set_yticklabels(REWARD_COMPONENT_KEYS, fontsize=8)
    axis.set_title("Every reward component")
    figure.colorbar(image, ax=axis, label="per-step reward")

    axis = axes[1, 0]
    for metric, fallback, label in (
        ("signal/root_x", qpos[:, 0], "x"),
        ("signal/root_y", qpos[:, 1], "y"),
        ("signal/root_z", qpos[:, 2], "z"),
    ):
        axis.plot(time, _series(arrays, metric, fallback), label=label)
    obstacle_relative = _series(
        arrays, "signal/obstacle_relative_x", np.full_like(time, np.nan)
    )
    if np.isfinite(obstacle_relative).any():
        axis.plot(time, obstacle_relative, linestyle=":", label="obstacle relative x")
    axis.axhline(0.5, color="k", linestyle="--", linewidth=1, label="apex height 0.5 m")
    axis.axhline(2.5, color="#7b61ff", linestyle="--", linewidth=0.8, label="jump x min")
    axis.axhline(3.1, color="#7b61ff", linestyle=":", linewidth=0.8, label="jump x max")
    axis.set_title("Root position")
    axis.set_ylabel("m")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    for metric, label in (
        ("signal/roll", "roll"),
        ("signal/pitch", "pitch"),
        ("signal/yaw", "yaw"),
    ):
        axis.plot(time, np.rad2deg(_series(arrays, metric, np.zeros_like(time))), label=label)
    axis.set_title("Root attitude")
    axis.set_ylabel("degrees")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[2, 0]
    for metric, fallback, label in (
        ("signal/forward_velocity", qvel[:, 0], "vx"),
        ("signal/lateral_velocity", qvel[:, 1], "vy"),
        ("signal/vertical_velocity", qvel[:, 2], "vz"),
    ):
        axis.plot(time, _series(arrays, metric, fallback), label=label)
    axis.axhline(0.05, color="g", linestyle="--", linewidth=1)
    axis.axhline(-0.05, color="r", linestyle="--", linewidth=1)
    axis.set_title("Linear velocity")
    axis.set_ylabel("m/s")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[2, 1]
    for metric, label in (
        ("signal/roll_rate", "roll rate"),
        ("signal/pitch_rate", "pitch rate"),
        ("signal/yaw_rate", "yaw rate"),
    ):
        axis.plot(time, _series(arrays, metric, np.zeros_like(time)), label=label)
    axis.set_title("Angular rates")
    axis.set_ylabel("rad/s")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    axis = axes[3, 0]
    for index, label in enumerate(("steer", "drive", "hip", "knee")):
        axis.plot(time, arrays["action"][:, index], label=f"action {label}")
    axis.set_title("Policy actions")
    axis.set_ylabel("normalized")
    axis.set_xlabel("time (s)")
    axis.legend(loc="best", ncols=2)
    axis.grid(alpha=0.25)

    axis = axes[3, 1]
    for index, label in enumerate(("steer", "drive", "hip", "knee")):
        axis.plot(time, arrays["ctrl"][:, index], label=f"ctrl {label}")
    axis.set_title("Applied actuator controls")
    axis.set_xlabel("time (s)")
    axis.legend(loc="best", ncols=2)
    axis.grid(alpha=0.25)

    axis = axes[4, 0]
    for metric, label in (
        ("signal/hip_velocity", "hip velocity"),
        ("signal/knee_velocity", "knee velocity"),
        ("signal/hip_force", "hip force"),
        ("signal/knee_force", "knee force"),
        ("signal/joint_power", "joint power"),
    ):
        axis.plot(time, _series(arrays, metric, np.zeros_like(time)), label=label)
    axis.set_title("Joint velocity, force, and mechanical power")
    axis.set_xlabel("time (s)")
    axis.legend(loc="best", fontsize=8, ncols=2)
    axis.grid(alpha=0.25)

    axis = axes[4, 1]
    event_names = (
        "event/jump_signal",
        "event/jump_zone_seen",
        "event/jump_zone_consumed",
        "event/ascending_seen",
        "event/height_seen",
        "event/apex_seen",
        "reset/source_airborne_rsi",
        "terminal/success",
        "terminal/physical_failure",
        "terminal/timeout",
    )
    for offset, name in enumerate(event_names):
        axis.step(
            time,
            _series(arrays, name, np.zeros_like(time)) + 1.25 * offset,
            where="post",
            label=name.removeprefix("event/"),
        )
    axis.set_title("One-shot jump and Apex events")
    axis.set_xlabel("time (s)")
    axis.set_yticks([])
    axis.legend(loc="best", fontsize=8, ncols=2)
    axis.grid(alpha=0.25)

    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_trace_dashboard(
    trace: EpisodeTrace,
    path: Path,
    *,
    reward_scaling: float,
    fps: int = 50,
) -> DiagnosticReport:
    """Saves a diagnostic PNG plus the complete synchronized numeric payload."""

    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("diagnostic dashboard path must end in .png")
    output.parent.mkdir(parents=True, exist_ok=True)
    data_path = output.with_suffix(".npz")
    arrays = trace_arrays(trace, reward_scaling=reward_scaling, fps=fps)
    split = split_trace_at_apex(trace)
    base_stem = output.stem.removesuffix("_diagnostic")
    pre_path = output.with_name(f"{base_stem}_pre_apex.npz")
    post_path = output.with_name(f"{base_stem}_post_apex.npz")

    _save_npz(data_path, arrays)
    _save_npz(pre_path, _segment_arrays(arrays, split.pre_frame_indices))
    _save_npz(post_path, _segment_arrays(arrays, split.post_frame_indices))

    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".png", delete=False
    ) as stream:
        temporary_plot = Path(stream.name)
    try:
        _draw_dashboard(arrays, temporary_plot)
        os.replace(temporary_plot, output)
    finally:
        if temporary_plot.exists():
            temporary_plot.unlink()

    return DiagnosticReport(
        plot=output.resolve(),
        data=data_path.resolve(),
        plot_sha256=sha256_file(output),
        data_sha256=sha256_file(data_path),
        pre_apex_data=pre_path.resolve(),
        post_apex_data=post_path.resolve(),
        pre_apex_data_sha256=sha256_file(pre_path),
        post_apex_data_sha256=sha256_file(post_path),
        sample_count=len(trace.frames),
        apex_frame_index=split.apex_frame_index,
        pre_apex_sample_count=len(split.pre_frame_indices),
        post_apex_sample_count=len(split.post_frame_indices),
        pre_apex_environment_transitions=split.pre_environment_transitions,
        post_apex_environment_transitions=split.post_environment_transitions,
        reward_scaling=float(reward_scaling),
        fps=int(fps),
    )


def _frame_metric(frame: EpisodeFrame, name: str, fallback: float = 0.0) -> float:
    return float(frame.metrics.get(name, fallback))


def telemetry_panel(
    frame: EpisodeFrame,
    *,
    width: int,
    height: int,
    tick: int,
    reward_scaling: float = 0.1,
    apex_frame_index: int = -1,
) -> np.ndarray:
    """Renders synchronized reward/state telemetry beside one physical frame."""

    if width <= 0 or height <= 0:
        raise ValueError("telemetry dimensions must be positive")
    dpi = 100
    figure = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#101218"
    )
    text_axis = figure.add_axes((0.02, 0.04, 0.46, 0.92), facecolor="#101218")
    bar_axis = figure.add_axes((0.57, 0.08, 0.41, 0.84), facecolor="#101218")
    text_axis.axis("off")
    clipped = _frame_metric(frame, "reward", frame.reward)
    unclipped = _frame_metric(frame, "reward/unclipped", frame.reward)
    qpos = frame.qpos
    qvel = frame.qvel
    phase = (
        "pre-Apex"
        if apex_frame_index < 0 or tick < apex_frame_index
        else ("first Apex" if tick == apex_frame_index else "post-Apex")
    )
    event_bits = " ".join(
        f"{label}:{int(_frame_metric(frame, metric) > 0.5)}"
        for label, metric in (
            ("J", "event/jump_signal"),
            ("Z", "event/jump_zone_seen"),
            ("H", "event/height_seen"),
            ("A", "event/ascending_seen"),
            ("X", "event/apex_seen"),
        )
    )
    lines = (
        f"tick {tick}  {phase}  end={frame.end_code}",
        f"R raw {unclipped:+.3f}",
        f"R clip {clipped:+.3f}",
        f"R PPO  {clipped * reward_scaling:+.3f}",
        "",
        "position [m]",
        f"x {_frame_metric(frame, 'signal/root_x', qpos[0]):+.3f}",
        f"y {_frame_metric(frame, 'signal/root_y', qpos[1]):+.3f}",
        f"z {_frame_metric(frame, 'signal/root_z', qpos[2]):+.3f}",
        "attitude [deg]",
        " ".join(
            f"{name} {np.rad2deg(_frame_metric(frame, metric)):+.1f}"
            for name, metric in (
                ("r", "signal/roll"),
                ("p", "signal/pitch"),
                ("y", "signal/yaw"),
            )
        ),
        "velocity [m/s]",
        " ".join(
            f"{name} {_frame_metric(frame, metric, qvel[index]):+.2f}"
            for index, (name, metric) in enumerate(
                (
                    ("vx", "signal/forward_velocity"),
                    ("vy", "signal/lateral_velocity"),
                    ("vz", "signal/vertical_velocity"),
                )
            )
        ),
        f"events {event_bits}",
        "reset RSI "
        f"{int(_frame_metric(frame, 'reset/source_airborne_rsi') > 0.5)}  "
        f"terminal S/F/T {int(frame.success)}/{int(frame.physical_failure)}/{int(frame.timeout)}",
        "joint "
        f"tau {_frame_metric(frame, 'signal/hip_force'):+.1f}/"
        f"{_frame_metric(frame, 'signal/knee_force'):+.1f}  "
        f"power {_frame_metric(frame, 'signal/joint_power'):+.1f}",
        "action " + " ".join(f"{value:+.2f}" for value in frame.action),
        "ctrl   " + " ".join(f"{value:+.2f}" for value in frame.ctrl),
    )
    text_axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        color="#f0f3f8",
        family="monospace",
        fontsize=max(5.0, min(8.5, height / 27.0)),
    )

    values = np.asarray(
        [frame.reward_components[name] for name in REWARD_COMPONENT_KEYS],
        dtype=np.float64,
    )
    positions = np.arange(len(REWARD_COMPONENT_KEYS))
    colors = np.where(values >= 0.0, "#58c77b", "#ef6a6a")
    bar_axis.barh(positions, values, color=colors)
    bar_axis.axvline(0.0, color="#d8dee9", linewidth=0.6)
    bar_axis.set_yticks(positions)
    bar_axis.set_yticklabels(REWARD_COMPONENT_KEYS, fontsize=4.5, color="#f0f3f8")
    bar_axis.invert_yaxis()
    bar_axis.tick_params(axis="x", labelsize=5, colors="#f0f3f8")
    bar_axis.set_title("reward components", color="#f0f3f8", fontsize=7)
    for spine in bar_axis.spines.values():
        spine.set_color("#667085")

    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    plt.close(figure)
    if rgb.shape[:2] != (height, width):
        raise RuntimeError(f"unexpected telemetry canvas shape: {rgb.shape}")
    return rgb
