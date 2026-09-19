"""Persistent, auditable learning-curve artifacts for formal PPO runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np

from .diagnostics import sha256_file


@dataclass(frozen=True)
class TrainingCurveReport:
    plot_path: Path
    data_path: Path
    report_path: Path
    plot_sha256: str
    data_sha256: str
    episode_sample_count: int
    ppo_sample_count: int
    episode_metric_semantics: str = "rolling_last_100_completed_episodes"


_EPISODE_KEYS = (
    "episode/sum_reward",
    "episode/length",
    "episode/reset/source_airborne_rsi",
)
_PPO_KEYS = (
    "training/kl_mean",
    "training/policy_loss",
    "training/v_loss",
    "training/total_loss",
    "training/policy_dist_mean_std",
    "training/sps",
)


def _load_rows(path: Path, required_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing training evidence: {path.name}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"training evidence is empty: {path.name}")
    steps = [int(row["training_transitions"]) for row in rows]
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise ValueError(f"{path.name} steps must be strictly increasing")
    for row in rows:
        metrics = row.get("metrics", {})
        for key in required_keys:
            if key not in metrics:
                raise ValueError(f"{path.name} is missing {key}")
            value = float(metrics[key])
            if not math.isfinite(value):
                raise ValueError(f"{path.name} contains nonfinite {key}")
    return rows


def _metric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row["metrics"][key]) for row in rows], dtype=np.float64)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_figure(path: Path, figure: Any) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        figure.savefig(temporary, dpi=150)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_training_curves(run_dir: Path) -> TrainingCurveReport:
    """Converts raw callbacks into synchronized PNG, NPZ, and JSON evidence."""

    root = Path(run_dir)
    episode_rows = _load_rows(root / "episode_metrics.jsonl", _EPISODE_KEYS)
    ppo_rows = _load_rows(root / "metrics.jsonl", _PPO_KEYS)

    episode_steps = np.asarray(
        [int(row["training_transitions"]) for row in episode_rows], dtype=np.int64
    )
    ppo_steps = np.asarray(
        [int(row["training_transitions"]) for row in ppo_rows], dtype=np.int64
    )
    episode_length = _metric(episode_rows, "episode/length")
    if np.any(episode_length <= 0.0):
        raise ValueError("episode/length must remain positive")
    reset_count = _metric(episode_rows, "episode/reset/source_airborne_rsi")
    reset_fraction = reset_count / episode_length
    if np.any((reset_fraction < 0.0) | (reset_fraction > 1.0)):
        raise ValueError("airborne RSI fraction must remain in [0, 1]")

    arrays = {
        "episode_training_transitions": episode_steps,
        "episode_mean_reward": _metric(episode_rows, "episode/sum_reward"),
        "episode_mean_length": episode_length,
        "episode_airborne_rsi_fraction": reset_fraction,
        "ppo_training_transitions": ppo_steps,
        "ppo_kl_mean": _metric(ppo_rows, "training/kl_mean"),
        "ppo_policy_loss": _metric(ppo_rows, "training/policy_loss"),
        "ppo_value_loss": _metric(ppo_rows, "training/v_loss"),
        "ppo_total_loss": _metric(ppo_rows, "training/total_loss"),
        "ppo_policy_mean_std": _metric(
            ppo_rows, "training/policy_dist_mean_std"
        ),
        "ppo_sps": _metric(ppo_rows, "training/sps"),
    }

    data_path = root / "training_curves.npz"
    plot_path = root / "training_curves.png"
    report_path = root / "training_curves.json"
    _atomic_npz(data_path, arrays)

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    figure.suptitle("Phase U PPO learning curves")
    axes[0, 0].plot(episode_steps, arrays["episode_mean_reward"])
    axes[0, 0].set_title("Mean episode reward")
    axes[0, 1].plot(episode_steps, arrays["episode_mean_length"])
    axes[0, 1].set_title("Mean episode length")
    rsi_axis = axes[0, 1].twinx()
    rsi_axis.plot(episode_steps, reset_fraction, color="tab:orange", alpha=0.6)
    rsi_axis.set_ylabel("airborne RSI fraction")
    axes[1, 0].plot(ppo_steps, arrays["ppo_kl_mean"])
    axes[1, 0].set_title("Approximate KL")
    axes[1, 1].plot(ppo_steps, arrays["ppo_policy_loss"], label="policy")
    axes[1, 1].plot(ppo_steps, arrays["ppo_value_loss"], label="value")
    axes[1, 1].plot(ppo_steps, arrays["ppo_total_loss"], label="total")
    axes[1, 1].set_title("PPO losses")
    axes[1, 1].legend()
    axes[2, 0].plot(ppo_steps, arrays["ppo_policy_mean_std"])
    axes[2, 0].set_title("Policy distribution mean std")
    axes[2, 1].plot(ppo_steps, arrays["ppo_sps"])
    axes[2, 1].set_title("Training throughput")
    for axis in axes.flat:
        axis.set_xlabel("environment transitions")
        axis.grid(alpha=0.25)
    _atomic_figure(plot_path, figure)
    plt.close(figure)

    report = TrainingCurveReport(
        plot_path=plot_path.resolve(),
        data_path=data_path.resolve(),
        report_path=report_path.resolve(),
        plot_sha256=sha256_file(plot_path),
        data_sha256=sha256_file(data_path),
        episode_sample_count=len(episode_rows),
        ppo_sample_count=len(ppo_rows),
    )
    report_path.write_text(
        json.dumps(
            {**asdict(report), "required_series": sorted(arrays)},
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
