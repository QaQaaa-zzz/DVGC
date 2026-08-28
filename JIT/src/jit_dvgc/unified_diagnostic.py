"""TRAIN-only fixed-panel diagnostics for a frozen unified pilot checkpoint."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from .checkpoint import load_checkpoint
from .config import file_sha256, load_config
from .constants import END_REASONS
from .handoff_snapshot import load_snapshot
from .ppo import make_checkpoint_policy
from .soft_tube import SoftTubeArtifact, load_soft_tube
from .tube_rsi_smoke import fixed_indices
from .unified_env import UnifiedTubeRSIEnv
from .unified_training import (
    UnifiedPilotConfig,
    checkpoint_identity,
    load_unified_pilot_config,
    read_json,
    validate_pilot_gate,
)


def validate_panel_artifact(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema") != "jit_soft_tube_v1"
        or manifest.get("status") != "completed"
        or manifest.get("training_guidance_only") is not True
    ):
        raise ValueError("fixed panel requires a completed learned Soft Tube")
    if (
        manifest.get("test_data_used") is not False
        or manifest.get("validation_data_used") is not False
    ):
        raise ValueError("fixed panel requires a TRAIN-only Tube")


def _bool(info: Mapping[str, Any], name: str) -> bool:
    return bool(np.asarray(info.get(name, False)))


def _int(info: Mapping[str, Any], name: str) -> int:
    return int(np.asarray(info.get(name, 0)))


def _terminal_reason(state: Any, *, reached_horizon: bool) -> str:
    end_code = _int(state.info, "end_code")
    if end_code in END_REASONS and end_code != 0:
        return END_REASONS[end_code]
    if reached_horizon:
        return "horizon"
    if bool(np.asarray(state.done)):
        return "other_terminal"
    return "running"


def _terminal_class(state: Any, *, reached_horizon: bool) -> str:
    if _bool(state.info, "success"):
        return "success"
    if _bool(state.info, "physical_failure"):
        return "physical_failure"
    if _bool(state.info, "timeout"):
        return "timeout"
    if reached_horizon and not bool(np.asarray(state.done)):
        return "horizon"
    return "task_failure"


def _finite_state(state: Any, action: Any | None = None) -> bool:
    values = (
        np.asarray(state.data.qpos),
        np.asarray(state.data.qvel),
        np.asarray(state.obs["state"]),
        np.asarray(state.reward),
    )
    if action is not None:
        values = (*values, np.asarray(action))
    return all(np.isfinite(value).all() for value in values)


def rollout_fixed_tube_panel(
    env: Any,
    policy: Callable[[Any, Any], Any],
    *,
    samples_per_phase: int = 8,
    horizon: int = 400,
    reset_fn: Callable[[Any, Any], Any] | None = None,
    step_fn: Callable[[Any, Any], Any] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Roll out fixed TRAIN starts until true terminal or the declared horizon."""
    validate_panel_artifact(env.tube_pool.artifact.manifest)
    if horizon <= 0:
        raise ValueError("fixed panel horizon must be positive")
    reset = jax.jit(env.reset_tube_index) if reset_fn is None else reset_fn
    step = jax.jit(env.step) if step_fn is None else step_fn
    phase_specs = (
        ("upstream", 0, int(env.tube_pool.upstream_count)),
        ("downstream", 1, int(env.tube_pool.downstream_count)),
    )
    records: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    total_interactions = 0
    phase_interactions = {"downstream": 0, "upstream": 0}

    for phase_name, phase_index, count in phase_specs:
        for ordinal, entry_index in enumerate(fixed_indices(count, samples_per_phase)):
            state = reset(np.int32(phase_index), np.int32(entry_index))
            if _int(state.info, "active_phase") != phase_index:
                raise ValueError("fixed panel reset phase mismatch")
            if _bool(state.info, "expert_switching_used"):
                raise ValueError("fixed panel used expert switching at reset")
            if not _finite_state(state):
                raise ValueError("fixed panel reset produced a nonfinite state")
            xs = [float(np.asarray(state.data.qpos)[0])]
            zs = [float(np.asarray(state.data.qpos)[2])]
            rewards: list[float] = []
            interactions = 0
            phase_transitioned = _bool(state.info, "phase_transitioned")
            apex_seen = False
            for tick in range(horizon):
                key = jax.random.PRNGKey(
                    1_100_000 + phase_index * 10_000 + ordinal * horizon + tick
                )
                result = policy(state.obs, key)
                action = result[0] if isinstance(result, tuple) else result
                state = step(state, action)
                interactions += 1
                total_interactions += 1
                phase_interactions[phase_name] += 1
                if not _finite_state(state, action):
                    raise ValueError("fixed panel step produced a nonfinite state")
                if _bool(state.info, "expert_switching_used"):
                    raise ValueError("fixed panel used expert switching")
                xs.append(float(np.asarray(state.data.qpos)[0]))
                zs.append(float(np.asarray(state.data.qpos)[2]))
                rewards.append(float(np.asarray(state.reward)))
                phase_transitioned |= _bool(state.info, "phase_transitioned")
                events = state.info.get("up_events")
                apex_seen |= bool(np.asarray(getattr(events, "apex_seen", False)))
                if bool(np.asarray(state.done)):
                    break
            terminal = _terminal_reason(state, reached_horizon=interactions == horizon)
            terminal_class = _terminal_class(
                state, reached_horizon=interactions == horizon
            )
            global_index = _int(state.info, "tube_global_index")
            record = {
                "phase": phase_name,
                "phase_index": phase_index,
                "entry_index": int(entry_index),
                "global_index": global_index,
                "environment_interactions": interactions,
                "terminal": terminal,
                "terminal_class": terminal_class,
                "success": _bool(state.info, "success"),
                "physical_failure": _bool(state.info, "physical_failure"),
                "timeout": _bool(state.info, "timeout"),
                "end_code": _int(state.info, "end_code"),
                "phase_transitioned": bool(phase_transitioned),
                "apex_seen": bool(apex_seen),
                "return": float(sum(rewards)),
                "start_x": xs[0],
                "start_z": zs[0],
                "final_x": xs[-1],
                "final_z": zs[-1],
                "finite": True,
            }
            records.append(record)
            trajectories.append(
                {
                    "phase": phase_name,
                    "entry_index": int(entry_index),
                    "global_index": global_index,
                    "x": xs,
                    "z": zs,
                    "terminal": terminal,
                    "terminal_class": terminal_class,
                }
            )

    end_counts = Counter(row["terminal"] for row in records)
    terminal_class_counts = Counter(row["terminal_class"] for row in records)
    report = {
        "schema": "jit_pi_unified_train_panel_v1",
        "status": "completed",
        "rollouts": len(records),
        "phase_rollouts": {
            "downstream": sum(row["phase"] == "downstream" for row in records),
            "upstream": sum(row["phase"] == "upstream" for row in records),
        },
        "environment_interactions": total_interactions,
        "phase_interactions": phase_interactions,
        "training_transitions": 0,
        "success_count": sum(row["success"] for row in records),
        "physical_failure_count": sum(row["physical_failure"] for row in records),
        "timeout_count": sum(row["timeout"] for row in records),
        "apex_seen_count": sum(row["apex_seen"] for row in records),
        "phase_transition_count": sum(row["phase_transitioned"] for row in records),
        "end_reason_counts": dict(sorted(end_counts.items())),
        "terminal_class_counts": dict(sorted(terminal_class_counts.items())),
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
        "soft_tube_manifest_sha256": env.tube_pool.artifact.manifest.get(
            "manifest_sha256"
        ),
        "records": records,
    }
    return report, tuple(trajectories)


def plot_xz_visitation(
    tube_points: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    output: Path,
) -> Path:
    """Plot literal TRAIN support plus checkpoint visitation; no interpolation."""
    import matplotlib.pyplot as plt

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    phase_style = {
        "upstream": ("#1677ff", "o"),
        "downstream": ("#f28e2b", "^"),
    }
    for phase, (color, marker) in phase_style.items():
        rows = [row for row in tube_points if row["phase"] == phase]
        ax.scatter(
            [row["x"] for row in rows],
            [row["z"] for row in rows],
            s=22,
            marker=marker,
            color=color,
            alpha=0.22,
            label=f"Soft Tube {phase} support",
            zorder=1,
        )
    terminal_colors = {
        "success": "#2ca02c",
        "physical_failure": "#d62728",
        "timeout": "#9467bd",
        "horizon": "#8c564b",
        "other_terminal": "#111111",
    }
    for row in trajectories:
        color = phase_style[row["phase"]][0]
        ax.plot(row["x"], row["z"], color=color, alpha=0.65, linewidth=1.2)
        ax.scatter(
            [row["x"][0]],
            [row["z"][0]],
            color=color,
            s=28,
            marker=phase_style[row["phase"]][1],
            zorder=3,
        )
        ax.scatter(
            [row["x"][-1]],
            [row["z"][-1]],
            color=terminal_colors.get(row["terminal_class"], "#111111"),
            s=42,
            marker="x",
            linewidths=1.5,
            zorder=4,
        )
    ax.axvline(3.6, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlabel("root x [m]")
    ax.set_ylabel("root z [m]")
    ax.set_title(
        "pi_unified fixed TRAIN-panel visitation over learned Soft Tube\n"
        "literal states and rollouts; no continuous/safety boundary implied"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def _tube_points(artifact: SoftTubeArtifact) -> tuple[dict[str, Any], ...]:
    points = []
    for row in artifact.entries:
        snapshot = load_snapshot(Path(row["snapshot"]))
        points.append(
            {
                "phase": row["phase"],
                "x": float(snapshot.qpos[0]),
                "z": float(snapshot.qpos[2]),
                "sampling_weight": float(row["sampling_weight"]),
            }
        )
    return tuple(points)


def _load_runtime(
    config: UnifiedPilotConfig,
) -> tuple[Any, Any, SoftTubeArtifact, Mapping[str, Any]]:
    up_config = load_config(Path(config.up_config_path))
    down_config = load_config(Path(config.down_config_path))
    if up_config.config_sha256 != config.up_config_sha256:
        raise ValueError("diagnostic upstream config identity mismatch")
    if down_config.config_sha256 != config.down_config_sha256:
        raise ValueError("diagnostic downstream config identity mismatch")
    artifact = load_soft_tube(Path(config.soft_tube_path))
    smoke_path = Path(config.tube_rsi_smoke_report)
    if file_sha256(smoke_path) != config.tube_rsi_smoke_report_sha256:
        raise ValueError("diagnostic Tube-RSI smoke identity mismatch")
    smoke = read_json(smoke_path)
    validate_pilot_gate(config, artifact.manifest, smoke)
    validate_panel_artifact(artifact.manifest)
    return up_config, down_config, artifact, smoke


def run_unified_fixed_panel(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    samples_per_phase: int = 8,
) -> dict[str, Any]:
    """Restore one unified checkpoint and run a bounded TRAIN-only panel."""
    config = load_unified_pilot_config(config_path)
    up_config, down_config, artifact, _ = _load_runtime(config)
    if jax.default_backend() != "gpu":
        raise RuntimeError("the unified fixed panel requires the visible JAX GPU")
    env = UnifiedTubeRSIEnv(
        up_config,
        down_config,
        artifact,
        runtime_naccdmax=config.runtime_naccdmax,
    )
    checkpoint_path = Path(checkpoint)
    payload = load_checkpoint(
        checkpoint_path, expected=checkpoint_identity(config, env)
    )
    if payload.training_transitions != config.ppo.requested_transitions:
        raise ValueError("fixed panel checkpoint is not the completed pilot")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    maximum_interactions = 2 * samples_per_phase * config.ppo.episode_horizon
    declaration = {
        "schema": "jit_pi_unified_train_panel_declaration_v1",
        "purpose": "deterministic TRAIN-only Tube visitation from frozen pilot checkpoint",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_payload_sha256": file_sha256(checkpoint_path / "payload.pkl"),
        "soft_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
        "samples_per_phase": samples_per_phase,
        "maximum_environment_interactions": maximum_interactions,
        "training_transitions": 0,
        "stopping_condition": "each fixed rollout stops at true terminal or 400-step ceiling",
        "expert_switching_used": False,
        "test_data_used": False,
        "validation_data_used": False,
    }
    (output / "declaration.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy = make_checkpoint_policy(env, payload, deterministic=True)
    try:
        report, trajectories = rollout_fixed_tube_panel(
            env,
            jax.jit(policy),
            samples_per_phase=samples_per_phase,
            horizon=config.ppo.episode_horizon,
        )
        if report["environment_interactions"] > maximum_interactions:
            raise ValueError("fixed panel exceeded its interaction ceiling")
        plot_path = plot_xz_visitation(
            _tube_points(artifact), trajectories, output / "xz_visitation.png"
        )
        (output / "trajectories.json").write_text(
            json.dumps(trajectories, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_report = {
            **report,
            "checkpoint": declaration["checkpoint"],
            "checkpoint_payload_sha256": declaration["checkpoint_payload_sha256"],
            "maximum_environment_interactions": maximum_interactions,
            "xz_visitation": str(plot_path.resolve()),
            "claim_boundary": {
                "train_panel_only": True,
                "learnability_claim": False,
                "final_policy_claim": False,
                "jce_jel_claim": False,
            },
        }
        (output / "report.json").write_text(
            json.dumps(final_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return final_report
    except BaseException as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    **declaration,
                    "status": "engineering_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
