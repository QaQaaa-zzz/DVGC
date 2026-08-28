"""Bounded restore/step smoke for the unified Tube-RSI environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .unified_env import UnifiedTubeRSIEnv


def _fixed_indices(count: int, requested: int) -> tuple[int, ...]:
    if requested <= 0 or count < requested:
        raise ValueError("Tube-RSI smoke requires enough distinct entries per phase")
    if requested == 1:
        return (0,)
    return tuple(
        int(index)
        for index in np.linspace(0, count - 1, requested, dtype=np.int64)
    )


def run_tube_rsi_smoke(
    env: UnifiedTubeRSIEnv,
    output_dir: Path,
    *,
    samples_per_phase: int = 8,
) -> dict[str, Any]:
    """Restore and step fixed TRAIN entries with exact zero-PPO accounting."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    declaration = {
        "schema": "jit_tube_rsi_smoke_declaration_v1",
        "purpose": "unified real-snapshot reset and one-step integration smoke",
        "upstream_samples": int(samples_per_phase),
        "downstream_samples": int(samples_per_phase),
        "maximum_environment_interactions": int(2 * samples_per_phase),
        "stopping_condition": "one reset and one step for every declared fixed index",
        "training_transitions": 0,
        "test_data_used": False,
        "expert_switching_used": False,
        "soft_tube_manifest_sha256": env.tube_pool.artifact.manifest.get(
            "manifest_sha256"
        ),
    }
    (output / "declaration.json").write_text(
        json.dumps(declaration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    phase_specs = (
        ("upstream", 0, env.tube_pool.upstream_count),
        ("downstream", 1, env.tube_pool.downstream_count),
    )
    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    action = np.zeros(4, dtype=np.float32)
    records: list[dict[str, Any]] = []
    try:
        for phase_name, phase_index, count in phase_specs:
            for entry_index in _fixed_indices(count, samples_per_phase):
                state = reset(np.int32(phase_index), np.int32(entry_index))
                next_state = step(state, action)
                finite = bool(
                    np.all(np.isfinite(np.asarray(next_state.data.qpos)))
                    and np.all(np.isfinite(np.asarray(next_state.data.qvel)))
                    and np.all(np.isfinite(np.asarray(next_state.obs["state"])))
                    and np.isfinite(float(next_state.reward))
                )
                if not finite:
                    raise ValueError("Tube-RSI smoke produced a nonfinite state")
                if int(state.info["active_phase"]) != phase_index:
                    raise ValueError("Tube-RSI reset phase mismatch")
                if bool(state.info["expert_switching_used"]) or bool(
                    next_state.info["expert_switching_used"]
                ):
                    raise ValueError("Tube-RSI smoke used expert switching")
                records.append(
                    {
                        "phase": phase_name,
                        "phase_index": phase_index,
                        "entry_index": entry_index,
                        "global_index": int(state.info["tube_global_index"]),
                        "reward": float(next_state.reward),
                        "done": bool(next_state.done),
                        "end_code": int(next_state.info["end_code"]),
                        "finite": finite,
                    }
                )
    except BaseException as exc:
        failure = {
            **declaration,
            "status": "engineering_error",
            "error": f"{type(exc).__name__}: {exc}",
            "environment_interactions": len(records),
            "records": records,
        }
        (output / "report.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise
    report = {
        "schema": "jit_tube_rsi_smoke_v1",
        "status": "completed",
        "tube_rsi_smoke": "GO",
        "environment_interactions": len(records),
        "training_transitions": 0,
        "phase_interactions": {
            phase: sum(row["phase"] == phase for row in records)
            for phase in ("upstream", "downstream")
        },
        "test_data_used": False,
        "validation_data_used": False,
        "expert_switching_used": False,
        "soft_tube_manifest_sha256": declaration["soft_tube_manifest_sha256"],
        "records": records,
    }
    if len(records) != declaration["maximum_environment_interactions"]:
        raise ValueError("Tube-RSI smoke interaction accounting did not close")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
