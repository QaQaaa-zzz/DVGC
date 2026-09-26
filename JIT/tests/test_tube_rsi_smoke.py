from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.handoff_snapshot import save_snapshot
from jit_dvgc.soft_tube import (
    PHASE_MIXTURE,
    SOFT_TUBE_SCHEMA,
    SoftTubeArtifact,
    WEIGHT_FLOOR,
    WEIGHT_SCALE,
)
from jit_dvgc.upstream_boundary import physical_state_sha256
from jit_dvgc.upstream_boundary import canonical_sha256


def _runtime_artifact(jit_root: Path, tmp_path: Path):
    up_config = load_config(jit_root / "configs/phase_u_continuation_smoke.json")
    down_config = load_config(jit_root / "configs/descent_recovery_smoke.json")
    source_env = TwoPhaseBikeEnv(up_config)
    up_state = jax.jit(source_env.reset_natural)(jax.random.PRNGKey(201))
    down_state = jax.jit(source_env.reset_airborne_rsi)(jax.random.PRNGKey(202))
    snapshots = []
    for phase, state, marker in (
        ("upstream", up_state, "up"),
        ("downstream", down_state, "down"),
    ):
        snapshot = source_env.capture_handoff_snapshot(
            state,
            policy_sha256="a" * 64,
            parent_trajectory=f"test-{marker}",
            policy_identity="test-only",
        )
        path = tmp_path / f"snapshot-{marker}"
        save_snapshot(path, snapshot)
        score = 0.25 if phase == "upstream" else 0.75
        snapshots.append(
            {
                "phase": phase,
                "split": "train",
                "snapshot": str(path),
                "source_bank": f"test-{marker}",
                "state_sha256": physical_state_sha256(snapshot),
                "parent_group_id": f"test-{marker}",
                "seed": 1000001,
                "role": "ascending_entry" if phase == "upstream" else "nearest_apex",
                "tick": int(snapshot.tick),
                "value_score": score,
                "sampling_weight": WEIGHT_FLOOR + WEIGHT_SCALE * score,
                "value_model_target": "V_up" if phase == "upstream" else "V_down",
            }
        )
    root = tmp_path / "soft-tube"
    root.mkdir()
    diagnostics = {}
    manifest = {
        "schema": SOFT_TUBE_SCHEMA,
        "status": "completed",
        "test_data_used": False,
        "validation_data_used": False,
        "phase_mixture": PHASE_MIXTURE,
        "entries_sha256": canonical_sha256({"entries": snapshots}),
        "diagnostics_sha256": canonical_sha256(diagnostics),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    for name, payload in (
        ("entries.json", snapshots),
        ("diagnostics.json", diagnostics),
        ("manifest.json", manifest),
    ):
        (root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    artifact = SoftTubeArtifact(root, manifest, tuple(snapshots), diagnostics)
    return up_config, down_config, source_env, artifact


def test_phase_transition_uses_observed_first_apex_and_never_an_expert_router():
    from jit_dvgc.unified_env import next_training_phase

    assert int(next_training_phase(0, False, False, False)) == 0
    assert int(next_training_phase(0, False, True, False)) == 1
    assert int(next_training_phase(0, False, True, True)) == 0
    assert int(next_training_phase(1, False, True, False)) == 1


@pytest.mark.gpu
def test_unified_env_restores_and_steps_both_phases_with_one_action_path(
    jit_root, tmp_path
):
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, source_env, artifact = _runtime_artifact(jit_root, tmp_path)
    env = UnifiedTubeRSIEnv(up_config, down_config, artifact)
    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    upstream = reset(np.int32(0), np.int32(0))
    downstream = reset(np.int32(1), np.int32(0))
    action = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    next_upstream = step(upstream, action)
    next_downstream = step(downstream, action)

    assert int(upstream.info["active_phase"]) == 0
    assert int(downstream.info["active_phase"]) == 1
    assert np.all(np.isfinite(np.asarray(next_upstream.data.qpos)))
    assert np.all(np.isfinite(np.asarray(next_downstream.data.qpos)))
    assert np.all(np.isfinite(np.asarray(next_upstream.obs["state"])))
    assert np.all(np.isfinite(np.asarray(next_downstream.obs["state"])))
    assert np.isfinite(float(next_upstream.reward))
    assert np.isfinite(float(next_downstream.reward))
    np.testing.assert_allclose(next_upstream.data.ctrl, next_downstream.data.ctrl)
    assert int(next_upstream.info["expert_switching_used"]) == 0
    assert int(next_downstream.info["expert_switching_used"]) == 0


@pytest.mark.gpu
def test_unified_env_accepts_pilot_specific_ccd_capacity(jit_root, tmp_path):
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, source_env, artifact = _runtime_artifact(jit_root, tmp_path)
    env = UnifiedTubeRSIEnv(
        up_config, down_config, artifact, runtime_naccdmax=1024
    )
    state = env.reset_tube_index(np.int32(0), np.int32(0))
    jax.block_until_ready(state)
    assert int(state.data._impl.naccdmax) == 1024


@pytest.mark.gpu
def test_smoke_closes_exact_interaction_accounting_without_training(
    jit_root, tmp_path
):
    from jit_dvgc.tube_rsi_smoke import run_tube_rsi_smoke
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, source_env, artifact = _runtime_artifact(jit_root, tmp_path)
    env = UnifiedTubeRSIEnv(up_config, down_config, artifact)
    output = tmp_path / "smoke"
    report = run_tube_rsi_smoke(env, output, samples_per_phase=1)

    assert report["status"] == "completed"
    assert report["tube_rsi_smoke"] == "GO"
    assert report["environment_interactions"] == 2
    assert report["training_transitions"] == 0
    assert report["phase_interactions"] == {"upstream": 1, "downstream": 1}
    assert report["test_data_used"] is False
    assert report["expert_switching_used"] is False
    assert json.loads((output / "report.json").read_text()) == report


@pytest.mark.gpu
def test_unified_env_rejects_different_runtime_xml_identities(jit_root, tmp_path):
    from dataclasses import replace
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, source_env, artifact = _runtime_artifact(jit_root, tmp_path)
    wrong_down = replace(
        down_config,
        model={**down_config.model, "xml_sha256": "f" * 64},
    )
    with pytest.raises(ValueError, match="XML identity mismatch"):
        UnifiedTubeRSIEnv(up_config, wrong_down, artifact)


@pytest.mark.gpu
def test_smoke_cli_runs_the_real_unified_environment(jit_root, tmp_path):
    up_config, down_config, source_env, artifact = _runtime_artifact(jit_root, tmp_path)
    output = tmp_path / "cli-smoke"
    env = dict(os.environ)
    env.update(
        PYTHONPATH="JIT/src",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MUJOCO_GL="egl",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "JIT/cli/smoke_tube_rsi.py",
            "--up-config",
            str(jit_root / "configs/phase_u_continuation_smoke.json"),
            "--down-config",
            str(jit_root / "configs/descent_recovery_smoke.json"),
            "--soft-tube",
            str(artifact.root),
            "--output-dir",
            str(output),
            "--samples-per-phase",
            "1",
        ],
        cwd=jit_root.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["tube_rsi_smoke"] == "GO"
    assert report["environment_interactions"] == 2
