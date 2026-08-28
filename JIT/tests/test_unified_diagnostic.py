from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jp


def _state(phase: int, step: int = 0):
    done = step >= 2
    return SimpleNamespace(
        data=SimpleNamespace(
            qpos=jp.asarray([3.0 + 0.1 * step, 0.0, 0.5 + 0.02 * step]),
            qvel=jp.zeros(3),
        ),
        obs={"state": jp.zeros(76), "privileged_state": jp.zeros(106)},
        reward=jp.asarray(float(step)),
        done=jp.asarray(done),
        info={
            "active_phase": jp.asarray(phase),
            "tube_global_index": jp.asarray(phase),
            "expert_switching_used": jp.asarray(False),
            "phase_transitioned": jp.asarray(False),
            "success": jp.asarray(False),
            "physical_failure": jp.asarray(done),
            "timeout": jp.asarray(False),
            "end_code": jp.asarray(2 if done else 0),
            "episode_step": jp.asarray(step),
        },
    )


def test_fixed_panel_runs_both_train_phases_to_true_terminal():
    from jit_dvgc.unified_diagnostic import rollout_fixed_tube_panel

    env = SimpleNamespace(
        tube_pool=SimpleNamespace(
            upstream_count=1,
            downstream_count=1,
            artifact=SimpleNamespace(
                manifest={
                    "schema": "jit_soft_tube_v1",
                    "status": "completed",
                    "training_guidance_only": True,
                    "manifest_sha256": "soft-hash",
                    "test_data_used": False,
                    "validation_data_used": False,
                }
            ),
        )
    )

    def reset(phase, _entry):
        return _state(int(phase), 0)

    def step(state, _action):
        return _state(int(state.info["active_phase"]), int(state.info["episode_step"]) + 1)

    report, trajectories = rollout_fixed_tube_panel(
        env,
        lambda _obs, _key: (jp.zeros(4), {}),
        samples_per_phase=1,
        horizon=10,
        reset_fn=reset,
        step_fn=step,
    )

    assert report["status"] == "completed"
    assert report["environment_interactions"] == 4
    assert report["phase_interactions"] == {"downstream": 2, "upstream": 2}
    assert report["phase_rollouts"] == {"downstream": 1, "upstream": 1}
    assert report["end_reason_counts"] == {"nonfinite": 2}
    assert report["terminal_class_counts"] == {"physical_failure": 2}
    assert report["expert_switching_used"] is False
    assert report["test_data_used"] is False
    assert report["validation_data_used"] is False
    assert len(trajectories) == 2
    assert all(len(row["x"]) == 3 for row in trajectories)


def test_fixed_panel_rejects_non_train_tube_identity():
    from jit_dvgc.unified_diagnostic import validate_panel_artifact

    valid = {
        "schema": "jit_soft_tube_v1",
        "status": "completed",
        "test_data_used": False,
        "validation_data_used": False,
        "training_guidance_only": True,
    }
    validate_panel_artifact(valid)
    for field in ("test_data_used", "validation_data_used"):
        import pytest

        with pytest.raises(ValueError, match="TRAIN-only"):
            validate_panel_artifact({**valid, field: True})


def test_xz_overlay_writes_literal_trajectory_plot(tmp_path):
    from jit_dvgc.unified_diagnostic import plot_xz_visitation

    tube_points = (
        {"phase": "upstream", "x": 2.5, "z": 0.12, "sampling_weight": 1.0},
        {"phase": "downstream", "x": 3.4, "z": 0.65, "sampling_weight": 0.2},
    )
    trajectories = (
        {
            "phase": "upstream",
            "entry_index": 0,
            "x": [2.5, 2.6],
            "z": [0.12, 0.2],
            "terminal": "physical_failure",
            "terminal_class": "physical_failure",
        },
        {
            "phase": "downstream",
            "entry_index": 0,
            "x": [3.4, 3.5],
            "z": [0.65, 0.55],
            "terminal": "timeout",
            "terminal_class": "timeout",
        },
    )
    output = tmp_path / "xz.png"
    result = plot_xz_visitation(tube_points, trajectories, output)
    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 1000
