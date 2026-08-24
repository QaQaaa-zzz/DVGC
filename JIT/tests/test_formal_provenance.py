from __future__ import annotations

import json

import numpy as np
import pytest

from jit_dvgc.config import canonical_sha256
from jit_dvgc.constants import REWARD_COMPONENT_KEYS
from jit_dvgc.evaluation import EpisodeFrame, EpisodeTrace, save_episode_trace
from jit_dvgc.provenance import (
    InteractionAccounting,
    RunDeclaration,
    close_run,
    mark_run_running,
    predeclare_run,
)


def _declaration(tmp_path, **overrides):
    resolved = {"phase": "propulsion_ascent", "model": {}}
    values = dict(
        run_id="formal",
        purpose="formal_propulsion_ascent_ppo",
        output_dir=tmp_path / "formal",
        config_sha256=canonical_sha256(resolved),
        xml_sha256="a" * 64,
        reference_sha256="b" * 64,
        training_transition_ceiling=998_400,
        stopping_conditions=("stop_on_nonfinite_metric",),
        resume_command="formal command",
    )
    values.update(overrides)
    return RunDeclaration(**values), resolved


def test_mark_running_preserves_predeclared_identity(tmp_path):
    declaration, resolved = _declaration(tmp_path)
    run_dir = predeclare_run(declaration, resolved_config=resolved)

    mark_run_running(
        run_dir,
        process_id=1234,
        metadata={"started_utc": "2026-08-24T00:00:00Z"},
    )

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status == {
        "status": "running",
        "process_id": 1234,
        "started_utc": "2026-08-24T00:00:00Z",
    }
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == "formal"
    assert manifest["starting_training_transition"] == 0
    assert manifest["resume_semantics"] == "fresh"


def test_running_transition_is_one_way_and_requires_a_real_pid(tmp_path):
    declaration, resolved = _declaration(tmp_path)
    run_dir = predeclare_run(declaration, resolved_config=resolved)
    with pytest.raises(ValueError, match="positive"):
        mark_run_running(run_dir, process_id=0, metadata={})
    mark_run_running(run_dir, process_id=1, metadata={})
    with pytest.raises(ValueError, match="predeclared"):
        mark_run_running(run_dir, process_id=2, metadata={})


def test_warm_start_metadata_is_immutable_and_truthful(tmp_path):
    parent = "/tmp/checkpoints/transition_256000"
    declaration, resolved = _declaration(
        tmp_path,
        parent_checkpoint=parent,
        starting_training_transition=256_000,
        resume_semantics="parameter_warm_start_optimizer_reset",
        segment_seed=820111,
        training_transition_ceiling=742_400,
    )
    run_dir = predeclare_run(declaration, resolved_config=resolved)
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["parent_checkpoint"] == parent
    assert manifest["starting_training_transition"] == 256_000
    assert manifest["resume_semantics"] == "parameter_warm_start_optimizer_reset"
    assert manifest["segment_seed"] == 820111


@pytest.mark.parametrize(
    "overrides",
    [
        {"starting_training_transition": -1},
        {"starting_training_transition": 25_600},
        {
            "starting_training_transition": 25_600,
            "parent_checkpoint": "/tmp/checkpoint",
            "resume_semantics": "bit_exact",
        },
        {"parent_checkpoint": "/tmp/checkpoint"},
    ],
)
def test_invalid_resume_declaration_is_rejected(tmp_path, overrides):
    declaration, resolved = _declaration(tmp_path, **overrides)
    with pytest.raises(ValueError, match="start|resume|parent"):
        predeclare_run(declaration, resolved_config=resolved)


def test_close_run_accepts_running_state(tmp_path):
    declaration, resolved = _declaration(tmp_path)
    run_dir = predeclare_run(declaration, resolved_config=resolved)
    mark_run_running(run_dir, process_id=1, metadata={})
    close_run(
        run_dir,
        status="aborted",
        accounting=InteractionAccounting(0, 0, 0, 0),
        reason="unit test",
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "aborted"


def _frame(tick: int) -> EpisodeFrame:
    terminal = tick == 3
    return EpisodeFrame(
        qpos=np.full(12, tick, dtype=np.float64),
        qvel=np.full(11, tick, dtype=np.float64),
        ctrl=np.full(4, tick, dtype=np.float64),
        action=np.full(4, tick / 10.0, dtype=np.float32),
        reward=float(tick),
        reward_components={key: float(tick) for key in REWARD_COMPONENT_KEYS},
        metrics={"signal/roll": 0.01 * tick, "event/apex_seen": float(terminal)},
        terminated=terminal,
        truncated=False,
        end_code=1 if terminal else 0,
        success=terminal,
        physical_failure=False,
        timeout=False,
    )


def test_episode_trace_artifact_contains_every_state_and_metric(tmp_path):
    trace = EpisodeTrace(
        seed=920001,
        frames=tuple(_frame(tick) for tick in range(4)),
        environment_transitions=3,
    )
    artifact = save_episode_trace(trace, tmp_path / "seed_920001")

    assert artifact.npz_path.is_absolute()
    assert artifact.metadata_path.is_absolute()
    with np.load(artifact.npz_path) as payload:
        assert payload["qpos"].shape == (4, 12)
        assert payload["qvel"].shape == (4, 11)
        assert payload["ctrl"].shape == (4, 4)
        assert payload["action"].shape == (4, 4)
        assert payload["reward"].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert payload["terminated"].tolist() == [False, False, False, True]
        assert "reward_component__drive" in payload.files
        assert "metric__signal__slash__roll" in payload.files
        assert "metric__event__slash__apex_seen" in payload.files
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["seed"] == 920001
    assert metadata["environment_transitions"] == 3
    assert metadata["captured_state_count"] == 4
    assert metadata["terminal"]["success"] is True
    assert metadata["npz_sha256"] == artifact.npz_sha256
