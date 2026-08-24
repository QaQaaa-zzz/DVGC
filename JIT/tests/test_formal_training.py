from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import jax.numpy as jp
import numpy as np
import pytest

from jit_dvgc.checkpoint import (
    CheckpointIdentity,
    CheckpointPayload,
    save_checkpoint,
)
from jit_dvgc.config import load_config
from jit_dvgc.constants import (
    ACTION_ORDER,
    ACTOR_FRAME_FIELDS,
    ACTOR_OBSERVATION_SIZE,
    PRIVILEGED_OBSERVATION_SIZE,
)
from jit_dvgc.formal_training import (
    FormalReport,
    FormalRunController,
    PanelResult,
    run_phase_u_formal,
    validate_formal_report,
)
from jit_dvgc.provenance import verify_run


def _identity():
    return CheckpointIdentity(
        config_sha256="1" * 64,
        xml_sha256="2" * 64,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        action_order=ACTION_ORDER,
    )


def _report(**overrides):
    values = dict(
        requested_training_transitions=998_400,
        starting_training_transition=0,
        completed_training_transitions=998_400,
        segment_training_transitions=998_400,
        brax_evaluation_transitions=0,
        fixed_evaluation_transitions=40,
        checkpoint_transitions=(0, 102_400, 256_000, 512_000, 742_400, 998_400),
        evaluated_transitions=(102_400, 256_000, 512_000, 742_400, 998_400),
        final_metrics={"training/sps": 1.0},
        checkpoint_restored=True,
        resume_semantics="fresh",
    )
    values.update(overrides)
    return FormalReport(**values)


def test_formal_report_requires_exact_target_and_panels():
    report = validate_formal_report(_report())
    assert report.completed_training_transitions == 998_400
    assert report.segment_training_transitions == 998_400


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"completed_training_transitions": 972_800}, "target"),
        ({"segment_training_transitions": 972_800}, "segment"),
        ({"brax_evaluation_transitions": 1}, "Brax evaluation"),
        ({"fixed_evaluation_transitions": 39}, "fixed evaluation"),
        ({"checkpoint_transitions": (0, 998_400)}, "checkpoint schedule"),
        ({"evaluated_transitions": (998_400,)}, "evaluation schedule"),
        ({"final_metrics": {"training/kl": float("nan")}}, "nonfinite"),
        ({"checkpoint_restored": False}, "restore"),
    ],
)
def test_formal_report_rejects_incomplete_evidence(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_formal_report(_report(**overrides))


def _controller(jit_root, tmp_path, *, start=0):
    config = load_config(jit_root / "configs" / "phase_u_formal.json")
    evaluated = []

    def evaluate_panel(step, _make_policy, _params):
        evaluated.append(step)
        return PanelResult(
            absolute_transition=step,
            environment_transitions=8,
            summary={"rollouts": 8, "environment_transitions": 8},
        )

    controller = FormalRunController(
        config=config,
        run_dir=tmp_path,
        identity=_identity(),
        starting_training_transition=start,
        evaluate_panel=evaluate_panel,
    )
    return controller, evaluated


def test_controller_executes_only_declared_checkpoint_and_evaluation_schedule(
    jit_root, tmp_path
):
    controller, evaluated = _controller(jit_root, tmp_path)
    params = ({"normalizer": 0}, {"actor": 1}, {"critic": 2})
    make_policy = object()

    controller.on_policy_params(0, make_policy, params)
    for relative in range(25_600, 998_400 + 1, 25_600):
        controller.on_policy_params(relative, make_policy, params)
        controller.on_progress(relative, {"training/kl": relative / 1_000_000})

    assert controller.completed_training_transitions == 998_400
    assert controller.checkpoint_transitions == (
        0,
        102_400,
        256_000,
        512_000,
        742_400,
        998_400,
    )
    assert tuple(evaluated) == (
        102_400,
        256_000,
        512_000,
        742_400,
        998_400,
    )
    assert controller.fixed_evaluation_transitions == 40
    restored = controller.restore_final_checkpoint()
    assert restored.training_transitions == 998_400
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 39
    assert rows[-1]["training_transitions"] == 998_400


def test_controller_rejects_duplicate_out_of_order_and_nonfinite_callbacks(
    jit_root, tmp_path
):
    controller, _ = _controller(jit_root, tmp_path)
    params = (None, None, None)
    controller.on_policy_params(0, object(), params)
    with pytest.raises(ValueError, match="consecutive"):
        controller.on_policy_params(0, object(), params)
    controller.on_policy_params(25_600, object(), params)
    with pytest.raises(ValueError, match="nonfinite"):
        controller.on_progress(25_600, {"training/kl": float("inf")})


def test_warm_controller_uses_absolute_offsets_and_only_future_evaluations(
    jit_root, tmp_path
):
    controller, evaluated = _controller(jit_root, tmp_path, start=256_000)
    params = ({"normalizer": 0}, {"actor": 1}, {"critic": 2})
    controller.on_policy_params(0, object(), params)
    for relative in range(25_600, 742_400 + 1, 25_600):
        controller.on_policy_params(relative, object(), params)
        controller.on_progress(relative, {"training/loss": 1.0})

    assert controller.completed_training_transitions == 998_400
    assert controller.checkpoint_transitions == (256_000, 512_000, 742_400, 998_400)
    assert tuple(evaluated) == (512_000, 742_400, 998_400)
    assert controller.fixed_evaluation_transitions == 24


class _FakeEnv:
    def __init__(self, _config):
        self._bundle = SimpleNamespace(xml_sha256="e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192")

    def reset(self, _key):
        return SimpleNamespace(
            obs={
                "state": jp.zeros(ACTOR_OBSERVATION_SIZE),
                "privileged_state": jp.zeros(PRIVILEGED_OBSERVATION_SIZE),
            }
        )


def _fake_make_policy(_params, deterministic=False):
    assert deterministic

    def policy(_observation, _key):
        return jp.zeros(4), {}

    return policy


def _fake_panel(_env, run_dir, config, step, _make_policy, _params):
    panel_dir = run_dir / "evaluations" / f"transition_{step}"
    panel_dir.mkdir(parents=True)
    artifacts = []
    for seed in config.ppo.held_out_seeds:
        npz_path = panel_dir / f"seed_{seed}.npz"
        np.savez_compressed(npz_path, qpos=np.zeros((2, 12)))
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        metadata_path = panel_dir / f"seed_{seed}.json"
        metadata = {
            "seed": seed,
            "environment_transitions": 1,
            "captured_state_count": 2,
            "npz_path": str(npz_path.resolve()),
            "npz_sha256": digest,
            "terminal": {
                "terminated": True,
                "truncated": False,
                "end_code": 3,
                "success": False,
                "physical_failure": True,
                "timeout": False,
            },
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        artifacts.append(
            {
                "seed": seed,
                "metadata_path": str(metadata_path.resolve()),
                "npz_path": str(npz_path.resolve()),
                "npz_sha256": digest,
                "environment_transitions": 1,
                "captured_state_count": 2,
            }
        )
    summary = {
        "absolute_transition": step,
        "held_out_seeds": list(config.ppo.held_out_seeds),
        "rollouts": 8,
        "environment_transitions": 8,
        "apex_success_rate": 0.0,
        "end_reason_counts": {"roll_limit": 8},
        "trace_artifacts": artifacts,
    }
    (panel_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if step == config.ppo.requested_transitions:
        video_path = panel_dir / "representative.mp4"
        state_path = panel_dir / "representative.npz"
        video_path.write_bytes(b"unit-video")
        np.savez_compressed(state_path, qpos=np.zeros((2, 12)))
        video_report = {
            "video": str(video_path.resolve()),
            "state_trace": str(state_path.resolve()),
            "captured_state_count": 2,
            "encoded_frame_count": 2,
            "environment_transitions": 1,
            "fps": 50,
        }
        (panel_dir / "video_report.json").write_text(
            json.dumps(video_report), encoding="utf-8"
        )
    return PanelResult(step, 8, summary)


def _fake_trainer(expected_steps, expected_evals):
    def train(**kwargs):
        assert kwargs["num_timesteps"] == expected_steps
        assert kwargs["num_evals"] == expected_evals
        assert kwargs["num_envs"] == 1024
        assert kwargs["unroll_length"] == 25
        assert kwargs["batch_size"] == 128
        assert kwargs["num_minibatches"] == 8
        assert kwargs["num_updates_per_batch"] == 1
        assert kwargs["learning_rate"] == 0.0001
        assert kwargs["run_evals"] is False
        params = ({"normalizer": 0}, {"actor": 1}, {"critic": 2})
        kwargs["policy_params_fn"](0, _fake_make_policy, params)
        for step in range(25_600, expected_steps + 1, 25_600):
            kwargs["policy_params_fn"](step, _fake_make_policy, params)
            kwargs["progress_fn"](step, {"training/kl": 0.01})
        return _fake_make_policy, params, {"training/kl": 0.01}

    return train


def test_formal_runner_closes_exact_fresh_segment_with_injected_trainer(
    jit_root, tmp_path
):
    result = run_phase_u_formal(
        jit_root / "configs" / "phase_u_formal.json",
        "formal_unit",
        run_root=tmp_path,
        trainer=_fake_trainer(998_400, 40),
        env_factory=_FakeEnv,
        panel_evaluator=_fake_panel,
        backend_name=lambda: "gpu",
    )

    run_dir = tmp_path / "formal_unit"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (run_dir / "formal_report.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert status["interaction_accounting"] == {
        "brax_evaluation": 0,
        "diagnostic": 0,
        "fixed_evaluation": 40,
        "training": 998_400,
    }
    assert manifest["resume_semantics"] == "fresh"
    assert manifest["segment_seed"] == 820101
    assert report["checkpoint_restored"] is True
    assert report["completed_training_transitions"] == 998_400
    assert result["run_dir"] == str(run_dir.resolve())


def test_formal_runner_warm_start_records_optimizer_reset_and_absolute_offset(
    jit_root, tmp_path
):
    config = load_config(jit_root / "configs" / "phase_u_formal.json")
    identity = CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=config.model["xml_sha256"],
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        action_order=ACTION_ORDER,
    )
    parent = tmp_path / "parent" / "transition_256000"
    save_checkpoint(
        parent,
        CheckpointPayload(identity, 256_000, {"n": 0}, {"a": 1}, {"c": 2}),
    )

    run_phase_u_formal(
        jit_root / "configs" / "phase_u_formal.json",
        "warm_unit",
        restore_checkpoint=parent,
        run_root=tmp_path,
        trainer=_fake_trainer(742_400, 30),
        env_factory=_FakeEnv,
        panel_evaluator=_fake_panel,
        backend_name=lambda: "gpu",
    )

    run_dir = tmp_path / "warm_unit"
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert manifest["starting_training_transition"] == 256_000
    assert manifest["parent_checkpoint"] == str(parent.resolve())
    assert manifest["resume_semantics"] == "parameter_warm_start_optimizer_reset"
    assert manifest["segment_seed"] == 820111
    assert status["interaction_accounting"]["training"] == 742_400
    assert status["interaction_accounting"]["fixed_evaluation"] == 24
    verified = verify_run(run_dir)
    assert verified["formal_checkpoint_transitions"] == [
        256_000,
        512_000,
        742_400,
        998_400,
    ]


def test_formal_verifier_rejects_checkpoint_panel_trace_and_report_drift(
    jit_root, tmp_path
):
    run_phase_u_formal(
        jit_root / "configs" / "phase_u_formal.json",
        "verify_unit",
        run_root=tmp_path,
        trainer=_fake_trainer(998_400, 40),
        env_factory=_FakeEnv,
        panel_evaluator=_fake_panel,
        backend_name=lambda: "gpu",
    )
    run_dir = tmp_path / "verify_unit"
    verified = verify_run(run_dir)
    assert verified["formal_checkpoint_transitions"][-1] == 998_400
    assert verified["fixed_evaluation_transitions"] == 40

    checkpoint = run_dir / "checkpoints/transition_102400/identity.json"
    original_checkpoint = checkpoint.read_text(encoding="utf-8")
    checkpoint.unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        verify_run(run_dir)
    checkpoint.write_text(original_checkpoint, encoding="utf-8")

    summary_path = run_dir / "evaluations/transition_102400/summary.json"
    original_summary = summary_path.read_text(encoding="utf-8")
    summary = json.loads(original_summary)
    summary["held_out_seeds"] = summary["held_out_seeds"][:-1]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out seeds"):
        verify_run(run_dir)
    summary_path.write_text(original_summary, encoding="utf-8")

    trace_path = run_dir / "evaluations/transition_102400/seed_920001.json"
    original_trace = trace_path.read_text(encoding="utf-8")
    trace = json.loads(original_trace)
    trace["environment_transitions"] = 2
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="trace transition"):
        verify_run(run_dir)
    trace_path.write_text(original_trace, encoding="utf-8")

    video_report_path = run_dir / "evaluations/transition_998400/video_report.json"
    original_video_report = video_report_path.read_text(encoding="utf-8")
    video_report_path.unlink()
    with pytest.raises(ValueError, match="video"):
        verify_run(run_dir)
    video_report_path.write_text(original_video_report, encoding="utf-8")

    report_path = run_dir / "formal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint_restored"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="restore"):
        verify_run(run_dir)
