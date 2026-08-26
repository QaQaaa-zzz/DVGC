from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jp
import mediapy as media
import numpy as np
import pytest

from jit_dvgc.checkpoint import (
    CheckpointIdentity,
    CheckpointPayload,
    save_checkpoint,
)
from jit_dvgc.config import canonical_sha256, load_config
from jit_dvgc.constants import (
    ACTION_ORDER,
    ACTOR_FRAME_FIELDS,
    ACTOR_TASK_FIELDS,
    ACTOR_OBSERVATION_SIZE,
    PRIVILEGED_OBSERVATION_SIZE,
    REWARD_COMPONENT_KEYS,
)
from jit_dvgc.formal_training import (
    FormalReport,
    FormalRunController,
    PanelResult,
    run_phase_u_formal,
    validate_formal_report,
)
from jit_dvgc.ppo import wrap_for_jit_training
from jit_dvgc.provenance import verify_run


V4_15M_CHECKPOINTS = (
    0,
    737_280,
    2_998_272,
    7_495_680,
    11_993_088,
    15_015_936,
)


def _identity():
    return CheckpointIdentity(
        config_sha256="1" * 64,
        xml_sha256="2" * 64,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
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
        diagnostic_transitions=0,
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


def test_absolute_5m_report_uses_resolved_schedule(jit_root):
    config = load_config(jit_root / "configs" / "phase_u_absolute_5m.json")
    report = FormalReport(
        requested_training_transitions=4_988_928,
        starting_training_transition=0,
        completed_training_transitions=4_988_928,
        segment_training_transitions=4_988_928,
        brax_evaluation_transitions=0,
        fixed_evaluation_transitions=40,
        diagnostic_transitions=40,
        checkpoint_transitions=(
            0,
            245_760,
            983_040,
            2_506_752,
            3_981_312,
            4_988_928,
        ),
        evaluated_transitions=(
            245_760,
            983_040,
            2_506_752,
            3_981_312,
            4_988_928,
        ),
        final_metrics={"training/sps": 1.0},
        checkpoint_restored=True,
        resume_semantics="fresh",
    )

    assert validate_formal_report(report, config=config) == report


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


def test_v4_controller_records_episode_metrics_without_policy_order_dependency(
    jit_root, tmp_path
):
    config = load_config(jit_root / "configs" / "phase_u_continuation_10m.json")
    controller = FormalRunController(
        config=config,
        run_dir=tmp_path,
        identity=_identity(),
        starting_training_transition=0,
        evaluate_panel=lambda *_args: PanelResult(0, 1, {}),
    )

    controller.on_episode_progress(24_576, {"episode/sps": 1000.0})
    assert not (tmp_path / "episode_metrics.jsonl").exists()
    controller.on_episode_progress(
        49_152,
        {
            "episode/sum_reward": 10.0,
            "episode/length": 20.0,
            "episode/reset/source_airborne_rsi": 1.0,
            "episode/sps": 1000.0,
        },
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "episode_metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert rows[0]["training_transitions"] == 49_152


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
        self._bundle = SimpleNamespace(xml_sha256="0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a")

    def reset(self, _key):
        return SimpleNamespace(
            obs={
                "state": jp.zeros(ACTOR_OBSERVATION_SIZE),
                "privileged_state": jp.zeros(PRIVILEGED_OBSERVATION_SIZE),
            }
        )

    reset_natural = reset


def _fake_make_policy(_params, deterministic=False):
    assert deterministic

    def policy(_observation, _key):
        return jp.zeros(4), {}

    return policy


def _fake_panel_in(
    run_dir, config, step, *, panel_root: str
):
    panel_dir = run_dir / panel_root / f"transition_{step}"
    panel_dir.mkdir(parents=True)
    artifacts = []
    for seed in config.ppo.held_out_seeds:
        npz_path = panel_dir / f"seed_{seed}.npz"
        reset_source = float(panel_root.startswith("diagnostics/airborne_rsi"))
        arrays = {
            "qpos": np.zeros((2, 12)),
            "qvel": np.zeros((2, 11)),
            "ctrl": np.zeros((2, 4)),
            "action": np.zeros((2, 4)),
            "reward": np.zeros(2),
            "terminated": np.array([False, True]),
            "truncated": np.zeros(2, dtype=bool),
            "end_code": np.array([0, 3]),
            "success": np.zeros(2, dtype=bool),
            "physical_failure": np.ones(2, dtype=bool),
            "timeout": np.zeros(2, dtype=bool),
            "metric__reset__slash__source_airborne_rsi": np.full(
                2, reset_source
            ),
        }
        for reward_key in REWARD_COMPONENT_KEYS:
            arrays[f"reward_component__{reward_key}"] = np.zeros(2)
        np.savez_compressed(npz_path, **arrays)
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
        plot_path = panel_dir / "representative_diagnostic.png"
        state_path = panel_dir / "representative_diagnostic.npz"
        media.write_video(
            video_path,
            np.zeros((2, 8, 8, 3), dtype=np.uint8),
            fps=50,
        )
        media.write_image(plot_path, np.zeros((8, 8, 3), dtype=np.uint8))
        diagnostic_arrays = {
            "time_seconds": np.array([0.0, 0.02]),
            "reward_clipped": np.zeros(2),
            "reward_unclipped": np.zeros(2),
            "reward_scaled": np.zeros(2),
            "qpos": np.zeros((2, 12)),
            "qvel": np.zeros((2, 11)),
            "ctrl": np.zeros((2, 4)),
            "action": np.zeros((2, 4)),
            "terminal_terminated": np.array([False, True]),
            "terminal_truncated": np.zeros(2, dtype=bool),
            "terminal_end_code": np.array([0, 3]),
            "terminal_success": np.zeros(2, dtype=bool),
            "terminal_physical_failure": np.ones(2, dtype=bool),
            "terminal_timeout": np.zeros(2, dtype=bool),
            "metric__reset__source_airborne_rsi": np.full(2, reset_source),
            "apex_frame_index": np.array([-1], dtype=np.int32),
            "segment_pre_apex": np.ones(2, dtype=bool),
            "segment_post_apex": np.zeros(2, dtype=bool),
        }
        for reward_key in REWARD_COMPONENT_KEYS:
            diagnostic_arrays[f"reward_component__{reward_key}"] = np.zeros(2)
        np.savez_compressed(state_path, **diagnostic_arrays)
        pre_path = panel_dir / "representative_pre_apex.npz"
        post_path = panel_dir / "representative_post_apex.npz"
        pre_arrays = {
            name: value.copy()
            for name, value in diagnostic_arrays.items()
            if name != "apex_frame_index"
        }
        pre_arrays["source_frame_index"] = np.arange(2, dtype=np.int32)
        pre_arrays["apex_frame_index"] = np.array([-1], dtype=np.int32)
        post_arrays = {
            name: value[:0].copy()
            for name, value in diagnostic_arrays.items()
            if name != "apex_frame_index" and value.shape[0] == 2
        }
        post_arrays["source_frame_index"] = np.empty(0, dtype=np.int32)
        post_arrays["apex_frame_index"] = np.array([-1], dtype=np.int32)
        np.savez_compressed(pre_path, **pre_arrays)
        np.savez_compressed(post_path, **post_arrays)
        video_report = {
            "video": str(video_path.resolve()),
            "state_trace": str(state_path.resolve()),
            "diagnostic_plot": str(plot_path.resolve()),
            "diagnostic_data": str(state_path.resolve()),
            "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "diagnostic_plot_sha256": hashlib.sha256(
                plot_path.read_bytes()
            ).hexdigest(),
            "diagnostic_data_sha256": hashlib.sha256(
                state_path.read_bytes()
            ).hexdigest(),
            "pre_apex_data": str(pre_path.resolve()),
            "post_apex_data": str(post_path.resolve()),
            "pre_apex_data_sha256": hashlib.sha256(
                pre_path.read_bytes()
            ).hexdigest(),
            "post_apex_data_sha256": hashlib.sha256(
                post_path.read_bytes()
            ).hexdigest(),
            "captured_state_count": 2,
            "encoded_frame_count": 2,
            "environment_transitions": 1,
            "apex_frame_index": -1,
            "pre_apex_sample_count": 2,
            "post_apex_sample_count": 0,
            "pre_apex_environment_transitions": 1,
            "post_apex_environment_transitions": 0,
            "fps": 50,
            "representative_seed": config.ppo.held_out_seeds[0],
            "representative_episode_npz": artifacts[0]["npz_path"],
            "representative_episode_npz_sha256": artifacts[0]["npz_sha256"],
            "reset_source_airborne_rsi": bool(reset_source),
        }
        (panel_dir / "video_report.json").write_text(
            json.dumps(video_report), encoding="utf-8"
        )
    return PanelResult(step, 8, summary)


def _fake_panel(_env, run_dir, config, step, _make_policy, _params):
    return _fake_panel_in(run_dir, config, step, panel_root="evaluations")


def _fake_rsi_panel(_env, run_dir, config, step, _make_policy, _params):
    return _fake_panel_in(
        run_dir,
        config,
        step,
        panel_root="diagnostics/airborne_rsi",
    )


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
        assert kwargs["log_training_metrics"] is False
        params = ({"normalizer": 0}, {"actor": 1}, {"critic": 2})
        kwargs["policy_params_fn"](0, _fake_make_policy, params)
        for step in range(25_600, expected_steps + 1, 25_600):
            kwargs["policy_params_fn"](step, _fake_make_policy, params)
            kwargs["progress_fn"](step, {"training/kl": 0.01})
        return _fake_make_policy, params, {"training/kl": 0.01}

    return train


def _fake_absolute_trainer():
    def train(**kwargs):
        assert kwargs["num_timesteps"] == 4_988_928
        assert kwargs["num_evals"] == 204
        assert kwargs["num_envs"] == 384
        assert kwargs["unroll_length"] == 64
        assert kwargs["batch_size"] == 16
        assert kwargs["num_minibatches"] == 24
        assert kwargs["num_updates_per_batch"] == 8
        assert kwargs["learning_rate"] == 0.0001
        assert kwargs["entropy_cost"] == 0.01
        assert kwargs["discounting"] == 0.99
        assert kwargs["clipping_epsilon"] == 0.2
        assert kwargs["max_grad_norm"] == 0.5
        assert kwargs["restore_params"] is None
        params = ({"normalizer": 0}, {"actor": 1}, {"critic": 2})
        kwargs["policy_params_fn"](0, _fake_make_policy, params)
        for step in range(24_576, 4_988_928 + 1, 24_576):
            kwargs["policy_params_fn"](step, _fake_make_policy, params)
            kwargs["progress_fn"](step, {"training/kl": 0.01})
        return _fake_make_policy, params, {"training/kl": 0.01}

    return train


def _fake_continuation_10m_trainer():
    def train(**kwargs):
        assert kwargs["num_timesteps"] == 15_015_936
        assert kwargs["num_evals"] == 612
        assert kwargs["log_training_metrics"] is True
        assert kwargs["training_metrics_steps"] == 24_576
        assert kwargs["restore_params"] is None
        assert kwargs["wrap_env_fn"] is wrap_for_jit_training
        params = ({"normalizer": 0}, {"actor": 1}, {"critic": 2})
        kwargs["policy_params_fn"](0, _fake_make_policy, params)
        for index, step in enumerate(range(24_576, 15_015_936 + 1, 24_576), 1):
            kwargs["progress_fn"](
                step,
                {
                    "episode/sum_reward": float(index),
                    "episode/length": 100.0,
                    "episode/reset/source_airborne_rsi": 8.0,
                    "episode/sps": 1000.0,
                },
            )
            kwargs["policy_params_fn"](step, _fake_make_policy, params)
            kwargs["progress_fn"](
                step,
                {
                    "training/kl_mean": 0.01,
                    "training/policy_loss": -0.2,
                    "training/v_loss": 0.4,
                    "training/total_loss": 0.0,
                    "training/policy_dist_mean_std": 0.8,
                    "training/sps": 1000.0,
                },
            )
        return _fake_make_policy, params, {"training/kl_mean": 0.01}

    return train


def _run_continuation_v4_unit(jit_root, tmp_path, run_id):
    run_phase_u_formal(
        jit_root / "configs" / "phase_u_continuation_10m.json",
        run_id,
        run_root=tmp_path,
        trainer=_fake_continuation_10m_trainer(),
        env_factory=_FakeEnv,
        panel_evaluator=_fake_panel,
        diagnostic_panel_evaluator=_fake_rsi_panel,
        backend_name=lambda: "gpu",
    )
    return tmp_path / run_id


@pytest.fixture(scope="module")
def completed_v4_run_for_manifest_mutation(tmp_path_factory):
    jit_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path_factory.mktemp("completed_v4_manifest")
    return _run_continuation_v4_unit(
        jit_root, run_root, "completed_v4_manifest_mutation"
    )


@pytest.mark.parametrize(
    ("artifact", "field", "mutated_value"),
    [
        pytest.param(
            "manifest",
            "parent_checkpoint",
            "/tmp/adversarial_parent_checkpoint",
            id="parent-checkpoint",
        ),
        pytest.param(
            "manifest",
            "starting_training_transition",
            24_576,
            id="starting-training-transition",
        ),
        pytest.param(
            "manifest",
            "resume_semantics",
            "parameter_warm_start_optimizer_reset",
            id="resume-semantics",
        ),
        pytest.param(
            "manifest", "segment_seed", 820_402, id="segment-seed"
        ),
        pytest.param(
            "manifest",
            "resume_command",
            (
                "/home/qy/mujoco_playground/.venv/bin/python "
                "JIT/cli/train_phase_expert.py --phase propulsion_ascent "
                "--config JIT/configs/phase_u_continuation_10m.json "
                "--run-id completed_v4_manifest_mutation --formal "
                "--restore-checkpoint /tmp/adversarial_parent_checkpoint"
            ),
            id="restore-bearing-resume-command",
        ),
        pytest.param(
            "resume_command.txt",
            None,
            "different persisted command\n",
            id="persisted-resume-command-mismatch",
        ),
    ],
)
def test_completed_v4_provenance_rejects_nonfresh_manifest_or_command(
    completed_v4_run_for_manifest_mutation,
    artifact,
    field,
    mutated_value,
):
    run_dir = completed_v4_run_for_manifest_mutation
    manifest_path = run_dir / "run_manifest.json"
    resume_path = run_dir / "resume_command.txt"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_resume = resume_path.read_text(encoding="utf-8")
    try:
        if artifact == "manifest":
            manifest = json.loads(original_manifest)
            manifest[field] = mutated_value
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            resume_path.write_text(mutated_value, encoding="utf-8")

        with pytest.raises(ValueError, match="formal v4 fresh-start provenance"):
            verify_run(run_dir)
    finally:
        manifest_path.write_text(original_manifest, encoding="utf-8")
        resume_path.write_text(original_resume, encoding="utf-8")


def test_v4_formal_restore_is_rejected_before_runtime_or_run_creation(
    jit_root, tmp_path
):
    def unexpected_backend():
        raise AssertionError("backend must not be inspected for a fresh-only restore")

    def unexpected_env(_config):
        raise AssertionError("environment must not be created for a fresh-only restore")

    run_id = "v4_restore_forbidden"
    with pytest.raises(ValueError, match="fresh-only.*restore"):
        run_phase_u_formal(
            jit_root / "configs" / "phase_u_continuation_10m.json",
            run_id,
            restore_checkpoint=tmp_path / "any_parent_checkpoint",
            run_root=tmp_path,
            env_factory=unexpected_env,
            backend_name=unexpected_backend,
        )

    assert not (tmp_path / run_id).exists()


def test_continuation_v4_runner_is_fresh_and_persists_learning_curves(
    jit_root, tmp_path
):
    run_dir = _run_continuation_v4_unit(
        jit_root, tmp_path, "continuation_v4_unit"
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_checkpoint"] is None
    assert manifest["starting_training_transition"] == 0
    assert manifest["training_transition_ceiling"] == 15_015_936
    assert "--restore-checkpoint" not in manifest["resume_command"]
    assert (run_dir / "training_curves.png").is_file()
    assert (run_dir / "training_curves.npz").is_file()
    assert (run_dir / "training_curves.json").is_file()
    verified = verify_run(run_dir)
    assert verified["absolute_training_transition"] == 15_015_936
    assert verified["formal_checkpoint_transitions"] == list(V4_15M_CHECKPOINTS)
    assert verified["training_curves"]["ppo_sample_count"] == 611

    report_path = run_dir / "training_curves.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    original = report_path.read_text(encoding="utf-8")
    report["data_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="curve data hash"):
        verify_run(run_dir)
    report_path.write_text(original, encoding="utf-8")

    video_path = run_dir / "evaluations/transition_15015936/video_report.json"
    original_video = video_path.read_text(encoding="utf-8")
    video = json.loads(original_video)
    video["pre_apex_data"], video["post_apex_data"] = (
        video["post_apex_data"],
        video["pre_apex_data"],
    )
    video["pre_apex_data_sha256"], video["post_apex_data_sha256"] = (
        video["post_apex_data_sha256"],
        video["pre_apex_data_sha256"],
    )
    video_path.write_text(json.dumps(video), encoding="utf-8")
    with pytest.raises(ValueError, match="Apex source indices|sample count"):
        verify_run(run_dir)
    video_path.write_text(original_video, encoding="utf-8")


def test_completed_v4_provenance_rejects_previous_method_values(
    jit_root, tmp_path
):
    run_dir = _run_continuation_v4_unit(
        jit_root, tmp_path, "continuation_v4_method_drift_unit"
    )
    resolved_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "run_manifest.json"
    original_resolved = resolved_path.read_text(encoding="utf-8")
    original_manifest = manifest_path.read_text(encoding="utf-8")
    previous_v4_values = (
        ("model", "naccdmax", 48),
        ("reset", "airborne_rsi_probability", 0.05),
        ("events", "jump_zone_x_max", 3.1),
        ("reward", "height_coeff", 20.0),
    )
    for section, field, previous_value in previous_v4_values:
        resolved = json.loads(original_resolved)
        resolved[section][field] = previous_value
        manifest = json.loads(original_manifest)
        manifest["config_sha256"] = canonical_sha256(resolved)
        resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="approved v4"):
            verify_run(run_dir)
    resolved_path.write_text(original_resolved, encoding="utf-8")
    manifest_path.write_text(original_manifest, encoding="utf-8")


def test_absolute_5m_runner_is_fresh_and_uses_dynamic_target(jit_root, tmp_path):
    run_phase_u_formal(
        jit_root / "configs" / "phase_u_absolute_5m.json",
        "absolute_5m_unit",
        run_root=tmp_path,
        trainer=_fake_absolute_trainer(),
        env_factory=_FakeEnv,
        panel_evaluator=_fake_panel,
        diagnostic_panel_evaluator=_fake_rsi_panel,
        backend_name=lambda: "gpu",
    )

    run_dir = tmp_path / "absolute_5m_unit"
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    report = json.loads(
        (run_dir / "formal_report.json").read_text(encoding="utf-8")
    )
    assert manifest["parent_checkpoint"] is None
    assert manifest["starting_training_transition"] == 0
    assert manifest["resume_semantics"] == "fresh"
    assert manifest["training_transition_ceiling"] == 4_988_928
    assert manifest["stopping_conditions"][0] == (
        "stop_at_absolute_transition_4988928"
    )
    assert "--restore-checkpoint" not in manifest["resume_command"]
    assert status["interaction_accounting"]["training"] == 4_988_928
    assert status["interaction_accounting"]["fixed_evaluation"] == 40
    assert status["interaction_accounting"]["diagnostic"] == 40
    assert report["completed_training_transitions"] == 4_988_928
    assert report["diagnostic_transitions"] == 40
    assert report["checkpoint_transitions"][-1] == 4_988_928
    verified = verify_run(run_dir)
    assert verified["absolute_training_transition"] == 4_988_928
    assert verified["formal_checkpoint_transitions"] == [
        0,
        245_760,
        983_040,
        2_506_752,
        3_981_312,
        4_988_928,
    ]

    diagnostic_summary_path = (
        run_dir
        / "diagnostics/airborne_rsi/transition_245760/summary.json"
    )
    original_summary = diagnostic_summary_path.read_text(encoding="utf-8")
    diagnostic_summary = json.loads(original_summary)
    diagnostic_summary["environment_transitions"] += 1
    diagnostic_summary_path.write_text(
        json.dumps(diagnostic_summary), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="RSI diagnostic panel total"):
        verify_run(run_dir)
    diagnostic_summary_path.write_text(original_summary, encoding="utf-8")

    diagnostic_video_path = (
        run_dir
        / "diagnostics/airborne_rsi/transition_4988928/video_report.json"
    )
    original_video = diagnostic_video_path.read_text(encoding="utf-8")
    diagnostic_video = json.loads(original_video)
    diagnostic_video["captured_state_count"] += 1
    diagnostic_video_path.write_text(
        json.dumps(diagnostic_video), encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="RSI diagnostic.*(frame accounting|sample count)"
    ):
        verify_run(run_dir)
    diagnostic_video_path.write_text(original_video, encoding="utf-8")

    rsi_trace_path = (
        run_dir
        / "diagnostics/airborne_rsi/transition_245760/seed_930001.npz"
    )
    rsi_metadata_path = rsi_trace_path.with_suffix(".json")
    original_trace = rsi_trace_path.read_bytes()
    original_metadata = rsi_metadata_path.read_text(encoding="utf-8")
    with np.load(rsi_trace_path) as payload:
        wrong_source_arrays = {
            name: payload[name].copy() for name in payload.files
        }
    wrong_source_arrays[
        "metric__reset__slash__source_airborne_rsi"
    ] = np.zeros(2)
    np.savez_compressed(rsi_trace_path, **wrong_source_arrays)
    rsi_metadata = json.loads(original_metadata)
    rsi_metadata["npz_sha256"] = hashlib.sha256(
        rsi_trace_path.read_bytes()
    ).hexdigest()
    rsi_metadata_path.write_text(json.dumps(rsi_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="RSI diagnostic reset source"):
        verify_run(run_dir)
    rsi_trace_path.write_bytes(original_trace)
    rsi_metadata_path.write_text(original_metadata, encoding="utf-8")

    with np.load(rsi_trace_path) as payload:
        truncated_arrays = {name: payload[name][:1] for name in payload.files}
    np.savez_compressed(rsi_trace_path, **truncated_arrays)
    rsi_metadata = json.loads(original_metadata)
    rsi_metadata["npz_sha256"] = hashlib.sha256(
        rsi_trace_path.read_bytes()
    ).hexdigest()
    rsi_metadata_path.write_text(json.dumps(rsi_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="RSI diagnostic.*sample count"):
        verify_run(run_dir)
    rsi_trace_path.write_bytes(original_trace)
    rsi_metadata_path.write_text(original_metadata, encoding="utf-8")

    final_video_report = (
        run_dir / "evaluations/transition_4988928/video_report.json"
    )
    original_final_video = final_video_report.read_text(encoding="utf-8")
    aliased = json.loads(original_final_video)
    arbitrary_npz = aliased["diagnostic_data"]
    arbitrary_hash = hashlib.sha256(Path(arbitrary_npz).read_bytes()).hexdigest()
    aliased["video"] = arbitrary_npz
    aliased["video_sha256"] = arbitrary_hash
    aliased["diagnostic_plot"] = arbitrary_npz
    aliased["diagnostic_plot_sha256"] = arbitrary_hash
    final_video_report.write_text(json.dumps(aliased), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact type|distinct"):
        verify_run(run_dir)
    final_video_report.write_text(original_final_video, encoding="utf-8")

    final_diagnostic_path = Path(
        json.loads(original_final_video)["diagnostic_data"]
    )
    original_final_diagnostic = final_diagnostic_path.read_bytes()
    with np.load(final_diagnostic_path) as payload:
        wrong_final_source = {
            name: payload[name].copy() for name in payload.files
        }
    wrong_final_source["metric__reset__source_airborne_rsi"] = np.ones(2)
    np.savez_compressed(final_diagnostic_path, **wrong_final_source)
    wrong_final_report = json.loads(original_final_video)
    wrong_final_report["diagnostic_data_sha256"] = hashlib.sha256(
        final_diagnostic_path.read_bytes()
    ).hexdigest()
    final_video_report.write_text(
        json.dumps(wrong_final_report), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="natural.*reset source"):
        verify_run(run_dir)
    final_diagnostic_path.write_bytes(original_final_diagnostic)
    final_video_report.write_text(original_final_video, encoding="utf-8")


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
        actor_task_fields=ACTOR_TASK_FIELDS,
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

    diagnostic_plot = (
        run_dir
        / "evaluations/transition_998400/representative_diagnostic.png"
    )
    original_plot = diagnostic_plot.read_bytes()
    diagnostic_plot.write_bytes(original_plot + b"drift")
    with pytest.raises(ValueError, match="diagnostic_plot.*hash"):
        verify_run(run_dir)
    diagnostic_plot.write_bytes(original_plot)

    resolved_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "run_manifest.json"
    original_resolved = resolved_path.read_text(encoding="utf-8")
    original_manifest = manifest_path.read_text(encoding="utf-8")
    resolved = json.loads(original_resolved)
    resolved["reset"]["airborne_rsi_probability"] = 0.5
    manifest = json.loads(original_manifest)
    manifest["config_sha256"] = canonical_sha256(resolved)
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="approved v2"):
        verify_run(run_dir)
    resolved_path.write_text(original_resolved, encoding="utf-8")
    manifest_path.write_text(original_manifest, encoding="utf-8")

    report_path = run_dir / "formal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint_restored"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="restore"):
        verify_run(run_dir)
