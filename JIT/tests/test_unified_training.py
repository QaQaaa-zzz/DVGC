from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jp
import pytest


def _pilot_config():
    from jit_dvgc.unified_training import UnifiedPilotConfig, UnifiedPPOConfig

    initialization = {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }
    return UnifiedPilotConfig(
        schema="jit_pi_unified_pilot_v1",
        raw={
            "schema": "jit_pi_unified_pilot_v1",
            "initialization": initialization,
            "claim_boundary": {
                "engineering_integrity_only": True,
                "test_data_used": False,
            },
        },
        config_sha256="pilot-config-hash",
        up_config_path="up.json",
        up_config_sha256="up-hash",
        down_config_path="down.json",
        down_config_sha256="down-hash",
        soft_tube_path="soft-tube",
        soft_tube_manifest_sha256="soft-hash",
        tube_rsi_smoke_report="smoke-report.json",
        tube_rsi_smoke_report_sha256="smoke-report-hash",
        runtime_naccdmax=1024,
        ppo=UnifiedPPOConfig(
            num_parallel_envs=1024,
            episode_horizon=400,
            unroll_length=25,
            batch_size=128,
            num_minibatches=8,
            num_updates_per_batch=1,
            requested_transitions=25_600,
            learning_rate=1.0e-4,
            entropy_cost=1.0e-3,
            reward_scaling=0.1,
            discounting=0.995,
            gae_lambda=0.97,
            clipping_epsilon=0.1,
            max_grad_norm=0.75,
            seed=821001,
        ),
    )


def test_checked_in_pilot_is_one_exact_fresh_block(jit_root):
    from jit_dvgc.unified_training import load_unified_pilot_config

    config = load_unified_pilot_config(jit_root / "configs/pi_unified_pilot.json")
    assert config.ppo.block_transitions == 25_600
    assert config.ppo.requested_transitions == 25_600
    assert config.ppo.num_parallel_envs == 1024
    assert config.ppo.seed == 821001
    assert config.runtime_naccdmax == 1024
    assert config.raw["initialization"] == {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }
    assert config.raw["claim_boundary"]["test_data_used"] is False
    assert config.raw["claim_boundary"]["engineering_integrity_only"] is True


def test_gate_requires_completed_soft_tube_and_go_smoke():
    from jit_dvgc.unified_training import validate_pilot_gate

    config = _pilot_config()
    soft = {
        "schema": "jit_soft_tube_v1",
        "status": "completed",
        "manifest_sha256": "soft-hash",
        "test_data_used": False,
        "validation_data_used": False,
        "training_guidance_only": True,
        "certified_safe": False,
    }
    smoke = {
        "schema": "jit_tube_rsi_smoke_v1",
        "status": "completed",
        "tube_rsi_smoke": "GO",
        "soft_tube_manifest_sha256": "soft-hash",
        "environment_interactions": 16,
        "training_transitions": 0,
        "test_data_used": False,
        "validation_data_used": False,
        "expert_switching_used": False,
    }
    validate_pilot_gate(config, soft, smoke)
    with pytest.raises(ValueError, match="Tube-RSI smoke is not GO"):
        validate_pilot_gate(config, soft, {**smoke, "tube_rsi_smoke": "NO_GO"})
    with pytest.raises(ValueError, match="TEST exclusion"):
        validate_pilot_gate(config, {**soft, "test_data_used": True}, smoke)
    with pytest.raises(ValueError, match="Soft Tube identity"):
        validate_pilot_gate(
            config, soft, {**smoke, "soft_tube_manifest_sha256": "other"}
        )


def test_trainer_kwargs_are_fresh_single_policy_and_exact():
    from jit_dvgc.ppo import make_network_factory, wrap_for_jit_training
    from jit_dvgc.unified_training import build_unified_trainer_kwargs

    config = _pilot_config()
    env = object()
    kwargs = build_unified_trainer_kwargs(config, env, progress_fn=lambda *_: None)
    assert kwargs["environment"] is env
    assert kwargs["num_timesteps"] == 25_600
    assert kwargs["restore_params"] is None
    assert kwargs["restore_value_fn"] is False
    assert kwargs["network_factory"].func is make_network_factory().func
    assert kwargs["wrap_env_fn"] is wrap_for_jit_training
    assert kwargs["num_evals"] == 0
    assert kwargs["run_evals"] is False


def test_pilot_runner_saves_and_restores_one_fresh_checkpoint(
    tmp_path, monkeypatch
):
    import jit_dvgc.unified_training as module
    from jit_dvgc.checkpoint import load_checkpoint

    config = _pilot_config()
    up = SimpleNamespace(
        config_sha256="up-hash",
        model={"xml_sha256": "xml-hash", "reference_sha256": "reference-hash"},
    )
    down = SimpleNamespace(config_sha256="down-hash")
    artifact = SimpleNamespace(
        manifest={
            "schema": "jit_soft_tube_v1",
            "status": "completed",
            "manifest_sha256": "soft-hash",
            "test_data_used": False,
            "validation_data_used": False,
            "training_guidance_only": True,
            "certified_safe": False,
        }
    )
    smoke = {
        "schema": "jit_tube_rsi_smoke_v1",
        "status": "completed",
        "tube_rsi_smoke": "GO",
        "soft_tube_manifest_sha256": "soft-hash",
        "environment_interactions": 16,
        "training_transitions": 0,
        "test_data_used": False,
        "validation_data_used": False,
        "expert_switching_used": False,
    }

    class FakeEnv:
        actor_observation_size = 76
        privileged_observation_size = 106
        action_size = 4
        _bundle = SimpleNamespace(xml_sha256="xml-hash")

    monkeypatch.setattr(module, "load_unified_pilot_config", lambda _path: config)
    monkeypatch.setattr(module, "load_config", lambda path: up if str(path) == "up.json" else down)
    monkeypatch.setattr(module, "load_soft_tube", lambda _path: artifact)
    monkeypatch.setattr(module, "read_json", lambda _path: smoke)
    monkeypatch.setattr(module, "file_sha256", lambda _path: "smoke-report-hash")
    monkeypatch.setattr(module.jax, "default_backend", lambda: "gpu")

    captured = {}

    def trainer(**kwargs):
        captured.update(kwargs)
        kwargs["progress_fn"](25_600, {"training/total_loss": jp.asarray(1.0)})
        return object(), ("normalizer", "actor", "critic"), {}

    result = module.run_unified_pilot(
        tmp_path / "pilot.json",
        "pilot-run",
        run_root=tmp_path,
        trainer=trainer,
        env_factory=lambda *_args, **_kwargs: FakeEnv(),
    )
    assert captured["restore_params"] is None
    assert result["report"]["training_transitions"] == 25_600
    assert result["report"]["checkpoint_restored"] is True
    assert result["report"]["expert_switching_used"] is False
    assert result["report"]["test_data_used"] is False
    assert result["report"]["final_metrics"] == {"training/total_loss": 1.0}
    checkpoint = tmp_path / "pilot-run/checkpoints/transition_25600"
    restored = load_checkpoint(checkpoint, expected=module.checkpoint_identity(config, FakeEnv()))
    assert restored.actor_params == "actor"
    assert restored.critic_params == "critic"
