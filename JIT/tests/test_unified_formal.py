from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jp


def test_checked_in_formal_contract_is_exact_10m_plus(jit_root):
    from jit_dvgc.unified_formal import load_unified_formal_config

    config = load_unified_formal_config(jit_root / "configs/pi_unified_formal.json")
    assert config.ppo.block_transitions == 25_600
    assert config.ppo.requested_transitions == 10_009_600
    assert config.ppo.requested_transitions // config.ppo.block_transitions == 391
    assert config.ppo.num_parallel_envs == 1024
    assert config.ppo.seed == 821101
    assert config.runtime_naccdmax == 1024
    assert config.formal.checkpoint_transitions == (
        0,
        1_024_000,
        2_508_800,
        5_017_600,
        7_500_800,
        10_009_600,
    )
    assert config.formal.train_panel_transitions == (
        1_024_000,
        2_508_800,
        5_017_600,
        7_500_800,
        10_009_600,
    )
    assert config.formal.samples_per_phase == 8
    assert config.raw["initialization"] == {
        "actor": "fresh",
        "critic": "fresh",
        "optimizer": "fresh",
        "restore_checkpoint": None,
    }
    assert config.raw["claim_boundary"]["test_data_used"] is False
    assert config.raw["claim_boundary"]["validation_data_used"] is False


def test_formal_trainer_kwargs_cover_every_exact_block(jit_root):
    from jit_dvgc.unified_formal import (
        build_unified_formal_trainer_kwargs,
        load_unified_formal_config,
    )

    config = load_unified_formal_config(jit_root / "configs/pi_unified_formal.json")
    callbacks = SimpleNamespace(
        on_progress=lambda *_: None,
        on_policy_params=lambda *_: None,
    )
    env = object()
    kwargs = build_unified_formal_trainer_kwargs(config, env, callbacks)
    assert kwargs["environment"] is env
    assert kwargs["num_timesteps"] == 10_009_600
    assert kwargs["num_evals"] == 392
    assert kwargs["training_metrics_steps"] == 25_600
    assert kwargs["restore_params"] is None
    assert kwargs["run_evals"] is False
    assert kwargs["policy_params_fn"] is callbacks.on_policy_params


def test_formal_controller_checkpoints_and_diagnoses_only_declared_milestones(
    jit_root, tmp_path
):
    from jit_dvgc.checkpoint import CheckpointIdentity
    from jit_dvgc.formal_training import PanelResult
    from jit_dvgc.unified_formal import (
        UnifiedFormalController,
        load_unified_formal_config,
    )

    config = load_unified_formal_config(jit_root / "configs/pi_unified_formal.json")
    saved = []
    panels = []

    def save(path, payload):
        saved.append((path.name, payload.training_transitions))

    def panel(step, _make_policy, _params):
        panels.append(step)
        return PanelResult(step, 16, {"status": "completed"})

    identity = CheckpointIdentity("cfg", "xml", (), (), ())
    controller = UnifiedFormalController(
        config=config,
        run_dir=tmp_path,
        identity=identity,
        evaluate_train_panel=panel,
        checkpoint_saver=save,
    )
    params = ("normalizer", "actor", "critic")
    for step in range(0, config.ppo.requested_transitions + 1, 25_600):
        if step:
            controller.on_progress(
                step,
                {
                    "episode/reward": jp.asarray(float(step)),
                    "episode/length": jp.asarray(10.0),
                },
            )
        controller.on_policy_params(step, object(), params)
        if step:
            controller.on_progress(
                step,
                {
                    "training/total_loss": jp.asarray(float(step)),
                },
            )

    expected_checkpoints = config.formal.checkpoint_transitions
    assert tuple(step for _, step in saved) == expected_checkpoints
    assert tuple(panels) == config.formal.train_panel_transitions
    assert controller.completed_training_transitions == 10_009_600
    assert controller.train_panel_interactions == 80
    assert controller.checkpoint_transitions == expected_checkpoints
    assert controller.train_panel_transitions == config.formal.train_panel_transitions
    assert controller.final_metrics["training/total_loss"] == 10_009_600.0
    assert len((tmp_path / "episode_metrics.jsonl").read_text().splitlines()) == 391


def test_formal_wrapper_loads_explicit_core_replay_contract(tmp_path):
    from jit_dvgc.training.formal import _load_core_replay_contract

    contract = {
        "schema": "jit_tube_rsi_core_replay_v1",
        "selection": "phase_then_source_then_entry",
        "core_probability": 0.5,
        "expansion_probability": 0.5,
        "core_within_source": "uniform",
        "expansion_within_source": "value_weighted",
        "source_core_definition": "first_core_retained_count_entries",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tube_sampling": contract}) + "\n", encoding="utf-8")
    assert _load_core_replay_contract(path) == contract

    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}\n", encoding="utf-8")
    assert _load_core_replay_contract(legacy) is None


def test_failed_train_panel_accounting_is_recovered_from_persisted_report(tmp_path):
    from jit_dvgc.training.formal import _reconcile_failed_train_panel_accounting

    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "engineering_error",
                "reason": "ValueError: plotting failed after rollout",
                "interaction_accounting": {
                    "training": 1_024_000,
                    "brax_evaluation": 0,
                    "fixed_evaluation": 0,
                    "diagnostic": 0,
                },
                "environment_transitions": 1_024_000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    panel_dir = tmp_path / "train_panels" / "transition_1024000"
    panel_dir.mkdir(parents=True)
    (panel_dir / "report.json").write_text(
        json.dumps(
            {
                "training_checkpoint_transition": 1_024_000,
                "environment_interactions": 449,
                "training_transitions": 0,
                "test_data_used": False,
                "validation_data_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _reconcile_failed_train_panel_accounting(tmp_path) == 449
    repaired = json.loads(status_path.read_text(encoding="utf-8"))
    assert repaired["interaction_accounting"]["diagnostic"] == 449
    assert repaired["environment_transitions"] == 1_024_449
    assert repaired["accounting_reconciliation"] == {
        "source": "persisted_train_panel_reports",
        "previous_diagnostic_interactions": 0,
        "reconciled_diagnostic_interactions": 449,
    }