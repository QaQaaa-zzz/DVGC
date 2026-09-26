from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jit_dvgc.checkpoint import CheckpointPayload, save_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.unified_formal import load_unified_formal_config
from jit_dvgc.unified_policy_freeze import (
    _checkpoint_identity,
    freeze_unified_policy,
    load_frozen_unified_manifest,
)


def _completed_source(jit_root: Path, tmp_path: Path):
    config_path = jit_root / "configs/pi_unified_round1_natural10.json"
    config = load_unified_formal_config(config_path)
    run_id = config.raw["run_declaration"]["run_id"]
    run_dir = tmp_path / run_id
    checkpoint = run_dir / "checkpoints" / f"transition_{config.ppo.requested_transitions}"
    identity = _checkpoint_identity(config)
    save_checkpoint(
        checkpoint,
        CheckpointPayload(
            identity=identity,
            training_transitions=config.ppo.requested_transitions,
            observation_normalizer={"mean": np.asarray([0.0, 1.0], np.float32)},
            actor_params={"layer": np.arange(6, dtype=np.float32).reshape(2, 3)},
            critic_params={"layer": np.arange(4, dtype=np.float32).reshape(2, 2)},
        ),
    )
    report = {
        "schema": "jit_pi_unified_formal_report_v1",
        "status": "completed",
        "requested_training_transitions": config.ppo.requested_transitions,
        "completed_training_transitions": config.ppo.requested_transitions,
        "checkpoint_transitions": list(config.formal.checkpoint_transitions),
        "train_panel_transitions": list(config.formal.train_panel_transitions),
        "train_panel_interactions": 0,
        "brax_evaluation_transitions": 0,
        "reset_mixture": config.reset_mixture.as_dict(),
        "test_data_used": False,
        "validation_data_used": False,
        "expert_switching_used": False,
        "checkpoint_restored": True,
        "final_metrics": {"loss": 1.0},
    }
    (run_dir / "formal_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config_path, config, run_dir, checkpoint


def test_freeze_unified_policy_binds_completed_checkpoint_without_promoting_claims(
    jit_root, tmp_path
):
    config_path, config, _, checkpoint = _completed_source(jit_root, tmp_path)
    output = tmp_path / "frozen_pi0"

    manifest = freeze_unified_policy(
        output,
        config_path=config_path,
        checkpoint=checkpoint,
        iteration=0,
    )

    policy = manifest["policy"]
    assert manifest["schema"] == "jit_frozen_unified_policy_v1"
    assert manifest["status"] == "frozen"
    assert manifest["immutable_parameters"] is True
    assert manifest["copied_checkpoint"] is False
    assert manifest["training_transitions"] == 0
    assert manifest["environment_interactions"] == 0
    assert policy["name"] == "pi_0"
    assert policy["iteration"] == 0
    assert policy["policy_role"] == "envelope_expansion_authority"
    assert policy["source_training_run_id"] == config.raw["run_declaration"]["run_id"]
    assert policy["source_training_transitions"] == 10_009_600
    assert policy["formal_config_sha256"] == config.config_sha256
    assert policy["source_reset_mixture"] == config.reset_mixture.as_dict()
    assert policy["payload_sha256"] == file_sha256(checkpoint / "payload.pkl")
    assert manifest["claim_boundary"] == {
        "envelope_expansion_authority": True,
        "pi_unified_star_claim": False,
        "jce_jel_claim": False,
        "certified_safe_tube_claim": False,
    }

    loaded = load_frozen_unified_manifest(output / "frozen_unified_policy.json")
    assert loaded == manifest


def test_freeze_unified_policy_rejects_run_id_directory_drift(jit_root, tmp_path):
    config_path, config, _, checkpoint = _completed_source(jit_root, tmp_path)
    wrong_run = tmp_path / "wrong_run"
    wrong_checkpoint = wrong_run / "checkpoints" / checkpoint.name
    wrong_checkpoint.parent.mkdir(parents=True)
    for name in ("payload.pkl", "identity.json"):
        (wrong_checkpoint.parent / checkpoint.name).mkdir(parents=True, exist_ok=True)
        (wrong_checkpoint / name).write_bytes((checkpoint / name).read_bytes())
    (wrong_run / "formal_report.json").write_text(
        (checkpoint.parent.parent / "formal_report.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run directory does not match config run_id"):
        freeze_unified_policy(
            tmp_path / "never_created",
            config_path=config_path,
            checkpoint=wrong_checkpoint,
            iteration=0,
        )
    assert config.raw["run_declaration"]["run_id"] != wrong_run.name


def test_freeze_unified_policy_rejects_nonindependent_training_provenance(
    jit_root, tmp_path
):
    config_path, _, run_dir, checkpoint = _completed_source(jit_root, tmp_path)
    report_path = run_dir / "formal_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["validation_data_used"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="used validation data"):
        freeze_unified_policy(
            tmp_path / "never_created",
            config_path=config_path,
            checkpoint=checkpoint,
            iteration=0,
        )


def _development_source(jit_root, tmp_path):
    from dataclasses import replace
    from jit_dvgc.checkpoint import load_checkpoint
    config_path, config, run_dir, final = _completed_source(jit_root, tmp_path)
    config_copy = tmp_path / 'diagnostic-config.json'
    config_copy.write_bytes(config_path.read_bytes())
    config_path = config_copy
    checkpoint = final.parent / 'transition_1024000'
    payload = load_checkpoint(final, expected=_checkpoint_identity(config))
    save_checkpoint(checkpoint, replace(payload, training_transitions=1024000))
    return config_path, config, run_dir, checkpoint


def _freeze_development(tmp_path, config_path, checkpoint):
    import jit_dvgc.unified_policy_freeze as freeze
    assert hasattr(freeze, 'freeze_development_checkpoint'), 'development checkpoint API missing'
    return freeze.freeze_development_checkpoint(
        tmp_path / 'diagnostic', config_path=config_path,
        checkpoint=checkpoint, name='checkpoint_1024000')


def test_development_checkpoint_loads_with_exact_source_identity_without_promotion(jit_root, tmp_path):
    config_path, config, run_dir, checkpoint = _development_source(jit_root, tmp_path)
    manifest = _freeze_development(tmp_path, config_path, checkpoint)
    assert manifest['schema'] == 'jit_frozen_development_checkpoint_v1'
    record = manifest['policy']
    assert record['policy_role'] == 'development_checkpoint'
    assert record['data_role'] == 'train'
    assert record['iteration'] == 0
    assert record['source_training_transitions'] == 1024000
    assert record['source_requested_training_transitions'] == 10009600
    assert record['formal_config_sha256'] == config.config_sha256
    assert record['formal_config_file_sha256'] == file_sha256(config_path)
    assert record['source_formal_report_sha256'] == file_sha256(run_dir / 'formal_report.json')
    assert record['checkpoint_identity_sha256'] == file_sha256(checkpoint / 'identity.json')
    assert manifest['claim_boundary']['envelope_expansion_authority'] is False
    assert load_frozen_unified_manifest(tmp_path / 'diagnostic/frozen_unified_policy.json') == manifest
    with pytest.raises(ValueError, match='exact completed final checkpoint'):
        freeze_unified_policy(tmp_path / 'formal', config_path=config_path, checkpoint=checkpoint, iteration=7)


@pytest.mark.parametrize('mutation', ['incomplete', 'schedule', 'payload_transition', 'sidecar', 'undeclared'])
def test_development_freeze_rejects_invalid_source(jit_root, tmp_path, mutation):
    from dataclasses import replace
    from jit_dvgc.checkpoint import load_checkpoint
    config_path, config, run_dir, checkpoint = _development_source(jit_root, tmp_path)
    report_path = run_dir / 'formal_report.json'
    report = json.loads(report_path.read_text())
    if mutation == 'incomplete':
        report['status'] = 'running'
    elif mutation == 'schedule':
        report['checkpoint_transitions'].remove(1024000)
    elif mutation == 'payload_transition':
        payload = load_checkpoint(checkpoint, expected=_checkpoint_identity(config))
        (checkpoint / 'payload.pkl').unlink()
        (checkpoint / 'identity.json').unlink()
        checkpoint.rmdir()
        save_checkpoint(checkpoint, replace(payload, training_transitions=2508800))
    elif mutation == 'sidecar':
        sidecar_path = checkpoint / 'identity.json'
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['training_transitions'] = 2508800
        sidecar_path.write_text(json.dumps(sidecar))
    elif mutation == 'undeclared':
        new = checkpoint.with_name('transition_123')
        checkpoint.rename(new)
        checkpoint = new
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _freeze_development(tmp_path, config_path, checkpoint)
    assert not (tmp_path / 'diagnostic').exists()


@pytest.mark.parametrize('mutation', ['report', 'role', 'claim', 'schema', 'data_role', 'config_bytes', 'payload', 'identity', 'expert_switching'])
def test_development_manifest_rejects_tamper_even_with_rehashed_manifest(jit_root, tmp_path, mutation):
    from jit_dvgc.unified_policy_freeze import _canonical_sha256
    config_path, _, run_dir, checkpoint = _development_source(jit_root, tmp_path)
    manifest = _freeze_development(tmp_path, config_path, checkpoint)
    if mutation == 'report':
        report_path = run_dir / 'formal_report.json'
        report = json.loads(report_path.read_text())
        report['final_metrics']['loss'] = 2.0
        report_path.write_text(json.dumps(report))
    elif mutation == 'role':
        manifest['policy']['policy_role'] = 'envelope_expansion_authority'
    elif mutation == 'claim':
        manifest['claim_boundary']['envelope_expansion_authority'] = True
    elif mutation == 'schema':
        manifest['schema'] = 'jit_frozen_unified_policy_v1'
    elif mutation == 'data_role':
        manifest['policy']['data_role'] = 'test'
    elif mutation == 'config_bytes':
        config_path.write_text(config_path.read_text() + '\n')
    elif mutation == 'payload':
        with (checkpoint / 'payload.pkl').open('ab') as stream:
            stream.write(b'corruption')
    elif mutation == 'identity':
        sidecar = checkpoint / 'identity.json'
        sidecar.write_text(sidecar.read_text() + '\n')
    elif mutation == 'expert_switching':
        manifest['expert_switching_used'] = True
    manifest['freeze_protocol_sha256'] = _canonical_sha256({k:v for k,v in manifest.items() if k != 'freeze_protocol_sha256'})
    path = tmp_path / 'diagnostic/frozen_unified_policy.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_frozen_unified_manifest(path)


@pytest.mark.parametrize('name', ['../checkpoint', 'pi_7', '', 'checkpoint/name'])
def test_development_checkpoint_rejects_unsafe_or_formal_policy_name(jit_root, tmp_path, name):
    import jit_dvgc.unified_policy_freeze as freeze
    config_path, _, _, checkpoint = _development_source(jit_root, tmp_path)
    assert hasattr(freeze, 'freeze_development_checkpoint'), 'development checkpoint API missing'
    with pytest.raises(ValueError):
        freeze.freeze_development_checkpoint(tmp_path/'diagnostic', config_path=config_path, checkpoint=checkpoint, name=name)
