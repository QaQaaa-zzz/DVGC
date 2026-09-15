"""A runtime rebind preserves source weights without inventing completed training."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jit_dvgc.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint
from jit_dvgc.config import file_sha256, load_config as load_phase_config
from jit_dvgc.jump_evidence_validation import read, write
import jit_dvgc.unified_policy_freeze as frozen

from .test_probe_recovery_window import recovery_source, generate


@pytest.fixture
def initialized_source(recovery_source, tmp_path, jit_root):
    source = recovery_source("descent_stable_forward_0p5s.json", descent_updates={
        "recovery_ticks": 100, "min_post_contact_forward_progress": 1.,
    })
    raw = generate(source)
    config = frozen._load_policy_formal_config(source[3])
    run_dir = tmp_path / "source_run" / "recovery_fixture"
    checkpoint = run_dir / "checkpoints/transition_0"
    save_checkpoint(checkpoint, CheckpointPayload(
        identity=frozen._checkpoint_identity(config), training_transitions=0,
        observation_normalizer={"mean": np.array([0., 1.], np.float32)},
        actor_params={"weights": np.arange(6, dtype=np.float32).reshape(2, 3)},
        critic_params={"weights": np.array([2., 3.], np.float32)},
    ))
    write(run_dir / "formal_report.json", dict(
        schema="jit_pi_unified_formal_report_v1", status="completed",
        requested_training_transitions=3200, completed_training_transitions=3200,
        checkpoint_restored=True, reset_mixture=config.reset_mixture.as_dict(),
        expert_switching_used=False, validation_data_used=False, test_data_used=False,
        checkpoint_transitions=[0, 3200], train_panel_transitions=[3200],
        checkpoint_evaluation=raw["checkpoint_evaluation"],
    ))
    source_dir = tmp_path / "source_frozen"
    manifest = frozen.freeze_development_checkpoint(
        source_dir, config_path=source[3], checkpoint=checkpoint, name="fixture_source",
        allow_initialization=True,
    )
    target = deepcopy(raw)
    down_path = jit_root / "configs/descent_stable_forward_0p5s.json"
    target["inputs"].update(down_config_path=str(down_path),
                           down_config_sha256=load_phase_config(down_path).config_sha256)
    target["input_files"][str(down_path)] = file_sha256(down_path)
    target["run_declaration"]["run_id"] = "rebound_fixture"
    target_path = tmp_path / "target_config.json"
    write(target_path, target)
    return target_path, source_dir / "frozen_unified_policy.json", manifest


def freeze_target(tmp_path, initialized_source):
    config_path, source_path, _ = initialized_source
    return frozen.freeze_initialization_policy(
        tmp_path / "rebound", config_path=config_path,
        source_frozen_policy=source_path, name="source_rebound",
    )


def test_initialization_rebind_has_zero_cost_preserved_parameters_and_no_formal_report(
    tmp_path, initialized_source,
):
    manifest = freeze_target(tmp_path, initialized_source)
    assert manifest["schema"] == "jit_frozen_initialization_policy_v1"
    assert manifest["new_training_transitions"] == manifest["environment_interactions"] == 0
    record = manifest["policy"]
    assert record["policy_role"] == "initialization_only"
    assert record["source_training_transitions"] == 0
    assert record["formal_config_sha256"] != initialized_source[2]["policy"]["formal_config_sha256"]
    for field in ("actor_sha256", "normalizer_sha256", "critic_sha256"):
        assert record[field] == initialized_source[2]["policy"][field]
    assert not list((tmp_path / "rebound").rglob("formal_report.json"))
    receipt = read(record["initialization_receipt"])
    assert receipt["new_training_transitions"] == receipt["environment_interactions"] == 0
    assert receipt["source_policy"] == initialized_source[2]["policy"]
    assert frozen.load_frozen_unified_manifest(tmp_path / "rebound/frozen_unified_policy.json") == manifest
    assert frozen.verify_frozen_unified_record(record).actor_sha256 == record["actor_sha256"]


@pytest.mark.parametrize("component", ["actor_params", "observation_normalizer", "critic_params"])
def test_rebound_checkpoint_with_changed_source_parameters_is_rejected(
    tmp_path, initialized_source, component,
):
    manifest = freeze_target(tmp_path, initialized_source)
    record = manifest["policy"]
    identity = frozen._checkpoint_identity(frozen._load_policy_formal_config(initialized_source[0]))
    payload = load_checkpoint(Path(record["checkpoint"]), expected=identity)
    changed = {key: value + 1 for key, value in getattr(payload, component).items()}
    replacement = tmp_path / "changed/checkpoints/transition_0"
    save_checkpoint(replacement, replace(payload, **{component: changed}))
    tampered = {**record, "checkpoint": str(replacement)}
    with pytest.raises(ValueError, match="source parameters"):
        frozen.verify_frozen_unified_record(tampered)


@pytest.mark.parametrize("target", ["source_manifest", "source_report", "receipt", "new_config"])
def test_initialization_rebind_rejects_provenance_or_config_drift(tmp_path, initialized_source, target):
    manifest = freeze_target(tmp_path, initialized_source)
    record = manifest["policy"]
    paths = {
        "source_manifest": initialized_source[1],
        "source_report": Path(initialized_source[2]["policy"]["source_formal_report"]),
        "receipt": Path(record["initialization_receipt"]),
        "new_config": initialized_source[0],
    }
    path = paths[target]
    if target == "receipt":
        receipt = read(path)
        receipt["new_training_transitions"] = 3200
        write(path, receipt)
    else:
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        frozen.verify_frozen_unified_record(record)


def test_initialization_policy_works_with_existing_probe_bank(tmp_path, initialized_source):
    from jit_dvgc.probe_bank import lock_probe_bank, load_probe_bank
    manifest = freeze_target(tmp_path, initialized_source)
    path = tmp_path / "bank.json"
    spec = dict(version="initialization_test", task=dict(
        xml_sha256=manifest["policy"]["xml_sha256"], start_contract_sha256="a" * 64,
        centerline_sha256="b" * 64, resolution_sha256="c" * 64,
        success_criterion="stable_forward_recovery", continuation_start_semantics="fresh_continuation_v1",
    ), max_ticks=400, label_interaction_budget=400, max_candidates_per_process=1,
        members=[dict(frozen_policy=str(tmp_path / "rebound/frozen_unified_policy.json"),
                      roles=["proposer", "evaluator"])])
    locked = lock_probe_bank(spec, path)
    assert load_probe_bank(path) == locked
    assert locked["members"][0]["policy"]["policy_role"] == "initialization_only"


def test_initialization_endpoint_rebind_cannot_change_actuator_contract(tmp_path, initialized_source):
    config_path, _, _ = initialized_source
    config = read(config_path)
    phase = read(config["inputs"]["down_config_path"])
    phase["action"]["base_rear_speed"] = 13.
    phase_path = tmp_path / "changed_action.json"
    write(phase_path, phase)
    config["inputs"].update(down_config_path=str(phase_path),
                           down_config_sha256=load_phase_config(phase_path).config_sha256)
    config["input_files"][str(phase_path)] = file_sha256(phase_path)
    write(config_path, config)
    with pytest.raises(ValueError, match="downstream runtime contract drift"):
        freeze_target(tmp_path, initialized_source)
    assert not (tmp_path / "rebound").exists()


@pytest.mark.parametrize("mutation", ["new_training", "claim", "copied_checkpoint", "role"])
def test_initialization_manifest_rejects_resealed_training_or_authority_claim(
    tmp_path, initialized_source, mutation,
):
    manifest = freeze_target(tmp_path, initialized_source)
    if mutation == "new_training":
        manifest["new_training_transitions"] = 3200
    elif mutation == "claim":
        manifest["claim_boundary"]["envelope_expansion_authority"] = True
    elif mutation == "copied_checkpoint":
        manifest["copied_checkpoint"] = False
    else:
        manifest["policy"]["policy_role"] = "development_checkpoint"
    manifest["freeze_protocol_sha256"] = frozen._canonical_sha256(
        {key: value for key, value in manifest.items() if key != "freeze_protocol_sha256"}
    )
    path = tmp_path / "rebound/frozen_unified_policy.json"
    write(path, manifest)
    with pytest.raises(ValueError):
        frozen.load_frozen_unified_manifest(path)
