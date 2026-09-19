from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from jit_dvgc.checkpoint import CheckpointIdentity, CheckpointPayload, save_checkpoint
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from jit_dvgc.handoff_bank import pytree_sha256
from jit_dvgc.phase_u_warm_start import load_phase_u_actor_initialization


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_manifest(path, payload):
    protocol = {
        key: value for key, value in payload.items() if key != "freeze_protocol_sha256"
    }
    payload["freeze_protocol_sha256"] = _canonical_sha256(protocol)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def frozen_phase_u_policy(tmp_path):
    identity = CheckpointIdentity(
        config_sha256="1" * 64,
        xml_sha256="2" * 64,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    normalizer = {
        "mean": np.array([1.25, -0.5], dtype=np.float32),
        "std": np.array([0.75, 2.0], dtype=np.float32),
    }
    actor = {"layer": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    critic = {"value": np.array([99.0], dtype=np.float32)}
    checkpoint = tmp_path / "source" / "checkpoints" / "transition_128000"
    save_checkpoint(
        checkpoint,
        CheckpointPayload(identity, 128_000, normalizer, actor, critic),
    )
    source_config = tmp_path / "source_config.json"
    source_config.write_text('{"source":"config"}\n', encoding="utf-8")
    source_report = tmp_path / "formal_report.json"
    source_report.write_text('{"status":"completed"}\n', encoding="utf-8")
    policy = {
        "name": "source_development_checkpoint",
        "iteration": 6,
        "policy_role": "development_checkpoint",
        "checkpoint": str(checkpoint.resolve()),
        "formal_config": str(source_config.resolve()),
        "formal_config_sha256": identity.config_sha256,
        "xml_sha256": identity.xml_sha256,
        "source_training_run_id": "source",
        "source_training_transitions": 128_000,
        "source_reset_mixture": {"natural_reset_probability": 0.2},
        "payload_sha256": _file_sha256(checkpoint / "payload.pkl"),
        "normalizer_sha256": pytree_sha256(normalizer),
        "actor_sha256": pytree_sha256(actor),
        "critic_sha256": pytree_sha256(critic),
        "actor_frame_fields": list(ACTOR_FRAME_FIELDS),
        "actor_task_fields": list(ACTOR_TASK_FIELDS),
        "action_order": list(ACTION_ORDER),
        "data_role": "train",
        "formal_config_file_sha256": _file_sha256(source_config),
        "source_formal_report": str(source_report.resolve()),
        "source_formal_report_sha256": _file_sha256(source_report),
        "checkpoint_identity_sha256": _file_sha256(checkpoint / "identity.json"),
        "source_requested_training_transitions": 256_000,
    }
    manifest = {
        "schema": "jit_frozen_development_checkpoint_v1",
        "status": "frozen",
        "immutable_parameters": True,
        "copied_checkpoint": False,
        "training_transitions": 0,
        "environment_interactions": 0,
        "expert_switching_used": False,
        "policy": policy,
        "claim_boundary": {
            "envelope_expansion_authority": False,
            "pi_unified_star_claim": False,
            "jce_jel_claim": False,
            "certified_safe_tube_claim": False,
        },
    }
    manifest_path = tmp_path / "frozen_unified_policy.json"
    _write_manifest(manifest_path, manifest)
    return {
        "path": manifest_path,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "normalizer": normalizer,
        "actor": actor,
        "critic": critic,
        "source_config": source_config,
        "source_report": source_report,
    }


def test_loader_restores_exact_normalizer_and_actor_with_fresh_training_state(
    frozen_phase_u_policy,
):
    source = frozen_phase_u_policy
    initialization = load_phase_u_actor_initialization(source["path"])

    assert initialization.restore_params[0] is not source["normalizer"]
    assert initialization.restore_params[1] is not source["actor"]
    np.testing.assert_array_equal(
        initialization.restore_params[0]["mean"], source["normalizer"]["mean"]
    )
    np.testing.assert_array_equal(
        initialization.restore_params[1]["layer"], source["actor"]["layer"]
    )
    assert len(initialization.restore_params) == 2
    assert initialization.parent_transition == 128_000
    assert initialization.provenance["actor_initialized"] is True
    assert initialization.provenance["normalizer_initialized"] is True
    assert initialization.provenance["critic_fresh"] is True
    assert initialization.provenance["optimizer_fresh"] is True
    assert "critic_params" not in initialization.provenance


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("actor", "actor"),
        ("normalizer", "normalizer"),
        ("xml", "xml"),
        ("action_order", "action_order"),
        ("actor_frame_fields", "actor_frame_fields"),
        ("actor_task_fields", "actor_task_fields"),
        ("payload", "payload"),
        ("transition", "transition"),
        ("manifest_status", "frozen"),
    ],
)
def test_loader_rejects_identity_or_manifest_drift(
    frozen_phase_u_policy, mutation, message
):
    source = frozen_phase_u_policy
    manifest = copy.deepcopy(source["manifest"])
    policy = manifest["policy"]
    if mutation == "actor":
        policy["actor_sha256"] = "a" * 64
    elif mutation == "normalizer":
        policy["normalizer_sha256"] = "b" * 64
    elif mutation == "xml":
        policy["xml_sha256"] = "c" * 64
    elif mutation == "action_order":
        policy["action_order"] = [*policy["action_order"][:-1], "wrong"]
    elif mutation == "actor_frame_fields":
        policy["actor_frame_fields"] = [*policy["actor_frame_fields"], "wrong"]
    elif mutation == "actor_task_fields":
        policy["actor_task_fields"] = ["wrong"]
    elif mutation == "payload":
        policy["payload_sha256"] = "d" * 64
    elif mutation == "transition":
        policy["source_training_transitions"] += 1
    elif mutation == "manifest_status":
        manifest["status"] = "running"
    _write_manifest(source["path"], manifest)

    with pytest.raises(ValueError, match=message):
        load_phase_u_actor_initialization(source["path"])


@pytest.mark.parametrize("artifact", ["source_config", "source_report"])
@pytest.mark.parametrize("mutation", ["missing", "content"])
def test_loader_rejects_missing_or_changed_source_artifacts(
    frozen_phase_u_policy, artifact, mutation
):
    source = frozen_phase_u_policy
    path = source[artifact]
    if mutation == "missing":
        path.unlink()
    else:
        path.write_text('{"drift":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="source.*(config|report).*(missing|hash)"):
        load_phase_u_actor_initialization(source["path"])
