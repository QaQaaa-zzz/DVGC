from __future__ import annotations

import hashlib
import json
import pickle

import pytest

from jit_dvgc.checkpoint import (
    CheckpointIdentity,
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS


def _identity(**overrides) -> CheckpointIdentity:
    values = dict(
        config_sha256="1" * 64,
        xml_sha256="2" * 64,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    values.update(overrides)
    return CheckpointIdentity(**values)


def _write_unpickle_marker(path: str):
    from pathlib import Path

    Path(path).write_text("unpickled", encoding="utf-8")


class _MarkerPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (_write_unpickle_marker, (str(self.marker),))


def test_checkpoint_round_trip_preserves_all_training_payloads(tmp_path):
    payload = CheckpointPayload(
        identity=_identity(),
        training_transitions=25_600,
        observation_normalizer={"mean": [0.0], "std": [1.0]},
        actor_params={"actor": [1, 2]},
        critic_params={"critic": [3, 4]},
    )
    save_checkpoint(tmp_path / "checkpoint", payload)

    restored = load_checkpoint(tmp_path / "checkpoint", expected=_identity())
    assert restored == payload
    assert (tmp_path / "checkpoint" / "identity.json").is_file()
    assert (tmp_path / "checkpoint" / "payload.pkl").is_file()


def test_checkpoint_rejects_config_identity_mismatch(tmp_path):
    payload = CheckpointPayload(
        identity=_identity(),
        training_transitions=0,
        observation_normalizer=None,
        actor_params=None,
        critic_params=None,
    )
    save_checkpoint(tmp_path / "checkpoint", payload)

    with pytest.raises(ValueError, match="config_sha256"):
        load_checkpoint(
            tmp_path / "checkpoint",
            expected=_identity(config_sha256="0" * 64),
        )


def test_checkpoint_rejects_jump_signal_identity_mismatch(tmp_path):
    payload = CheckpointPayload(
        identity=_identity(),
        training_transitions=0,
        observation_normalizer=None,
        actor_params=None,
        critic_params=None,
    )
    save_checkpoint(tmp_path / "checkpoint", payload)
    with pytest.raises(ValueError, match="actor_task_fields"):
        load_checkpoint(
            tmp_path / "checkpoint",
            expected=_identity(actor_task_fields=("different_signal",)),
        )


def test_checkpoint_rejects_sidecar_identity_before_unpickling(tmp_path):
    directory = tmp_path / "checkpoint"
    save_checkpoint(
        directory,
        CheckpointPayload(_identity(), 0, None, None, None),
    )
    marker = tmp_path / "unpickle_marker"
    payload_path = directory / "payload.pkl"
    payload_path.write_bytes(pickle.dumps(_MarkerPayload(marker)))
    sidecar_path = directory / "identity.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="config_sha256"):
        load_checkpoint(
            directory,
            expected=_identity(config_sha256="0" * 64),
        )
    assert not marker.exists()
