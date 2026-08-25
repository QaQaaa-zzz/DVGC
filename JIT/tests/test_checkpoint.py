from __future__ import annotations

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
