from __future__ import annotations


def test_checkpoint_identity_uses_config_hash_xml_and_runtime_contract():
    from types import SimpleNamespace

    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    from jit_dvgc.expert_freeze import checkpoint_identity_from_config

    config = SimpleNamespace(config_sha256="c" * 64, model={"xml_sha256": "x" * 64})
    identity = checkpoint_identity_from_config(config)
    assert identity.config_sha256 == "c" * 64
    assert identity.xml_sha256 == "x" * 64
    assert identity.actor_frame_fields == ACTOR_FRAME_FIELDS
    assert identity.actor_task_fields == ACTOR_TASK_FIELDS
    assert identity.action_order == ACTION_ORDER
