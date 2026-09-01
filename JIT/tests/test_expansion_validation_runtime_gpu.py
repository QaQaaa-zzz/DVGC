from __future__ import annotations

import jax
import numpy as np
import pytest


pytestmark = pytest.mark.gpu


def test_real_validation_anchors_restore_into_unified_runtime_without_steps(
    jit_root, repository_root
):
    """Restore all five held-out anchors without consuming validation rollout steps."""
    if jax.default_backend() != "gpu":
        pytest.skip("requires JAX GPU")

    from jit_dvgc.expansion_validation_protocol import (
        audit_expansion_validation_protocol,
        load_expansion_validation_protocol_config,
    )
    from jit_dvgc.expansion_validation_runtime import (
        _expected_unified_actor_observation,
        load_validation_anchor_snapshots,
        restore_validation_anchor_as_unified,
    )
    from jit_dvgc.unified_formal import build_unified_formal_environment
    from jit_dvgc.unified_policy_freeze import load_frozen_unified_manifest

    config_path = jit_root / "configs/envelope_iter0_expansion_validation.json"
    audit = audit_expansion_validation_protocol(config_path)
    assert audit["status"] == "protocol_ready"
    config = load_expansion_validation_protocol_config(config_path)
    protocol = config["protocol"]
    frozen = load_frozen_unified_manifest(repository_root / protocol["frozen_policy"])
    policy_record = frozen["policy"]
    _formal, _artifact, env = build_unified_formal_environment(
        repository_root / policy_record["formal_config"]
    )
    anchors = load_validation_anchor_snapshots(protocol)
    assert len(anchors) == 5

    restored = 0
    cached_mismatch_by_phase = {"upstream": 0, "downstream": 0}
    for phase in ("upstream", "downstream"):
        for index, _anchor in enumerate(protocol["sources"][phase]["anchors"]):
            snapshot = anchors[(phase, index)]
            state = restore_validation_anchor_as_unified(
                snapshot,
                phase=phase,
                env=env,
                parent_group_index=index,
            )
            jax.block_until_ready(state)
            actor = np.asarray(jax.device_get(state.obs["state"]), dtype=np.float32)
            expected_actor = _expected_unified_actor_observation(snapshot, phase=phase)
            np.testing.assert_allclose(actor, expected_actor, rtol=0.0, atol=1.0e-5)
            cached = np.asarray(snapshot.observation, dtype=np.float32)
            if not np.allclose(cached, expected_actor, rtol=0.0, atol=1.0e-5):
                cached_mismatch_by_phase[phase] += 1
            assert actor.shape == (76,)
            assert np.isfinite(actor).all()
            assert int(np.asarray(jax.device_get(state.info["active_phase"]))) == (
                0 if phase == "upstream" else 1
            )
            assert int(np.asarray(jax.device_get(state.info["start_phase"]))) == (
                0 if phase == "upstream" else 1
            )
            assert int(np.asarray(jax.device_get(state.info["episode_step"]))) == 0
            assert int(np.asarray(jax.device_get(state.info["phase_episode_step"]))) == 0
            assert not bool(np.asarray(jax.device_get(state.info["expert_switching_used"])))
            assert not bool(np.asarray(jax.device_get(state.done)))
            restored += 1

    assert restored == 5
    assert cached_mismatch_by_phase == {"upstream": 0, "downstream": 2}
