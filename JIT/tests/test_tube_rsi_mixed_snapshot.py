from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv
from jit_dvgc.handoff_snapshot import save_snapshot
from jit_dvgc.soft_tube import (
    PHASE_MIXTURE,
    SOFT_TUBE_SCHEMA,
    SoftTubeArtifact,
    WEIGHT_FLOOR,
    WEIGHT_SCALE,
)
from jit_dvgc.unified_envelope_snapshot import (
    capture_unified_envelope_snapshot,
    physical_state_sha256 as unified_state_sha256,
    restore_unified_envelope_snapshot,
    save_unified_envelope_snapshot,
)
from jit_dvgc.upstream_boundary import physical_state_sha256


def _entry(
    *,
    phase: str,
    snapshot,
    state_sha256: str,
    score: float,
    value_model_target: str,
    expansion: bool,
):
    entry = {
        "phase": phase,
        "split": "train",
        "snapshot": str(snapshot),
        "state_sha256": state_sha256,
        "value_score": score,
        "sampling_weight": WEIGHT_FLOOR + WEIGHT_SCALE * score,
        "value_model_target": value_model_target,
    }
    if expansion:
        threshold = (
            0.9333483934566058
            if phase == "upstream"
            else 0.8721734129976408
        )
        entry.update(
            {
                "snapshot_schema": "jit_unified_envelope_snapshot_v1",
                "continuation_label": 1,
                "score_source": {
                    "kind": "policy_conditioned_continuation_field",
                    "field_name": value_model_target,
                    "acceptance_threshold_exclusive": threshold,
                    "selection_rule": (
                        "TRAIN_label_positive_and_score_strictly_greater_than_threshold"
                    ),
                },
            }
        )
    return entry


def _legacy_artifact(jit_root, tmp_path):
    up_config = load_config(jit_root / "configs/phase_u_continuation_smoke.json")
    down_config = load_config(jit_root / "configs/descent_recovery_smoke.json")
    source_env = TwoPhaseBikeEnv(up_config)
    up_state = jax.jit(source_env.reset_natural)(jax.random.PRNGKey(301))
    down_state = jax.jit(source_env.reset_airborne_rsi)(jax.random.PRNGKey(302))

    rows = []
    snapshots = {}
    for phase, state, marker, score in (
        ("upstream", up_state, "legacy-up", 0.25),
        ("downstream", down_state, "legacy-down", 0.75),
    ):
        snapshot = source_env.capture_handoff_snapshot(
            state,
            policy_sha256="a" * 64,
            parent_trajectory=f"test-{marker}",
            policy_identity="test-only",
        )
        path = tmp_path / marker
        save_snapshot(path, snapshot)
        rows.append(
            _entry(
                phase=phase,
                snapshot=path,
                state_sha256=physical_state_sha256(snapshot),
                score=score,
                value_model_target="V_up" if phase == "upstream" else "V_down",
                expansion=False,
            )
        )
        snapshots[phase] = snapshot

    artifact = SoftTubeArtifact(
        tmp_path / "legacy-soft-tube",
        {
            "schema": SOFT_TUBE_SCHEMA,
            "status": "completed",
            "test_data_used": False,
            "validation_data_used": False,
            "phase_mixture": PHASE_MIXTURE,
        },
        tuple(rows),
        {},
    )
    return up_config, down_config, source_env, artifact, rows, snapshots


def _contextualized_snapshot(state, env, *, phase: str, marker: str):
    root_x = state.data.qpos[env._bundle.model_index.root_qpos_address]
    up_events = state.info["up_events"].replace(
        jump_zone_seen=jnp.asarray(True),
        ascending_seen=jnp.asarray(True),
        apex_seen=jnp.asarray(phase == "downstream"),
        episode_step=jnp.asarray(13 if phase == "upstream" else 21, jnp.int32),
    )
    down_events = state.info["down_events"].replace(
        airborne_seen=jnp.asarray(True),
        valid_contact_seen=jnp.asarray(phase == "downstream"),
        contact_x=jnp.asarray(root_x),
        post_contact_ticks=jnp.asarray(0 if phase == "upstream" else 2, jnp.int32),
        recovery_success=jnp.asarray(False),
    )
    info = {
        **state.info,
        "up_events": up_events,
        "down_events": down_events,
        "active_phase": jnp.asarray(0 if phase == "upstream" else 1, jnp.int32),
        "start_phase": jnp.asarray(0, jnp.int32),
        "phase_transitioned": jnp.asarray(phase == "downstream"),
        "episode_step": jnp.asarray(13 if phase == "upstream" else 21, jnp.int32),
        "phase_episode_step": jnp.asarray(7 if phase == "upstream" else 4, jnp.int32),
        "episode_return": jnp.asarray(1.75 if phase == "upstream" else 2.5, jnp.float32),
        "source_tick": jnp.asarray(77 if phase == "upstream" else 88, jnp.int32),
    }
    contextual = state.replace(info=info)
    return capture_unified_envelope_snapshot(
        contextual,
        env=env,
        parent_trajectory=f"mixed-{marker}",
        parent_state_sha256=("d" if phase == "upstream" else "e") * 64,
        config_sha256="c" * 64,
        policy_actor_sha256="b" * 64,
        policy_payload_sha256="f" * 64,
        policy_iteration=0,
    )


def _mixed_artifact(jit_root, tmp_path):
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, source_env, legacy, legacy_rows, _ = _legacy_artifact(
        jit_root, tmp_path
    )
    source_unified = UnifiedTubeRSIEnv(up_config, down_config, legacy)
    upstream_state = jax.jit(source_unified.reset_tube_index)(
        np.int32(0), np.int32(0)
    )
    downstream_state = jax.jit(source_unified.reset_tube_index)(
        np.int32(1), np.int32(0)
    )
    upstream_snapshot = _contextualized_snapshot(
        upstream_state, source_unified, phase="upstream", marker="up"
    )
    downstream_snapshot = _contextualized_snapshot(
        downstream_state, source_unified, phase="downstream", marker="down"
    )

    up_path = tmp_path / "unified-up"
    down_path = tmp_path / "unified-down"
    save_unified_envelope_snapshot(up_path, upstream_snapshot)
    save_unified_envelope_snapshot(down_path, downstream_snapshot)

    rows = [
        legacy_rows[0],
        _entry(
            phase="upstream",
            snapshot=up_path,
            state_sha256=unified_state_sha256(upstream_snapshot),
            score=0.95,
            value_model_target="C_up^0",
            expansion=True,
        ),
        legacy_rows[1],
        _entry(
            phase="downstream",
            snapshot=down_path,
            state_sha256=unified_state_sha256(downstream_snapshot),
            score=0.95,
            value_model_target="C_down^0",
            expansion=True,
        ),
    ]
    artifact = SoftTubeArtifact(
        tmp_path / "mixed-soft-tube",
        {
            "schema": SOFT_TUBE_SCHEMA,
            "status": "completed",
            "test_data_used": False,
            "validation_data_used": False,
            "phase_mixture": PHASE_MIXTURE,
        },
        tuple(rows),
        {},
    )
    return up_config, down_config, artifact, {
        "upstream": upstream_snapshot,
        "downstream": downstream_snapshot,
    }


@pytest.mark.gpu
def test_mixed_snapshot_pool_contract_is_jit_vmap_compatible(jit_root, tmp_path):
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, artifact, _ = _mixed_artifact(jit_root, tmp_path)
    env = UnifiedTubeRSIEnv(up_config, down_config, artifact)
    indices = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    sample = jax.jit(jax.vmap(env.tube_pool.snapshot_pool.sample_at_index))(indices)

    np.testing.assert_array_equal(
        np.asarray(sample["preserve_unified_context"]),
        np.asarray([False, True, False, True]),
    )
    assert int(sample["episode_step"][1]) == 13
    assert int(sample["phase_episode_step"][1]) == 7
    assert int(sample["episode_step"][3]) == 21
    assert int(sample["phase_episode_step"][3]) == 4
    assert bool(sample["down_events"]["airborne_seen"][3])
    assert bool(sample["down_events"]["valid_contact_seen"][3])


@pytest.mark.gpu
def test_unified_expansion_reset_preserves_exact_context_and_next_step(
    jit_root, tmp_path
):
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, artifact, snapshots = _mixed_artifact(jit_root, tmp_path)
    env = UnifiedTubeRSIEnv(up_config, down_config, artifact)
    reset = jax.jit(env.reset_tube_index)
    step = jax.jit(env.step)
    action = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)

    for phase_index, phase in ((0, "upstream"), (1, "downstream")):
        snapshot = snapshots[phase]
        restored = reset(np.int32(phase_index), np.int32(1))

        assert int(restored.info["active_phase"]) == snapshot.active_phase
        assert int(restored.info["start_phase"]) == snapshot.start_phase
        assert bool(restored.info["phase_transitioned"]) == snapshot.phase_transitioned
        assert int(restored.info["episode_step"]) == snapshot.episode_step
        assert int(restored.info["phase_episode_step"]) == snapshot.phase_episode_step
        assert float(restored.info["episode_return"]) == pytest.approx(
            snapshot.episode_return
        )
        for name, expected in snapshot.up_events.items():
            np.testing.assert_array_equal(
                np.asarray(getattr(restored.info["up_events"], name)),
                np.asarray(expected),
            )
        for name, expected in snapshot.down_events.items():
            np.testing.assert_array_equal(
                np.asarray(getattr(restored.info["down_events"], name)),
                np.asarray(expected),
            )

        canonical = restore_unified_envelope_snapshot(snapshot, env)
        next_restored = step(restored, action)
        next_canonical = step(canonical, action)

        np.testing.assert_allclose(
            np.asarray(next_restored.data.qpos),
            np.asarray(next_canonical.data.qpos),
            atol=1.0e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(next_restored.data.qvel),
            np.asarray(next_canonical.data.qvel),
            atol=1.0e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(next_restored.obs["state"]),
            np.asarray(next_canonical.obs["state"]),
            atol=1.0e-5,
            rtol=0.0,
        )
        assert float(next_restored.reward) == pytest.approx(
            float(next_canonical.reward), abs=1.0e-5
        )
        assert float(next_restored.done) == pytest.approx(
            float(next_canonical.done), abs=1.0e-6
        )


@pytest.mark.gpu
def test_mixed_snapshot_natural_reset_selection_keeps_one_pytree_contract(
    jit_root, tmp_path
):
    from jit_dvgc.unified_env import UnifiedTubeRSIEnv

    up_config, down_config, artifact, _ = _mixed_artifact(jit_root, tmp_path)
    env = UnifiedTubeRSIEnv(
        up_config,
        down_config,
        artifact,
        natural_reset_probability=0.5,
    )
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(99), 4))
    assert np.all(np.isfinite(np.asarray(states.data.qpos)))
    assert states.obs["state"].shape[0] == 4
