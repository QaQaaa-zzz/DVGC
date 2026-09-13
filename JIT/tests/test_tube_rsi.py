from __future__ import annotations

import jax
import numpy as np
import pytest

from jit_dvgc.soft_tube import SoftTubeArtifact, build_soft_tube

from .test_soft_tube import COMPATIBILITY, _fixture


def _artifact(tmp_path):
    return build_soft_tube(
        _fixture(tmp_path),
        tmp_path / "soft_tube",
        score_up=lambda _model, rows: np.asarray([0.05, 0.95]),
        score_down=lambda _model, rows: np.asarray([0.5]),
    )


def _core_replay_artifact(tmp_path):
    artifact = _artifact(tmp_path)
    upstream = [dict(row) for row in artifact.entries if row["phase"] == "upstream"]
    downstream = [dict(row) for row in artifact.entries if row["phase"] == "downstream"]
    assert len(upstream) == 2
    assert len(downstream) == 1
    entries = (
        upstream[0],
        downstream[0],
        upstream[1],
        dict(downstream[0]),
    )
    manifest = {
        **artifact.manifest,
        "artifact_role": "policy_conditioned_core_retaining_soft_tube_iteration",
        "source_tube_entry_count": 2,
        "core_retained_count": 2,
        "expansion_count": 2,
        "entry_count": 4,
        "upstream_count": 2,
        "downstream_count": 2,
    }
    return SoftTubeArtifact(artifact.root, manifest, entries, artifact.diagnostics)


def _core_replay_contract():
    return {
        "schema": "jit_tube_rsi_core_replay_v1",
        "selection": "phase_then_source_then_entry",
        "core_probability": 0.5,
        "expansion_probability": 0.5,
        "core_within_source": "uniform",
        "expansion_within_source": "value_weighted",
        "source_core_definition": "first_core_retained_count_entries",
    }


def test_sampler_is_fixed_seed_deterministic_and_phase_balanced(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool

    pool = TubeRSIPool.from_artifact(_artifact(tmp_path), compatibility=COMPATIBILITY)
    keys = jax.random.split(jax.random.PRNGKey(101), 4096)
    first = jax.vmap(pool.sample)(keys)
    second = jax.vmap(pool.sample)(keys)

    np.testing.assert_array_equal(first["tube_phase"], second["tube_phase"])
    np.testing.assert_array_equal(
        first["tube_entry_index"], second["tube_entry_index"]
    )
    phases = np.asarray(first["tube_phase"])
    assert set(phases.tolist()) == {0, 1}
    assert abs(float(np.mean(phases)) - 0.5) < 0.04


def test_within_phase_sampling_preserves_low_score_support_and_favors_high_score(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool

    pool = TubeRSIPool.from_artifact(_artifact(tmp_path), compatibility=COMPATIBILITY)
    keys = jax.random.split(jax.random.PRNGKey(102), 4096)
    samples = jax.vmap(lambda key: pool.sample_phase(key, 0))(keys)
    indices = np.asarray(samples["tube_entry_index"])
    counts = np.bincount(indices, minlength=2)
    assert counts[0] > 0
    assert counts[1] > counts[0]


def test_source_balanced_replay_guarantees_half_phase_mass_to_retained_core(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool, describe_tube_sampling

    artifact = _core_replay_artifact(tmp_path)
    contract = _core_replay_contract()
    summary = describe_tube_sampling(artifact, contract)
    assert summary["phases"]["upstream"]["core_probability"] == pytest.approx(0.5)
    assert summary["phases"]["downstream"]["core_probability"] == pytest.approx(0.5)

    pool = TubeRSIPool.from_artifact(
        artifact,
        compatibility=COMPATIBILITY,
        core_replay_contract=contract,
    )
    up_probs = np.asarray(jax.nn.softmax(pool.upstream_sampling_logits))
    down_probs = np.asarray(jax.nn.softmax(pool.downstream_sampling_logits))
    assert float(np.sum(up_probs[: pool.upstream_core_count])) == pytest.approx(0.5)
    assert float(np.sum(down_probs[: pool.downstream_core_count])) == pytest.approx(0.5)
    assert pool.upstream_core_count == 1
    assert pool.downstream_core_count == 1


def test_fixed_index_sampling_restores_exact_saved_arrays_without_mutation(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool

    artifact = _artifact(tmp_path)
    pool = TubeRSIPool.from_artifact(artifact, compatibility=COMPATIBILITY)
    source_up = pool.snapshot_pool.snapshot(0).qpos.copy()
    source_down = pool.snapshot_pool.snapshot(pool.upstream_count).qpos.copy()

    up = pool.sample_at(0, 0)
    down = pool.sample_at(1, 0)

    np.testing.assert_array_equal(up["qpos"], source_up)
    np.testing.assert_array_equal(down["qpos"], source_down)
    assert int(up["tube_phase"]) == 0
    assert int(down["tube_phase"]) == 1
    assert int(up["tube_global_index"]) == 0
    assert int(down["tube_global_index"]) == pool.upstream_count
    np.testing.assert_array_equal(pool.snapshot_pool.snapshot(0).qpos, source_up)
    np.testing.assert_array_equal(
        pool.snapshot_pool.snapshot(pool.upstream_count).qpos, source_down
    )


def test_sampler_accepts_policy_conditioned_fields_from_later_iterations(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool

    artifact = _artifact(tmp_path)
    entries = [dict(entry) for entry in artifact.entries]
    upstream = next(row for row in entries if row["phase"] == "upstream")
    upstream["value_model_target"] = "C_up^1"
    upstream["continuation_label"] = 1
    upstream["score_source"] = {
        "kind": "policy_conditioned_continuation_field",
        "field_name": "C_up^1",
        "acceptance_threshold_exclusive": float(upstream["value_score"]) - 1.0e-6,
        "selection_rule": "TRAIN_label_positive_and_score_strictly_greater_than_threshold",
    }
    later = SoftTubeArtifact(
        artifact.root, artifact.manifest, tuple(entries), artifact.diagnostics
    )
    TubeRSIPool.from_artifact(later, compatibility=COMPATIBILITY)

    upstream["value_model_target"] = "C_down^1"
    upstream["score_source"] = {
        **upstream["score_source"],
        "field_name": "C_down^1",
    }
    invalid = SoftTubeArtifact(
        artifact.root, artifact.manifest, tuple(entries), artifact.diagnostics
    )
    with pytest.raises(ValueError, match="cross-phase or invalid continuation field"):
        TubeRSIPool.from_artifact(invalid, compatibility=COMPATIBILITY)


def test_sampler_accepts_verified_policy_family_landing_train_support(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool

    artifact = _artifact(tmp_path)
    entries = [dict(entry) for entry in artifact.entries]
    expansion = next(row for row in entries if row["phase"] == "upstream")
    expansion.update(
        value_model_target="policy_family_first_valid_landing",
        value_score=1.0,
        sampling_weight=1.0,
        continuation_label=1,
        jump_start_reachability_proven=True,
        source_train_role_manifest_sha256="b" * 64,
        score_source={
            "kind": "observed_policy_family_first_valid_landing_label",
            "policy_family_sha256": "a" * 64,
            "selection_rule": "TRAIN_label_positive",
            "threshold_source": None,
            "fitted_classifier_used": False,
        },
    )
    landing = SoftTubeArtifact(
        artifact.root, artifact.manifest, tuple(entries), artifact.diagnostics
    )

    TubeRSIPool.from_artifact(landing, compatibility=COMPATIBILITY)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda entry: entry.update(split="validation"), "non-TRAIN"),
        (lambda entry: entry.update(sampling_weight=0.0), "positive sampling weight"),
        (lambda entry: entry.update(phase="apex"), "unsupported phase"),
    ],
)
def test_sampler_rejects_ineligible_artifact_entries(tmp_path, mutation, message):
    from jit_dvgc.tube_rsi import TubeRSIPool

    artifact = _artifact(tmp_path)
    entries = [dict(entry) for entry in artifact.entries]
    mutation(entries[0])
    invalid = SoftTubeArtifact(
        artifact.root, artifact.manifest, tuple(entries), artifact.diagnostics
    )
    with pytest.raises(ValueError, match=message):
        TubeRSIPool.from_artifact(invalid, compatibility=COMPATIBILITY)


def test_sampler_fails_closed_on_snapshot_compatibility_drift(tmp_path):
    from jit_dvgc.tube_rsi import TubeRSIPool

    with pytest.raises(ValueError, match="compatibility identity mismatch"):
        TubeRSIPool.from_artifact(
            _artifact(tmp_path), compatibility={"xml_sha256": "wrong"}
        )
