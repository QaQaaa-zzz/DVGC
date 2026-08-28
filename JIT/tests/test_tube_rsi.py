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
