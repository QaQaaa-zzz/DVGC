from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _cell_masses(groups, labels, weights):
    result = {}
    for group, label in sorted(set(zip(groups, labels))):
        mask = np.asarray(
            [g == group and int(y) == int(label) for g, y in zip(groups, labels)]
        )
        result[(group, int(label))] = float(np.sum(weights[mask]))
    return result


def test_observed_cell_weighting_preserves_historical_mixed_cell_behavior() -> None:
    from jit_dvgc.iterative_weighting_compat import observed_cell_balanced_weights
    from jit_dvgc.policy_conditioned_continuation_field import _cell_balanced_weights

    groups = ("g1", "g1", "g1", "g2", "g2", "g2", "g2", "g2")
    y = np.asarray([0, 1, 1, 0, 0, 1, 1, 1], dtype=np.float32)
    expected = _cell_balanced_weights(groups, y)
    actual = observed_cell_balanced_weights(groups, y)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_observed_cell_weighting_accepts_outcome_pure_parent_without_fabrication() -> None:
    from jit_dvgc.iterative_weighting_compat import observed_cell_balanced_weights
    from jit_dvgc.policy_conditioned_continuation_field import _cell_balanced_weights

    groups = ("pure_positive", "pure_positive", "pure_positive", "mixed", "mixed", "mixed")
    y = np.asarray([1, 1, 1, 0, 0, 1], dtype=np.float32)

    with pytest.raises(ValueError, match="both labels in every parent group"):
        _cell_balanced_weights(groups, y)

    weights = observed_cell_balanced_weights(groups, y)
    masses = _cell_masses(groups, y, weights)
    assert set(masses) == {
        ("pure_positive", 1),
        ("mixed", 0),
        ("mixed", 1),
    }
    assert max(masses.values()) - min(masses.values()) < 1e-6
    assert np.isclose(np.mean(weights), 1.0)
    assert np.isfinite(weights).all()
    assert np.all(weights > 0.0)


def test_prefit_archive_preserves_empty_failure_directory(tmp_path: Path) -> None:
    from jit_dvgc.iterative_weighting_compat import _archive_empty_prefit_failure

    output = tmp_path / "continuation_C1"
    output.mkdir()
    archive = _archive_empty_prefit_failure(output)
    assert archive == tmp_path / "continuation_C1_failed_prefit_parent_cell_contract"
    assert not output.exists()
    marker = json.loads((archive / "engineering_failure.json").read_text())
    assert marker["status"] == "preserved_engineering_failure_before_model_fit"
    assert marker["model_parameters_written"] is False
    assert marker["training_rows_changed"] is False


def test_prefit_archive_refuses_nonempty_partial_fit(tmp_path: Path) -> None:
    from jit_dvgc.iterative_weighting_compat import _archive_empty_prefit_failure

    output = tmp_path / "continuation_C1"
    output.mkdir()
    (output / "partial.txt").write_text("preserve me")
    with pytest.raises(FileExistsError, match="nonempty incomplete"):
        _archive_empty_prefit_failure(output)
    assert (output / "partial.txt").read_text() == "preserve me"


def test_standard_iterative_profile_is_76x64x64x1_and_legacy_is_unchanged() -> None:
    from jit_dvgc.iterative_continuation_fields import MODEL_PROFILES

    legacy = MODEL_PROFILES["legacy_tiny_tanh"]
    standard = MODEL_PROFILES["standard_mlp_64x64_tanh"]
    assert legacy["hidden_sizes"] == [8]
    assert legacy["parameter_count"] == 625
    assert legacy["architecture"] == "76->8_tanh->1"
    assert standard["hidden_sizes"] == [64, 64]
    assert standard["parameter_count"] == 9153
    assert standard["architecture"] == "76->64_tanh->64_tanh->1"
    assert standard["post_failure_architecture_revision"] is True


def test_iterative_score_supports_standard_two_hidden_layer_field(tmp_path: Path) -> None:
    from jit_dvgc.iterative_continuation_fields import _score

    field = tmp_path / "field.npz"
    np.savez(
        field,
        mean=np.zeros((76,), dtype=np.float32),
        std=np.ones((76,), dtype=np.float32),
        w1=np.zeros((76, 64), dtype=np.float32),
        b1=np.zeros((64,), dtype=np.float32),
        w2=np.zeros((64, 64), dtype=np.float32),
        b2=np.zeros((64,), dtype=np.float32),
        w3=np.zeros((64,), dtype=np.float32),
        b3=np.asarray(0.0, dtype=np.float32),
    )
    rows = [{"actor_observation": [0.0] * 76}]
    scores = _score(field, rows)
    assert scores.shape == (1,)
    np.testing.assert_allclose(scores, np.asarray([0.5]), rtol=0.0, atol=1e-12)
