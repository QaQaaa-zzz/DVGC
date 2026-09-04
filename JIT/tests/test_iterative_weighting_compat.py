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
