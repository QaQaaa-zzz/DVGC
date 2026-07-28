import numpy as np
import pytest

from cli.build_descent_terminal_targets_from_tube_v1 import terminal_matrix


def test_terminal_matrix_uses_declared_finite_11d_features():
    rows = [{"physical_feature": np.arange(16, dtype=float)} for _ in range(3)]
    matrix = terminal_matrix(rows)
    assert matrix.shape == (3, 11)
    assert np.array_equal(matrix[0], [0, 2, 3, 4, 6, 8, 9, 10, 11, 13, 14])


def test_terminal_matrix_rejects_nonfinite():
    row = np.arange(16, dtype=float); row[3] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        terminal_matrix([{"physical_feature": row}])
