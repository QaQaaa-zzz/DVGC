from cli.audit_descent_balanced_p1 import _scale

import numpy as np


def test_robust_scale_never_drops_below_physical_floor():
    values = np.asarray([[0., 1.], [0., 2.], [0., 100.]])
    result = _scale(values, np.asarray([.1, .2]))
    assert result[0] == .1
    assert result[1] >= .2
