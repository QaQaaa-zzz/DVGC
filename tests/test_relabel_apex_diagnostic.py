import numpy as np

from cli.relabel_apex_policies import support_diagnostic


def test_support_diagnostic_uses_per_anchor_radius_and_contributions():
    metadata = {"support_features": [[0., 0.], [10., 0.]], "stage_entry_matcher": {
        "scale": [1., 2.], "radii": [.5, 2.], "radius": 2.,
    }}
    result = support_diagnostic(np.asarray([11., 2.]), metadata)
    assert result["anchor_index"] == 1
    assert result["distance"] == np.sqrt(2.) / 2.
    assert result["squared_scaled_contributions"] == [0.25, 0.25]
