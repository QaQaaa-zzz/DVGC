import pickle

from cli.audit_descent_balanced_p1 import _records, _scale
from dvgc.bank import SnapshotBank

import numpy as np


def test_robust_scale_never_drops_below_physical_floor():
    values = np.asarray([[0., 1.], [0., 2.], [0., 100.]])
    result = _scale(values, np.asarray([.1, .2]))
    assert result[0] == .1
    assert result[1] >= .2


def test_mixed_v4_artifact_is_not_loaded_as_snapshot_bank(tmp_path, monkeypatch):
    path = tmp_path / "timing_explicit_snapshots.pkl"
    path.write_bytes(pickle.dumps([{"snapshot_v4": {"physical_feature": [1.]}}]))
    monkeypatch.setattr(SnapshotBank, "load", lambda _: (_ for _ in ()).throw(AssertionError()))
    rows = {"rows": [{"source_artifact": str(path), "source_index": 0,
                       "physical_state_sha256": "state"}]}
    assert _records(rows)["state"]["record"]["physical_feature"] == [1.]
