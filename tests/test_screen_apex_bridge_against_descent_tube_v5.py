from dvgc.bank import SnapshotBank
from cli.screen_apex_bridge_against_descent_tube_v5 import SOURCE, validate_source


def test_fixed_apex_bridge_source_is_parent_disjoint_and_physically_stable():
    source = SnapshotBank.load(SOURCE)
    validate_source(source.records)
    assert len(source.records) == 4
    assert len({row["trajectory_parent_id"] for row in source.records}) == 4
