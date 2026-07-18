import pytest

from dvgc.snapshot_provenance import validate_snapshot_source_records


def test_declared_but_unobserved_policy_is_rejected():
    with pytest.raises(ValueError,match="do not match"):
        validate_snapshot_source_records([{"id":"old","snapshot_source_policy_hash":"old"}],
                                         {"snapshot_source_policy_hashes":["old","unused"]})
