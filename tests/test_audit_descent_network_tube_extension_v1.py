import pytest

from cli.audit_descent_network_tube_extension_v1 import union_records


def test_tube_union_preserves_base_and_adds_new_identity():
    merged = union_records([{"id": "base", "value": 1}], [{"id": "new", "value": 2}])
    assert {row["id"] for row in merged} == {"base", "new"}


def test_tube_union_rejects_identity_collision():
    with pytest.raises(ValueError, match="collision"):
        union_records([{"id": "same"}], [{"id": "same"}])
