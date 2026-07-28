import pytest
from cli.audit_descent_natural_bridge_candidates_v1 import union_records

def test_union_is_non_overwriting_and_rejects_identity_collision():
 assert [r['id'] for r in union_records([{'id':'a'}],[{'id':'b'}])]==['a','b']
 with pytest.raises(ValueError):union_records([{'id':'a'}],[{'id':'a'}])
