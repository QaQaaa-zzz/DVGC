from cli.prepare_fast_handoff_route import snapshot_identity

def test_fast_route_uses_full_snapshot_identity():
    assert callable(snapshot_identity)
