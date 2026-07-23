from dvgc.rollout import inferred_apex_seen


def test_dynamic_snapshot_with_none_reference_index_uses_explicit_apex_latch():
    assert inferred_apex_seen({"reference_index": None, "apex_seen": 1}) == 1
    assert inferred_apex_seen({"reference_index": None, "apex_seen": 0}) == 0


def test_apex_latch_fallback_is_none_safe():
    assert inferred_apex_seen({"reference_index": None, "source_index": None}) == 0
    assert inferred_apex_seen({"reference_index": 220}) == 1
    assert inferred_apex_seen({"source_index": 219}) == 0
