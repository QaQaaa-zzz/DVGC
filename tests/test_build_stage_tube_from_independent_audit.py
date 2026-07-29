from cli.build_stage_tube_from_independent_audit import exact_safe


def test_exact_safe_requires_all_preregistered_branches():
    assert exact_safe({"label": "positive", "s": 32, "n": 32}, 32)
    assert not exact_safe({"label": "positive", "s": 31, "n": 32}, 32)
    assert not exact_safe({"label": "boundary", "s": 32, "n": 32}, 32)
    assert not exact_safe({"label": "positive", "s": 8, "n": 8}, 32)
