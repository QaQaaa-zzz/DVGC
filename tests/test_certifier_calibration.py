from dvgc.certifier_calibration import (
    current_conjunction, pooled_heterogeneity, sequential_confidence,
    protocol_hash, wilson_lower,
)


def branches(successes, total=32):
    return [{"final_recovery": i < successes} for i in range(total)]


def test_rules_keep_target_probability_fixed_and_detect_heterogeneity():
    stable = branches(32)
    assert current_conjunction(stable, .7)
    assert pooled_heterogeneity(stable, .7)
    assert sequential_confidence(stable, .7)
    heterogeneous = ([{"final_recovery": True}] * 16 +
                     [{"final_recovery": i < 8} for i in range(16)])
    assert not pooled_heterogeneity(heterogeneous, .7)


def test_wilson_and_protocol_hash_are_deterministic():
    assert 0 <= wilson_lower(8, 8) <= 1
    assert protocol_hash("stage_conjunction", .7) == protocol_hash("stage_conjunction", .7)
    assert protocol_hash("stage_conjunction", .7) != protocol_hash("sequential_adaptive", .7)
