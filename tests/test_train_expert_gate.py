from cli.train_expert import expert_gate_passed


def test_chain_only_gate_does_not_require_landing_retention_or_composite_final():
    report = {"chain_rate": .05, "composite_final_rate": 0.0, "timeout_rate": 0.0}
    target = {"chain_rate": .10}
    assert expert_gate_passed("chain_only", report, target, False, [])
    assert not expert_gate_passed("legacy_composite", report, target, False, [])


def test_chain_only_gate_still_rejects_timeout_or_missing_target_support():
    report = {"chain_rate": .05, "composite_final_rate": .20, "timeout_rate": .01}
    assert not expert_gate_passed("chain_only", report, {"chain_rate": .10}, True, [])
    report["timeout_rate"] = 0.0
    assert not expert_gate_passed("chain_only", report, {"chain_rate": 0.0}, True, [])
