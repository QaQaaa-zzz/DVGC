import json

from cli.certify_descent_reachability_network_pilot_v1 import _verify_runtime, final_safety_decision


def test_runtime_verifier_reports_all_frozen_contracts_current():
    checks = _verify_runtime()
    assert checks
    assert all(checks.values()), json.dumps(checks, indent=2)


def test_final_safety_p1_does_not_require_chain_entry():
    branches = [
        {"final_recovery": True, "downstream_entry": False},
        {"final_recovery": True, "downstream_entry": True},
        {"final_recovery": True, "downstream_entry": False},
        {"final_recovery": False, "downstream_entry": True},
    ]
    decision = final_safety_decision(branches)
    assert decision == {"pass": True, "successes": 3, "branches": 4, "reasons": []}
