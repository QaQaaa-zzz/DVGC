import json

from cli.certify_descent_reachability_network_pilot_v1 import _verify_runtime


def test_runtime_verifier_reports_all_frozen_contracts_current():
    checks = _verify_runtime()
    assert checks
    assert all(checks.values()), json.dumps(checks, indent=2)
