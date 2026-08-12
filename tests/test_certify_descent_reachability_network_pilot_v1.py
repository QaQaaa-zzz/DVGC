import pytest

from cli.certify_descent_reachability_network_pilot_v1 import _verify_runtime, final_safety_decision
from cli.run_backward_descent_nominal_pilot import EXPECTED
from dvgc.config import file_sha256, load_config


def test_legacy_runtime_verifier_fails_closed_after_model_migration():
    assert file_sha256(load_config("configs/default.json").xml_path) != EXPECTED["xml"]
    with pytest.raises(SystemExit, match="runtime/provenance gate failed"):
        _verify_runtime()


def test_final_safety_p1_does_not_require_chain_entry():
    branches = [
        {"final_recovery": True, "downstream_entry": False},
        {"final_recovery": True, "downstream_entry": True},
        {"final_recovery": True, "downstream_entry": False},
        {"final_recovery": False, "downstream_entry": True},
    ]
    decision = final_safety_decision(branches)
    assert decision == {"pass": True, "successes": 3, "branches": 4, "reasons": []}
