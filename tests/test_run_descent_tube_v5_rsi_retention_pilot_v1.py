import copy

from dvgc.bank import SnapshotBank
from cli.run_descent_tube_v5_rsi_retention_pilot_v1 import (
    FRONTIER,
    TUBE,
    acceptance,
    build_training_bank,
)


def test_training_bank_uses_only_safe_tube_states_and_balances_regions():
    tube = SnapshotBank.load(TUBE)
    frontier = SnapshotBank.load(FRONTIER)
    bank = build_training_bank(tube, "hash", frontier, "frontier-hash")
    assert len(bank.records) == len(tube.records) + len(frontier.records) == 27
    assert abs(sum(row["reset_weight"] for row in bank.records) - 1.0) < 1e-7
    assert {row["descent_layer"] for row in bank.records} == {"early", "middle", "late"}
    assert all(row["artifact_role"] == "proposal_support_bank" for row in bank.records)
    safe = [row for row in bank.records if row["bootstrap_group"] == "provisional_safe"]
    boundary = [row for row in bank.records if row["bootstrap_group"] == "boundary"]
    assert abs(sum(row["reset_weight"] for row in safe) - .8) < 1e-7
    assert abs(sum(row["reset_weight"] for row in boundary) - .2) < 1e-7
    assert len(boundary) == 3 and all(row["construction_screen_only"] for row in boundary)
    assert all("certified_safe" not in row and "safe_claim_allowed" not in row for row in safe)


def test_acceptance_requires_retention_drift_and_improvement():
    baseline = {"P1_states": 10, "final_branches": 30,
                "regions": {name: {"P1": 2} for name in ("early", "middle", "late")}}
    final = copy.deepcopy(baseline); final["final_branches"] = 31
    frontier_before = {"P1_states": 0, "final_branches": 0}
    frontier_after = {"P1_states": 1, "final_branches": 3}
    checks = acceptance(baseline, final, {"rms": .01, "max": .04}, True,
                        frontier_before, frontier_after)
    assert all(checks.values())
    final["final_branches"] = 29
    assert not all(acceptance(baseline, final, {"rms": .01, "max": .04}, True,
                              frontier_before, frontier_after).values())
