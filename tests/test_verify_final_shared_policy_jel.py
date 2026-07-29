from cli.verify_final_shared_policy_jel import STAGES, verify_rows


def _row(stage, index):
    construction = [{"seed": index * 100 + branch, "final_recovery": True,
                     "certification_round": "construction"} for branch in range(32)]
    audit = [{"seed": 10000 + index * 100 + branch, "final_recovery": True,
              "certification_round": "independent_audit"} for branch in range(32)]
    return {
        "id": stage, "phase_rsi_stage": stage, "artifact_role": "final_shared_policy_jel",
        "certified_safe": True, "safe_claim_allowed": True,
        "formal_shared_policy_jel": True, "training_only": False,
        "policy_params_sha256": "policy", "final": {
            "label": "safe", "successes": 64, "failures": 0, "branches": 64,
        },
        "construction_final_branches": construction,
        "independent_audit_final_branches": audit,
        "certification_branches": construction + audit,
    }


def test_complete_five_phase_rows_pass_structure_verification():
    counts, reasons = verify_rows([_row(stage, index) for index, stage in enumerate(STAGES)], "policy")
    assert counts == {stage: 1 for stage in STAGES}
    assert reasons == []


def test_missing_phase_and_nonfinal_branch_are_rejected():
    rows = [_row(stage, index) for index, stage in enumerate(STAGES[:-1])]
    rows[0]["certification_branches"][0]["final_recovery"] = False
    _, reasons = verify_rows(rows, "policy")
    assert any("non-Final branch" in reason for reason in reasons)
    assert "all five phases are not represented" in reasons
