from copy import deepcopy

from cli.verify_descent_tube_v5 import verify_records
from dvgc.bank import beta_posterior, posterior_label


POLICY = "policy"
KWARGS = {
    "policy_identity": POLICY,
    "branches": 32,
    "min_branches": 8,
    "safe_threshold": .7,
    "dead_threshold": .3,
    "boundary_max_width": .35,
}


def _outcome(successes):
    posterior = beta_posterior(successes, 32 - successes)
    return {
        "successes": successes,
        "failures": 32 - successes,
        "branches": 32,
        "posterior": posterior,
        "label": posterior_label(
            posterior, 32, min_branches=8, safe_threshold=.7,
            dead_threshold=.3, boundary_max_width=.35,
        ),
    }


def _row(record_id="state", final_successes=32):
    branches = []
    for index in range(32):
        final = index < final_successes
        branches.append({
            "branch_index": index,
            "branch_seed": 1000 + index,
            "seed_namespace": "independent",
            "final_recovery": final,
            "chain_success": index < 20,
            "end_reason": "recovery" if final else "roll_limit",
        })
    return {
        "id": record_id,
        "artifact_role": "certified_tube",
        "certified_safe": True,
        "safe_claim_allowed": True,
        "training_only": False,
        "tube_metrics_eligible": True,
        "independent_audit": True,
        "policy_version": POLICY,
        "final": _outcome(final_successes),
        "chain": _outcome(20),
        "certification_branches": branches,
    }


def test_posterior_safe_state_need_not_be_exact_32_of_32():
    summary, reasons = verify_records([_row(final_successes=28)], **KWARGS)
    assert reasons == []
    assert summary["exact_32_of_32_states"] == 0
    assert summary["posterior_safe_with_failures_states"] == 1
    assert summary["physical_failure_reasons"] == {"roll_limit": 4}


def test_verifier_rejects_evidence_mismatch_and_seed_reuse():
    first = _row("first")
    second = _row("second")
    second["final"]["successes"] = 31
    summary, reasons = verify_records([first, second], **KWARGS)
    assert summary["unique_branch_seeds"] == 32
    assert any("Final posterior/evidence mismatch" in reason for reason in reasons)
    assert "branch seeds absent or duplicated across Tube" in reasons


def test_timeout_is_not_allowed_inside_certified_safe_evidence():
    row = deepcopy(_row(final_successes=28))
    row["certification_branches"][-1]["end_reason"] = "timeout"
    _, reasons = verify_records([row], **KWARGS)
    assert any("forbidden timeout branch" in reason for reason in reasons)
